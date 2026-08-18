#!/usr/bin/env python3
"""组合级回测 CLI:美股单边多头 Top-N + 逆波动率加权,周频调仓。"""
import argparse

from engine.config import (DEFAULT_UNIVERSE, DEFAULT_TOP_N, MOMENTUM_LOOKBACK,
                           MOMENTUM_SKIP, VOL_WINDOW, DEFAULT_COMMISSION,
                           DEFAULT_SLIPPAGE, DEFAULT_CAPITAL, DATA_DIR)
from engine.data import load_panel, slice_panel
from engine.factors import momentum, realized_vol
from engine.strategy import StrategyConfig
from engine.backtest import CostModel, run_backtest
from engine.evaluation import build_metrics, equal_weight_buyhold


def main():
    p = argparse.ArgumentParser(description="组合级回测:美股单边多头 Top-N + 逆波动率加权,周频调仓")
    p.add_argument("--symbols", nargs="*", default=DEFAULT_UNIVERSE)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--sell-band", type=int, default=0, help="卖出阈值放宽幅度(滞后带)")
    p.add_argument("--min-holding-weeks", type=int, default=0, help="最短持仓周数")
    p.add_argument("--momentum-lookback", type=int, default=MOMENTUM_LOOKBACK)
    p.add_argument("--momentum-skip", type=int, default=MOMENTUM_SKIP)
    p.add_argument("--vol-window", type=int, default=VOL_WINDOW)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    p.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    p.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    args = p.parse_args()

    panel = load_panel(args.symbols, DATA_DIR)
    factor_scores = {
        "momentum": momentum(panel, args.momentum_lookback, args.momentum_skip),
        "realized_vol": realized_vol(panel, args.vol_window),
    }

    # 截断预热期:从因子第一个有效日(动量与波动率均非 NaN)开始,
    # 避免策略 warmup 期空仓 vs benchmark 满仓的不公平对比。
    valid = factor_scores["momentum"].notna() & factor_scores["realized_vol"].notna()
    has_any = valid.any(axis=1)
    if has_any.any():
        start = has_any[has_any].index[0]
        panel = slice_panel(panel, start)
        factor_scores = {k: v.loc[start:] for k, v in factor_scores.items()}

    cfg = StrategyConfig(top_n=args.top_n, sell_band=args.sell_band,
                         min_holding_weeks=args.min_holding_weeks)

    res_gross = run_backtest(panel, factor_scores, cfg,
                             CostModel(commission=0.0, slippage=0.0), args.capital)
    res_net = run_backtest(panel, factor_scores, cfg,
                           CostModel(commission=args.commission, slippage=args.slippage),
                           args.capital)
    bench = equal_weight_buyhold(panel, args.capital)

    m_gross = build_metrics(res_gross.nav, args.capital, res_gross.turnovers)
    m_net = build_metrics(res_net.nav, args.capital, res_net.turnovers)
    m_bench = build_metrics(bench, args.capital, [])

    rows = [("策略(不含成本)", m_gross), ("策略(含成本)", m_net), ("等权 buy&hold", m_bench)]

    print(f"标的: {', '.join(panel.symbols)}")
    print(f"参数: Top-N={cfg.top_n}, 滞后带 {cfg.sell_band}, 最短持仓 {cfg.min_holding_weeks}周, "
          f"动量 {args.momentum_lookback}/{args.momentum_skip}, "
          f"波动率窗口 {args.vol_window}, 佣金 {args.commission:.4%}, 滑点 {args.slippage:.4%}")
    print("-" * 64)
    print(f"{'':<16}{'年化收益':>12}{'Sharpe':>10}{'最大回撤':>12}{'年化换手':>12}")
    for label, m in rows:
        to = m["turnover_annual"]
        to_str = f"{to:>10.0%}" if to else f"{'—':>12}"
        print(f"{label:<16}{m['annual_return']:>11.2%}{m['sharpe']:>10.2f}"
              f"{m['max_drawdown']:>11.2%}{to_str}")
    print("-" * 64)
    print(f"回测区间: {panel.dates[0].date()} ~ {panel.dates[-1].date()} "
          f"({len(panel.dates)} 个交易日, {len(res_net.turnovers)} 次调仓)")


if __name__ == "__main__":
    main()
