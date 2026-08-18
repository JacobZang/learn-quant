#!/usr/bin/env python3
"""趋势过滤 + 行业动量轮动(组合策略)。

思路:用 SPY 的 200 日均线做「市场开关」——
  - 市场在 MA200 之上(牛市)→ 正常做行业横截面动量 Top-N
  - 市场在 MA200 之下(熊市)→ 空仓转现金
把「横截面动量(选什么)」和「趋势过滤(做不做)」两个维度组合起来。

无未来函数:决策日 T 收盘判断市场状态 + 算行业动量,T+1 开盘成交。

用法:
  .venv/bin/python trend_filtered_rotation.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.config import DATA_DIR                                              # noqa: E402
from engine.data import load_panel, slice_panel                                 # noqa: E402
from engine.factors import momentum, realized_vol                               # noqa: E402
from engine.strategy import StrategyConfig, select_weights, rebalance_dates     # noqa: E402
from engine.backtest import CostModel, Portfolio, run_backtest                  # noqa: E402
from engine.evaluation import build_metrics, equal_weight_buyhold               # noqa: E402

SECTORS = ["XLK.US", "XLF.US", "XLE.US", "XLV.US", "XLI.US", "XLY.US",
           "XLP.US", "XLB.US", "XLRE.US", "XLU.US", "XLC.US"]


def run_filtered(panel, factors, cfg, cost, market_ok, capital):
    """市场过滤版:market_ok[t]=True 才做轮动,否则目标权重全 0(空仓)。"""
    rebal = set(rebalance_dates(panel.dates))
    port = Portfolio(capital, panel.symbols)
    pending = None
    navs, turnovers = [], []
    for t in panel.dates:
        if pending is not None:
            turn = port.execute(pending, panel.open.loc[t], cost)
            turnovers.append((t, turn))
            pending = None
        navs.append(port.nav(panel.close.loc[t]))
        if t in rebal:
            if market_ok[t]:
                pending = select_weights(panel, factors, t, cfg)
            else:
                pending = pd.Series(0.0, index=panel.symbols)
    return pd.Series(navs, index=panel.dates), turnovers


def yearly(nav):
    return nav.resample("YE").last().pct_change()


def main():
    panel = load_panel(SECTORS, DATA_DIR)
    factors = {"momentum": momentum(panel, 252, 21), "realized_vol": realized_vol(panel, 63)}

    spy = pd.read_csv(os.path.join("data", "SPY.US.csv"), parse_dates=["date"]).set_index("date")
    market_ok = spy["close"] > spy["close"].rolling(200).mean()  # SPY > MA200

    # 截断 warmup(行业动量 252 天)
    valid = factors["momentum"].notna() & factors["realized_vol"].notna()
    start = valid.any(axis=1)[valid.any(axis=1)].index[0]
    panel = slice_panel(panel, start)
    factors = {k: v.loc[start:] for k, v in factors.items()}
    market_ok = market_ok.reindex(panel.dates).ffill()

    cost = CostModel(0.0, 0.0005)
    cfg = StrategyConfig(top_n=5, sell_band=2, min_holding_weeks=4)
    capital = 1_000_000

    res_no = run_backtest(panel, factors, cfg, cost, capital)
    nav_filt, turns_filt = run_filtered(panel, factors, cfg, cost, market_ok, capital)
    bench = equal_weight_buyhold(panel, capital)

    nd = (panel.dates[-1] - panel.dates[0]).days
    print(f"行业轮动 + SPY MA200 趋势过滤  区间 {panel.dates[0].date()}~{panel.dates[-1].date()}")
    print("-" * 66)
    print(f"{'':<20}{'年化收益':>10}{'Sharpe':>9}{'最大回撤':>11}{'年化换手':>10}")
    for label, nav, turns in [
        ("行业动量(无过滤)", res_no.nav, res_no.turnovers),
        ("行业动量+趋势过滤", nav_filt, turns_filt),
        ("等权 buy&hold", bench, []),
    ]:
        m = build_metrics(nav, capital, turns)
        to = f"{m['turnover_annual']:>9.0%}" if turns else f"{'—':>10}"
        print(f"{label:<20}{m['annual_return']:>9.2%}{m['sharpe']:>9.2f}{m['max_drawdown']:>10.2%}{to}")
    print("-" * 66)

    print("\n分年收益(无过滤 vs 有过滤 vs 持有):")
    y_no, y_filt, y_bh = yearly(res_no.nav), yearly(nav_filt), yearly(bench)
    print(f"{'年份':>6}{'无过滤':>10}{'有过滤':>10}{'持有':>10}")
    for yr in y_no.index:
        a, b, c = y_no[yr], y_filt[yr], y_bh[yr]
        if pd.isna(a) or pd.isna(b) or pd.isna(c):
            continue
        print(f"{yr.year:>6}{a:>9.1%}{b:>9.1%}{c:>9.1%}")


if __name__ == "__main__":
    main()
