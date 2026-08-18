#!/usr/bin/env python3
"""Walk-forward 参数寻优:滚动训练/验证,评估换手控制参数是否稳健。

回答的核心问题:网格里那个「收益最高」的参数点是真 alpha 还是运气?

方法(handoff 第 1 / 5 条):
  - 多折滚动:每折只用 train 段选参数,只在 valid 段看结果(严格隔离)。
  - 记录全部搜索轮次,不只看最优。
  - 评估 train/valid 排名相关性:相关 ≈ 0 说明「在 train 上挑参数」这个动作
    本身是高噪音,那个最高点不可信。

用法:
  .venv/bin/python walk_forward.py
"""
import sys
from itertools import product

import pandas as pd

from engine.config import DEFAULT_UNIVERSE, DATA_DIR, MOMENTUM_LOOKBACK, MOMENTUM_SKIP, VOL_WINDOW
from engine.data import load_panel, slice_panel
from engine.factors import momentum, realized_vol
from engine.strategy import StrategyConfig
from engine.backtest import CostModel, run_backtest
from engine.evaluation import build_metrics


def cfg_key(cfg):
    return f"band{cfg.sell_band}_min{cfg.min_holding_weeks}"


def walk_forward(panel, factor_scores, param_grid, cost, capital, n_folds=3):
    """滚动 walk-forward。每折:train 段跑全部参数选 best,valid 段跑全部参数算排名相关。"""
    n = len(panel.dates)
    valid_len = n // (n_folds + 1)
    results = []

    for f in range(n_folds):
        train_end = n - (n_folds - f) * valid_len
        valid_end = min(train_end + valid_len, n)

        train_panel = slice_panel(panel, end=panel.dates[train_end - 1])
        valid_panel = slice_panel(panel, start=panel.dates[train_end],
                                  end=panel.dates[valid_end - 1])

        train_sharpe, valid_sharpe = {}, {}
        for cfg in param_grid:
            key = cfg_key(cfg)
            r = run_backtest(train_panel, factor_scores, cfg, cost, capital)
            train_sharpe[key] = build_metrics(r.nav, capital, r.turnovers)["sharpe"]
            r = run_backtest(valid_panel, factor_scores, cfg, cost, capital)
            valid_sharpe[key] = build_metrics(r.nav, capital, r.turnovers)["sharpe"]

        best_key = max(train_sharpe, key=train_sharpe.get)
        # Spearman:用 rank 后的 Pearson 相关(纯 pandas,无需 scipy)
        corr = pd.Series(train_sharpe).rank().corr(pd.Series(valid_sharpe).rank())

        results.append({
            "fold": f,
            "train_range": f"{train_panel.dates[0].date()}~{train_panel.dates[-1].date()}",
            "valid_range": f"{valid_panel.dates[0].date()}~{valid_panel.dates[-1].date()}",
            "train_best": best_key,
            "valid_best_sharpe": valid_sharpe[best_key],
            "rank_corr": corr,
            "train_sharpe": train_sharpe,
            "valid_sharpe": valid_sharpe,
        })
    return results


def main():
    cost = CostModel(commission=0.0, slippage=0.0005)
    capital = 1_000_000

    panel = load_panel(DEFAULT_UNIVERSE, DATA_DIR)
    factor_scores = {"momentum": momentum(panel, MOMENTUM_LOOKBACK, MOMENTUM_SKIP),
                     "realized_vol": realized_vol(panel, VOL_WINDOW)}
    valid = factor_scores["momentum"].notna() & factor_scores["realized_vol"].notna()
    has = valid.any(axis=1)
    start = has[has].index[0]
    panel = slice_panel(panel, start)
    factor_scores = {k: v.loc[start:] for k, v in factor_scores.items()}

    param_grid = [StrategyConfig(top_n=5, sell_band=b, min_holding_weeks=m)
                  for b, m in product([0, 1, 2, 3], [0, 2, 4, 8])]

    results = walk_forward(panel, factor_scores, param_grid, cost, capital, n_folds=3)

    print("=" * 70)
    print("Walk-forward 参数寻优 (3 折滚动, 选参指标 = 含成本 Sharpe)")
    print("=" * 70)
    best_keys = []
    for r in results:
        best_keys.append(r["train_best"])
        print(f"\n折 {r['fold'] + 1}:")
        print(f"  train {r['train_range']}  ->  valid {r['valid_range']}")
        print(f"  train 选出的最优: {r['train_best']}")
        print(f"  该参数在 valid 的 Sharpe: {r['valid_best_sharpe']:.2f}")
        print(f"  train/valid 排名相关(Spearman): {r['rank_corr']:+.2f}")

    print("\n" + "=" * 70)
    print("跨折汇总")
    print("=" * 70)
    print(f"3 折 train 选出的最优参数: {best_keys}")
    unique = set(best_keys)
    print(f"参数一致性: {len(unique)}/3 折选出相同参数 {'(稳健)' if len(unique) == 1 else '(不稳健,高噪音)'}")
    corrs = [r["rank_corr"] for r in results]
    print(f"train/valid 排名相关: {[f'{c:+.2f}' for c in corrs]}  均值 {sum(corrs)/len(corrs):+.2f}")

    # 关键判断:如果排名相关接近 0 或为负,说明「在 train 挑参数」无预测力
    avg_corr = sum(corrs) / len(corrs)
    if avg_corr < 0.2:
        verdict = "排名相关接近 0 → train 上的「最优参数」对 valid 几乎没有预测力,网格最高点是运气"
    else:
        verdict = "排名相关为正 → 参数选择有一定稳健性,但仍需更长数据确认"
    print(f"\n结论: {verdict}")


if __name__ == "__main__":
    main()
