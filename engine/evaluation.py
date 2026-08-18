"""评估指标 + benchmark。"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from backtest_ma_cross import annualized_return  # 复用已修的自然日年化公式


def max_drawdown(nav):
    peak = nav.cummax()
    return float(((peak - nav) / peak).max())


def sharpe(nav, rf=0.0, ann=252):
    """日收益年化 Sharpe。注意:年化用 sqrt(252)(交易日口径),与 CAGR 的
    自然日口径不同 —— 这是刻意的,两者分别遵循各自行业惯例。"""
    daily = nav.pct_change().dropna()
    std = daily.std(ddof=1)
    if std == 0:
        return 0.0
    return float((daily.mean() - rf / ann) / std * (ann ** 0.5))


def turnover_annual(turnovers):
    """年化单边换手 = 平均单次单边换手 * 52(周)。"""
    if not turnovers:
        return 0.0
    avg = float(np.mean([t for _, t in turnovers]))
    return avg * 52.0


def equal_weight_buyhold(panel, capital):
    """首日开盘等额买入所有有数据的标的,持有到底,返回 NAV 序列。"""
    entry = panel.open.iloc[0]
    entry = entry[entry.notna()]
    dollars = capital / len(entry)
    shares = dollars / entry
    return (shares * panel.close[entry.index]).sum(axis=1)


def build_metrics(nav, capital, turnovers):
    natural_days = (nav.index[-1] - nav.index[0]).days
    return {
        "annual_return": annualized_return(float(nav.iloc[-1]), capital, natural_days),
        "sharpe": sharpe(nav),
        "max_drawdown": max_drawdown(nav),
        "turnover_annual": turnover_annual(turnovers),
    }
