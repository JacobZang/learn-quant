#!/usr/bin/env python3
"""宽基动量择时(时序动量 / 趋势跟踪)。

策略:风险资产过去 N 日收益 > 0 → 全仓持有;否则转现金(0 收益)。
核心价值:趋势跟踪在熊市转现金,显著降低回撤(以牛市少量踏空为代价)。

无未来函数:决策日 T 收盘看动量(只用 <=T 数据),T+1 开盘成交。

用法:
  .venv/bin/python momentum_timing.py --symbol SPY.US --lookback 252
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.evaluation import annualized_return, max_drawdown, sharpe  # noqa: E402


def load(symbol):
    df = pd.read_csv(os.path.join("data", f"{symbol}.csv"), parse_dates=["date"])
    return df.set_index("date")


def run(symbol, lookback, capital=1_000_000, freq="M"):
    """返回时序动量策略的每日 NAV 序列。"""
    df = load(symbol)
    close, open_ = df["close"], df["open"]
    mom = close / close.shift(lookback) - 1  # 过去 lookback 日收益(向后看,安全)

    period = "M" if freq == "M" else "W"
    reb = set(pd.Series(0, index=close.index).groupby(close.index.to_period(period)).tail(1).index)

    cash, shares = float(capital), 0.0
    in_market, pending = False, None
    navs = []

    for t in close.index:
        if pending is not None:  # 执行前一日收盘定的信号(今日开盘)
            if pending and not in_market:
                shares = cash / open_[t]
                cash = 0.0
                in_market = True
            elif not pending and in_market:
                cash = shares * open_[t]
                shares = 0.0
                in_market = False
            pending = None
        navs.append(cash + shares * close[t])
        if t in reb and not pd.isna(mom[t]):
            pending = mom[t] > 0  # T 收盘定信号

    return pd.Series(navs, index=close.index)


def yearly_returns(nav):
    """年末 NAV 的年收益率。"""
    return nav.resample("YE").last().pct_change()


def main():
    p = argparse.ArgumentParser(description="宽基动量择时(时序动量)")
    p.add_argument("--symbol", default="SPY.US")
    p.add_argument("--lookback", type=int, default=252, help="动量回看天数(默认 252=12个月)")
    p.add_argument("--capital", type=float, default=1_000_000)
    args = p.parse_args()

    df = load(args.symbol)
    close = df["close"]
    nav = run(args.symbol, args.lookback, args.capital)
    bh = close / close.iloc[0] * args.capital

    natural_days = (nav.index[-1] - nav.index[0]).days
    rows = [
        ("动量择时", nav),
        ("买入持有", bh),
    ]
    print(f"标的: {args.symbol}  动量回看 {args.lookback} 日({args.lookback//21}个月)  "
          f"区间 {nav.index[0].date()}~{nav.index[-1].date()}")
    print("-" * 62)
    print(f"{'':<12}{'年化收益':>11}{'Sharpe':>9}{'最大回撤':>11}{'期末净值':>12}")
    for label, s in rows:
        m = {
            "ann": annualized_return(float(s.iloc[-1]), args.capital, natural_days),
            "sharpe": sharpe(s),
            "mdd": max_drawdown(s),
        }
        print(f"{label:<12}{m['ann']:>10.2%}{m['sharpe']:>9.2f}{m['mdd']:>10.2%}{s.iloc[-1]/args.capital:>11.3f}")
    print("-" * 62)

    # 分年收益,重点看熊市年份
    print("\n分年收益(择时 vs 持有):")
    y_strat = yearly_returns(nav)
    y_bh = yearly_returns(bh)
    print(f"{'年份':>6}{'择时':>10}{'持有':>10}{'超额':>10}")
    for yr in y_strat.index:
        s = y_strat[yr]
        b = y_bh[yr]
        if pd.isna(s) or pd.isna(b):
            continue
        print(f"{yr.year:>6}{s:>9.1%}{b:>9.1%}{s-b:>+9.1%}")


if __name__ == "__main__":
    main()
