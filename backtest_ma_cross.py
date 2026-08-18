#!/usr/bin/env python3
"""
双均线交叉策略回测 Demo
========================
策略逻辑:
  快线上穿慢线(金叉) -> 买入
  快线下穿慢线(死叉) -> 卖出

关键设计 —— 避免「未来函数」(look-ahead bias):
  信号用「当日收盘价」算出,但成交放在「次日开盘价」。
  因为现实中你在 T 日收盘那一刻才知道均线已经交叉,不可能再回到收盘价去成交,
  最早也只能 T+1 开盘才下单。直接用当日收盘价成交 = 偷看未来,回测收益会虚高。

用法:
  python3 backtest_ma_cross.py --data data/AAPL.US.csv --fast 20 --slow 60
"""

import argparse
import csv
from datetime import date


def load_csv(path):
    """读取 CSV,返回按时间升序的各字段列表。"""
    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dates.append(row["date"])
            opens.append(float(row["open"]))
            highs.append(float(row["high"]))
            lows.append(float(row["low"]))
            closes.append(float(row["close"]))
            volumes.append(int(row["volume"]))
    return dates, opens, highs, lows, closes, volumes


def moving_average(values, window):
    """简单移动平均(SMA),前 window-1 个位置返回 None。"""
    out = [None] * len(values)
    s = sum(values[:window])
    out[window - 1] = s / window
    for i in range(window, len(values)):
        s += values[i] - values[i - window]
        out[i] = s / window
    return out


def max_drawdown(equity):
    """最大回撤:从历史最高点跌到后续最低点的最大跌幅(比例,正数)。"""
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    return mdd


def annualized_return(final_equity, capital, natural_days):
    """按自然日数计算的年化收益(CAGR)。

    分母必须是自然日(日历日),不能用交易日 bar 数 —— 用 bar 数会把年化
    高估约 30%(如 31% vs 正确的 21%),这是本文件历史上修过的一个 bug。
    """
    if natural_days <= 0:
        return 0.0
    return (final_equity / capital) ** (365.0 / natural_days) - 1


def backtest(dates, opens, closes, fast_n, slow_n, capital, commission):
    """执行回测,返回 (equity曲线, 交易记录, 指标dict)。"""
    fast_ma = moving_average(closes, fast_n)
    slow_ma = moving_average(closes, slow_n)

    cash = float(capital)
    shares = 0.0
    in_position = False
    pending = None          # 前一日收盘产生的信号,今日开盘执行
    trades = []             # (date, side, price, shares)
    equity = []             # 每个交易日的收盘权益

    for i in range(len(dates)):
        # 1) 先执行前一日收盘确认的信号(今日开盘价成交)
        if pending == "buy":
            price = opens[i] * (1 + commission)       # 买入价含佣金上浮
            shares = cash / price
            cash = 0.0
            trades.append((dates[i], "BUY", opens[i], shares))
            in_position = True
        elif pending == "sell":
            price = opens[i] * (1 - commission)       # 卖出价含佣金下浮
            cash = shares * price
            trades.append((dates[i], "SELL", opens[i], shares))
            shares = 0.0
            in_position = False
        pending = None

        # 2) 记录当日收盘权益(现金 + 持仓市值)
        equity.append(cash + shares * closes[i])

        # 3) 用当日收盘价判断是否产生新信号(次日执行)
        #    慢线窗口更大,所以只要 slow_ma[i] 就绪,快线也必然就绪
        if slow_ma[i] is not None and slow_ma[i - 1] is not None:
            golden = fast_ma[i] > slow_ma[i] and fast_ma[i - 1] <= slow_ma[i - 1]
            death = fast_ma[i] < slow_ma[i] and fast_ma[i - 1] >= slow_ma[i - 1]
            if golden and not in_position:
                pending = "buy"
            elif death and in_position:
                pending = "sell"

    final_equity = equity[-1]

    # 统计:配对买卖,计算每轮盈亏与胜率
    wins = 0
    rounds = 0
    buy_price = None
    for _, side, price, _ in trades:
        if side == "BUY":
            buy_price = price
        elif side == "SELL" and buy_price is not None:
            rounds += 1
            if price > buy_price:
                wins += 1
            buy_price = None

    bars = len(dates)
    natural_days = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days
    total_ret = final_equity / capital - 1
    annual = annualized_return(final_equity, capital, natural_days)
    bh_ret = closes[-1] / closes[0] - 1

    return equity, trades, {
        "period": f"{dates[0]} ~ {dates[-1]}",
        "bars": bars,
        "natural_days": natural_days,
        "init_capital": capital,
        "final_equity": round(final_equity, 2),
        "total_return": total_ret,
        "annual_return": annual,
        "max_drawdown": max_drawdown(equity),
        "buy_hold_return": bh_ret,
        "num_trades": len([t for t in trades if t[1] in ("BUY", "SELL")]),
        "num_rounds": rounds,
        "win_rate": (wins / rounds) if rounds else 0.0,
        "trades": trades,
    }


def main():
    p = argparse.ArgumentParser(description="双均线交叉策略回测")
    p.add_argument("--data", required=True, help="CSV 路径,如 data/AAPL.US.csv")
    p.add_argument("--fast", type=int, default=20, help="快线窗口(默认 20)")
    p.add_argument("--slow", type=int, default=60, help="慢线窗口(默认 60)")
    p.add_argument("--capital", type=float, default=100000, help="初始资金(默认 10 万)")
    p.add_argument("--commission", type=float, default=0.0, help="单边佣金率(默认 0,如 0.0005=万5)")
    args = p.parse_args()

    dates, opens, highs, lows, closes, volumes = load_csv(args.data)
    _, trades, m = backtest(dates, opens, closes, args.fast, args.slow, args.capital, args.commission)

    print(f"回测区间: {m['period']}  (共 {m['bars']} 个交易日)")
    print(f"参数:     MA{args.fast} / MA{args.slow}, 佣金 {args.commission:.4%}")
    print("-" * 52)
    print(f"初始资金:   {m['init_capital']:>14,.2f}")
    print(f"期末权益:   {m['final_equity']:>14,.2f}")
    print(f"总收益率:   {m['total_return']:>13.2%}")
    print(f"年化收益率: {m['annual_return']:>13.2%}")
    print(f"最大回撤:   {m['max_drawdown']:>13.2%}")
    print(f"买入持有:   {m['buy_hold_return']:>13.2%}  (同期 benchmark)")
    print("-" * 52)
    print(f"交易笔数:   {m['num_trades']}  (完整买卖 {m['num_rounds']} 轮)")
    print(f"胜率:       {m['win_rate']:.1%}")
    print("-" * 52)

    if trades:
        print("交易明细 (T日收盘信号 -> T+1 开盘成交):")
        for date, side, price, sh in trades:
            print(f"  {date}  {side:4s}  {price:>8.2f}  x {sh:,.2f} 股")


if __name__ == "__main__":
    main()
