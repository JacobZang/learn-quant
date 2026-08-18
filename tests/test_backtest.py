"""回归测试:锁住已修复的两个 bug 及核心回测正确性。

- 时区:港股 UTC 时间戳必须 +8h 取日期,不能直接切 [:10](否则出现「周日」)。
- 年化:分母必须是自然日,不能用交易日 bar 数(否则高估约 30%)。
- 无未来函数:信号 T 日收盘确认,成交必须在 T+1 日开盘。

运行:
  python3 -m unittest discover -s tests -v
"""
import csv
import os
import sys
import unittest
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backtest_ma_cross as bt   # noqa: E402
import build_data as bd          # noqa: E402


class TestTimeZone(unittest.TestCase):
    def test_hk_utc_shifts_to_next_day(self):
        # 港股 16:00 UTC == 香港次日 00:00,取日期必须 +1 天
        self.assertEqual(bd.exchange_date("2024-01-01T16:00:00Z", "700.HK"), "2024-01-02")

    def test_us_date_unchanged(self):
        # 美股 05:00 UTC == 美东当日 00:00(冬令时),日期不变
        self.assertEqual(bd.exchange_date("2024-01-02T05:00:00Z", "AAPL.US"), "2024-01-02")


class TestDataQuality(unittest.TestCase):
    def test_no_weekend_dates(self):
        for sym in ("AAPL.US", "700.HK"):
            path = os.path.join(ROOT, "data", f"{sym}.csv")
            with open(path) as f:
                for row in csv.DictReader(f):
                    d = date.fromisoformat(row["date"])
                    self.assertLess(d.weekday(), 5, f"{sym} 出现周末日期 {row['date']}")


class TestAnnualizedReturn(unittest.TestCase):
    def test_natural_days_used(self):
        # 365 自然日、涨 10% -> 年化恰为 10%
        self.assertAlmostEqual(bt.annualized_return(110000, 100000, 365), 0.10, places=6)

    def test_zero_days_returns_zero(self):
        self.assertEqual(bt.annualized_return(110000, 100000, 0), 0.0)

    def test_backtest_reports_natural_days_consistently(self):
        dates, opens, _, _, closes, _ = bt.load_csv(os.path.join(ROOT, "data/AAPL.US.csv"))
        _, _, m = bt.backtest(dates, opens, closes, 20, 60, 100000, 0.0)
        expected = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days
        self.assertEqual(m["natural_days"], expected)
        # 年化必须与 natural_days 自洽(而非 bar 数)
        self.assertAlmostEqual(
            m["annual_return"],
            bt.annualized_return(m["final_equity"], m["init_capital"], m["natural_days"]),
        )


class TestNoLookahead(unittest.TestCase):
    def test_signal_executes_next_day(self):
        """金叉在 T 日收盘确认,第一笔 BUY 必须落在 T+1 日开盘。"""
        start = date(2024, 1, 1)
        n = 50
        dates = [(start + timedelta(days=i)).isoformat() for i in range(n)]
        closes = [100.0] * 30 + [100.0 + 20.0 * (i - 29) for i in range(30, n)]
        opens = list(closes)  # 无跳空

        fast_n, slow_n = 5, 20
        fast_ma = bt.moving_average(closes, fast_n)
        slow_ma = bt.moving_average(closes, slow_n)

        # 找到程序本身认定的金叉日 k
        k = None
        for i in range(1, n):
            if slow_ma[i] is not None and slow_ma[i - 1] is not None:
                if fast_ma[i] > slow_ma[i] and fast_ma[i - 1] <= slow_ma[i - 1]:
                    k = i
                    break
        self.assertIsNotNone(k, "合成数据应产生金叉")

        _, trades, _ = bt.backtest(dates, opens, closes, fast_n, slow_n, 100000, 0.0)
        buys = [t for t in trades if t[1] == "BUY"]
        self.assertTrue(buys, "应至少有一笔 BUY")
        self.assertEqual(buys[0][0], dates[k + 1],
                         f"BUY 应在金叉次日 {dates[k+1]} 成交,实际 {buys[0][0]}")


class TestMovingAverage(unittest.TestCase):
    def test_ma_values(self):
        self.assertEqual(bt.moving_average([1, 2, 3, 4, 5], 3),
                         [None, None, 2.0, 3.0, 4.0])

    def test_max_drawdown(self):
        self.assertAlmostEqual(bt.max_drawdown([100, 120, 90, 110]), 0.25)


if __name__ == "__main__":
    unittest.main()
