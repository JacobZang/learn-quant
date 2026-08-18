"""测试数据层:多标的对齐、ffill、traded 掩码、后上市标的排除。"""
import csv
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.data import load_panel  # noqa: E402


class TestLoadPanelReal(unittest.TestCase):
    def test_shape_and_alignment(self):
        symbols = ["AAPL.US", "MSFT.US", "NVDA.US"]
        panel = load_panel(symbols, os.path.join(ROOT, "data"))
        self.assertEqual(list(panel.symbols), symbols)
        self.assertEqual(panel.close.shape, (len(panel.dates), 3))
        # ffill 之后(除 leading NaN)不应有 NaN
        self.assertFalse(panel.close.iloc[1:].isna().any().any())
        # 首日三只都真实成交
        self.assertTrue(panel.traded.iloc[0].all())

    def test_no_weekend_dates(self):
        panel = load_panel(["AAPL.US"], os.path.join(ROOT, "data"))
        for d in panel.dates:
            self.assertLess(d.weekday(), 5, f"出现周末 {d}")


class TestLateListing(unittest.TestCase):
    def test_late_listing_excluded_before_listing(self):
        tmp = tempfile.mkdtemp()

        def write_csv(sym, dates, prices):
            with open(os.path.join(tmp, f"{sym}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                for d, px in zip(dates, prices):
                    w.writerow([d, px, px, px, px, 100])

        base = date(2024, 1, 1)
        dates_a = [(base + timedelta(days=i)).isoformat() for i in range(10)]
        dates_b = [(base + timedelta(days=i)).isoformat() for i in range(4, 10)]
        write_csv("A.US", dates_a, [10.0 + i for i in range(10)])
        write_csv("B.US", dates_b, [20.0 + i for i in range(6)])

        panel = load_panel(["A.US", "B.US"], tmp)

        # B 上市前(1/1~1/4)traded=False,close 是 leading NaN(未被 ffill 回填)
        self.assertFalse(panel.traded.loc["2024-01-01", "B.US"])
        self.assertTrue(pd.isna(panel.close.loc["2024-01-01", "B.US"]))
        # B 上市后(1/5)traded=True,close 有值
        self.assertTrue(panel.traded.loc["2024-01-05", "B.US"])
        self.assertFalse(pd.isna(panel.close.loc["2024-01-05", "B.US"]))


if __name__ == "__main__":
    unittest.main()
