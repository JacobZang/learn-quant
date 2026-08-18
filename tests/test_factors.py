"""测试因子:公式手算值 + 无未来函数(截断不变性 / 未来价格扰动不变性)。"""
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.data import PricePanel           # noqa: E402
from engine.factors import momentum, realized_vol  # noqa: E402


def make_panel(data):
    """从 {symbol: [close...]} 构造简单 PricePanel(open=high=low=close)。"""
    idx = pd.date_range("2024-01-01", periods=len(next(iter(data.values()))), freq="D")
    close = pd.DataFrame(data, index=idx)
    return PricePanel(close=close, open=close.copy(), high=close.copy(), low=close.copy(),
                      volume=close.copy(), traded=close.notna(),
                      symbols=list(data.keys()), dates=idx)


class TestMomentum(unittest.TestCase):
    def test_formula(self):
        panel = make_panel({"A": list(range(1, 11)), "B": list(range(10, 0, -1))})
        mom = momentum(panel, lookback=3, skip=1)
        # mom[t] = close[t-1] / close[t-3] - 1
        t = pd.Timestamp("2024-01-04")  # 第 4 天(index 3)
        self.assertAlmostEqual(mom.loc[t, "A"], 3.0 / 1.0 - 1)   # close[2]/close[0]
        self.assertAlmostEqual(mom.loc[t, "B"], 8.0 / 10.0 - 1)  # close[2]/close[0]

    def test_truncation_invariance(self):
        """把面板截断到 t 重算因子,与全量面板 t 行必须逐值相等(无未来函数)。"""
        data = {"A": list(range(1, 21)), "B": list(range(20, 0, -1))}
        full = momentum(make_panel(data), lookback=5, skip=2)

        t = full.index[14]
        trunc = momentum(make_panel({k: v[:15] for k, v in data.items()}), lookback=5, skip=2)
        self.assertAlmostEqual(full.loc[t, "A"], trunc.iloc[-1]["A"])
        self.assertAlmostEqual(full.loc[t, "B"], trunc.iloc[-1]["B"])

    def test_future_price_invariance(self):
        """把 t 之后的价格改成极端值,t 行的因子取值必须不变。"""
        data = {"A": list(range(1, 21)), "B": list(range(20, 0, -1))}
        mom_before = momentum(make_panel(data), lookback=5, skip=2)

        mutated = {k: list(v) for k, v in data.items()}
        mutated["A"][15:] = [9999.0] * 5
        mutated["B"][15:] = [0.0] * 5
        mom_after = momentum(make_panel(mutated), lookback=5, skip=2)

        t = mom_before.index[14]
        self.assertAlmostEqual(mom_before.loc[t, "A"], mom_after.loc[t, "A"])
        self.assertAlmostEqual(mom_before.loc[t, "B"], mom_after.loc[t, "B"])


class TestRealizedVol(unittest.TestCase):
    def test_truncation_invariance(self):
        data = {"A": list(range(1, 21)), "B": list(range(20, 0, -1))}
        full = realized_vol(make_panel(data), window=5)
        t = full.index[14]
        trunc = realized_vol(make_panel({k: v[:15] for k, v in data.items()}), window=5)
        self.assertAlmostEqual(full.loc[t, "A"], trunc.iloc[-1]["A"])
        self.assertAlmostEqual(full.loc[t, "B"], trunc.iloc[-1]["B"])


if __name__ == "__main__":
    unittest.main()
