"""测试策略层:周频决策日、Top-N 选股、逆波动率加权。"""
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.data import PricePanel                      # noqa: E402
from engine.strategy import StrategyConfig, rebalance_dates, select_weights  # noqa: E402


def make_panel(close_vals):
    """close_vals: {symbol: [close...]},构造单日或多日 panel。"""
    idx = pd.date_range("2024-01-01", periods=len(next(iter(close_vals.values()))), freq="D")
    close = pd.DataFrame(close_vals, index=idx)
    return PricePanel(close=close, open=close.copy(), high=close.copy(), low=close.copy(),
                      volume=close.copy(), traded=close.notna(),
                      symbols=list(close_vals.keys()), dates=idx)


class TestRebalanceDates(unittest.TestCase):
    def test_weekly_last_day(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")  # 两个完整交易周,无周末
        rb = list(rebalance_dates(dates))
        # 第一周最后交易日 = 1/5 周五;第二周 = 1/12 周五
        self.assertEqual(rb, [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")])

    def test_holiday_front_shift(self):
        # 周五(1/5)休市,则 1/4 周四为最后交易日
        dates = pd.DatetimeIndex([
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"),
        ])
        self.assertEqual(list(rebalance_dates(dates)), [pd.Timestamp("2024-01-04")])


class TestSelectWeights(unittest.TestCase):
    def test_top_n_inverse_vol(self):
        symbols = ["A", "B", "C", "D"]
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-10")])
        close = pd.DataFrame([[10, 20, 30, 40]], index=idx, columns=symbols)
        panel = PricePanel(close=close, open=close.copy(), high=close.copy(), low=close.copy(),
                           volume=close.copy(), traded=close.notna(),
                           symbols=symbols, dates=idx)

        # 动量 D(0.4) > C(0.3) > B(0.2) > A(0.1),Top-2 选 D、C
        mom = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]], index=idx, columns=symbols)
        # 波动率 C=0.2, D=0.4 -> 逆波动率权重 C:1/0.2=5, D:1/0.4=2.5
        vol = pd.DataFrame([[0.2, 0.3, 0.2, 0.4]], index=idx, columns=symbols)
        factor_scores = {"momentum": mom, "realized_vol": vol}

        w = select_weights(panel, factor_scores, idx[0], StrategyConfig(top_n=2))

        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertEqual(set(w[w > 0].index), {"C", "D"})
        self.assertAlmostEqual(w["C"], 5.0 / 7.5)
        self.assertAlmostEqual(w["D"], 2.5 / 7.5)
        self.assertEqual(w["A"], 0.0)
        self.assertEqual(w["B"], 0.0)

    def test_empty_universe(self):
        panel = make_panel({"A": [1.0, 2.0], "B": [2.0, 3.0]})
        # 因子全 NaN -> universe 为空 -> 返回全 0
        nan = pd.DataFrame([[None, None], [None, None]], index=panel.dates, columns=panel.symbols)
        w = select_weights(panel, {"momentum": nan, "realized_vol": nan},
                           panel.dates[0], StrategyConfig(top_n=2))
        self.assertTrue((w == 0.0).all())


def single_day_panel(symbols, mom_vals, vol_vals):
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-10")])
    close = pd.DataFrame([[10.0] * len(symbols)], index=idx, columns=symbols)
    panel = PricePanel(close=close, open=close.copy(), high=close.copy(), low=close.copy(),
                       volume=close.copy(), traded=close.notna(), symbols=symbols, dates=idx)
    factors = {
        "momentum": pd.DataFrame([mom_vals], index=idx, columns=symbols),
        "realized_vol": pd.DataFrame([vol_vals], index=idx, columns=symbols),
    }
    return panel, factors


class TestStickySelection(unittest.TestCase):
    def test_sell_band_holds_marginal(self):
        # 动量 D=40(rank1) C=30(rank2) A=20(rank3) B=10(rank4)
        panel, factors = single_day_panel(["A", "B", "C", "D"], [20, 10, 30, 40], [0.2] * 4)
        cfg = StrategyConfig(top_n=2, sell_band=1)  # sell_threshold=3
        w = select_weights(panel, factors, panel.dates[0], cfg,
                           holdings={"A"}, holding_age={"A": 5})
        # A(rank3)在带内保留;D(1)/C(2)买入;B(4)不买
        self.assertEqual(set(w[w > 0].index), {"A", "C", "D"})

    def test_sell_band_sells_beyond_band(self):
        # A=10(rank4)掉出带(>3),应卖出
        panel, factors = single_day_panel(["A", "B", "C", "D"], [10, 20, 30, 40], [0.2] * 4)
        cfg = StrategyConfig(top_n=2, sell_band=1)
        w = select_weights(panel, factors, panel.dates[0], cfg,
                           holdings={"A"}, holding_age={"A": 5})
        self.assertEqual(set(w[w > 0].index), {"C", "D"})

    def test_min_holding_keeps_recent(self):
        # A 掉到 rank4,但持仓未满 min_holding_weeks 时保留
        panel, factors = single_day_panel(["A", "B", "C", "D"], [10, 20, 30, 40], [0.2] * 4)
        cfg = StrategyConfig(top_n=2, sell_band=1, min_holding_weeks=4)
        w = select_weights(panel, factors, panel.dates[0], cfg,
                           holdings={"A"}, holding_age={"A": 1})
        self.assertIn("A", set(w[w > 0].index))
        w2 = select_weights(panel, factors, panel.dates[0], cfg,
                            holdings={"A"}, holding_age={"A": 5})
        self.assertNotIn("A", set(w2[w2 > 0].index))

    def test_degenerate_matches_stateless(self):
        # band=0 & min=0 时,有状态 == 无状态(向后兼容)
        panel, factors = single_day_panel(["A", "B", "C", "D"], [20, 10, 30, 40], [0.2] * 4)
        cfg = StrategyConfig(top_n=2, sell_band=0, min_holding_weeks=0)
        w_stateful = select_weights(panel, factors, panel.dates[0], cfg,
                                    holdings={"A"}, holding_age={"A": 3})
        w_stateless = select_weights(panel, factors, panel.dates[0], cfg)
        pd.testing.assert_series_equal(w_stateful, w_stateless)


if __name__ == "__main__":
    unittest.main()
