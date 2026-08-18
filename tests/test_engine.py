"""测试回测引擎:成本模型、组合调仓、换手、端到端冒烟。"""
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.config import DEFAULT_UNIVERSE, DATA_DIR          # noqa: E402
from engine.data import load_panel                           # noqa: E402
from engine.factors import momentum, realized_vol            # noqa: E402
from engine.strategy import StrategyConfig                   # noqa: E402
from engine.backtest import CostModel, Portfolio, run_backtest  # noqa: E402


class TestCostModel(unittest.TestCase):
    def test_haircut_and_fee(self):
        cost = CostModel(commission=0.001, slippage=0.01)
        self.assertAlmostEqual(cost.buy_price(100.0), 101.0)
        self.assertAlmostEqual(cost.sell_price(100.0), 99.0)
        self.assertAlmostEqual(cost.fee(10000.0, "BUY"), 10.0)


class TestPortfolio(unittest.TestCase):
    def test_execute_build_from_cash(self):
        port = Portfolio(100000, ["A", "B"])
        target = pd.Series({"A": 0.5, "B": 0.5})
        open_row = pd.Series({"A": 100.0, "B": 100.0})
        turn = port.execute(target, open_row, CostModel(commission=0.0, slippage=0.0))
        # 从空仓到 50/50,单边换手 = 0.5 * (0.5 + 0.5) = 0.5
        self.assertAlmostEqual(turn, 0.5)
        self.assertAlmostEqual(port.shares["A"], 500.0)
        self.assertAlmostEqual(port.shares["B"], 500.0)
        self.assertAlmostEqual(port.cash, 0.0)

    def test_slippage_reduces_shares(self):
        port = Portfolio(100000, ["A"])
        target = pd.Series({"A": 1.0})
        open_row = pd.Series({"A": 100.0})
        cost = CostModel(commission=0.0, slippage=0.01)  # 1% 滑点
        port.execute(target, open_row, cost)
        # 买入价 101,只能买 100000/101 股
        self.assertAlmostEqual(port.shares["A"], 100000 / 101)

    def test_nav_consistency(self):
        port = Portfolio(100000, ["A"])
        port.shares["A"] = 100.0
        port.cash = 0.0
        close_row = pd.Series({"A": 105.0})
        self.assertAlmostEqual(port.nav(close_row), 10500.0)


class TestEndToEnd(unittest.TestCase):
    def test_smoke(self):
        panel = load_panel(DEFAULT_UNIVERSE, os.path.join(ROOT, DATA_DIR))
        factor_scores = {"momentum": momentum(panel), "realized_vol": realized_vol(panel)}
        cfg = StrategyConfig(top_n=5)
        res = run_backtest(panel, factor_scores, cfg, CostModel(), 1_000_000)

        self.assertAlmostEqual(res.nav.iloc[0], 1_000_000)  # 首日纯现金
        self.assertFalse(res.nav.isna().any())               # 全程无 NaN
        self.assertTrue((res.nav > 0).all())                 # NAV 恒正
        self.assertGreater(len(res.turnovers), 0)            # 有调仓

    def test_no_lookahead_weekly(self):
        """决策日 T 收盘定权重,T+1 开盘才成交:验证第一笔成交发生在第一个决策日之后。"""
        panel = load_panel(DEFAULT_UNIVERSE, os.path.join(ROOT, DATA_DIR))
        factor_scores = {"momentum": momentum(panel), "realized_vol": realized_vol(panel)}
        from engine.strategy import rebalance_dates
        rebal = set(rebalance_dates(panel.dates))
        res = run_backtest(panel, factor_scores, StrategyConfig(top_n=5), CostModel(), 1_000_000)
        first_trade = res.turnovers[0][0]
        # 第一个决策日之后才可能有成交
        first_rebal = min(rebal)
        self.assertGreater(first_trade, first_rebal)

    def test_turnover_decreases_with_stickiness(self):
        from engine.evaluation import turnover_annual
        panel = load_panel(DEFAULT_UNIVERSE, os.path.join(ROOT, DATA_DIR))
        factor_scores = {"momentum": momentum(panel), "realized_vol": realized_vol(panel)}
        loose = run_backtest(panel, factor_scores, StrategyConfig(top_n=5), CostModel(), 1_000_000)
        sticky = run_backtest(panel, factor_scores,
                              StrategyConfig(top_n=5, sell_band=2, min_holding_weeks=4),
                              CostModel(), 1_000_000)
        self.assertLess(turnover_annual(sticky.turnovers), turnover_annual(loose.turnovers))


if __name__ == "__main__":
    unittest.main()
