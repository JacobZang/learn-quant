"""组合回测引擎:Portfolio + 可插拔成本 + 主循环。"""
from dataclasses import dataclass

import pandas as pd

from .strategy import rebalance_dates, select_weights


@dataclass
class CostModel:
    """可插拔成本模型(单边)。

    commission: 单边佣金率(按成交额)
    slippage:   单边滑点(价格比例,默认 5bp)
    预留(首版不实现): sell_tax_rate(港股印花税)、lot_size(每手取整)
    """
    commission: float = 0.0
    slippage: float = 0.0005

    def buy_price(self, ref):
        return ref * (1.0 + self.slippage)

    def sell_price(self, ref):
        return ref * (1.0 - self.slippage)

    def fee(self, notional, side):
        return notional * self.commission


class Portfolio:
    """按股数精确追踪的组合(首版允许小数股)。"""

    def __init__(self, capital, symbols):
        self.cash = float(capital)
        self.shares = pd.Series(0.0, index=list(symbols))

    def nav(self, price_row):
        return self.cash + float((self.shares * price_row).sum())

    def weights(self, price_row):
        mv = self.shares * price_row
        total = mv.sum()
        if total <= 0:
            return pd.Series(0.0, index=mv.index)
        return mv / total

    def execute(self, target, open_row, cost):
        """按开盘价从当前持仓调到 target 权重,先卖后买,返回单边换手率。"""
        nav_before = self.nav(open_row)
        cur_w = self.weights(open_row)

        # 先卖:减持到目标,回笼现金
        for s in open_row.index:
            if cur_w[s] > target[s]:
                notional = (cur_w[s] - target[s]) * nav_before
                price = cost.sell_price(open_row[s])
                sh = notional / price
                self.shares[s] -= sh
                self.cash += sh * price - cost.fee(sh * price, "SELL")

        # 后买:增持到目标
        for s in open_row.index:
            if target[s] > cur_w[s]:
                notional = (target[s] - cur_w[s]) * nav_before
                price = cost.buy_price(open_row[s])
                sh = notional / price
                self.shares[s] += sh
                self.cash -= sh * price + cost.fee(sh * price, "BUY")

        return 0.5 * float((target - cur_w).abs().sum())


@dataclass
class BacktestResult:
    nav: pd.Series            # 每日收盘 NAV
    turnovers: list           # [(date, 单边换手率), ...]


def run_backtest(panel, factor_scores, cfg, cost, capital=1_000_000,
                 strategy_fn=None, rebalance_dates_fn=None):
    """主循环:决策日 T 收盘定目标权重,T+1 开盘成交(无未来函数)。

    strategy_fn:       权重函数,签名为 (panel, factor_scores, date, cfg, holdings, holding_age)
                       -> pd.Series(symbol->weight)。默认 select_weights。
    rebalance_dates_fn: 调仓日期函数,签名为 (dates) -> DatetimeIndex。
                       默认 rebalance_dates(周频)。可通过 cfg 类型自动推断。
    """
    if strategy_fn is None:
        strategy_fn = select_weights
    if rebalance_dates_fn is None:
        from .strategy import AllocationConfig
        if isinstance(cfg, AllocationConfig):
            from .strategy import rebalance_dates_periodic
            rebalance_dates_fn = lambda d: rebalance_dates_periodic(d, cfg.rebalance_freq)
        else:
            rebalance_dates_fn = rebalance_dates

    rebal = set(rebalance_dates_fn(panel.dates))
    port = Portfolio(capital, panel.symbols)
    pending = None
    current_holdings = set()
    holding_age = {}
    navs = []
    turnovers = []

    for t in panel.dates:
        # 1) 执行前一日收盘定的目标(今日开盘成交)
        if pending is not None:
            turn = port.execute(pending, panel.open.loc[t], cost)
            turnovers.append((t, turn))
            # 更新持仓状态:新买入 age=0,持续持有 age+1,卖出移除
            new_holdings = {s for s in pending.index if pending[s] > 0}
            holding_age = {s: (holding_age[s] + 1 if s in current_holdings else 0)
                           for s in new_holdings}
            current_holdings = new_holdings
            pending = None
        # 2) 记录当日收盘 NAV
        navs.append(port.nav(panel.close.loc[t]))
        # 3) 若是决策日,用当日收盘价定新目标(次日执行)
        if t in rebal:
            pending = strategy_fn(panel, factor_scores, t, cfg, current_holdings, holding_age)

    return BacktestResult(
        nav=pd.Series(navs, index=panel.dates),
        turnovers=turnovers,
    )
