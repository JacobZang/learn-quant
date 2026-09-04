"""策略层:周频调仓 + Top-N 选股 + 逆波动率加权。"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyConfig:
    top_n: int = 5
    weight_by_inverse_vol: bool = True
    sell_band: int = 0          # 卖出阈值放宽幅度:卖出阈值 = top_n + sell_band
    min_holding_weeks: int = 0  # 最短持仓周数(0 = 不限制)


def rebalance_dates(dates):
    """每个自然周的最后一个交易日 = 决策日(通常是周五,节假日自动前移)。"""
    s = pd.Series(0, index=dates)
    return s.groupby(dates.to_period("W")).tail(1).index


@dataclass
class AllocationConfig:
    """固定权重资产配置(再平衡策略),与动量选股的 StrategyConfig 并列。

    weights:        {symbol: weight} 字典,权重归一化后和为 1
    rebalance_freq: 再平衡频率 "weekly" | "monthly" | "quarterly"
    """
    weights: dict = None
    rebalance_freq: str = "monthly"

    def __post_init__(self):
        if not self.weights:
            raise ValueError("AllocationConfig.weights 不能为空")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("权重之和必须为正")
        self.weights = {k: v / total for k, v in self.weights.items()}


_FREQ_PERIOD = {"weekly": "W", "monthly": "M", "quarterly": "Q"}


def rebalance_dates_periodic(dates, freq="monthly"):
    """每个自然月/季/周最后一个交易日 = 决策日(定期再平衡)。

    只支持 AllocationConfig 支持的三种频率;与 rebalance_dates(周频)语义一致。
    """
    period = _FREQ_PERIOD[freq]
    s = pd.Series(0, index=dates)
    return s.groupby(dates.to_period(period)).tail(1).index


def allocate_weights(panel, factor_scores, date, cfg, holdings=None, holding_age=None):
    """固定权重分配:忽略因子与持仓状态,直接返回 cfg.weights。

    签名与 select_weights 兼容,便于插入 run_backtest 的 strategy_fn 槽位。
    未出现在 panel.symbols 或 cfg.weights 里的标的补 0。
    """
    w = pd.Series(0.0, index=panel.symbols)
    for sym, weight in cfg.weights.items():
        if sym in w.index:
            w[sym] = weight
    return w


def select_weights(panel, factor_scores, date, cfg, holdings=None, holding_age=None):
    """在决策日 `date` 收盘,返回目标权重 Series(symbol -> weight,和为 1)。

    选股流程: universe 过滤(真实成交且因子非 NaN) -> 选股(带换手控制)
    -> 逆波动率归一化加权,未选中的补 0。

    换手控制(holdings/holding_age 非 None 且 band/min 不全为 0 时生效):
      - 未持有:动量排名 <= top_n 才买入
      - 持有中:排名 <= top_n + sell_band,或持仓周数 < min_holding_weeks,才保留
    两者抑制边缘股票在阈值附近的频繁进出,降低换手。
    """
    mom = factor_scores["momentum"].loc[date]
    vol = factor_scores["realized_vol"].loc[date]

    universe = [
        s for s in panel.symbols
        if panel.traded.loc[date, s] and pd.notna(mom[s]) and pd.notna(vol[s])
    ]
    if not universe:
        return pd.Series(0.0, index=panel.symbols)

    if holdings is None or (cfg.sell_band == 0 and cfg.min_holding_weeks == 0):
        # 退化:完全再平衡(旧行为)
        selected = mom[universe].nlargest(cfg.top_n).index
    else:
        ranked = mom[universe].rank(ascending=False)  # 1 = 动量最高
        sell_threshold = cfg.top_n + cfg.sell_band
        selected = []
        for s in universe:
            r = ranked[s]
            if s in holdings:
                age = holding_age.get(s, 0)
                if r <= sell_threshold or age < cfg.min_holding_weeks:
                    selected.append(s)  # 保留
            elif r <= cfg.top_n:
                selected.append(s)  # 买入
        selected = pd.Index(selected)

    if len(selected) == 0:
        return pd.Series(0.0, index=panel.symbols)

    if cfg.weight_by_inverse_vol:
        inv_vol = 1.0 / vol[selected]
        weights = inv_vol / inv_vol.sum()
    else:
        weights = pd.Series(1.0 / len(selected), index=selected)

    return weights.reindex(panel.symbols, fill_value=0.0)
