"""数据模型:把多个标的的日线 CSV 加载并对齐成宽表 PricePanel。"""
from dataclasses import dataclass
import os

import pandas as pd


@dataclass
class PricePanel:
    """宽表:每张 DataFrame 的 index=日期, columns=symbols。

    - close/open/high/low 已 ffill(只前填,禁 bfill —— 未来函数红线)。
    - traded 是 ffill 之前的真实成交掩码(True = 当日该标的真实有成交)。
      后上市标的在上市前 traded=False,会被排除出选股 universe。
    """
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    traded: pd.DataFrame
    symbols: list
    dates: pd.DatetimeIndex


def load_panel(symbols, data_dir="data"):
    """读各标的 CSV,按日期 union 对齐成宽表。

    交易日对齐 = 各标的实际 K 线日期的 union(不引入官方交易日历,因
    MCP 的 trading_days 工具会把休市日也列为交易日,不可靠)。
    """
    dfs = {}
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym}.csv")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        dfs[sym] = df

    union = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in dfs.values()])))

    def wide(field):
        return pd.DataFrame(
            {sym: df[field].reindex(union) for sym, df in dfs.items()},
            index=union,
        )

    close = wide("close")
    open_ = wide("open")
    high = wide("high")
    low = wide("low")
    volume = wide("volume")

    traded = close.notna()  # ffill 之前记下真实成交日

    return PricePanel(
        close=close.ffill(),
        open=open_.ffill(),
        high=high.ffill(),
        low=low.ffill(),
        volume=volume.ffill().fillna(0.0),
        traded=traded,
        symbols=list(symbols),
        dates=union,
    )


def slice_panel(panel, start=None, end=None):
    """截断 panel 到 [start, end] 区间(含端点),返回新的 PricePanel。"""
    close = panel.close.loc[start:end]
    open_ = panel.open.loc[start:end]
    high = panel.high.loc[start:end]
    low = panel.low.loc[start:end]
    volume = panel.volume.loc[start:end]
    traded = panel.traded.loc[start:end]
    return PricePanel(
        close=close, open=open_, high=high, low=low,
        volume=volume, traded=traded,
        symbols=panel.symbols, dates=close.index,
    )
