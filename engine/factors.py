"""横截面因子。所有因子必须满足「无未来函数」:只用 T 日及之前的数据算出 T 行取值。

shift 纪律(红线):
  - rolling(...) / pct_change() / shift(k>=0) 安全(向后看或取更早数据)。
  - 禁止 shift(-1) / bfill / rolling(center=True)。
"""
import pandas as pd


def momentum(panel, lookback=126, skip=21):
    """中期动量: mom[t] = close[t-skip] / close[t-lookback] - 1。

    用 126/21(约 6 个月动量、跳过最近 1 个月)而非经典 252/21,因数据仅约
    2.6 年,126 天预热已吃掉半年;数据变长后可切 252。
    """
    return panel.close.shift(skip) / panel.close.shift(lookback) - 1.0


def realized_vol(panel, window=63, ann_factor=252):
    """实现波动率(年化): 过去 window 个交易日日收益的标准差 * sqrt(ann_factor)。"""
    ret = panel.close.pct_change()  # t-1 -> t,安全
    return ret.rolling(window).std(ddof=1) * (ann_factor ** 0.5)
