import akshare as ak
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_qfq_data(symbol, start_date, end_date, period="daily"):
    # 获取贵州茅台 前复权 日线行情
    df = ak.stock_zh_a_hist(
        symbol="600519", 
        period="daily", 
        start_date=start_date, 
        end_date=end_date, 
        adjust="qfq"   # 'qfq' 前复权, 'hfq' 后复权, None 不复权
    )

    # 列名映射：中文 -> 英文
    df.rename(columns={
        "日期": "date",
        "股票代码": "ts_code",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "chg",
        "换手率": "turnover"
    }, inplace=True)

    # 日期格式转换
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # 删除无关列
    df.drop(columns=["ts_code"], inplace=True)

    filename = f"data/{symbol}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    # 读取数据
    df = pd.read_csv(filename, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def make_dataset(df, window=30):
    # ------- 构造衍生指标 -------
    # 5日均线 & 20日均线
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    # 均线差
    df["MA_diff"] = df["MA5"] - df["MA20"]

    # 删除前面缺失值
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ==== 构造输入特征 ====
    features = df[["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "turnover", "MA_diff"]].values

    # 标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 构造样本和标签
    X, y = [], []
    for i in range(window, len(features)-1):
        X.append(features[i-window:i])
        label = 1 if df.loc[i+1, "close"] > df.loc[i, "close"] else 0
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("样本数:", X.shape, "标签分布:", np.bincount(y))
    return X, y

def cut_dataset(X, y, train_ratio=0.7, val_ratio=0.15):

    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test