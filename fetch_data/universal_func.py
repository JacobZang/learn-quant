import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filename):    
    # 读取数据
    df = pd.read_csv(filename, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def build_featured_dataset(df, window=30):
    # 均线
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA5_20_diff"] = df["MA5"] - df["MA20"]

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])

    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + RS))

    # ATR
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift(1))
    df["L-PC"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    df.drop(columns=["H-L","H-PC","L-PC","TR"], inplace=True)
    df.dropna(inplace=True)
    feature_cols = ["open","high","low","close","volume", "MA5_20_diff","RSI","ATR","OBV"]

    # 标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values)

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

def cut_dataset(X, y, train_ratio=0.7, val_ratio=0.25):

    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test