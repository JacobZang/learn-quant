import akshare as ak
import pandas as pd
import fetch_data.universal_func as universal_func

def get_adjusted_stock_data_to_csv(symbol, start_date, end_date, adjust, period="daily"):
    # 获取贵州茅台 前复权 日线行情
    df = ak.stock_zh_a_hist(
        symbol=symbol, 
        period=period, 
        start_date=start_date, 
        end_date=end_date, 
        adjust=adjust
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

    filename = f"data/{symbol}_adjusted.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    
def get_processed_adjusted_data(symbol, window):
    filename = f"data/{symbol}_adjusted.csv"
    df = universal_func.load_data(filename)
    X, y = universal_func.build_featured_dataset(df, window=window)
    return universal_func.cut_dataset(X, y)