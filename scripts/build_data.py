#!/usr/bin/env python3
"""
从 Longbridge MCP 的 history_candlesticks_by_date 返回文件生成 CSV。

正确处理交易所时区:
  MCP 返回的 timestamp 是 UTC,直接切 [:10] 会导致港股日期偏移一天(出现「周日」)。
  美股虽然 UTC 切日期恰好正确,但为了统一,也一律转交易所时区再取日期。

市场 -> 时区:
  US -> America/New_York, HK -> Asia/Hong_Kong, CN -> Asia/Shanghai, SG -> Asia/Singapore

用法:
  python3 scripts/build_data.py <mcp_result.json> <SYMBOL> <output.csv>
"""
import csv
import json
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

MARKET_TZ = {
    "US": "America/New_York",
    "HK": "Asia/Hong_Kong",
    "CN": "Asia/Shanghai",
    "SG": "Asia/Singapore",
}


def parse_mcp_result(path):
    """MCP 结果文件结构是 [{"type":"text","text":"[...]"}]。"""
    raw = json.load(open(path, encoding="utf-8"))
    return json.loads(raw[0]["text"])


def exchange_date(ts, symbol):
    """把 UTC 时间戳转成交易所当地时区,再取日期。"""
    market = symbol.split(".")[-1].upper()
    tz = ZoneInfo(MARKET_TZ.get(market, "UTC"))
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
    return dt.date().isoformat()


def main():
    path, symbol, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = parse_mcp_result(path)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow([
                exchange_date(r["timestamp"], symbol),
                r["open"], r["high"], r["low"], r["close"], r["volume"],
            ])
    print(f"{symbol} -> {out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
