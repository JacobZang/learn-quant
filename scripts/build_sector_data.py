#!/usr/bin/env python3
"""把 3 段分段拉取的行业 ETF 数据拼接成完整 10 年 CSV。

分段原因:长桥 history_candlesticks_by_date 单次最多返回 1000 条(约 4 年),
所以 2016~2026 分 3 段拉,此脚本去重 + 按时间排序 + 时区转日期后落盘。
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import parse_mcp_result, exchange_date  # noqa: E402

SYMBOLS = ["XLK.US", "XLF.US", "XLE.US", "XLV.US", "XLI.US", "XLY.US",
           "XLP.US", "XLB.US", "XLRE.US", "XLU.US", "XLC.US"]

BASE = os.path.expanduser(
    "~/.claude/projects/-home-jacobe-repos-quantest/098247c2-0ea4-4ed2-9dd8-a3cd0ffb79b7/tool-results")

# 三段:2016-01~2019-09 / 2019-10~2023-06 / 2023-07~2026-08
SEG1 = {
    "XLK.US": "call_00_eR8J9dvEhI9nBn6cHx197442.json",
    "XLF.US": "call_01_dYfbu3PrWcecOLyha57G9267.json",
    "XLE.US": "call_02_OfDsMbmrLpOsIkJCg3g87184.json",
    "XLV.US": "call_03_FrNRbfCC7xLI6ASeeF8Y7347.json",
    "XLI.US": "call_04_KzRnL8aFreON2qF1ern85122.json",
    "XLY.US": "call_05_7XKrTRvqbhYjYw7iFiWz1454.json",
    "XLP.US": "call_06_O7T8KGG0TDVsvGydSq1m7121.json",
    "XLB.US": "call_07_9N3KVmFFb3Ipzjfh0gPJ7803.json",
    "XLRE.US": "call_08_a7e1UtgorwDrX1rxonHB3814.json",
    "XLU.US": "call_09_A9FAYanHpbPeRxtzd5xF4128.json",
    "XLC.US": "call_10_2jksvHSXEDYjUUlqjGJy5322.json",
}
SEG2 = {
    "XLK.US": "call_00_mqFZHPhFA4LWVJA08J3w6433.json",
    "XLF.US": "call_01_fSP0T4LavnC5qeYHYWEw5385.json",
    "XLE.US": "call_02_hjW4dViqKMMPBksu6RHL1998.json",
    "XLV.US": "call_03_V0iZnHn7ivCK0Ln79NTM9120.json",
    "XLI.US": "call_04_iPhjCPk2w3pBMhLAwAIw2377.json",
    "XLY.US": "call_05_HzW7RYshNVFrQBg5ZBv11518.json",
    "XLP.US": "call_06_4nZ6vO1qYSjsiXxi5eoJ9823.json",
    "XLB.US": "call_07_UrMOWbOFP14Ipk20UeJl8830.json",
    "XLRE.US": "call_08_C0AHcgziPkO6jCiSREWf8304.json",
    "XLU.US": "call_09_0VGFCA7KE9pAl2YysCOm5737.json",
    "XLC.US": "call_10_zAL1WK1AvSoega8eimnh2767.json",
}
SEG3 = {
    "XLK.US": "call_00_tfZETbZuAcENC0sDndJf6073.json",
    "XLF.US": "call_01_KIN78fI14Adjhny1jTjr6790.json",
    "XLE.US": "call_02_NVKyuXgPQbjqK62Z6pGJ5122.json",
    "XLV.US": "call_03_46mn25WfGqWQC52UfTpS2037.json",
    "XLI.US": "call_04_oAS07KwX7axu4vQAqmb10561.json",
    "XLY.US": "call_05_x85K9RYk5q9ifFfe5n6a1968.json",
    "XLP.US": "call_06_IKBhXxxmtVPydm853Jfj7771.json",
    "XLB.US": "call_07_BWu5oLjyQQ3NgsxqYBoN8795.json",
    "XLRE.US": "call_08_OljIFn5KxrorDJSaGPVC3218.json",
    "XLU.US": "call_09_5kVD18w5fQHSxV2AElX34817.json",
    "XLC.US": "call_10_2IAXWqXz5R478RwGUi0m1938.json",
}


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    for sym in SYMBOLS:
        rows = []
        for seg in (SEG1, SEG2, SEG3):
            rows.extend(parse_mcp_result(os.path.join(BASE, seg[sym])))
        # 去重(段边界可能重叠)+ 按时间排序
        dedup = {r["timestamp"]: r for r in rows}
        rows = sorted(dedup.values(), key=lambda r: r["timestamp"])
        with open(os.path.join(out_dir, f"{sym}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for r in rows:
                w.writerow([exchange_date(r["timestamp"], sym), r["open"], r["high"],
                            r["low"], r["close"], r["volume"]])
        print(f"{sym}: {len(rows)} rows  {rows[0]['timestamp'][:10]} ~ {rows[-1]['timestamp'][:10]}")


if __name__ == "__main__":
    main()
