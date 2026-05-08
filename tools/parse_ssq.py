#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将“双色球历史开奖数据（Tab 分隔 txt / xlsx）”解析为前端可加载的 JSON。

用法：
  python3 tools/parse_ssq.py /path/to/ssq_history.txt data/draws.json
  python3 tools/parse_ssq.py /path/to/ssq_history.xlsx data/draws.json
"""

import json
import os
import re
import sys
from datetime import datetime

import pandas as pd

def to_int(s: str):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def to_date(s: str):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except Exception:
        return s


def is_data_line(line: str) -> bool:
    # 数据行以“序号\t期号\t年份...”开头
    return bool(re.match(r"^\d+\t\d{7}\t\d{4}\t", line))


def parse_txt(input_path: str):
    draws = []
    with open(input_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not is_data_line(line):
                continue
            parts = line.split("\t")
            if len(parts) < 11:
                continue

            issue = parts[1].strip()
            year = to_int(parts[2])
            date = to_date(parts[3])
            reds = [to_int(x) for x in parts[4:10]]
            blue = to_int(parts[10])

            if not issue or any(r is None for r in reds) or blue is None:
                continue

            draws.append(
                {
                    "issue": issue,
                    "year": year,
                    "date": date,
                    "reds": sorted(reds),
                    "blue": blue,
                }
            )
    return draws


def _norm_date(v):
    if pd.isna(v) or v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip()
    return s or None


def parse_xlsx(input_path: str):
    df = pd.read_excel(input_path, sheet_name="开奖数据")
    required = ["期号", "年份", "开奖日期", "红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}")

    draws = []
    for _, row in df.iterrows():
        issue = str(row["期号"]).strip()
        year = int(row["年份"]) if not pd.isna(row["年份"]) else None
        date = _norm_date(row["开奖日期"])
        reds = [int(row[f"红球{i}"]) for i in range(1, 7)]
        blue = int(row["蓝球"])
        draws.append({"issue": issue, "year": year, "date": date, "reds": sorted(reds), "blue": blue})
    return draws


def main():
    if len(sys.argv) < 2:
        print("用法：python3 tools/parse_ssq.py 输入.txt [输出.json]", file=sys.stderr)
        sys.exit(2)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "data/draws.json"

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        draws = parse_xlsx(input_path)
    else:
        draws = parse_txt(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "source": os.path.basename(input_path),
                    "count": len(draws),
                    "generatedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
                "draws": draws,
            },
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    print(f"✅ Parsed {len(draws)} draws -> {output_path}")


if __name__ == "__main__":
    main()
