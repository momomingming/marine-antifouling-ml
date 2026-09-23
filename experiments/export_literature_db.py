#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文献库 → 应用端导出
====================
`data/literature/literature_unified.csv` 是**唯一事实源**（21 列，含 methods/results/
conclusions 等结构化字段）；应用只读精简的 `literature_db.csv`（前 11 列）。
本脚本负责把 unified 的更新导出到各部署副本，避免手工复制造成漂移。

用法:
  python experiments/export_literature_db.py                 # 导出到 deploy_job + hf_space
  python experiments/export_literature_db.py --check         # 只比对差异，不写
"""
import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data", "literature", "literature_unified.csv")

# 应用端字段（与 app.py load_literature_db 消费的列一致）
APP_COLS = ['paperId', 'title', 'titleZh', 'authors', 'journal', 'year',
            'doi', 'citationCount', 'abstract', 'url', 'source']

TARGETS = [
    os.path.join(REPO, "deployments", "deploy_job", "literature_db.csv"),
    os.path.join(REPO, "deployments", "hf_space", "literature_db.csv"),
]


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def fingerprint(rows):
    return f"{len(rows)} 行 / 非空abstract {sum(1 for r in rows if (r.get('abstract') or '').strip())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只比对，不写文件')
    args = ap.parse_args()

    if not os.path.exists(SRC):
        print("缺少", SRC)
        return 1
    src_rows = read_rows(SRC)
    miss = [f for f in APP_COLS if f not in (src_rows[0].keys() if src_rows else {})]
    if miss:
        print("源文件缺少应用所需列:", miss)
        return 1
    print(f"源 unified: {fingerprint(src_rows)}")

    for t in TARGETS:
        old = fingerprint(read_rows(t)) if os.path.exists(t) else "(不存在)"
        new = fingerprint(src_rows)
        if args.check:
            flag = "同" if old == new else "差"
            print(f"  [{flag}] {os.path.relpath(t, REPO)}: {old}  →  {new}")
            continue
        with open(t, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=APP_COLS, extrasaction='ignore')
            w.writeheader()
            for r in src_rows:
                w.writerow({k: r.get(k, '') for k in APP_COLS})
        print(f"  [写] {os.path.relpath(t, REPO)}: {old}  →  {new}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
