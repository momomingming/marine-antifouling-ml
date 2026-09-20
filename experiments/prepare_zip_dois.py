#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务③扩库 - 本地预处理 Step 1
从两份微信文献 zip 抽出所有 PDF 的文件名 DOI/PII 候选，生成待补全清单。
不依赖网络。真正的元数据补全由 fetch_crossref_metadata.py 在通网后执行。
"""
import zipfile, re, os, csv

ZIP_BASE = r"D:\xwechat_files\wxid_rgx3r4rn7zfg22_8bc9\msg\file\2026-09"
ZIPS = ["marine antifouling.zip", "marine antifouling2.zip"]
OUT = r"D:\锂硫电池\marine-antifouling-ml\data\literature\zip_doi_candidates.csv"

# ScienceDirect PII: S + 14~18 位数字 + 可选校验字母
PII_RE = re.compile(r'S\d{13,18}[A-Za-z0-9]?', re.I)
# 文件名里直接带完整 DOI（少见但可能有）
DOI_RE = re.compile(r'10\.\d{4,9}/[^\s/]+\.[A-Za-z0-9./()\-]+', re.I)

rows = []
total = 0
pii_cnt = 0
for zname in ZIPS:
    zpath = os.path.join(ZIP_BASE, zname)
    if not os.path.exists(zpath):
        print("MISSING", zpath); continue
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if not n.lower().endswith('.pdf'):
                continue
            total += 1
            base = os.path.basename(n)
            pii = PII_RE.search(base)
            doi = DOI_RE.search(base)
            rows.append({
                "source_zip": zname,
                "filename": base,
                "filename_pii": pii.group(0) if pii else "",
                "filename_doi": doi.group(0) if doi else "",
            })
            if pii:
                pii_cnt += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cols = ["source_zip", "filename", "filename_pii", "filename_doi"]
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

print(f"总PDF: {total} | 文件名含ScienceDirect PII: {pii_cnt} | 文件名含完整DOI: {sum(1 for r in rows if r['filename_doi'])}")
print(f"候选清单已写: {OUT}")
