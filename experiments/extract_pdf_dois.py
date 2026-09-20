#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务③扩库 - 本地预处理 Step 2（重活，建议后台跑）
从两份 zip 的 PDF 正文首页抽取真实 DOI + 标题首行，增量写入 CSV。
不依赖网络（仅本地 PDF 解析）。扫描版 PDF 无文本层会留空，需后续 OCR。
输出: data/literature/zip_pdf_metadata.csv
"""
import zipfile, re, os, csv, io

ZIP_BASE = r"D:\xwechat_files\wxid_rgx3r4rn7zfg22_8bc9\msg\file\2026-09"
ZIPS = ["marine antifouling.zip", "marine antifouling2.zip"]
UNIFIED = r"D:\锂硫电池\marine-antifouling-ml\data\literature\literature_unified.csv"
OUT = r"D:\锂硫电池\marine-antifouling-ml\data\literature\zip_pdf_metadata.csv"

from pypdf import PdfReader

PII_RE = re.compile(r'S\d{13,18}[A-Za-z0-9]?', re.I)
DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>\]\)]{5,}', re.I)
TITLE_JUNK = re.compile(r'^(journal|sciencedirect|available online|received|accepted|abstract|keywords|https?://|doi|www\.)', re.I)

# 已入库 DOI 集合（用于统计新增）
unified_dois = set()
if os.path.exists(UNIFIED):
    with open(UNIFIED, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("doi") or "").strip().lower()
            if d:
                unified_dois.add(d)

cols = ["source_zip", "filename", "filename_pii", "pdf_doi", "pdf_title", "text_layers"]
write_header = not os.path.exists(OUT)
found = 0
new_doi = 0
scanned_only = 0
with open(OUT, "a", encoding="utf-8-sig", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols)
    if write_header:
        w.writeheader()
    for zname in ZIPS:
        zpath = os.path.join(ZIP_BASE, zname)
        if not os.path.exists(zpath):
            continue
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if not n.lower().endswith(".pdf"):
                    continue
                base = os.path.basename(n)
                pii = PII_RE.search(base)
                rec = {"source_zip": zname, "filename": base,
                       "filename_pii": pii.group(0) if pii else "",
                       "pdf_doi": "", "pdf_title": "", "text_layers": "1"}
                try:
                    data = z.read(n)
                    reader = PdfReader(io.BytesIO(data))
                    text = ""
                    for pg in reader.pages[:2]:
                        try:
                            text += (pg.extract_text() or "") + "\n"
                        except Exception:
                            pass
                    if not text.strip():
                        rec["text_layers"] = "0"   # 扫描版，无文本层
                        scanned_only += 1
                        w.writerow(rec); continue
                    m = DOI_RE.search(text)
                    if m:
                        doi = m.group(0).rstrip(".,;)$%").strip()
                        rec["pdf_doi"] = doi
                        found += 1
                        if doi.lower() not in unified_dois:
                            new_doi += 1
                    # 标题首行启发式
                    for line in text.splitlines():
                        line = line.strip()
                        if len(line) >= 15 and not TITLE_JUNK.match(line):
                            rec["pdf_title"] = line[:200]
                            break
                except Exception as e:
                    rec["text_layers"] = "err:" + type(e).__name__
                w.writerow(rec)
print(f"PDF正文DOI抽取完成: 成功抽取DOI={found} | 其中新增(不在统一库)={new_doi} | 扫描版无文本={scanned_only}")
print(f"明细已增量写入: {OUT}")
