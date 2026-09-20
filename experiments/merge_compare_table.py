"""
合并 ScienceDirect/Scopus 导出的 compare-table 到统一文献库
==========================================================
输入 CSV 列: PII, DOI, EID, Publication Date, Open Access, Content Depth,
             Source(实为文章标题), Abstract, Methods, Results, Conclusions
统一库在原有 schema 上扩展: methods/results/conclusions/openAccess/contentDepth/eid/pii

策略:
  - 主键 = DOI(小写), 无 DOI 时回退标题
  - 已存在 → 补全空缺字段(更长的 abstract 优先, 新增的结构化字段优先)
  - 不存在 → 新增条目

用法:
  python experiments/merge_compare_table.py [源CSV路径]
"""
import os
import re
import sys
import csv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED = os.path.join(REPO, "data", "literature", "literature_unified.csv")
DEFAULT_SRC = r"D:\edge下载\compare-table.csv"

# 统一库字段 (扩展后)
COLS = [
    "paperId", "title", "titleZh", "authors", "journal", "year", "doi",
    "citationCount", "abstract", "url", "source", "filename", "pages",
    "size_mb", "methods", "results", "conclusions", "openAccess",
    "contentDepth", "eid", "pii",
]
NEW_COLS = ["methods", "results", "conclusions", "openAccess", "contentDepth", "eid", "pii"]
# 部署侧子集 schema (与 deployments/hf_space/literature_db.csv 保持一致)
DEPLOY_COLS = ["paperId", "title", "titleZh", "authors", "journal", "year",
               "doi", "citationCount", "abstract", "url", "source"]

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s]+")


def norm(s):
    return (s or "").strip().lower()


def parse_year(s):
    m = re.search(r"(\d{4})", s or "")
    return m.group(1) if m else ""


def doi_of(url):
    m = DOI_RE.search(url or "")
    return m.group(0).rstrip(".").rstrip("/") if m else ""


def load_unified(path):
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = list(r.fieldnames or COLS)
    return rows, cols


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    if not os.path.exists(src):
        print(f"源文件不存在: {src}")
        return 1

    with open(src, encoding="utf-8-sig", newline="") as f:
        incoming = list(csv.DictReader(f))
    print(f"源文件: {src} | 行数: {len(incoming)}")

    rows, cols = load_unified(UNIFIED)
    # schema 扩展: 补齐新增列
    for c in NEW_COLS:
        if c not in cols:
            cols.append(c)
    for r in rows:
        for c in cols:
            r.setdefault(c, "")

    index = {}
    for i, r in enumerate(rows):
        d = norm(r.get("doi"))
        k = ("doi", d) if d else ("title", norm(r.get("title")))
        if k[1]:
            index.setdefault(k, i)

    added = enriched = skipped = 0
    for src_row in incoming:
        doi = doi_of(src_row.get("DOI", ""))
        title = (src_row.get("Source") or "").strip()   # ScienceDirect 导出把标题放在 Source 列
        key = ("doi", norm(doi)) if doi else ("title", norm(title))
        if not key[1]:
            skipped += 1
            continue

        rec = {
            "paperId": "", "title": title, "titleZh": "", "authors": "",
            "journal": "", "year": parse_year(src_row.get("Publication Date", "")),
            "doi": doi, "citationCount": "", "abstract": (src_row.get("Abstract") or "").strip(),
            "url": (src_row.get("DOI") or "").strip(), "source": "ScienceDirect",
            "filename": "", "pages": "", "size_mb": "",
            "methods": (src_row.get("Methods") or "").strip(),
            "results": (src_row.get("Results") or "").strip(),
            "conclusions": (src_row.get("Conclusions") or "").strip(),
            "openAccess": (src_row.get("Open Access") or "").strip(),
            "contentDepth": (src_row.get("Content Depth") or "").strip(),
            "eid": (src_row.get("EID") or "").strip(),
            "pii": (src_row.get("PII") or "").strip(),
        }

        if key in index:
            tgt = rows[index[key]]
            changed = False
            for c in NEW_COLS:                       # 新增结构化字段: 空缺才补
                if rec[c] and not (tgt.get(c) or "").strip():
                    tgt[c] = rec[c]; changed = True
            if rec["title"] and not (tgt.get("title") or "").strip():
                tgt["title"] = rec["title"]; changed = True
            if len(rec["abstract"]) > len(tgt.get("abstract") or ""):
                tgt["abstract"] = rec["abstract"]; changed = True
            if rec["year"] and not (tgt.get("year") or "").strip():
                tgt["year"] = rec["year"]; changed = True
            if rec["url"] and not (tgt.get("url") or "").strip():
                tgt["url"] = rec["url"]; changed = True
            enriched += 1 if changed else 0
        else:
            index[key] = len(rows)
            rows.append(rec)
            added += 1

    with open(UNIFIED, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 同步部署侧副本 (deploy_job 原本缺失该文件 → 文献面板为空)
    targets = [
        os.path.join(REPO, "deployments", "hf_space", "literature_db.csv"),
        os.path.join(REPO, "deployments", "deploy_job", "literature_db.csv"),
    ]
    for t in targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        with open(t, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DEPLOY_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  同步部署副本: {os.path.relpath(t, REPO)}")

    has_doi = sum(1 for r in rows if r.get("doi"))
    has_abs = sum(1 for r in rows if (r.get("abstract") or "").strip())
    has_mr = sum(1 for r in rows if (r.get("methods") or "").strip())
    print(f"\n完成: 统一库 {len(rows)} 篇 | 新增 {added} | 补全 {enriched} | 跳过 {skipped}")
    print(f"字段覆盖: doi {has_doi} | abstract {has_abs} | methods/results {has_mr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
