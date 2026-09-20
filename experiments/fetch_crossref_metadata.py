#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务③扩库 - 通网后执行 Step 3（需外网）
读取 experiments Step2 的 zip_pdf_metadata.csv（含 pdf_doi / pdf_title），
逐条去 CrossRef 拉元数据，并入 data/literature/literature_unified.csv。

用法:
  python experiments/fetch_crossref_metadata.py
前置: 网络可达 api.crossref.org（开代理后运行）

CrossRef 礼貌池: 带 mailto 参数 + 每次请求间隔 >=1s。
"""
import csv, os, re, sys, time, json
from urllib.parse import quote
import urllib.request

METADATA = r"D:\锂硫电池\marine-antifouling-ml\data\literature\zip_pdf_metadata.csv"
UNIFIED = r"D:\锂硫电池\marine-antifouling-ml\data\literature\literature_unified.csv"
MAILTO = "user@example.com"   # 改成你的邮箱以进入 CrossRef 礼貌池(更快)
SLEEP = 1.0

COLS = ['paperId','title','titleZh','authors','journal','year','doi','citationCount','abstract','url','source','filename','pages','size_mb']

TAG_RE = re.compile(r'<[^>]+>')

def clean_abstract(jats):
    if not jats:
        return ""
    return TAG_RE.sub('', jats).replace('\n', ' ').strip()

def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': f'marine-antifouling/1.0 (mailto:{MAILTO})'})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode('utf-8'))

def work_to_row(src_row, msg):
    title = (msg.get('title') or [''])[0]
    authors = "; ".join(f"{a.get('given','')} {a.get('family','')}".strip() for a in msg.get('author', []) if a.get('family'))
    container = (msg.get('container-title') or [''])[0]
    year = ''
    for key in ('published-print', 'published-online', 'issued'):
        dp = msg.get(key, {}).get('date-parts', [['']])[0]
        if dp and dp[0]:
            year = str(dp[0]); break
    doi = msg.get('DOI', '')
    cited = msg.get('is-referenced-by-count', '')
    abstract = clean_abstract(msg.get('abstract', ''))
    url = msg.get('URL', '')
    return {
        'paperId': '', 'title': title, 'titleZh': '',
        'authors': authors, 'journal': container, 'year': year,
        'doi': doi, 'citationCount': cited, 'abstract': abstract,
        'url': url, 'source': 'crossref-batch',
        'filename': src_row.get('filename', ''), 'pages': '', 'size_mb': '',
    }

def main():
    if not os.path.exists(METADATA):
        print("缺少", METADATA, "请先跑 extract_pdf_dois.py"); sys.exit(1)

    # 载入已入库 DOI/title 索引（用于去重 + 补全已有条目）
    idx = {}
    rows_unified = []
    if os.path.exists(UNIFIED):
        with open(UNIFIED, encoding='utf-8-sig', newline='') as f:
            rows_unified = list(csv.DictReader(f))
        for i, r in enumerate(rows_unified):
            d = (r.get('doi') or '').strip().lower()
            t = (r.get('title') or '').strip().lower()
            if d: idx[('doi', d)] = i
            if t: idx[('title', t)] = i

    added = 0
    enriched = 0
    with open(METADATA, encoding='utf-8-sig', newline='') as f:
        src_rows = list(csv.DictReader(f))

    for i, src in enumerate(src_rows, 1):
        doi = (src.get('pdf_doi') or '').strip()
        title = (src.get('pdf_title') or '').strip()
        key = ('doi', doi.lower()) if doi else ('title', title.lower()) if title else None
        try:
            if doi:
                url = f"https://api.crossref.org/works/{quote(doi)}?mailto={MAILTO}"
            elif title:
                url = f"https://api.crossref.org/works?query.bibliographic={quote(title)}&rows=1&mailto={MAILTO}"
            else:
                continue
            msg = fetch_json(url).get('message', {})
            if not doi and msg.get('score', 0) < 20:   # 标题模糊匹配过低，跳过
                print(f"  [{i}/{len(src_rows)}] 低置信跳过: {title[:50]}")
                time.sleep(SLEEP); continue
            row = work_to_row(src, msg)
            if not row['title']:
                time.sleep(SLEEP); continue
            k2 = ('doi', row['doi'].lower()) if row['doi'] else ('title', row['title'].lower()) if row['title'] else None
            if k2 and k2 in idx:
                # 补全已有条目缺失字段（abstract/被引数/year/url）
                b = rows_unified[idx[k2]]
                for fld in ('abstract', 'citationCount', 'year', 'url', 'authors', 'journal'):
                    v = (row.get(fld) or '').strip()
                    if v and not (b.get(fld) or '').strip():
                        b[fld] = v
                enriched += 1
                print(f"  [{i}/{len(src_rows)}] ~ 补全 {row['title'][:55]}")
            else:
                rows_unified.append(row)
                if row['doi']: idx[('doi', row['doi'].lower())] = len(rows_unified) - 1
                if row['title']: idx[('title', row['title'].lower())] = len(rows_unified) - 1
                added += 1
                print(f"  [{i}/{len(src_rows)}] + {row['title'][:60]} ({row.get('year','')})")
        except Exception as e:
            print(f"  [{i}/{len(src_rows)}] 失败 {doi or title[:40]}: {e}")
        time.sleep(SLEEP)

    with open(UNIFIED, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows_unified)
    print(f"\n完成: 新增 {added} 条 + 补全 {enriched} 条 -> {UNIFIED}（总计 {len(rows_unified)} 篇）")

if __name__ == '__main__':
    main()
