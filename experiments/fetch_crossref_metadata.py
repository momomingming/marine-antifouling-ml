#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务③扩库 — Step 3: CrossRef 元数据回填（需外网）
==================================================
读取 Step2 产出的 zip_pdf_metadata.csv（含 pdf_doi / pdf_title），
逐条查 CrossRef，并入 data/literature/literature_unified.csv。

相比初版的关键修复
------------------
1. **不再丢列**：初版硬编码 14 列 COLS，而 unified 现有 21 列
   （多出 methods/results/conclusions/openAccess/contentDepth/eid/pii），
   一旦写入会**静默清空这 7 列**。现改为读现有表头 + 动态追加新列。
2. **断点续跑**：进度记 progress.json，每 FLUSH 条增量落盘；中断后重跑自动跳过已处理项。
3. **自动备份**：首次运行前把 unified 备份为 literature_unified.bak-<时间戳>.csv。
4. **并发 + 重试**：4 路并发（CrossRef 礼貌池可接受），单请求失败指数退避重试 3 次。
   （实测单请求 ~10s，串行 390 条约 70 分钟，4 路可压到 ~18 分钟。）

用法
----
  python experiments/fetch_crossref_metadata.py                 # 全量回填（可中断续跑）
  python experiments/fetch_crossref_metadata.py --limit 5       # 只处理前 5 条未处理项（试跑）
  python experiments/fetch_crossref_metadata.py --dry-run       # 只查询+打印，不写任何文件
  python experiments/fetch_crossref_metadata.py --workers 2     # 降低并发
  环境变量 CROSSREF_MAILTO=you@mail.com  可进入 CrossRef 礼貌池（更稳）
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIT_DIR = os.path.join(REPO, "data", "literature")
METADATA = os.path.join(LIT_DIR, "zip_pdf_metadata.csv")
UNIFIED = os.path.join(LIT_DIR, "literature_unified.csv")
PROGRESS = os.path.join(LIT_DIR, ".crossref_progress.json")

MAILTO = os.environ.get("CROSSREF_MAILTO", "marine-antifouling@example.com")
SLEEP = 0.4          # 每请求间隔（配合 4 并发 ≈ 10 req/s 上限内）
FLUSH = 25           # 每处理 N 条落盘一次
RETRY = 5            # 单请求失败重试次数（429 限速需更长退避，见 fetch_json）

TAG_RE = re.compile(r'<[^>]+>')
_ENRICH_FIELDS = ('abstract', 'citationCount', 'year', 'url', 'authors', 'journal')


# ----------------------------------------------------------------- 工具
def clean_abstract(jats):
    if not jats:
        return ""
    return TAG_RE.sub('', jats).replace('\n', ' ').strip()


def norm_title(t):
    """标题规范化，用于模糊去重（去掉非字母数字，压缩空白）。"""
    return re.sub(r'[^0-9a-z\u4e00-\u9fff]+', '', (t or '').lower())


def fetch_json(url, retry=RETRY):
    """带退避重试的 GET。
    429（限速）单独给更长冷却：首轮全量跑出的 9 条"404"复测实为 429，
    3 次 1.5s 退避不足以穿过 CrossRef 的限速窗口。"""
    last = None
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': f'marine-antifouling/1.0 (mailto:{MAILTO})',
                'Accept': 'application/json',
            })
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:          # 429 / 5xx 等
            last = e
            wait = 12.0 if e.code == 429 else 3.0 * (attempt + 1)
            time.sleep(wait)
        except Exception as e:                       # noqa: BLE001 网络层任意异常都重试
            last = e
            time.sleep(3.0 * (attempt + 1))
    raise last


TITLE_SIM_MIN = 0.70       # 标题检索结果的相似度门槛（低于此判为误匹配，宁可跳过）


def _title_match(src_title, cr_title):
    """源 PDF 标题与 CrossRef 正式标题的相似度。

    仅靠 CrossRef 的 score 门槛（实测 20）不足以防误匹配——PDF 首行常是页眉脏数据，
    会匹配到毫不相关的论文。这里改为「前缀匹配优先 + 序列相似度兜底」：
      · 源标题被 PDF 截断时，正式标题通常以它为前缀 → 判 1.0
      · 否则算归一化后的 SequenceMatcher 相似度
    """
    a = norm_title(src_title)
    b = norm_title(cr_title)
    if not a or not b:
        return 0.0
    head = a[:min(len(a), 30)]
    if len(a) >= 12 and (b.startswith(head) or a.startswith(b[:min(len(b), 30)])):
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def clean_doi(d):
    """清洗 PDF 提取出的畸形 DOI。

    实测 `zip_pdf_metadata.csv` 里存在被截断或污染的 DOI，会直接 404：
      '10.1007/s11998-'                    ← 尾部被截断
      '10.1016/j.polymer.201'              ← 截断
      '10.1073/pnas.1201973109/-/DCSupplemental'  ← 附带补充材料后缀
      '10.11933/j.issn.1007−9289.20181130002'     ← 用了 U+2212 减号而非连字符
    这些无法靠重试救回，只能清洗后交给标题检索兜底。
    """
    s = (d or '').strip()
    s = s.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')
    for pre in ('https://doi.org/', 'http://dx.doi.org/', 'http://doi.org/', 'doi:'):
        if s.lower().startswith(pre):
            s = s[len(pre):]
    s = re.split(r'/-/', s)[0]        # 去掉 /-/DCSupplemental 之类的附加路径
    return s.strip().rstrip('.,;').strip()


def _search_by_title(title):
    """标题检索：正确解析 /works?query.bibliographic 的返回结构。

    注意该接口返回的是 message.items[]（列表），**不是**单个 work 对象——
    初版直接把它当 work 用，导致 row['title'] 恒为空、标题分支静默失效。
    """
    url = (f"https://api.crossref.org/works?"
           f"query.bibliographic={quote(title)}&rows=1&mailto={MAILTO}")
    msg = fetch_json(url).get('message', {})
    items = msg.get('items') or []
    if not items:
        return None, 0.0
    top = items[0]
    return top, float(top.get('score') or 0.0)


def fetch_work(doi, title):
    """按 DOI 精确查；DOI 畸形或 404 时回退到标题检索。返回 (work, how)。"""
    doi = clean_doi(doi)
    if doi:
        try:
            msg = fetch_json(f"https://api.crossref.org/works/{quote(doi)}"
                             f"?mailto={MAILTO}").get('message', {})
            if msg.get('DOI'):
                return msg, 'doi'
        except urllib.error.HTTPError as e:
            if e.code != 404:            # 429 等已在 fetch_json 内重试过，抛给上层
                raise
        except Exception:
            raise
    if title:
        work, score = _search_by_title(title)
        if work is not None:
            sim = _title_match(title, (work.get('title') or [''])[0])
            if sim >= TITLE_SIM_MIN:
                return work, f'title sim={sim:.2f}'
    return None, None


def work_to_row(src_row, msg):
    title = (msg.get('title') or [''])[0]
    authors = "; ".join(
        f"{a.get('given','')} {a.get('family','')}".strip()
        for a in msg.get('author', []) if a.get('family'))
    container = (msg.get('container-title') or [''])[0]
    year = ''
    for key in ('published-print', 'published-online', 'issued'):
        dp = (msg.get(key) or {}).get('date-parts', [['']])[0]
        if dp and dp[0]:
            year = str(dp[0])
            break
    return {
        'paperId': '', 'title': title, 'titleZh': '',
        'authors': authors, 'journal': container, 'year': year,
        'doi': msg.get('DOI', ''), 'citationCount': str(msg.get('is-referenced-by-count', '') or ''),
        'abstract': clean_abstract(msg.get('abstract', '')), 'url': msg.get('URL', ''),
        'source': 'crossref-batch', 'filename': src_row.get('filename', ''),
        'pages': '', 'size_mb': '',
    }


def load_unified():
    """读现有库：返回 (rows, fieldnames)。fieldnames 保持原序，绝不丢列。"""
    if not os.path.exists(UNIFIED):
        return [], ['paperId', 'title', 'titleZh', 'authors', 'journal', 'year',
                    'doi', 'citationCount', 'abstract', 'url', 'source',
                    'filename', 'pages', 'size_mb']
    with open(UNIFIED, encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames or [])
        rows = list(rd)
    return rows, fieldnames


def build_index(rows):
    """DOI / 规范化标题 → 行号。"""
    idx = {}
    for i, r in enumerate(rows):
        d = (r.get('doi') or '').strip().lower()
        t = norm_title(r.get('title'))
        if d:
            idx.setdefault(('doi', d), i)
        if t:
            idx.setdefault(('title', t), i)
    return idx


def save(rows, fieldnames):
    tmp = UNIFIED + '.tmp'
    with open(tmp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    os.replace(tmp, UNIFIED)


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='最多处理 N 条未处理项（0=全部）')
    ap.add_argument('--dry-run', action='store_true', help='只查询打印，不写文件')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--fresh', action='store_true', help='忽略进度文件，从头跑')
    args = ap.parse_args()

    if not os.path.exists(METADATA):
        print("缺少", METADATA, "请先跑 extract_pdf_dois.py")
        return 1

    rows, fieldnames = load_unified()
    print(f"现有库: {len(rows)} 条 / {len(fieldnames)} 列")

    def _blank(field, rs):
        """空值计数。统一 str() 兜底：CrossRef 的 is-referenced-by-count 是 int，
        与 CSV 读入的 str 混在同一批 rows 里，直接 .strip() 会抛 AttributeError。"""
        return sum(1 for r in rs if not str(r.get(field) or '').strip())

    missing_pre = {f: _blank(f, rows) for f in ('abstract', 'citationCount', 'year')}
    print("回填前空缺:", missing_pre)

    with open(METADATA, encoding='utf-8-sig', newline='') as f:
        src_rows = list(csv.DictReader(f))
    print(f"源清单: {len(src_rows)} 条")

    # 进度
    done = set()
    if os.path.exists(PROGRESS) and not args.fresh:
        try:
            done = set(json.load(open(PROGRESS, encoding='utf-8')).get('done', []))
            print(f"读到进度: 已完成 {len(done)} 条")
        except Exception:
            done = set()

    todo = [(i, s) for i, s in enumerate(src_rows) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("无待处理项。")
        return 0
    print(f"本次待处理: {len(todo)} 条（{args.workers} 并发）\n")

    # 备份（仅真实写入时）
    if not args.dry_run and rows:
        bak = os.path.join(
            LIT_DIR, f"literature_unified.bak-{time.strftime('%Y%m%d-%H%M%S')}.csv")
        with open(bak, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in fieldnames})
        print("已备份 ->", os.path.basename(bak), "\n")

    idx = build_index(rows)
    lock = threading.Lock()
    stats = {'added': 0, 'enriched': 0, 'skip': 0, 'fail': 0,
             'field': {k: 0 for k in _ENRICH_FIELDS}}
    pending = []          # 待落盘的新行（dry-run 下不落）

    def handle(item):
        i, src = item
        doi = (src.get('pdf_doi') or '').strip()
        title = (src.get('pdf_title') or '').strip()
        if not doi and not title:
            return i, 'skip', None, '无 DOI / 标题'
        try:
            work, how = fetch_work(doi, title)
            if work is None:
                return i, 'skip', None, f'DOI 无效且标题检索无可靠命中: {(doi or title)[:40]}'
            row = work_to_row(src, work)
            if not row['title']:
                return i, 'skip', None, f'空标题: {(doi or title)[:45]}'
            return i, 'ok', row, ('' if how == 'doi' else how)
        except Exception as e:                       # noqa: BLE001
            return i, 'fail', None, f'{type(e).__name__}: {e}'

    t0 = time.time()
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(handle, it): it for it in todo}
        for fut in as_completed(futs):
            i, status, row, note = fut.result()
            processed += 1
            if status == 'ok':
                k2 = (('doi', row['doi'].lower()) if row['doi']
                      else ('title', norm_title(row['title'])))
                with lock:
                    if k2 in idx:
                        base = rows[idx[k2]]
                        filled = []
                        for fld in _ENRICH_FIELDS:
                            v = str(row.get(fld) or '').strip()
                            if v and not str(base.get(fld) or '').strip():
                                base[fld] = v
                                stats['field'][fld] += 1
                                filled.append(fld)
                        stats['enriched'] += 1
                        print(f"  [{processed}/{len(todo)}] ~ 补全 {filled or '无新字段'} | "
                              f"{row['title'][:50]}{' ← ' + note if note else ''}")
                    else:
                        for k in fieldnames:
                            row.setdefault(k, '')
                        rows.append(row)
                        if row['doi']:
                            idx[('doi', row['doi'].lower())] = len(rows) - 1
                        if row['title']:
                            idx[('title', norm_title(row['title']))] = len(rows) - 1
                        stats['added'] += 1
                        print(f"  [{processed}/{len(todo)}] + {row['title'][:55]} "
                              f"({row.get('year','')}){' ← ' + note if note else ''}")
                    done.add(i)
                    if not args.dry_run and processed % FLUSH == 0:
                        save(rows, fieldnames)
                        json.dump({'done': sorted(done), 'updated': time.time()},
                                  open(PROGRESS, 'w', encoding='utf-8'))
                        print(f"      …已落盘（库 {len(rows)} 条 / 已处理 {processed}）")
            elif status == 'skip':
                stats['skip'] += 1
                done.add(i)
                print(f"  [{processed}/{len(todo)}] - 跳过 {note}")
            else:
                stats['fail'] += 1
                print(f"  [{processed}/{len(todo)}] ! 失败 {note}")
            time.sleep(SLEEP / max(1, args.workers))

    dt = time.time() - t0
    if not args.dry_run:
        save(rows, fieldnames)
        json.dump({'done': sorted(done), 'updated': time.time()},
                  open(PROGRESS, 'w', encoding='utf-8'))
        print(f"\n已写回 -> {UNIFIED}（总计 {len(rows)} 条 / {len(fieldnames)} 列）")
    missing_post = {f: _blank(f, rows) for f in ('abstract', 'citationCount', 'year')}
    print(f"回填后空缺: {missing_post}")
    print(f"\n完成: 新增 {stats['added']} | 补全 {stats['enriched']} | "
          f"跳过 {stats['skip']} | 失败 {stats['fail']} | 耗时 {dt:.0f}s")
    print("字段补全明细:", stats['field'])
    if args.dry_run:
        print("(--dry-run: 未写任何文件)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
