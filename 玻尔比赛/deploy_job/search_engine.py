# -*- coding: utf-8 -*-
"""search_engine.py — 海洋防污平台「物质输入」智能搜索模块

特性
----
1. 多字段检索：名称(name) / 别名(alias_of) / SMILES / 类别(cls) 全覆盖。
2. 模糊匹配：基于 difflib 序列比对的近似匹配，容忍拼写误差与顺序差异。
3. 相关度打分与排序：精确 > 别名 > 前缀 > 分词前缀 > 包含 > 类别 > SMILES > 近似。
4. 类别映射：英文类别键 -> 中文展示名。
5. 结果高亮摘要：生成带相关度、匹配类型、类别、SMILES 的 Markdown。
6. 零第三方依赖，索引仅在首次构建（模块级缓存），后续搜索均为 O(n) 且极快。

用法
----
    from search_engine import get_searcher
    searcher = get_searcher(material_records)          # records: list[dict]
    ranked = searcher.search("pdms", max_results=20)   # -> list[dict]
    choices = [searcher.to_choice(r) for r in ranked]  # 兼容原 Dropdown 格式
    md = searcher.format_results_md(ranked, "pdms")    # 展示用 Markdown
"""
from __future__ import annotations

import difflib
import re

# 类别英文键 -> 中文展示名
CLS_CN = {
    "self_polishing": "自抛光",
    "biocide": "杀生剂",
    "nanocomposite": "纳米复合",
    "smart": "智能响应",
    "noncomposite": "非复合",
}

# 分词分隔符（名称里常见的连接符）
_TOKEN_SPLIT = re.compile(r"[\s/\-·,]+")

# 模糊匹配最低相似度阈值（低于此值视为不相关，避免噪声）
FUZZY_THRESHOLD = 0.6


class MaterialSearcher:
    """对物质记录建立可检索索引并提供排名搜索。"""

    def __init__(self, records):
        self.records = []
        self.index = []
        self.build(records)

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------
    def build(self, records):
        self.records = []
        self.index = []
        for r in records or []:
            name = (r.get("name") or "").strip()
            smiles = (r.get("smiles") or "").strip()
            cls = (r.get("cls") or "")
            alias_of = r.get("alias_of")
            rec = dict(r)
            rec["name"] = name
            rec["smiles"] = smiles
            rec["cls"] = cls
            name_l = name.lower()
            tokens = [t for t in _TOKEN_SPLIT.split(name_l) if t]
            self.records.append(rec)
            self.index.append({
                "name_l": name_l,
                "tokens": tokens,
                "smiles_l": smiles.lower(),
                "cls_l": cls.lower(),
                "cls_cn": CLS_CN.get(cls, cls),
                "alias_of_l": (alias_of or "").lower(),
            })

    # ------------------------------------------------------------------
    # 相似度
    # ------------------------------------------------------------------
    @staticmethod
    def _ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    # ------------------------------------------------------------------
    # 单条打分
    # ------------------------------------------------------------------
    def _score(self, q: str, qtokens, idx):
        name_l = idx["name_l"]
        # 1) 精确名称
        if name_l == q:
            return 100.0, "精确名称"
        # 2) 别名命中
        if idx["alias_of_l"] == q:
            return 96.0, "别名"
        # 3) 名称前缀
        if name_l.startswith(q):
            return 92.0, "前缀"
        # 4) 名称分词前缀（如 "pdms" 命中 "poly ... pdms" 的末词）
        for t in idx["tokens"]:
            if t.startswith(q):
                return 88.0, "分词前缀"
        # 5) 名称包含
        if q in name_l:
            return 72.0, "名称包含"
        # 6) 类别匹配（英文键或中文名）
        if q in idx["cls_l"] or q in idx["cls_cn"].lower():
            return 62.0, "类别"
        # 7) SMILES 片段（适合直接粘 SMILES 查询）
        if q in idx["smiles_l"]:
            return 52.0, "SMILES"
        # 8) 模糊近似（仅在查询较长、且无上述命中时启用，避免噪声）
        if len(q) >= 3:
            best = 0.0
            best_type = ""
            r = self._ratio(q, name_l)
            if r > best:
                best, best_type = r, "名称近似"
            for t in idx["tokens"]:
                r2 = self._ratio(q, t)
                if r2 > best:
                    best, best_type = r2, "分词近似"
            if best >= FUZZY_THRESHOLD:
                return 40.0 + best * 10.0, best_type
        return 0.0, ""

    # ------------------------------------------------------------------
    # 对外搜索
    # ------------------------------------------------------------------
    def search(self, query, max_results=20, fuzzy_threshold=FUZZY_THRESHOLD):
        global FUZZY_THRESHOLD
        q = (query or "").strip().lower()
        if not q:
            # 空查询：浏览模式，按名称排序返回（上限放宽到 50）
            out = [self._result(rec, i, 0.0, "浏览")
                   for i, rec in enumerate(self.records)]
            out.sort(key=lambda x: x["name"])
            return out[:max(20, min(50, max_results))]
        qtokens = [t for t in _TOKEN_SPLIT.split(q) if t]
        results = []
        for i, idx in enumerate(self.index):
            score, mtype = self._score(q, qtokens, idx)
            if score > 0:
                results.append(self._result(self.records[i], i, score, mtype))
        results.sort(key=lambda x: (-x["score"], x["name"]))
        return results[:max_results]

    # ------------------------------------------------------------------
    # 结果包装与展示
    # ------------------------------------------------------------------
    def _result(self, rec, i, score, mtype):
        return {
            "name": rec["name"],
            "smiles": rec["smiles"],
            "cls": rec["cls"],
            "cls_cn": CLS_CN.get(rec["cls"], rec["cls"]),
            "score": round(score, 1),
            "match": mtype,
            "index": i,
        }

    def to_choice(self, rec) -> str:
        """兼容原 Dropdown 的取值格式：'name | smiles'（供 add_to_input 解析）。"""
        return f"{rec['name']} | {rec['smiles']}"

    def format_results_md(self, results, query=""):
        """生成排名结果 Markdown，用于搜索结果展示面板。"""
        if not results:
            return "未找到匹配物质，试试更短的关键词，或直接粘贴 SMILES 片段。"
        head = f"**搜索 “{query}” · 命中 {len(results)} 项（按相关度排序）**\n"
        lines = [head]
        for r in results[:12]:
            if r["score"] >= 85:
                mark = "★"          # 高相关
            elif r["score"] >= 60:
                mark = "☆"          # 中相关
            else:
                mark = "·"          # 弱相关/近似
            lines.append(
                f"{mark} **{r['name']}**　`{r['cls_cn']}`　_{r['match']}_　"
                f"(相关度 {r['score']})\n"
                f"　- SMILES: `{r['smiles']}`"
            )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 模块级单例：索引只构建一次
# ----------------------------------------------------------------------
_searcher = None


def get_searcher(records):
    global _searcher
    if _searcher is None:
        _searcher = MaterialSearcher(records)
    return _searcher


def reset_searcher():
    """在记录源变化时强制重建索引。"""
    global _searcher
    _searcher = None


if __name__ == "__main__":
    # 最小自测
    demo = [
        {"name": "聚二甲基硅氧烷 PDMS", "smiles": "C[Si](C)(C)O[Si](C)(C)C", "cls": "noncomposite"},
        {"name": "PDMS/ZnO", "smiles": "C[Si](C)(C)O[Si](C)(C)C.[Zn]", "cls": "nanocomposite"},
        {"name": "SPC", "smiles": "O=C(O)C(C)O", "cls": "biocide"},
        {"name": "PDMS (别名 PD)", "smiles": "C[Si](C)(C)O[Si](C)(C)C", "cls": "noncomposite", "alias_of": "聚二甲基硅氧烷 PDMS"},
    ]
    s = MaterialSearcher(demo)
    for q in ["pdms", "pdm", "聚二甲", "杀菌", "O=C(O)", "xyz"]:
        print(f"\n== query: {q} ==")
        for r in s.search(q, max_results=5):
            print(f"  {r['score']:>5}  {r['match']:<6}  {r['name']}  [{r['cls_cn']}]")
    print("\n" + s.format_results_md(s.search("pdms"), "pdms"))
