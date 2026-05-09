"""
记忆存储模块 — 借鉴 AGI 长期记忆机制

核心能力：
  - 每次分析运行后，将完整结果存入 ChromaDB 内存数据库
  - 查询某只股票的历史状态演变（L0→L5 各维度变化）
  - 追踪信号"速度"（如：光模块从 L3→L4 传导加速）
  - 时序对比：当前状态 vs 历史相似状态

使用 ChromaDB 的 Embedding 能力实现语义级相似搜索：
  搜索"类似当前光模块景气度扩散的市场状态"
  → 返回历史上最相似的时期及其后续走势

数据目录：./memory/  （Git 已忽略，属于本地持久化数据）
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd

# ChromaDB 安装检查
try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

# ── 配置 ──
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory")
SNAPSHOT_COLLECTION = "analysis_snapshots"     # 分析快照
TREND_COLLECTION = "signal_trends"             # 信号趋势
SIMILARITY_COLLECTION = "state_similarity"     # 状态相似搜索


class MemoryStore:
    """
    记忆存储
    将每次分析结果向量化存储，支持时序查询和相似搜索
    """

    def __init__(self, persist_dir: str = None):
        if not _HAS_CHROMA:
            raise ImportError(
                "ChromaDB 未安装。运行: pip install chromadb"
            )

        self.persist_dir = persist_dir or MEMORY_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        self.snapshots = self._get_collection(SNAPSHOT_COLLECTION)
        self.trends = self._get_collection(TREND_COLLECTION)
        self.states = self._get_collection(SIMILARITY_COLLECTION)

    def _get_collection(self, name: str):
        """获取集合（已存在则加载，否则创建）"""
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

    # ═══════════════════════════════════════════════════════════════════════
    # 1. 保存分析快照
    # ═══════════════════════════════════════════════════════════════════════

    def save_analysis(self, analysis_data: dict,
                      date: str = None) -> str:
        """
        保存一次完整的六层分析结果到记忆库

        analysis_data: export_json.py 产出的完整分析 dict
        date:          "YYYY-MM-DD" 格式日期，默认今天

        返回 doc_id
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        doc_id = f"{date}_{hashlib.md5(json.dumps(analysis_data, ensure_ascii=False).encode()).hexdigest()[:12]}"

        # 构建可搜索的文本表示
        search_text = self._build_search_text(analysis_data)

        # 提取元数据用于过滤查询
        metadata = self._extract_metadata(analysis_data, date)

        # 存入 ChromaDB
        self.snapshots.add(
            ids=[doc_id],
            documents=[search_text],
            metadatas=[metadata]
        )

        # 同时记录各股票的状态演进
        self._record_stock_trends(analysis_data, date)

        return doc_id

    def _build_search_text(self, data: dict) -> str:
        """将分析结果转为可搜索的文本"""
        parts = []

        if "risk" in data:
            r = data["risk"]
            parts.append(f"风险等级{r.get('level','N/A')} 评分{r.get('score',0)}")

        if "layer1" in data:
            l1 = data["layer1"]
            trigger = l1.get("triggered_signals", [])
            if trigger:
                parts.append(f"实业信号: {'; '.join(trigger[:5])}")

        if "layer2" in data:
            l2 = data["layer2"]
            parts.append(f"金融评分{l2.get('score',0)} 期货信号{l2.get('signal_count',0)}个")

        if "layer3" in data:
            l3 = data["layer3"]
            if isinstance(l3, dict):
                for ch in (l3.get("active") or l3.get("chains") or []):
                    parts.append(f"产业链{ch.get('chain','')} 激活强度{ch.get('strength',0)}")

        if "layer4" in data:
            l4 = data["layer4"]
            if isinstance(l4, list):
                for p in l4:
                    parts.append(f"个股{p.get('name','')} "
                                 f"评分{p.get('scores',{}).get('total',0)} "
                                 f"判定{p.get('verdict','')}")
            elif isinstance(l4, dict):
                for p in (l4.get("profiles") or []):
                    parts.append(f"个股{p.get('name','')} "
                                 f"评分{p.get('scores',{}).get('total',0)} "
                                 f"判定{p.get('verdict','')}")

        if "decisions" in data:
            decisions = data["decisions"]
        elif "layer5" in data:
            decisions = data["layer5"]
        else:
            decisions = []
        for d in (decisions or []):
            parts.append(f"决策{d.get('name','')} "
                         f"方向{d.get('verdict','')} "
                         f"仓位{d.get('position_pct',0)}% "
                         f"止损{d.get('stop_loss_pct',0)}%")

        return " | ".join(parts)

    def _extract_metadata(self, data: dict, date: str) -> dict:
        """提取元数据字段用于过滤查询"""
        meta = {"date": date}

        # 风险等级
        if "risk" in data:
            meta["risk_level"] = data["risk"].get("level", "N/A")
            meta["risk_score"] = data["risk"].get("score", 0)

        # Layer 1 评分
        if "layer1" in data:
            meta["l1_score"] = data["layer1"].get("score", 0)

        # Layer 2 评分
        if "layer2" in data:
            meta["l2_score"] = data["layer2"].get("score", 0)

        # Layer 3 活跃链数
        active_chains = 0
        if "layer3" in data:
            l3 = data["layer3"]
            if isinstance(l3, dict):
                for ch in (l3.get("active") or l3.get("chains") or []):
                    if ch.get("active"):
                        active_chains += 1
        meta["active_chains"] = active_chains

        # Layer 4 优秀股数
        real_count = 0
        if "layer4" in data:
            l4 = data["layer4"]
            if isinstance(l4, list):
                profiles_iter = l4
            elif isinstance(l4, dict):
                profiles_iter = l4.get("profiles") or []
            else:
                profiles_iter = []
            for p in profiles_iter:
                if p.get("verdict") == "REAL":
                    real_count += 1
        meta["real_stocks"] = real_count

        # 决策数
        decisions_data = data.get("decisions") or data.get("layer5") or []
        go_count = sum(1 for d in decisions_data
                       if d.get("verdict") == "GO")
        meta["go_decisions"] = go_count

        return meta

    # ═══════════════════════════════════════════════════════════════════════
    # 2. 个股趋势追踪
    # ═══════════════════════════════════════════════════════════════════════

    def _record_stock_trends(self, data: dict, date: str):
        """记录各只股票的评分/仓位变化趋势"""
        decisions = data.get("decisions") or data.get("layer5") or []
        if not decisions:
            return

        for dec in decisions:
            code = dec.get("code", "")
            name = dec.get("name", "")
            if not code:
                continue

            trend_text = (
                f"日期{date} {name}({code}) "
                f"L0_pass={dec.get('l0_pass')} "
                f"L1_pass={dec.get('l1_pass')} "
                f"L2_pass={dec.get('l2_pass')} "
                f"L3_pass={dec.get('l3_pass')} "
                f"L4_score={dec.get('l4_score',0)} "
                f"仓位={dec.get('position_pct',0)}% "
                f"方向={dec.get('verdict','')} "
                f"时序信号={dec.get('timing','')} "
                f"PE={dec.get('pe',0)} "
                f"现金流={dec.get('cash_flow_ratio','N/A')}"
            )

            trend_id = f"trend_{code}_{date.replace('-','')}"
            self.trends.add(
                ids=[trend_id],
                documents=[trend_text],
                metadatas=[{
                    "date": date,
                    "code": code,
                    "name": name,
                    "l0_pass": int(dec.get("l0_pass", False)),
                    "l1_pass": int(dec.get("l1_pass", False)),
                    "l2_pass": int(dec.get("l2_pass", False)),
                    "l3_pass": int(dec.get("l3_pass", False)),
                    "l4_score": dec.get("l4_score", 0),
                    "position_pct": dec.get("position_pct", 0),
                    "verdict": dec.get("verdict", ""),
                    "pe": dec.get("pe", 0) or 0,
                }]
            )

            # 存入状态相似搜索库
            self._index_for_similarity_search(code, name, dec, data, date)

    def _index_for_similarity_search(self, code: str, name: str,
                                     dec: dict, full_data: dict, date: str):
        """为相似状态搜索建立索引"""
        state_text = (
            f"{name}({code}) 风险{full_data.get('risk',{}).get('level','N/A')} "
            f"L0={dec.get('l0_pass')} L1={dec.get('l1_pass')} "
            f"L2={dec.get('l2_pass')} L3={dec.get('l3_pass')} "
            f"L4评分={dec.get('l4_score',0)} 仓位={dec.get('position_pct',0)}% "
            f"信号={dec.get('timing','')} 方向={dec.get('verdict','')}"
        )

        state_id = f"state_{code}_{date.replace('-','')}"
        self.states.add(
            ids=[state_id],
            documents=[state_text],
            metadatas=[{
                "date": date,
                "code": code,
                "verdict": dec.get("verdict", ""),
                "position": dec.get("position_pct", 0),
            }]
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. 查询接口
    # ═══════════════════════════════════════════════════════════════════════

    def get_stock_history(self, code: str) -> List[dict]:
        """
        获取某只股票的所有历史记录
        返回按时间排序的列表
        """
        results = self.trends.get(
            where={"code": code},
            include=["documents", "metadatas"]
        )
        if not results or not results["metadatas"]:
            return []

        records = []
        for i, meta in enumerate(results["metadatas"]):
            records.append({
                "date": meta.get("date", ""),
                "name": meta.get("name", ""),
                "l0_pass": bool(meta.get("l0_pass", False)),
                "l1_pass": bool(meta.get("l1_pass", False)),
                "l2_pass": bool(meta.get("l2_pass", False)),
                "l3_pass": bool(meta.get("l3_pass", False)),
                "l4_score": meta.get("l4_score", 0),
                "position_pct": meta.get("position_pct", 0),
                "verdict": meta.get("verdict", ""),
                "pe": meta.get("pe", 0),
                "raw_text": results["documents"][i] if results["documents"] else "",
            })

        return sorted(records, key=lambda x: x["date"])

    def get_velocity(self, code: str, window: int = 7) -> dict:
        """
        追踪股票信号变化的"速度"
        返回各维度在最近 window 天内的变化量

        价值：判断信号加速/减速，比单点判断更有预见性
        """
        history = self.get_stock_history(code)
        if len(history) < 2:
            return {"note": "数据不足，需至少2个时间点"}

        recent = history[-min(window, len(history)):]

        velocity = {
            "code": code,
            "name": recent[-1].get("name", ""),
            "period": f"{recent[0]['date']} ~ {recent[-1]['date']}",
            "points": len(recent),
            "trends": {
                "l4_score": {
                    "start": recent[0]["l4_score"],
                    "end": recent[-1]["l4_score"],
                    "delta": round(recent[-1]["l4_score"] - recent[0]["l4_score"], 1),
                },
                "position": {
                    "start": recent[0]["position_pct"],
                    "end": recent[-1]["position_pct"],
                    "delta": round(recent[-1]["position_pct"] - recent[0]["position_pct"], 1),
                },
                "verdict_chain": self._verdict_chain(recent),
            }
        }

        # 计算加速度（二阶导数）
        if len(recent) >= 4:
            deltas = [recent[i+1]["l4_score"] - recent[i]["l4_score"]
                      for i in range(len(recent)-1)]
            velocity["l4_acceleration"] = round(
                (deltas[-1] - deltas[0]) / max(len(deltas), 1), 2
            )

        return velocity

    def _verdict_chain(self, history: List[dict]) -> List[str]:
        """提取判定变化链"""
        return [h["verdict"] for h in history]

    def search_similar_states(self, query_text: str, n: int = 5) -> List[dict]:
        """
        搜索历史上相似的市场状态

        例如：
          search_similar_states("AI算力链光模块高景气，L3传导加速")

        返回历史上最相似的 n 个时期及其后续表现
        """
        results = self.states.query(
            query_texts=[query_text],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        similar = []
        for i in range(len(results["ids"][0])):
            similar.append({
                "id": results["ids"][0][i],
                "distance": round(results["distances"][0][i], 3),
                "document": results["documents"][0][i] if results["documents"] else "",
                "date": results["metadatas"][0][i].get("date"),
                "code": results["metadatas"][0][i].get("code"),
            })

        return similar

    def get_latest_snapshot(self) -> Optional[dict]:
        """获取最近一次分析快照"""
        results = self.snapshots.get(
            include=["metadatas", "documents"]
        )
        if not results or not results["metadatas"]:
            return None

        # 按日期取最新
        sorted_meta = sorted(results["metadatas"], key=lambda x: x.get("date", ""), reverse=True)
        latest_date = sorted_meta[0]["date"]

        latest = self.snapshots.get(
            where={"date": latest_date},
            include=["metadatas", "documents"]
        )
        if latest and latest["metadatas"]:
            return {
                "date": latest["metadatas"][0]["date"],
                "risk_level": latest["metadatas"][0].get("risk_level"),
                "risk_score": latest["metadatas"][0].get("risk_score"),
                "l1_score": latest["metadatas"][0].get("l1_score"),
                "l2_score": latest["metadatas"][0].get("l2_score"),
                "active_chains": latest["metadatas"][0].get("active_chains"),
                "real_stocks": latest["metadatas"][0].get("real_stocks"),
                "go_decisions": latest["metadatas"][0].get("go_decisions"),
                "summary": latest["documents"][0] if latest["documents"] else "",
            }
        return None

    def get_stats(self) -> dict:
        """获取记忆库统计信息"""
        return {
            "snapshots": self.snapshots.count(),
            "trends": self.trends.count(),
            "states": self.states.count(),
            "persist_dir": self.persist_dir,
        }

    def export_web_summary(self) -> dict:
        """
        导出 Web 前端可展示的记忆摘要
        返回：统计信息 + 最新快照 + 所有股票的最新趋势变化
        """
        summary = {"stats": self.get_stats()}

        # 最新快照
        summary["latest"] = self.get_latest_snapshot()

        # 获取所有出现过的股票代码
        try:
            trends_all = self.trends.get(
                include=["metadatas"]
            )
            codes_seen = set()
            if trends_all and trends_all["metadatas"]:
                for m in trends_all["metadatas"]:
                    c = m.get("code")
                    if c:
                        codes_seen.add(c)

            stock_trends = []
            for code in sorted(codes_seen):
                history = self.get_stock_history(code)
                if len(history) >= 2:
                    latest = history[-1]
                    prev = history[-2]
                    stock_trends.append({
                        "code": code,
                        "name": latest.get("name", ""),
                        "records": len(history),
                        "current": {
                            "l4_score": latest["l4_score"],
                            "position_pct": latest["position_pct"],
                            "verdict": latest["verdict"],
                        },
                        "delta": {
                            "l4_score": round(latest["l4_score"] - prev["l4_score"], 1),
                            "position_pct": round(latest["position_pct"] - prev["position_pct"], 1),
                        },
                    })
                elif len(history) == 1:
                    h = history[0]
                    stock_trends.append({
                        "code": code,
                        "name": h.get("name", ""),
                        "records": 1,
                        "current": {
                            "l4_score": h["l4_score"],
                            "position_pct": h["position_pct"],
                            "verdict": h["verdict"],
                        },
                        "delta": {"l4_score": 0, "position_pct": 0},
                    })

            summary["stocks"] = stock_trends
        except Exception:
            summary["stocks"] = []

        return summary


# ═══════════════════════════════════════════════════════════════════════
# CLI 测试工具
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  记忆存储模块")
    print(f"{'='*60}")

    if not _HAS_CHROMA:
        print("  [ERROR] ChromaDB 未安装，请运行: pip install chromadb")
        return

    store = MemoryStore()
    stats = store.get_stats()
    print(f"  持久化目录: {stats['persist_dir']}")
    print(f"  分析快照:   {stats['snapshots']} 条")
    print(f"  趋势记录:   {stats['trends']} 条")
    print(f"  状态索引:   {stats['states']} 条")

    # 查询最新快照
    latest = store.get_latest_snapshot()
    if latest:
        print(f"\n  最近分析: {latest['date']}")
        print(f"  风险: {latest['risk_level']} ({latest['risk_score']}分)")
        print(f"  L1实业: {latest['l1_score']}分")
        print(f"  L2金融: {latest['l2_score']}分")
        print(f"  活跃产业链: {latest['active_chains']}条")
        print(f"  优秀标的: {latest['real_stocks']}只")
        print(f"  买入决策: {latest['go_decisions']}个")
    else:
        print("\n  尚未存储任何分析快照")
        print("  (运行 export_json.py 后会自动存入)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
