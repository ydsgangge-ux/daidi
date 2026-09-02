"""
磁盘缓存助手 — 按自然日缓存行情/财务抓取结果

目的：
  - 避免同一天多次运行重复请求数据源（东财爱限流，当天重复请求易被拒连）
  - 大幅缩短重复运行的墙钟时间

所有缓存按 (市场, 代码, 日期) 组织，日期默认今天，跨进程共享。
当天缓存不足可手动删除 cache/ 目录强制重拉。
"""
import os
import pickle
from datetime import datetime

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _path(market: str, code: str, date: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{market}_{code}_{date}.pkl")


def load_kline(market: str, code: str, date: str = None) -> tuple:
    """
    读取当日K线缓存。返回 (source, df) 或 (None, None)。
    source: 数据来源标记（如 "em"/"sina"/"tx"/"cache"）
    """
    p = _path(market, code, date or _today())
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None, None


def save_kline(market: str, code: str, source: str, df: pd.DataFrame, date: str = None) -> None:
    """写入当日K线缓存。df 为空时不写。"""
    if df is None or len(df) == 0:
        return
    try:
        p = _path(market, code, date or _today())
        with open(p, "wb") as f:
            pickle.dump((source, df), f)
    except Exception:
        pass


def load_json(market: str, code: str, date: str = None):
    """读取当日 JSON 类缓存（如 LLM 结果）。"""
    p = _path(market, code, date or _today()).replace(".pkl", ".json")
    if not os.path.exists(p):
        return None
    try:
        import json
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(market: str, code: str, payload, date: str = None) -> None:
    """写入当日 JSON 类缓存。"""
    try:
        p = _path(market, code, date or _today()).replace(".pkl", ".json")
        import json
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass