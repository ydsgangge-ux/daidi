"""
Layer 0 — 风险管理
贯穿所有层的否决机制，任意触发则全框架暂停
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


@dataclass
class RiskStatus:
    level: str          # GREEN / YELLOW / RED
    score: int          # 0-100，越高越安全
    triggered: list     # 已触发的规则
    max_position: float # 当前允许的最大总仓位
    note: str


def get_market_env() -> dict:
    """获取大盘环境关键指标"""
    result = {}

    # 1. 上证指数近5日涨跌
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        df = df.tail(10).reset_index(drop=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        latest = float(df["close"].iloc[-1])
        prev5  = float(df["close"].iloc[-6]) if len(df) >= 6 else latest
        chg5d  = (latest - prev5) / prev5 * 100
        # 单日最大跌幅
        df["pct"] = df["close"].pct_change() * 100
        max_drop_1d = float(df["pct"].min())
        result["index_5d_chg"] = round(chg5d, 2)
        result["max_1d_drop"]  = round(max_drop_1d, 2)
        result["index_latest"] = round(latest, 2)
    except Exception as e:
        result["index_5d_chg"] = 0
        result["max_1d_drop"]  = 0
        result["index_latest"] = 0
        result["index_err"]    = str(e)

    # 2. 北向资金（近5日净流入）
    try:
        df_north = ak.stock_connect_position_statistics_em()
        # 取沪股通+深股通合计
        net_cols = [c for c in df_north.columns if "净买" in c or "净流" in c]
        if net_cols:
            val = pd.to_numeric(df_north[net_cols[0]].iloc[0], errors="coerce")
            result["north_net"] = round(float(val), 2) if pd.notna(val) else 0
        else:
            result["north_net"] = 0
    except Exception:
        result["north_net"] = None  # 无法获取时不触发规则

    # 3. VIX（通过黄金替代判断避险情绪）
    try:
        df_vix = ak.macro_usa_vix()
        df_vix["收盘"] = pd.to_numeric(df_vix["收盘"], errors="coerce")
        vix_val = float(df_vix["收盘"].dropna().iloc[-1])
        result["vix"] = round(vix_val, 2)
    except Exception:
        result["vix"] = None

    # 4. 人民币汇率（在岸）
    try:
        df_fx = ak.currency_boc_sina(symbol="美元人民币")
        df_fx["中间价"] = pd.to_numeric(df_fx["中间价"], errors="coerce")
        df_fx = df_fx.dropna(subset=["中间价"])
        latest_fx  = float(df_fx["中间价"].iloc[-1])
        prev_fx    = float(df_fx["中间价"].iloc[-6]) if len(df_fx) >= 6 else latest_fx
        fx_chg_5d  = (latest_fx - prev_fx) / prev_fx * 100
        result["usdcny"]        = round(latest_fx, 4)
        result["usdcny_5d_chg"] = round(fx_chg_5d, 3)
    except Exception:
        result["usdcny"]        = None
        result["usdcny_5d_chg"] = None

    return result


def evaluate_layer0(env: dict) -> RiskStatus:
    """根据环境数据计算Layer 0状态（规则参数从 strategy_config 读取）"""
    from strategy_config import L0_RULES, L0_POSITION_MAP
    triggered = []
    deductions = 0

    # 按 L0_RULES 逐条评估
    for field, comp, threshold, ded_pts, msg_template in L0_RULES:
        val = env.get(field)
        if val is None:
            continue
        triggered_flag = False
        if comp == "lt":
            triggered_flag = val < threshold
        elif comp == "gt":
            triggered_flag = val > threshold
        elif comp == "lte":
            triggered_flag = val <= threshold
        elif comp == "gte":
            triggered_flag = val >= threshold

        if triggered_flag:
            # 避免 VIX 两级规则都触发时两条消息完全一样
            msg = msg_template.format(value=val)
            triggered.append(msg)
            deductions += ded_pts

    score = max(0, 100 - deductions)

    # 按 L0_POSITION_MAP 查找等级和仓位
    level = "RED"
    max_pos = 0.0
    note = "红色警报，清仓或空仓，等待环境改善"
    idx_5d = env.get("index_5d_chg", 0)

    for min_score, lvl, sub_rules in sorted(L0_POSITION_MAP, key=lambda x: -x[0]):
        if score >= min_score:
            level = lvl
            for chg_threshold, pos, note_text in sub_rules:
                if idx_5d >= chg_threshold or chg_threshold == 999:
                    max_pos = pos
                    note = note_text
                    break
            break

    return RiskStatus(
        level=level,
        score=score,
        triggered=triggered,
        max_position=max_pos,
        note=note
    )


def run_layer0() -> Tuple[RiskStatus, dict]:
    env = get_market_env()
    status = evaluate_layer0(env)
    return status, env
