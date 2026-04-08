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
    """根据环境数据计算Layer 0状态"""
    triggered = []
    deductions = 0

    # 规则1：单日大跌
    if env.get("max_1d_drop", 0) < -3.0:
        triggered.append(f"单日最大跌幅 {env['max_1d_drop']}%，触发-3%阈值")
        deductions += 40

    # 规则2：近5日累计跌幅
    if env.get("index_5d_chg", 0) < -5.0:
        triggered.append(f"5日累计跌幅 {env['index_5d_chg']}%，触发-5%阈值")
        deductions += 25

    # 规则3：VIX
    vix = env.get("vix")
    if vix is not None:
        if vix > 25:
            triggered.append(f"VIX={vix}，超过25，全球恐慌")
            deductions += 35
        elif vix > 20:
            triggered.append(f"VIX={vix}，超过20，需警惕")
            deductions += 15

    # 规则4：汇率急贬
    fx_chg = env.get("usdcny_5d_chg")
    if fx_chg is not None and fx_chg > 1.0:
        triggered.append(f"人民币5日贬值{fx_chg}%，超过1%阈值")
        deductions += 20

    score = max(0, 100 - deductions)

    if score >= 70:
        level = "GREEN"
        if env.get("index_5d_chg", 0) > 2:
            max_pos, note = 0.60, "做多环境，最大仓位60%"
        else:
            max_pos, note = 0.30, "震荡环境，最大仓位30%"
    elif score >= 40:
        level = "YELLOW"
        max_pos = 0.10
        note = "黄色预警，最大仓位10%，不新建仓位"
    else:
        level = "RED"
        max_pos = 0.0
        note = "红色警报，清仓或空仓，等待环境改善"

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
