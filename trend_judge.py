"""
趋势判定引擎 — L0+L1+L2 综合分析
核心判断：当前市场应该 GO / WAIT / NO？
这是整个系统的"出不出手"关键决策点
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from llm_client import chat_json, is_available


@dataclass
class TrendJudgment:
    verdict: str         # "GO" / "WAIT" / "NO"
    trend: str           # "BULL" / "BEAR" / "SIDEWAYS"
    confidence: int      # 0-100
    l0_verdict: str      # "PASS" / "BLOCK"
    reasoning: str       # 一段话，判断逻辑
    key_signals: list    # 影响判断的 3-5 个最关键信号
    risks: list          # 当前最大的 2-3 个风险点
    opportunities: list  # 如果出手，最值得关注的 2-3 个方向
    action: str          # 具体操作建议
    score: int           # 0-100 综合得分（供前端显示）
    source: str          # "LLM" 或 "RULE_FALLBACK"
    timestamp: str = ""


# ── Prompt 模板 ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位资深A股市场策略分析师，你的核心分析原则：

1. **风险第一**：规避风险永远是第一位，不追求稳赢，而是提高胜率。宁可错过，不可做错。
2. **实业数据优先**：优先关注实业数据（用电量、PMI、物流运价、商品价格等），这些反映真实的社会需求和产业趋势，比金融指标更诚实。
3. **金融信号辅助**：结合美债收益率、中美利差、美元指数、黄金等国际金融信号，判断外部环境。
4. **L0风控是硬性否决**：如果风控层触发RED，无论其他信号如何，都必须给出NO。

分析要求：
- 综合所有信号，给出整体趋势判断（BULL/BEAR/SIDEWAYS）
- 给出明确的操作判定（GO/WAIT/NO）
- 必须引用具体数据和信号来支撑你的判断
- 列出关键风险点和潜在机会方向"""

USER_PROMPT_TEMPLATE = """## Layer 0 风控状态
- 风控等级：{l0_level}（GREEN=安全 / YELLOW=预警 / RED=禁止操作）
- 风控得分：{l0_score}/100
- 触发项：{l0_triggered}
- 上证指数5日变动：{index_5d}%
- 单日最大跌幅：{max_1d}%
{vix_info}
{fx_info}

## Layer 1 实业数据（当前信号）
### 食品板块
{food_signals}

### 大宗商品
{bulk_signals}

### 宏观货币
{macro_signals}

### 基建需求
{infra_signals}

### 全国用电量
{electricity_signals}

### 物流运价
{logistics_signals}

## Layer 2 国际/金融信号
{intl_signals}

### 股指期货基差
{futures_signals}

## L1/L2 背离检测
{divergence}

请严格按以下JSON格式输出分析结论，不要输出其他内容：
{{
  "verdict": "GO 或 WAIT 或 NO",
  "trend": "BULL 或 BEAR 或 SIDEWAYS",
  "confidence": 0到100的整数,
  "l0_verdict": "PASS 或 BLOCK",
  "reasoning": "200字以内的综合判断逻辑，必须引用具体信号数值",
  "key_signals": ["影响判断的3-5个最关键信号，格式：信号名:数值(含义)"],
  "risks": ["当前最大的2-3个风险点"],
  "opportunities": ["如果出手，最值得关注的2-3个方向"],
  "action": "50字以内的具体操作建议"
}}"""


# ── 格式化函数 ──────────────────────────────────────────────────────────

def _fmt_indicators(indicators: list) -> str:
    """格式化 Indicator 列表为文本"""
    if not indicators:
        return "  （无数据）"
    lines = []
    for i in indicators:
        sig_mark = {"UP": "📈", "DOWN": "📉", "NEUTRAL": "➖",
                     "ALERT_UP": "🔺", "ALERT_DOWN": "🔻"}.get(i.signal, "➖")
        yoy_str = f"，同比{i.yoy}%" if i.yoy else ""
        z_str = f"，偏离{i.zscore}σ" if i.zscore else ""
        lines.append(f"  {sig_mark} {i.name}: {i.value}{i.unit}{yoy_str}{z_str} [{i.signal}]")
    return "\n".join(lines)


def _fmt_signals(signals: list) -> str:
    """格式化 IntlSignal 列表为文本"""
    if not signals:
        return "  （无数据）"
    lines = []
    for s in signals:
        dir_mark = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(s.direction, "⚪")
        chg_str = f"，变动{s.change}" if s.change else ""
        lines.append(f"  {dir_mark} {s.name}: {s.value}{s.unit}{chg_str} [{s.direction}]")
    return "\n".join(lines)


# ── LLM 分析 ────────────────────────────────────────────────────────────

def llm_trend_judge(l0_status, l1_result, l2_result, env_data: dict) -> TrendJudgment:
    """调用 DeepSeek 进行趋势判定"""
    if not is_available():
        return rule_fallback(l0_status, l1_result, l2_result)

    # L1 分组
    food_str = _fmt_indicators(l1_result.food)
    bulk_str = _fmt_indicators(l1_result.bulk)
    macro_str = _fmt_indicators(l1_result.macro)
    infra_str = _fmt_indicators(l1_result.infra)
    elec_str = _fmt_indicators(l1_result.electricity)
    logi_str = _fmt_indicators(l1_result.logistics)

    # L2
    intl_str = _fmt_signals(l2_result.intl)
    fut_str = _fmt_signals(l2_result.futures)
    div_str = "; ".join(l2_result.divergence) if l2_result.divergence else "无"

    # L0
    vix_info = f"- VIX/QVIX：{env_data.get('vix')}" if env_data.get('vix') else ""
    fx_info = f"- 美元/人民币：{env_data.get('usdcny')}（5日变动{env_data.get('usdcny_5d_chg')}%）" if env_data.get('usdcny') else ""

    prompt = USER_PROMPT_TEMPLATE.format(
        l0_level=l0_status.level,
        l0_score=l0_status.score,
        l0_triggered="、".join(l0_status.triggered) if l0_status.triggered else "无",
        index_5d=env_data.get("index_5d_chg", "N/A"),
        max_1d=env_data.get("max_1d_drop", "N/A"),
        vix_info=vix_info,
        fx_info=fx_info,
        food_signals=food_str,
        bulk_signals=bulk_str,
        macro_signals=macro_str,
        infra_signals=infra_str,
        electricity_signals=elec_str,
        logistics_signals=logi_str,
        intl_signals=intl_str,
        futures_signals=fut_str,
        divergence=div_str,
    )

    print("  调用 DeepSeek 进行趋势判定...")
    result = chat_json(SYSTEM_PROMPT, prompt, temperature=0.2)

    if result is None:
        print("  [WARN] LLM 调用失败，使用规则兜底")
        return rule_fallback(l0_status, l1_result, l2_result)

    try:
        return TrendJudgment(
            verdict=result.get("verdict", "WAIT"),
            trend=result.get("trend", "SIDEWAYS"),
            confidence=int(result.get("confidence", 50)),
            l0_verdict=result.get("l0_verdict", "PASS" if l0_status.level != "RED" else "BLOCK"),
            reasoning=result.get("reasoning", ""),
            key_signals=result.get("key_signals", []),
            risks=result.get("risks", []),
            opportunities=result.get("opportunities", []),
            action=result.get("action", ""),
            score=int(result.get("confidence", 50)),
            source="LLM",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    except Exception as e:
        print(f"  [WARN] LLM 响应解析失败: {e}，使用规则兜底")
        return rule_fallback(l0_status, l1_result, l2_result)


# ── 规则兜底（LLM 不可用时） ────────────────────────────────────────────

def rule_fallback(l0_status, l1_result, l2_result) -> TrendJudgment:
    """纯规则的趋势判定（不依赖 LLM）"""
    import warnings
    warnings.filterwarnings("ignore")

    # L0 否决
    if l0_status.level == "RED":
        return TrendJudgment(
            verdict="NO", trend="BEAR", confidence=80,
            l0_verdict="BLOCK",
            reasoning=f"L0风控触发RED，{l0_status.note}。禁止任何新建仓操作。",
            key_signals=["L0风控RED"],
            risks=[l0_status.note],
            opportunities=[],
            action="空仓等待，不操作",
            score=10, source="RULE_FALLBACK",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

    # 收集信号
    all_l1 = (l1_result.food + l1_result.bulk + l1_result.macro
              + l1_result.infra + l1_result.electricity + l1_result.logistics)
    up_cnt = sum(1 for i in all_l1 if "UP" in i.signal)
    down_cnt = sum(1 for i in all_l1 if "DOWN" in i.signal)
    l1_total = len(all_l1)

    bullish = sum(1 for s in l2_result.intl if s.direction == "BULLISH")
    bearish = sum(1 for s in l2_result.intl if s.direction == "BEARISH")
    l2_total = len(l2_result.intl) if l2_result.intl else 1

    # 综合评分（权重和阈值从 config 读取）
    from strategy_config import (
        TREND_L1_WEIGHT, TREND_L2_WEIGHT, TREND_L0_WEIGHT,
        TREND_GO_THRESHOLD, TREND_NO_THRESHOLD,
    )
    l1_ratio = (up_cnt - down_cnt) / max(l1_total, 1)
    l2_ratio = (bullish - bearish) / max(l2_total, 1)
    composite = 50 + l1_ratio * TREND_L1_WEIGHT + l2_ratio * TREND_L2_WEIGHT + l0_status.score * TREND_L0_WEIGHT

    if composite >= TREND_GO_THRESHOLD:
        verdict, trend = "GO", "BULL"
    elif composite <= TREND_NO_THRESHOLD:
        verdict, trend = "NO", "BEAR"
    else:
        verdict, trend = "WAIT", "SIDEWAYS"

    if l0_status.level == "YELLOW":
        verdict = "WAIT"
        composite = min(composite, 50)

    key_sigs = []
    for i in all_l1:
        if "ALERT" in i.signal:
            key_sigs.append(f"L1异动: {i.name}={i.value}{i.unit}")
    for s in l2_result.intl:
        if s.direction in ("BULLISH", "BEARISH"):
            key_sigs.append(f"L2: {s.name}={s.value}{s.unit}[{s.direction}]")
    if not key_sigs:
        key_sigs = [f"L1={l1_result.score}分, L2={l2_result.score}分"]

    return TrendJudgment(
        verdict=verdict, trend=trend, confidence=int(min(max(composite, 5), 95)),
        l0_verdict="PASS",
        reasoning=f"规则兜底：L1={l1_result.score}分(UP{up_cnt}/DOWN{down_cnt})，"
                  f"L2={l2_result.score}分(多{bullish}/空{bearish})，"
                  f"L0={l0_status.score}分({l0_status.level})",
        key_signals=key_sigs[:5],
        risks=["LLM不可用，仅基于规则判断，建议配置DeepSeek API"],
        opportunities=[],
        action="建议配置 DEEPSEEK_API_KEY 以获得更深入的趋势分析",
        score=int(composite), source="RULE_FALLBACK",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    )


# ── 导出为 dict（供 JSON 序列化） ───────────────────────────────────────

def trend_to_dict(t: TrendJudgment) -> dict:
    return {
        "verdict":       t.verdict,
        "trend":         t.trend,
        "confidence":    t.confidence,
        "l0_verdict":    t.l0_verdict,
        "reasoning":     t.reasoning,
        "key_signals":   t.key_signals,
        "risks":         t.risks,
        "opportunities": t.opportunities,
        "action":        t.action,
        "score":         t.score,
        "source":        t.source,
        "timestamp":     t.timestamp,
    }
