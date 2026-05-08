"""
产业链周期预估模块
基于催化剂数据库 + 传导时序 + LLM 动态分析，估算产业链上涨周期

核心功能：
1. 催化剂日历：每条链的已知催化剂及时间
2. 传导时序模型：产业链各环节受益先后顺序
3. 周期阶段识别：启动期/主升浪/高位震荡/退潮期
4. 剩余上涨时间预估
5. 周期成熟度评分（0-100，越高越成熟=越危险）
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, date
from llm_client import chat_json, is_available


# ══════════════════════════════════════════════════════════════════════════════
# 催化剂日历（按链分组）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Catalyst:
    date: str          # 预计时间，如 "2025-Q2", "2025-06", "每年3月"
    event: str         # 事件名称
    impact: str        # HIGH / MEDIUM / LOW
    description: str   # 对产业链的影响
    passed: bool = False  # 是否已过（动态计算）


CATALYST_DB: Dict[str, List[dict]] = {
    "AI算力链": [
        {"date": "每年2-3月", "event": "英伟达GTC大会", "impact": "HIGH",
         "description": "发布新一代GPU/网络架构，指引全年算力需求方向"},
        {"date": "每年4月", "event": "云厂商Q1资本开支季报", "impact": "HIGH",
         "description": "微软/谷歌/亚马逊/Meta公布Capex，直接决定算力链景气度"},
        {"date": "每年7月", "event": "云厂商Q2资本开支季报", "impact": "HIGH",
         "description": "验证Q1趋势是否延续，中报业绩验证期"},
        {"date": "每年10月", "event": "云厂商Q3资本开支季报", "impact": "MEDIUM",
         "description": "指引全年Capex方向，决定Q4估值切换"},
        {"date": "每年11月", "event": "英伟达Q3财报", "impact": "HIGH",
         "description": "数据中心收入增速+下一代产品进展，AI算力链最重要催化剂"},
        {"date": "每季度", "event": "光模块出货量跟踪", "impact": "MEDIUM",
         "description": "800G/1.6T月度出货数据，验证需求景气度"},
        {"date": "每季度", "event": "台积电法说会", "impact": "MEDIUM",
         "description": "CoWoS产能分配+AI营收占比，上游验证信号"},
    ],
    "半导体材料链": [
        {"date": "每年1-2月", "event": "日本/美国出口管制政策更新", "impact": "HIGH",
         "description": "每次升级管制→国产替代紧迫性+政策资金加速"},
        {"date": "每季度", "event": "中芯国际/华虹季报", "impact": "HIGH",
         "description": "晶圆厂采购订单中材料国产化比例，最可靠验证信号"},
        {"date": "每半年", "event": "大基金投资动态", "impact": "MEDIUM",
         "description": "国家大基金投向，指引材料国产替代重点方向"},
        {"date": "每季度", "event": "全球半导体销售额（SIA）", "impact": "MEDIUM",
         "description": "行业景气度基准，材料需求跟随晶圆厂产能利用率"},
    ],
    "新能源车链": [
        {"date": "每月初", "event": "新能源车月度销量", "impact": "HIGH",
         "description": "比亚迪/特斯拉/理想等月度交付数据，直接反映终端需求"},
        {"date": "每季度", "event": "锂矿价格走势", "impact": "HIGH",
         "description": "碳酸锂期货+现货价格，决定全链成本和利润空间"},
        {"date": "每年3月/9月", "event": "特斯拉交付/发布会", "impact": "MEDIUM",
         "description": "新车型/新技术路线（如4680电池进展）"},
        {"date": "每季度", "event": "宁德时代财报", "impact": "MEDIUM",
         "description": "全球动力电池龙头，指引行业盈利趋势"},
    ],
    "出口制造链": [
        {"date": "每月", "event": "中国出口金额（美元计）", "impact": "HIGH",
         "description": "海关总署月度数据，出口链景气度最直接指标"},
        {"date": "每月", "event": "美国ISM制造业PMI", "impact": "HIGH",
         "description": "新订单子项领先出口需求6-8周"},
        {"date": "每月", "event": "SCFI运价指数", "impact": "MEDIUM",
         "description": "上海出口集装箱运价，验证出口景气度"},
        {"date": "不定期", "event": "美国关税政策变动", "impact": "HIGH",
         "description": "301关税/反补贴调查，出口链最大风险变量"},
    ],
    "工业自动化链": [
        {"date": "每季度", "event": "发那科/ABB订单数据", "impact": "MEDIUM",
         "description": "全球工业自动化需求先行指标"},
        {"date": "每年", "event": "特斯拉Optimus进展", "impact": "HIGH",
         "description": "人形机器人量产时间表，国产零部件最大催化"},
        {"date": "每季度", "event": "汇川技术/埃斯顿财报", "impact": "MEDIUM",
         "description": "国产工控龙头营收增速，国产替代进度验证"},
        {"date": "每月", "event": "中国制造业PMI", "impact": "MEDIUM",
         "description": "50荣枯线以上→资本开支意愿强→自动化需求增"},
    ],
    "航空航天材料链": [
        {"date": "每季度", "event": "C919交付量", "impact": "HIGH",
         "description": "商飞季度交付数据，直接驱动上游材料需求"},
        {"date": "每半年", "event": "军机产能释放信号", "impact": "HIGH",
         "description": "中航工业/航发集团产能建设进展"},
        {"date": "每季度", "event": "西部超导/中复神鹰财报", "impact": "MEDIUM",
         "description": "钛合金/碳纤维军品收入增速验证"},
    ],
    "新能源材料链（SiC）": [
        {"date": "每半年", "event": "天岳先进8英寸进展", "impact": "HIGH",
         "description": "8英寸SiC衬底量产确认，行业最大催化剂"},
        {"date": "每季度", "event": "特斯拉SiC用量更新", "impact": "HIGH",
         "description": "每台车SiC用量提升+供应商切换进度"},
        {"date": "每季度", "event": "SiC价格走势", "impact": "HIGH",
         "description": "6英寸/8英寸衬底价格，价格企稳是拐点信号"},
        {"date": "每年", "event": "英飞凌/Wolfspeed财报", "impact": "MEDIUM",
         "description": "全球SiC龙头产能规划和订单情况"},
    ],
    "铜/铝/有色金属链": [
        {"date": "每月", "event": "LME铜/铝库存数据", "impact": "HIGH",
         "description": "全球显性库存变动，低库存=强支撑"},
        {"date": "每月", "event": "中国铜/铝进口量", "impact": "MEDIUM",
         "description": "中国需求验证，全球最大消费国"},
        {"date": "每季度", "event": "紫金矿业/中国铝业财报", "impact": "MEDIUM",
         "description": "龙头矿企产量指引和成本数据"},
        {"date": "不定期", "event": "美联储利率决议", "impact": "HIGH",
         "description": "降息→美元弱→有色估值提升；加息则相反"},
    ],
    "生猪养殖链": [
        {"date": "每月", "event": "能繁母猪存栏数据", "impact": "HIGH",
         "description": "领先猪价6-10个月，存栏持续下降=猪价上涨前兆"},
        {"date": "每半月", "event": "生猪价格（全国均价）", "impact": "HIGH",
         "description": "直接反映供需格局，决定养殖利润"},
        {"date": "每季度", "event": "牧原/温氏出栏量+成本", "impact": "MEDIUM",
         "description": "龙头成本线决定行业盈利中枢"},
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 产业链历史周期参考（用于经验预估）
# ══════════════════════════════════════════════════════════════════════════════

CHAIN_CYCLE_REFERENCE: Dict[str, dict] = {
    "AI算力链": {
        "typical_cycle_months": 24,
        "phases": [
            {"name": "启动期", "duration_months": 3, "description": "云厂商Capex拐点+英伟达新品发布"},
            {"name": "主升浪", "duration_months": 9, "description": "光模块/服务器订单放量，业绩兑现"},
            {"name": "高位震荡", "duration_months": 6, "description": "业绩持续但增速放缓，估值消化"},
            {"name": "分化期", "duration_months": 6, "description": "龙头抗跌，概念股退潮，关注1.6T/下一代"},
        ],
        "note": "AI算力链受技术迭代驱动，周期较长且可能反复",
    },
    "半导体材料链": {
        "typical_cycle_months": 18,
        "phases": [
            {"name": "启动期", "duration_months": 3, "description": "出口管制升级/大基金投资消息"},
            {"name": "验证期", "duration_months": 6, "description": "晶圆厂验证通过，订单开始放量"},
            {"name": "业绩兑现", "duration_months": 6, "description": "国产化率提升，业绩持续高增"},
            {"name": "估值消化", "duration_months": 3, "description": "增速回落至30%以下，进入估值消化"},
        ],
        "note": "国产替代逻辑独立于半导体周期，但受全球需求影响",
    },
    "新能源车链": {
        "typical_cycle_months": 18,
        "phases": [
            {"name": "底部筑底", "duration_months": 4, "description": "锂价触底+库存出清信号"},
            {"name": "复苏期", "duration_months": 6, "description": "销量回暖+锂价回升，中游盈利修复"},
            {"name": "主升浪", "duration_months": 5, "description": "销量高增+新技术催化（固态电池等）"},
            {"name": "高位分化", "duration_months": 3, "description": "龙头稳、二线分化，关注新技术差异化"},
        ],
        "note": "当前处于锂价低迷期，等待筑底信号",
    },
    "出口制造链": {
        "typical_cycle_months": 12,
        "phases": [
            {"name": "信号期", "duration_months": 2, "description": "美国库存比下行+ISM新订单回升"},
            {"name": "订单期", "duration_months": 3, "description": "港口吞吐量提升+运价上涨"},
            {"name": "业绩期", "duration_months": 4, "description": "出口企业营收确认，业绩兑现"},
            {"name": "退潮期", "duration_months": 3, "description": "补库结束+关税风险升温"},
        ],
        "note": "出口链受海外需求和关税政策双重影响，周期较短",
    },
    "工业自动化链": {
        "typical_cycle_months": 24,
        "phases": [
            {"name": "国产替代启动", "duration_months": 6, "description": "PMI回暖+政策支持（设备更新）"},
            {"name": "订单放量", "duration_months": 8, "description": "工控龙头订单加速，进口替代比例提升"},
            {"name": "人形机器人催化", "duration_months": 6, "description": "Optimus量产进度+国产零部件验证"},
            {"name": "增速换挡", "duration_months": 4, "description": "基数效应显现，关注新技术突破"},
        ],
        "note": "国产替代逻辑独立于宏观周期，人形机器人是远期期权",
    },
    "航空航天材料链": {
        "typical_cycle_months": 36,
        "phases": [
            {"name": "政策催化", "duration_months": 6, "description": "军机交付提速+C919订单落地"},
            {"name": "产能建设", "duration_months": 12, "description": "上游材料扩产，订单持续释放"},
            {"name": "业绩兑现", "duration_months": 12, "description": "军品+民品双驱动，收入利润双升"},
            {"name": "估值消化", "duration_months": 6, "description": "军品定价机制+产能释放完成"},
        ],
        "note": "军工+大飞机双逻辑，周期最长，适合长期配置",
    },
    "新能源材料链（SiC）": {
        "typical_cycle_months": 24,
        "phases": [
            {"name": "等待期", "duration_months": 6, "description": "SiC价格下行+产能过剩出清"},
            {"name": "拐点确认", "duration_months": 4, "description": "价格企稳+下游需求回暖"},
            {"name": "放量期", "duration_months": 8, "description": "8英寸量产+车规级订单放量"},
            {"name": "成熟期", "duration_months": 6, "description": "国产替代率快速提升，关注盈利拐点"},
        ],
        "note": "当前处于价格下行+等待8英寸量产阶段",
    },
    # ---------- 铜/铝/有色金属链（原为错误格式，现补充完整的周期定义） ----------
    "铜/铝/有色金属链": {
        "typical_cycle_months": 18,
        "phases": [
            {"name": "信号期", "duration_months": 3, "description": "美元走弱+降息预期/库存低位推动"},
            {"name": "主升浪", "duration_months": 7, "description": "铜铝价格上行+矿企利润释放"},
            {"name": "高位震荡", "duration_months": 5, "description": "价格高位+需求验证博弈"},
            {"name": "退潮期", "duration_months": 3, "description": "库存回升+美元走强/需求放缓"},
        ],
        "note": "有色金属受全球定价+中国需求双重驱动，与美元/美债利率高度相关",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 周期阶段分析结果
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CyclePhase:
    phase: str          # EARLY / MID_RALLY / LATE / DECLINING
    phase_name: str     # 启动期 / 主升浪 / 高位震荡 / 退潮期
    phase_desc: str     # 阶段特征描述
    progress_pct: float # 当前周期进度百分比 0-100
    maturity: int       # 成熟度 0-100，越高越接近尾声
    remaining_months: float  # 预估剩余上涨时间（月）
    remaining_months_low: float  # 保守估计
    remaining_months_high: float # 乐观估计
    catalyst_count_90d: int      # 未来90天催化剂数量
    catalyst_count_high_90d: int # 未来90天高影响催化剂数量
    next_catalysts: List[dict]   # 最近的催化剂列表
    risk_warnings: List[str]     # 风险提示
    reasoning: str       # 分析理由（50-100字）


def _get_catalysts_for_chain(chain_name: str) -> List[dict]:
    """获取某条链的催化剂列表"""
    return CATALYST_DB.get(chain_name, [])


def _estimate_cycle_progress(chain_name: str) -> dict:
    """
    基于历史周期参考 + 策略配置中的起始日期，估算当前进度
    返回 {progress_pct, current_phase_index, current_phase_name}
    """
    from strategy_config import CHAIN_CYCLE_START as CYCLE_START_CFG

    ref = CHAIN_CYCLE_REFERENCE.get(chain_name)
    if not ref:
        return {"progress_pct": 50, "current_phase_index": 1, "current_phase_name": "未知"}

    try:
        # 从配置文件读取起始日期（如果没有则回退硬编码）
        cycle_start_str = CYCLE_START_CFG.get(chain_name, "2023-01")
        start = datetime.strptime(cycle_start_str, "%Y-%m")
        now = datetime.now()
        elapsed_months = (now.year - start.year) * 12 + (now.month - start.month)

        total = ref["typical_cycle_months"]
        progress = min(elapsed_months / total * 100, 100)

        # 确定当前阶段
        cumulative = 0
        phase_idx = 0
        for i, phase in enumerate(ref["phases"]):
            cumulative += phase["duration_months"]
            if elapsed_months <= cumulative:
                phase_idx = i
                break
        else:
            phase_idx = len(ref["phases"]) - 1

        return {
            "progress_pct": round(progress, 1),
            "current_phase_index": phase_idx,
            "current_phase_name": ref["phases"][phase_idx]["name"],
            "elapsed_months": elapsed_months,
        }
    except Exception:
        return {"progress_pct": 50, "current_phase_index": 1, "current_phase_name": "未知"}


def _get_upcoming_catalysts(chain_name: str, days: int = 90) -> List[dict]:
    """
    获取未来N天的催化剂列表
    返回按时间排序的催化剂
    """
    catalysts = _get_catalysts_for_chain(chain_name)
    upcoming = []
    now = datetime.now()

    for cat in catalysts:
        date_str = cat["date"]
        # 解析催化剂时间
        is_recurring = False
        cat_date = None

        if date_str.startswith("每年"):
            # 周期性事件，判断今年是否已过
            is_recurring = True
        elif date_str.startswith("每季度"):
            is_recurring = True
        elif date_str.startswith("每月") or date_str.startswith("每半月"):
            is_recurring = True
        elif date_str.startswith("每半年"):
            is_recurring = True
        elif date_str == "不定期":
            is_recurring = True

        if is_recurring:
            # 周期性事件默认在未来会有
            upcoming.append({
                "date": date_str,
                "event": cat["event"],
                "impact": cat["impact"],
                "description": cat["description"],
                "timing": "recurring",
            })
        else:
            try:
                cat_date = datetime.strptime(date_str, "%Y-%m")
                if cat_date > now:
                    upcoming.append({
                        "date": date_str,
                        "event": cat["event"],
                        "impact": cat["impact"],
                        "description": cat["description"],
                        "timing": "one-time",
                    })
            except ValueError:
                upcoming.append({
                    "date": date_str,
                    "event": cat["event"],
                    "impact": cat["impact"],
                    "description": cat["description"],
                    "timing": "recurring",
                })

    # 高影响排前面
    impact_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    upcoming.sort(key=lambda x: impact_order.get(x["impact"], 99))
    return upcoming


def _calc_maturity(progress_pct: float, phase_name: str) -> int:
    """计算成熟度（0-100）"""
    # 进度本身占60%
    base = int(progress_pct * 0.6)

    # 阶段加成
    phase_bonus = {
        "启动期": -10, "底部筑底": -10, "等待期": -15,
        "验证期": 0, "复苏期": 0, "拐点确认": 0, "信号期": 0,
        "主升浪": 5, "业绩兑现": 5, "放量期": 5, "订单放量": 5,
        "业绩期": 10, "订单期": 5,
        "高位震荡": 15, "分化期": 15, "增速换挡": 15,
        "高位分化": 15, "退潮期": 25, "估值消化": 20,
        "成熟期": 20, "退潮期": 25,
    }
    bonus = phase_bonus.get(phase_name, 0)

    return max(0, min(100, base + bonus))


def _calc_remaining_months(progress_pct: float, ref: dict,
                           phase_name: str) -> tuple:
    """计算剩余上涨时间（低/中/高估计）"""
    total = ref["typical_cycle_months"]
    elapsed = total * progress_pct / 100

    # 不同阶段的剩余时间逻辑不同
    if progress_pct < 25:
        # 早期：大部分时间还在前面
        mid = total - elapsed
        low = max(1, mid * 0.7)
        high = mid * 1.2
    elif progress_pct < 60:
        # 中期：仍有空间但开始收窄
        mid = (total - elapsed) * 0.85
        low = max(1, mid * 0.6)
        high = mid * 1.3
    elif progress_pct < 85:
        # 后期：剩余时间有限
        mid = (total - elapsed) * 0.5
        low = max(1, mid * 0.5)
        high = mid * 1.2
    else:
        # 尾声：可能随时结束
        mid = max(1, (total - elapsed) * 0.3)
        low = 0.5
        high = mid * 1.5

    return round(low, 1), round(mid, 1), round(high, 1)


def analyze_chain_cycle(chain_name: str, signal_strength: int = 0) -> CyclePhase:
    """
    分析单条产业链的周期阶段

    参数：
        chain_name: 产业链名称
        signal_strength: L3给的信号强度（0-100）

    返回：CyclePhase 对象
    """
    # 1. 周期进度
    progress_info = _estimate_cycle_progress(chain_name)
    progress = progress_info["progress_pct"]
    phase_name = progress_info["current_phase_name"]

    ref = CHAIN_CYCLE_REFERENCE.get(chain_name, {})
    phases = ref.get("phases", [])

    # 2. 催化剂
    upcoming = _get_upcoming_catalysts(chain_name, 90)
    catalyst_total = len(upcoming)
    catalyst_high = sum(1 for c in upcoming if c["impact"] == "HIGH")

    # 3. 成熟度
    maturity = _calc_maturity(progress, phase_name)

    # 4. 阶段判定
    if progress < 20:
        phase, phase_desc = "EARLY", "产业链处于启动期，催化剂刚落地，估值修复为主"
    elif progress < 50:
        phase, phase_desc = "MID_RALLY", "产业链进入主升浪，订单放量+业绩兑现，赚钱效应最强"
    elif progress < 75:
        phase, phase_desc = "LATE", "产业链进入高位震荡，增速放缓但仍正增长，估值消化中"
    else:
        phase, phase_desc = "DECLINING", "产业链接近尾声，需警惕增速大幅回落和资金撤离"

    # 用ref中的阶段描述覆盖
    if phases:
        idx = progress_info.get("current_phase_index", 0)
        if idx < len(phases):
            phase_desc = phases[idx]["description"]

    # 5. 剩余时间
    if ref:
        low, mid, high = _calc_remaining_months(progress, ref, phase_name)
    else:
        low, mid, high = 1.0, 3.0, 6.0

    # 6. 风险提示
    warnings = []
    if maturity > 70:
        warnings.append(f"周期成熟度{maturity}%，处于后半程，注意追高风险")
    if progress > 60:
        warnings.append("已进入周期后半段，后续上涨空间收窄")
    if catalyst_high == 0 and maturity > 40:
        warnings.append("近期无高影响催化剂，上涨动力可能不足")
    if maturity > 50 and phase == "MID_RALLY":
        warnings.append("主升浪已持续较久，留意获利回吐压力")

    # 7. 生成理由
    reasoning = _generate_reasoning(
        chain_name, phase, phase_name, progress, maturity,
        mid, catalyst_high, ref.get("note", "")
    )

    return CyclePhase(
        phase=phase,
        phase_name=phase_name,
        phase_desc=phase_desc,
        progress_pct=progress,
        maturity=maturity,
        remaining_months=mid,
        remaining_months_low=low,
        remaining_months_high=high,
        catalyst_count_90d=catalyst_total,
        catalyst_count_high_90d=catalyst_high,
        next_catalysts=upcoming[:5],
        risk_warnings=warnings,
        reasoning=reasoning,
    )


def _generate_reasoning(chain_name: str, phase: str, phase_name: str,
                        progress: float, maturity: int,
                        remaining: float, catalyst_high: int,
                        note: str) -> str:
    """生成分析理由"""
    parts = []

    # 阶段描述
    phase_map = {
        "EARLY": "处于启动早期",
        "MID_RALLY": "处于主升浪阶段",
        "LATE": "进入高位震荡期",
        "DECLINING": "接近周期尾声",
    }
    parts.append(f"{chain_name}{phase_map.get(phase, '未知阶段')}（{phase_name}）")

    # 进度
    parts.append(f"周期进度约{progress:.0f}%，成熟度{maturity}/100")

    # 剩余时间
    if remaining >= 3:
        parts.append(f"预估还有{remaining:.0f}个月左右的上涨窗口")
    elif remaining >= 1:
        parts.append(f"预估还有约{remaining:.0f}个月的上涨窗口")
    else:
        parts.append("上涨窗口接近尾声，需格外谨慎")

    # 催化剂
    if catalyst_high >= 2:
        parts.append(f"未来3个月有{catalyst_high}个高影响催化剂，有持续驱动")
    elif catalyst_high == 0:
        parts.append("未来3个月无明确高影响催化剂，需关注新增催化")

    if note:
        parts.append(note)

    return "；".join(parts)


def analyze_all_chains(chain_names: List[str],
                       chain_strengths: Dict[str, int] = None) -> Dict[str, CyclePhase]:
    """
    分析多条产业链的周期阶段

    参数：
        chain_names: 要分析的产业链名称列表
        chain_strengths: 各链的信号强度 {name: strength}

    返回：{chain_name: CyclePhase}
    """
    chain_strengths = chain_strengths or {}
    results = {}

    for name in chain_names:
        strength = chain_strengths.get(name, 0)
        try:
            results[name] = analyze_chain_cycle(name, strength)
        except Exception:
            results[name] = CyclePhase(
                phase="UNKNOWN", phase_name="分析异常", phase_desc="数据不足",
                progress_pct=50, maturity=50, remaining_months=3,
                remaining_months_low=1, remaining_months_high=6,
                catalyst_count_90d=0, catalyst_count_high_90d=0,
                next_catalysts=[], risk_warnings=["分析异常"],
                reasoning="产业链周期分析异常，建议人工确认",
            )

    return results


def cycle_to_dict(cp: CyclePhase) -> dict:
    """CyclePhase 转为字典（用于 JSON 序列化）"""
    return {
        "phase": cp.phase,
        "phase_name": cp.phase_name,
        "phase_desc": cp.phase_desc,
        "progress_pct": cp.progress_pct,
        "maturity": cp.maturity,
        "remaining_months": cp.remaining_months,
        "remaining_months_low": cp.remaining_months_low,
        "remaining_months_high": cp.remaining_months_high,
        "catalyst_count_90d": cp.catalyst_count_90d,
        "catalyst_count_high_90d": cp.catalyst_count_high_90d,
        "next_catalysts": cp.next_catalysts,
        "risk_warnings": cp.risk_warnings,
        "reasoning": cp.reasoning,
    }
