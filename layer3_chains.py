"""
Layer 3 — 产业链传导图谱（LLM 动态版）
静态知识库 + LLM 动态信号匹配
根据 L1/L2 实际信号判断哪条链在传导、传导到哪个节点
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from llm_client import chat_json, is_available
from trend_judge import _fmt_indicators, _fmt_signals


@dataclass
class ChainNode:
    stage: str          # 国际信号源 / 上游 / 中游 / 下游 / 风险节点
    name: str
    detail: str
    lead_weeks: str     # 领先时间
    a_stocks: List[str] = field(default_factory=list)
    active: bool = False


@dataclass
class IndustryChain:
    name: str
    description: str
    nodes: List[ChainNode]
    l1_triggers: List[str]  # 触发该链的L1信号关键词
    l2_triggers: List[str]  # 触发该链的L2信号关键词
    current_node: Optional[str] = None
    signal_strength: int = 0
    window: str = ""
    reasoning: str = ""     # LLM 给出的激活/不激活理由


@dataclass
class Layer3Result:
    active_chains: List[IndustryChain] = field(default_factory=list)
    inactive_chains: List[IndustryChain] = field(default_factory=list)
    top_nodes: List[dict] = field(default_factory=list)
    market_narrative: str = ""
    cross_chain_signals: list = field(default_factory=list)
    source: str = ""       # "LLM" 或 "RULE_FALLBACK"
    timestamp: str = ""


# ── 产业链知识库（作为 LLM 的参考资料） ───────────────────────────────────

def build_chain_library() -> List[IndustryChain]:
    """静态产业链知识库——描述每条链的结构和逻辑，供 LLM 参考"""
    return [

        IndustryChain(
            name="AI算力链",
            description="从云厂商资本支出到数据中心配套",
            l1_triggers=["用电", "电子", "PMI"],
            l2_triggers=["美债", "美元", "BULLISH"],
            nodes=[
                ChainNode("国际信号源", "云厂商Capex",
                          "微软/谷歌/亚马逊/Meta季报资本支出指引", "领先全链", []),
                ChainNode("国际信号源", "英伟达订单",
                          "台积电CoWoS产能分配、GB200出货节奏", "领先6-8周", []),
                ChainNode("上游", "HBM内存/PCB基板",
                          "SK海力士HBM3e出货量、AI服务器专用基板", "领先4-6周",
                          ["688382", "300666"]),
                ChainNode("中游", "光模块",
                          "800G/1.6T出货量持续放量", "当前热点",
                          ["300308", "300394", "002281"]),
                ChainNode("中游", "液冷散热",
                          "浸没式液冷渗透率提升", "同步-滞后2周",
                          ["002837", "300443"]),
                ChainNode("下游配套", "高压电源/UPS",
                          "数据中心用电配套招标", "滞后4-8周",
                          ["002479", "300002"]),
                ChainNode("下游配套", "IDC运营",
                          "数据中心上架率改善", "滞后8-16周",
                          ["300383", "600171"]),
            ],
        ),

        IndustryChain(
            name="新能源车链",
            description="从锂矿到整车的完整产业链",
            l1_triggers=["新能源", "锂", "铝"],
            l2_triggers=["原油", "NEUTRAL", "BULLISH"],
            nodes=[
                ChainNode("国际信号源", "锂矿价格",
                          "澳洲Pilbara/智利SQM锂精矿报价", "领先全链", []),
                ChainNode("国际信号源", "特斯拉产销量",
                          "全球EV需求风向标", "领先8-12周", []),
                ChainNode("上游", "碳酸锂/氢氧化锂",
                          "现货价+期货曲线+库存周期", "领先6-8周",
                          ["002466", "002240"]),
                ChainNode("上游", "正极/负极材料",
                          "容百科技/贝特瑞/璞泰来", "领先4-6周",
                          ["688005", "835185", "603659"]),
                ChainNode("中游", "电芯/PACK",
                          "宁德时代/亿纬锂能", "同步市场",
                          ["300750", "300014"]),
                ChainNode("中游", "隔膜/电解液",
                          "恩捷股份/天赐材料", "同步-滞后2周",
                          ["002812", "002709"]),
                ChainNode("下游", "整车",
                          "比亚迪/理想月度销量", "滞后4-8周",
                          ["002594", "601127"]),
                ChainNode("下游配套", "充电桩",
                          "特来电/星星充电", "滞后8-20周",
                          ["002522"]),
            ],
        ),

        IndustryChain(
            name="出口制造链",
            description="港口数据+美元走势+海外需求联动",
            l1_triggers=["出口", "港口", "物流", "BDI", "BCI"],
            l2_triggers=["美元", "BULLISH", "ISM"],
            nodes=[
                ChainNode("国际信号源", "美国零售库存比",
                          "库存/销售比下行→补库需求→中国出口订单", "领先8-12周", []),
                ChainNode("国际信号源", "美国ISM制造业PMI",
                          "新订单子项领先出口需求", "领先6-8周", []),
                ChainNode("上游传导", "BDI/BCI运价",
                          "干散货/集装箱运价+港口吞吐量双验证", "领先4-6周",
                          ["601919", "601872"]),
                ChainNode("中游", "港口吞吐量",
                          "上海/宁波/深圳集装箱周报", "领先2-4周",
                          ["600717", "001872"]),
                ChainNode("下游", "出口制造企业",
                          "家电/电子组装/机械设备", "滞后2-4周",
                          ["600690", "300866", "000651"]),
                ChainNode("风险节点", "贸易摩擦/关税",
                          "美国301关税/欧盟反补贴调查", "L0触发", []),
            ],
        ),

        IndustryChain(
            name="工业自动化链",
            description="区分周期逻辑与国产替代逻辑",
            l1_triggers=["制造业", "工业", "PMI", "用电"],
            l2_triggers=["NEUTRAL", "BULLISH"],
            nodes=[
                ChainNode("国际信号源", "发那科/ABB订单",
                          "全球工业自动化需求先行", "领先16-24周", []),
                ChainNode("上游零部件", "减速器",
                          "绿的谐波（谐波）/双环传动（RV）", "领先8-12周",
                          ["688017", "002472"]),
                ChainNode("上游零部件", "伺服电机/控制器",
                          "汇川技术/禾川科技", "领先6-8周",
                          ["300124", "688120"]),
                ChainNode("中游", "工业机器人本体",
                          "埃斯顿/汇川/华中数控", "同步-滞后4周",
                          ["002747", "300124", "300161"]),
                ChainNode("中游新兴", "人形机器人",
                          "特斯拉Optimus量产节奏驱动", "新兴节点",
                          ["688017", "300124", "002747"]),
                ChainNode("下游", "系统集成商",
                          "博众精工/天奇股份", "滞后8-16周",
                          ["688097", "002009"]),
            ],
        ),

        IndustryChain(
            name="半导体材料链",
            description="不跟半导体周期，跟验证进度；国产替代最强逻辑",
            l1_triggers=["半导体", "电子", "用电"],
            l2_triggers=["BEARISH", "美元", "利差"],  # 出口管制升级反而加速国产替代
            nodes=[
                ChainNode("国际信号源", "美国出口管制",
                          "每次升级→国产替代紧迫性+政策资金加速", "政策催化剂", []),
                ChainNode("国际信号源", "台积电/三星Capex",
                          "全球晶圆厂扩产→材料需求总量", "领先24-52周", []),
                ChainNode("上游", "光刻胶",
                          "晶瑞电材/南大光电/徐州博康", "国产替代主线",
                          ["300655", "300346"]),
                ChainNode("上游", "溅射靶材",
                          "江丰电子/有研新材", "扩产周期同步",
                          ["300666", "600206"]),
                ChainNode("中游", "CMP抛光材料",
                          "安集科技/鼎龙股份", "验证节点",
                          ["688019", "300054"]),
                ChainNode("中游", "电子特气",
                          "华特气体/雅克科技", "同步",
                          ["688268", "002409"]),
                ChainNode("下游验证", "晶圆厂采购比例",
                          "中芯国际/华虹季报采购订单是最可靠信号", "滞后12-24周",
                          ["688981", "688347"]),
            ],
        ),

        IndustryChain(
            name="航空航天材料链",
            description="C919交付提速+军机产能，长周期结构性机会",
            l1_triggers=["航空", "钛", "碳纤维", "基建"],
            l2_triggers=["BULLISH", "NEUTRAL"],
            nodes=[
                ChainNode("国际信号源", "波音/空客交付量",
                          "全球航空需求基准", "领先全链", []),
                ChainNode("上游", "碳纤维原丝",
                          "中复神鹰/光威复材/恒神股份", "领先16-24周",
                          ["688295", "300699", "688309"]),
                ChainNode("上游", "钛合金",
                          "西部超导/宝钛股份", "领先12-20周",
                          ["688122", "600456"]),
                ChainNode("中游", "复合材料制造",
                          "航天复材/中航高科", "同步",
                          ["300617", "600862"]),
                ChainNode("下游", "航空/航天主机厂",
                          "C919交付节奏/军机生产计划", "滞后20-40周",
                          ["600760", "000768"]),
            ],
        ),

        IndustryChain(
            name="新能源材料链（SiC）",
            description="碳化硅渗透率持续提升，国产替代+需求增量双驱动",
            l1_triggers=["新能源", "电子", "用电"],
            l2_triggers=["BULLISH", "NEUTRAL"],
            nodes=[
                ChainNode("国际信号源", "特斯拉SiC用量",
                          "Model 3/Y每台SiC用量/Wolfspeed产能", "领先8-12周", []),
                ChainNode("上游", "SiC衬底/外延",
                          "天岳先进/天科合达/露笑科技", "领先8-12周",
                          ["688234", "301049"]),
                ChainNode("中游", "SiC器件/模块",
                          "斯达半导/时代电气/比亚迪半导体", "同步-滞后4周",
                          ["603290", "688187"]),
                ChainNode("下游", "新能源车/储能",
                          "SiC渗透率提升持续驱动上游", "滞后12-20周",
                          ["002594", "300750"]),
            ],
        ),

        IndustryChain(
            name="铜/铝/有色金属链",
            description="全球定价+中国需求，宏观和产业双重驱动",
            l1_triggers=["铜", "铝", "基建", "房地产", "用电"],
            l2_triggers=["美元", "美债", "原油"],
            nodes=[
                ChainNode("国际信号源", "美元指数/美债",
                          "美元走弱+利率下行→有色金属估值提升", "领先4-8周", []),
                ChainNode("国际信号源", "LME铜/铝库存",
                          "全球显性库存变动", "领先2-4周", []),
                ChainNode("上游", "铜矿/铝土矿",
                          "紫金矿业/中国铝业/神火股份", "领先6-8周",
                          ["601899", "601600", "000933"]),
                ChainNode("中游", "铜加工/铝加工",
                          "铜陵有色/南山铝业", "同步",
                          ["000630", "600219"]),
                ChainNode("下游", "电力/新能源/建筑",
                          "电网投资+新能源装机+建筑竣工", "滞后4-12周",
                          ["600025", "601985"]),
            ],
        ),

        IndustryChain(
            name="生猪养殖链",
            description="猪周期驱动，关注产能去化进度",
            l1_triggers=["生猪", "玉米", "大豆", "农产品", "食品"],
            l2_triggers=["NEUTRAL"],
            nodes=[
                ChainNode("上游", "饲料原料",
                          "玉米/大豆价格决定养殖成本线", "同步",
                          ["002100", "000930"]),
                ChainNode("中游", "种猪/仔猪",
                          "能繁母猪存栏决定未来供应", "领先6-10个月",
                          ["002567"]),
                ChainNode("中游", "商品猪养殖",
                          "牧原/温氏/新希望出栏量", "同步市场",
                          ["002714", "300498", "000876"]),
                ChainNode("下游", "屠宰/肉制品",
                          "双汇/龙大美食", "滞后2-4周",
                          ["000895", "002726"]),
            ],
        ),
    ]


def _chain_to_llm_input(chain: IndustryChain) -> dict:
    """把产业链对象转为 LLM 可读的 JSON 描述"""
    return {
        "name": chain.name,
        "description": chain.description,
        "triggers": {
            "L1信号关键词": chain.l1_triggers,
            "L2信号关键词": chain.l2_triggers,
        },
        "nodes": [
            {
                "阶段": n.stage,
                "节点": n.name,
                "说明": n.detail,
                "领先时间": n.lead_weeks,
                "相关A股": n.a_stocks,
            }
            for n in chain.nodes
        ],
    }


# ── LLM 动态匹配 ─────────────────────────────────────────────────────────

CHAIN_SYSTEM_PROMPT = """你是一位A股产业链传导分析专家。你的任务是根据当前的实业数据信号和国际金融信号，判断哪些产业链正在被激活、传导到了哪个节点。

分析原则：
1. 信号必须有数据支撑，不要凭空推测
2. 产业链传导有先后顺序：国际信号源→上游→中游→下游
3. 传导强度取决于信号数量和一致性（多个同向信号 > 单个强信号）
4. 关注跨产业链的关联（如美元走弱同时利好有色金属和出口制造）
5. 不要所有链都给高分——大部分时候只有1-2条主线"""

CHAIN_USER_PROMPT_TEMPLATE = """## 当前 Layer 1 实业数据信号
### 食品
{food}
### 大宗商品
{bulk}
### 宏观货币
{macro}
### 基建
{infra}
### 用电量
{electricity}
### 物流运价
{logistics}

## 当前 Layer 2 国际/金融信号
{intl}

## 国际信号源（新闻数据，产业链传导起点）
### AI算力/半导体
{intl_ai}
### 半导体材料
{intl_semi}
### 新能源车/SiC
{intl_auto}
### 工业自动化
{intl_auto2}
### 航空航天
{intl_aero}
### 有色金属
{intl_metal}
### 港口出口
{intl_export}
### 生猪养殖
{intl_pig}

## 可选产业链知识库
{chains}

请严格按以下JSON格式输出，不要输出其他内容：
{{
  "chains": [
    {{
      "name": "链名（必须与知识库中的名称一致）",
      "active": true或false,
      "strength": 0到100的整数（信号强度）,
      "current_node": "当前传导到的节点名称",
      "reasoning": "50字以内：为什么激活或未激活，引用了哪些具体信号（包括国际信号源新闻数据）",
      "window": "30字以内的操作窗口建议"
    }}
  ],
  "cross_chain_signals": ["跨产业链的关联信号（1-3条）"],
  "market_narrative": "100字以内：用一段话总结当前市场的主线逻辑和资金方向，必须引用国际信号源的具体数据"
}}"""


def llm_match_chains(l1_result, l2_result, intl_signals=None) -> Layer3Result:
    """用 LLM 动态匹配产业链（含国际信号源新闻数据）"""
    if not is_available():
        return rule_match_chains(l1_result, l2_result)

    food_str = _fmt_indicators(l1_result.food)
    bulk_str = _fmt_indicators(l1_result.bulk)
    macro_str = _fmt_indicators(l1_result.macro)
    infra_str = _fmt_indicators(l1_result.infra)
    elec_str = _fmt_indicators(l1_result.electricity)
    logi_str = _fmt_indicators(l1_result.logistics)
    intl_str = _fmt_signals(l2_result.intl)

    # ── 国际信号源（外部传入或自行采集） ──
    if intl_signals is None:
        intl_signals = []
        try:
            from international_signals import run_intl_signals_collection
            intl_result = run_intl_signals_collection(days=14)
            intl_signals = intl_result.signals
            print(f"  → 国际信号：{intl_result.raw_news_count} 条新闻，"
                  f"{intl_result.llm_extracted} 条结构化信号")
        except Exception as e:
            print(f"  [WARN] 国际信号采集失败: {e}")

    from international_signals import signals_to_text_for_chain

    # 按类别格式化国际信号（每条链单独分组，让 LLM 精确匹配）
    intl_ai = signals_to_text_for_chain(intl_signals, "AI算力链")
    intl_semi = signals_to_text_for_chain(intl_signals, "半导体材料链")
    intl_auto = signals_to_text_for_chain(intl_signals, "新能源车链")
    intl_auto2 = signals_to_text_for_chain(intl_signals, "工业自动化链")
    intl_aero = signals_to_text_for_chain(intl_signals, "航空航天材料链")
    intl_sic = signals_to_text_for_chain(intl_signals, "新能源材料链（SiC）")
    intl_metal = signals_to_text_for_chain(intl_signals, "铜/铝/有色金属链")
    intl_export = signals_to_text_for_chain(intl_signals, "出口制造链")
    intl_pig = signals_to_text_for_chain(intl_signals, "生猪养殖链")

    chains_lib = build_chain_library()
    chains_json = json.dumps(
        [_chain_to_llm_input(c) for c in chains_lib],
        ensure_ascii=False, indent=2
    )

    prompt = CHAIN_USER_PROMPT_TEMPLATE.format(
        food=food_str, bulk=bulk_str, macro=macro_str,
        infra=infra_str, electricity=elec_str, logistics=logi_str,
        intl=intl_str, intl_ai=intl_ai, intl_semi=intl_semi,
        intl_auto=intl_auto, intl_auto2=intl_auto2,
        intl_aero=intl_aero, intl_sic=intl_sic,
        intl_metal=intl_metal, intl_export=intl_export,
        intl_pig=intl_pig,
        chains=chains_json,
    )

    print("  调用 DeepSeek 进行产业链匹配...")
    result = chat_json(CHAIN_SYSTEM_PROMPT, prompt, temperature=0.2)

    if result is None:
        print("  [WARN] LLM 调用失败，使用规则兜底")
        return rule_match_chains(l1_result, l2_result)

    # 解析结果并映射回原始链对象
    chain_map = {c.name: c for c in chains_lib}
    active, inactive = [], []

    for chain_data in result.get("chains", []):
        name = chain_data.get("name", "")
        if name in chain_map:
            c = chain_map[name]
            c.signal_strength = int(chain_data.get("strength", 0))
            c.current_node = chain_data.get("current_node", "")
            c.window = chain_data.get("window", "")
            c.reasoning = chain_data.get("reasoning", "")
            is_active = chain_data.get("active", False)

            # 标记活跃节点
            if is_active and c.current_node:
                for node in c.nodes:
                    node.active = (node.name == c.current_node)

            if is_active and c.signal_strength >= 40:
                active.append(c)
            else:
                inactive.append(c)

    top_nodes = []
    for chain in sorted(active, key=lambda c: c.signal_strength, reverse=True)[:3]:
        top_nodes.append({
            "chain": chain.name,
            "node": chain.current_node,
            "strength": chain.signal_strength,
            "window": chain.window,
            "reasoning": chain.reasoning,
            "a_stocks": list({s for n in chain.nodes if n.active for s in n.a_stocks}),
        })

    return Layer3Result(
        active_chains=active,
        inactive_chains=inactive,
        top_nodes=top_nodes,
        market_narrative=result.get("market_narrative", ""),
        cross_chain_signals=result.get("cross_chain_signals", []),
        source="LLM",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    )


# ── 规则兜底（LLM 不可用时） ────────────────────────────────────────────

def rule_match_chains(l1_result, l2_result) -> Layer3Result:
    """纯规则的产业链匹配"""
    chains = build_chain_library()
    active, inactive = [], []

    l1_up_names = [i.name for i in
                   (l1_result.food + l1_result.bulk + l1_result.macro +
                    l1_result.infra + l1_result.electricity + l1_result.logistics)
                   if "UP" in i.signal]

    l2_dominant = "BULLISH" if l2_result.score >= 55 else (
                  "BEARISH" if l2_result.score <= 45 else "NEUTRAL")

    for chain in chains:
        # 关键词匹配计数
        l1_hits = sum(1 for trigger in chain.l1_triggers
                      if any(trigger in name for name in l1_up_names))
        l2_hits = sum(1 for trigger in chain.l2_triggers
                      if trigger in l2_dominant or
                      any(trigger in s.name for s in l2_result.intl if s.direction == "BULLISH"))
        total_hits = l1_hits + l2_hits

        # 动态计算强度
        if total_hits >= 4:
            chain.signal_strength = min(90, 50 + total_hits * 10)
        elif total_hits >= 2:
            chain.signal_strength = min(70, 30 + total_hits * 10)
        elif total_hits >= 1:
            chain.signal_strength = 25
        else:
            chain.signal_strength = 10

        if chain.signal_strength >= 40:
            # 找到匹配的节点
            for node in chain.nodes:
                for trigger in chain.l1_triggers:
                    if trigger in " ".join(l1_up_names):
                        if not any(n.active for n in chain.nodes):
                            node.active = True
                            chain.current_node = node.name
                        break

            chain.window = "LLM不可用，仅基于信号关键词匹配，建议配置DeepSeek API"
            chain.reasoning = f"L1关键词命中{l1_hits}个，L2命中{l2_hits}个"
            active.append(chain)
        else:
            inactive.append(chain)

    top_nodes = []
    for chain in sorted(active, key=lambda c: c.signal_strength, reverse=True)[:3]:
        top_nodes.append({
            "chain": chain.name,
            "node": chain.current_node or "未知",
            "strength": chain.signal_strength,
            "window": chain.window,
            "reasoning": chain.reasoning,
            "a_stocks": list({s for n in chain.nodes if n.active for s in n.a_stocks}),
        })

    return Layer3Result(
        active_chains=active,
        inactive_chains=inactive,
        top_nodes=top_nodes,
        market_narrative="LLM不可用，建议配置 DEEPSEEK_API_KEY 以获得产业链动态分析",
        cross_chain_signals=[],
        source="RULE_FALLBACK",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    )


def run_layer3(l1_result, l2_result, intl_signals=None) -> Layer3Result:
    return llm_match_chains(l1_result, l2_result, intl_signals)
