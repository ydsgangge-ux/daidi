"""
国际信号源采集器 — 补充 L3 产业链的国际信号源节点数据
策略：akshare 新闻聚合 + 费城半导体指数 + LLM 结构化提取
"""
import akshare as ak
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from llm_client import chat_json, is_available


@dataclass
class IntlSourceSignal:
    """国际信号源结构化数据"""
    category: str        # 信号类别（云厂商/半导体/电动车/航空/机器人/有色）
    name: str            # 信号名称
    value: str           # 信号值（文字描述）
    direction: str       # BULLISH / BEARISH / NEUTRAL
    detail: str          # 详细描述
    source: str          # 来源
    date: str            # 新闻日期
    relevance_chains: List[str] = field(default_factory=list)  # 关联产业链


@dataclass
class IntlSignalsResult:
    """国际信号采集结果"""
    signals: List[IntlSourceSignal] = field(default_factory=list)
    sox_index: Optional[dict] = None
    raw_news_count: int = 0
    llm_extracted: int = 0
    timestamp: str = ""


# ── 搜索关键词配置（按产业链节点龙头企业精准搜索） ────────────────────────

SEARCH_KEYWORDS = {
    # ─── AI算力链 ───
    "AI算力_国际源": [
        # 云厂商 Capex
        "微软资本支出", "微软财报", "谷歌资本支出", "谷歌财报",
        "亚马逊AWS营收", "Meta资本支出",
        # 英伟达/台积电
        "英伟达营收", "英伟达订单", "英伟达GB200", "英伟达交付",
        "台积电营收", "台积电产能", "台积电CoWoS",
        # SK海力士 HBM
        "SK海力士HBM", "HBM3e",
    ],
    "AI算力_国内": [
        # 光模块龙头
        "中际旭创", "新易盛", "光迅科技", "800G光模块",
        # 液冷
        "英维克", "申菱环境",
        # IDC
        "光环新网", "数据港",
    ],

    # ─── 半导体材料链 ───
    "半导体材料_国际源": [
        "美国芯片出口管制", "半导体出口管制",
        "台积电扩产", "三星半导体投资",
        "全球晶圆厂", "中芯国际",
    ],
    "半导体材料_国内": [
        # 光刻胶
        "晶瑞电材", "南大光电",
        # 靶材
        "江丰电子", "有研新材",
        # CMP
        "安集科技", "鼎龙股份",
        # 特气
        "华特气体", "雅克科技",
    ],

    # ─── 新能源车链 ───
    "新能源车_国际源": [
        "特斯拉交付", "特斯拉销量", "特斯拉Q1", "特斯拉Q2",
        "Pilbara锂矿", "SQM锂", "碳酸锂价格", "锂矿拍卖",
    ],
    "新能源车_国内": [
        # 动力电池
        "宁德时代", "亿纬锂能",
        # 正极负极
        "容百科技", "贝特瑞", "璞泰来",
        # 隔膜电解液
        "恩捷股份", "天赐材料",
        # 整车
        "比亚迪销量", "理想汽车销量",
        # 充电桩
        "特锐德充电桩",
    ],

    # ─── 工业自动化链 ───
    "工业自动化_国际源": [
        "发那科订单", "发那科营收", "ABB订单", "ABB机器人",
        "西门子工业", "人形机器人",
    ],
    "工业自动化_国内": [
        # 龙头
        "汇川技术", "埃斯顿",
        # 零部件
        "绿的谐波", "双环传动", "禾川科技",
        # 系统
        "博众精工", "华中数控",
    ],

    # ─── 航空航天材料链 ───
    "航空航天_国际源": [
        "波音交付", "空客交付", "波音订单",
        "航空发动机", "商飞C919",
    ],
    "航空航天_国内": [
        # 碳纤维
        "中复神鹰", "光威复材",
        # 钛合金
        "西部超导", "宝钛股份",
        # 复材/主机厂
        "中航高科", "中航沈飞",
    ],

    # ─── 新能源材料链（SiC） ───
    "SiC": [
        "Wolfspeed", "天岳先进", "天科合达",
        "斯达半导", "时代电气",
        "碳化硅衬底", "碳化硅器件",
    ],

    # ─── 铜/铝/有色金属链 ───
    "有色金属_国际源": [
        "LME铜", "LME铝", "铜价", "铝价",
        "铜库存", "铝库存",
    ],
    "有色金属_国内": [
        "紫金矿业", "中国铝业", "神火股份",
        "铜陵有色", "南山铝业",
    ],

    # ─── 港口物流/出口制造链 ───
    "港口物流": [
        "集装箱运价", "BDI指数", "波罗的海",
        "中远海控", "上港集团",
        "宁波港", "出口数据",
    ],

    # ─── 生猪养殖链 ───
    "生猪养殖": [
        "牧原股份", "温氏股份", "新希望",
        "生猪价格", "能繁母猪",
        "玉米价格", "大豆价格",
    ],
}

# 每个产业链关注的信号类别
CHAIN_RELEVANCE = {
    "AI算力链":               ["AI算力_国际源", "AI算力_国内", "半导体材料_国际源"],
    "半导体材料链":           ["半导体材料_国际源", "半导体材料_国内", "AI算力_国际源"],
    "新能源车链":             ["新能源车_国际源", "新能源车_国内", "有色金属_国际源"],
    "出口制造链":             ["港口物流", "有色金属_国际源"],
    "工业自动化链":           ["工业自动化_国际源", "工业自动化_国内", "半导体材料_国内"],
    "航空航天材料链":         ["航空航天_国际源", "航空航天_国内"],
    "新能源材料链（SiC）":     ["SiC", "新能源车_国际源"],
    "铜/铝/有色金属链":       ["有色金属_国际源", "有色金属_国内", "港口物流"],
    "生猪养殖链":             ["生猪养殖"],
}


def fetch_news_by_keywords(keywords: List[str], days: int = 30) -> List[dict]:
    """
    按关键词批量抓取东方财富新闻
    返回去重后的新闻列表
    """
    cutoff = datetime.now() - timedelta(days=days)
    seen_titles = set()
    all_news = []

    for kw in keywords:
        try:
            df = ak.stock_news_em(symbol=kw)
            if df is None or len(df) == 0:
                continue

            for _, row in df.iterrows():
                title = str(row.get("新闻标题", ""))
                content = str(row.get("新闻内容", ""))
                date_str = str(row.get("发布时间", ""))
                source = str(row.get("文章来源", ""))

                # 去重
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # 日期过滤（只保留最近的）
                try:
                    news_date = pd.to_datetime(date_str)
                    if news_date < cutoff:
                        continue
                    date_fmt = news_date.strftime("%Y-%m-%d")
                except Exception:
                    date_fmt = date_str[:10] if len(date_str) >= 10 else date_str

                # 跳过太短的内容
                if len(content) < 20:
                    continue

                all_news.append({
                    "keyword": kw,
                    "title": title,
                    "content": content[:200],  # 截断避免太长
                    "date": date_fmt,
                    "source": source,
                })
        except Exception:
            continue

    # 按日期倒序
    all_news.sort(key=lambda x: x["date"], reverse=True)
    return all_news


def fetch_sox_index() -> Optional[dict]:
    """获取费城半导体指数"""
    try:
        df = ak.macro_global_sox_index()
        if df is not None and len(df) > 0:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            result = {
                "date": str(latest["日期"]),
                "value": float(latest["最新值"]),
                "daily_chg": round(float(latest["涨跌幅"]), 2),
            }
            if prev is not None:
                result["prev_value"] = float(prev["最新值"])
                result["m1_chg"] = round(float(latest["近3月涨跌幅"]), 2)
                result["y1_chg"] = round(float(latest["近1年涨跌幅"]), 2)
            return result
    except Exception:
        pass
    return None


def llm_extract_signals(news_list: List[dict]) -> List[IntlSourceSignal]:
    """用 LLM 从新闻文本中提取结构化信号"""
    if not is_available() or not news_list:
        return []

    # 构建新闻文本
    news_text = ""
    for i, n in enumerate(news_list[:50]):  # 最多50条，避免token超限
        news_text += f"\n{i+1}. [{n['date']}] {n['title']}\n   {n['content']}\n"

    system_prompt = """你是国际产业链信号分析专家。你的任务是从财经新闻中提取结构化的产业链信号数据。

分析原则：
1. 只提取有具体数据的新闻（如"交付35.8万辆"、"同比增长12.6%"等），忽略纯观点性文章
2. 判断信号方向：BULLISH（利好相关产业链）/ BEARISH（利空）/ NEUTRAL（中性）
3. 关联到最相关的产业链

信号类别只允许以下值：AI算力、半导体材料、新能源车、工业自动化、航空航天、新能源材料SiC、有色金属、港口物流、生猪养殖"""

    user_prompt = f"""以下是最近的产业链相关财经新闻：

{news_text}

请严格按以下JSON格式提取信号，不要输出其他内容：
{{
  "signals": [
    {{
      "category": "信号类别（AI算力/半导体材料/新能源车/工业自动化/航空航天/新能源材料SiC/有色金属/港口物流/生猪养殖）",
      "name": "信号名称（10字以内）",
      "value": "信号值（具体数据，如'交付35.8万辆'）",
      "direction": "BULLISH或BEARISH或NEUTRAL",
      "detail": "30字以内的详细描述和影响分析",
      "source": "新闻来源"
    }}
  ]
}}

注意：
- 只提取有实际数据支撑的信号，不要编造
- 最多提取15条最重要的信号
- 忽略重复的信号，保留信息量最大的
- 忽略与产业链无关的新闻"""

    print(f"  调用 LLM 从 {len(news_list)} 条新闻中提取信号...")
    result = chat_json(system_prompt, user_prompt, temperature=0.2)

    if result is None:
        return []

    signals = []
    for s in result.get("signals", []):
        category = s.get("category", "")
        # 自动关联产业链（支持模糊匹配：LLM返回"AI算力"能匹配"AI算力_国际源"）
        relevant_chains = []
        for chain, cats in CHAIN_RELEVANCE.items():
            if category in cats or any(category in c for c in cats):
                relevant_chains.append(chain)

        signals.append(IntlSourceSignal(
            category=category,
            name=s.get("name", ""),
            value=s.get("value", ""),
            direction=s.get("direction", "NEUTRAL"),
            detail=s.get("detail", ""),
            source=s.get("source", "新闻"),
            date=datetime.now().strftime("%Y-%m-%d"),
            relevance_chains=relevant_chains,
        ))

    return signals


def run_intl_signals_collection(days: int = 30) -> IntlSignalsResult:
    """
    执行完整的国际信号采集流程
    1. 按产业链关键词批量抓新闻
    2. 获取费城半导体指数
    3. LLM 提取结构化信号
    """
    result = IntlSignalsResult(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    # 1. 批量抓新闻
    print("  采集国际产业链新闻...")
    all_keywords = []
    for cat, kws in SEARCH_KEYWORDS.items():
        all_keywords.extend(kws)

    # 去重关键词
    all_keywords = list(dict.fromkeys(all_keywords))

    all_news = fetch_news_by_keywords(all_keywords, days=days)
    result.raw_news_count = len(all_news)
    print(f"  → 采集到 {len(all_news)} 条新闻")

    # 2. 费城半导体指数
    sox = fetch_sox_index()
    if sox:
        result.sox_index = sox
        sox_dir = "BULLISH" if sox.get("daily_chg", 0) > 0 else (
                  "BEARISH" if sox.get("daily_chg", 0) < -1 else "NEUTRAL")
        result.signals.append(IntlSourceSignal(
            category="AI算力",
            name="费城半导体指数",
            value=f"SOX {sox['value']:.0f}",
            direction=sox_dir,
            detail=f"费城半导体指数 {sox['value']:.0f}，日涨跌 {sox['daily_chg']:.2f}%，"
                   f"近1年 {sox.get('y1_chg', 0):.1f}%",
            source="费城证券交易所",
            date=sox["date"],
            relevance_chains=["AI算力链", "半导体材料链"],
        ))

    # 3. LLM 提取结构化信号
    if all_news:
        extracted = llm_extract_signals(all_news)
        result.signals.extend(extracted)
        result.llm_extracted = len(extracted)
        print(f"  → LLM 提取 {len(extracted)} 条结构化信号")

    return result


def signals_to_text_for_chain(signals: List[IntlSourceSignal],
                               chain_name: str) -> str:
    """将信号格式化为 L3 产业链匹配可用的文本"""
    relevant = []
    for s in signals:
        chains = s.relevance_chains
        # 兼容 LLM 返回单字符串而非列表的情况
        if isinstance(chains, str):
            chains = [chains]
        elif not isinstance(chains, (list, tuple)):
            chains = []
        if chain_name in chains:
            relevant.append(s)
    if not relevant:
        return "  （暂无相关新闻信号）"

    lines = []
    for s in relevant:
        dir_mark = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(s.direction, "⚪")
        lines.append(f"  {dir_mark} {s.name}: {s.value} [{s.direction}] {s.detail}")
    return "\n".join(lines)
