"""
Layer 2 — 金融期货 & 国际信号
美元指数 / 黄金 / 中美国债收益率 / USD/CNY / 股指期货基差
与 Layer 1 不重叠（不含国内农产品期货）
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


@dataclass
class IntlSignal:
    name: str
    value: float
    unit: str
    change: Optional[float]      # 近期变动
    direction: str               # BULLISH / BEARISH / NEUTRAL
    a_share_impact: str          # 对A股的传导含义
    source: str


@dataclass
class Layer2Result:
    intl: list         = field(default_factory=list)
    futures: list      = field(default_factory=list)
    bank: list         = field(default_factory=list)    # 银行/信用信号
    divergence: list   = field(default_factory=list)   # 与L1的背离提示
    score: int         = 50
    timestamp: str     = ""


def _safe_pct(latest, prev):
    """安全计算变动百分比"""
    if prev == 0 or pd.isna(prev) or pd.isna(latest):
        return None
    return round((latest - prev) / abs(prev) * 100, 2)


# ── L2 深层含义分析模板 ──────────────────────────────────────────────────────

_L2_IMPACT = {
    "美元指数 DXY": {
        "BULLISH": ("美元走弱→人民币升值→北向资金流入A股窗口打开。"
                    "美元跌的核心原因通常是：美联储降息预期升温、美国经济数据走弱。"
                    "利好：外资重仓股（贵州茅台、招商银行）、"
                    "新兴市场成长股（科技、医药）、"
                    "大宗商品（铜、铝、黄金以美元计价，美元跌→商品涨）。"
                    "关注：北向资金是否在美元走弱后1-2周内净流入——如果流入，确认信号"),
        "BEARISH": ("美元走强→人民币贬值→北向资金流出压力+外资重仓股承压。"
                    "美元强的核心原因通常是：美联储加息/不降息、美国经济强劲、避险需求。"
                    "利空：外资重仓股（茅台、招行）、科技成长（资金回流美国）。"
                    "但利好：出口制造链（海尔智家、安克创新——美元强→出口商品更有竞争力）。"
                    "风险：如果美元持续走强+中美利差扩大，A股可能大幅调整"),
        "NEUTRAL": ("美元平稳，人民币汇率稳定，外资流入流出处于平衡状态"),
    },
    "美元/离岸人民币 USD/CNH": {
        "BULLISH": ("人民币升值→中国资产对外资更便宜→北向资金可能加仓A股。"
                    "人民币升值还利好：航空（中国国航——美元债务贬值，汇兑收益）、"
                    "造纸（进口木浆成本下降）。"
                    "出口链利空（安克创新、海尔——海外收入换算的人民币减少）"),
        "BEARISH": ("人民币贬值→外资流出压力+进口成本上升。"
                    "贬值利好出口链（安克创新、海尔智家、中远海控——海外收入变相增加）。"
                    "但贬值反映中国经济相对美国走弱，整体情绪偏负面。"
                    "如果跌破7.3关键位，央行可能出手干预（外汇存款准备金率/逆周期因子）"),
        "NEUTRAL": ("汇率平稳，央行货币政策空间充足"),
    },
    "上海金现货 Au99.99": {
        "BULLISH": ("黄金上涨的三种驱动：①实际利率下行（美债收益率-通胀预期）"
                    "②地缘冲突（中东战争/俄乌局势紧张→避险资金涌入黄金）"
                    "③央行去美元化（中国/印度央行持续增持黄金储备）。"
                    "如果三者共振，金价可能突破历史新高。"
                    "关注：黄金股弹性通常是金价的2-3倍（紫金矿业、山东黄金、中金黄金）。"
                    "注意：黄金持续大涨=市场在定价极端风险，整体股市可能承压"),
        "BEARISH": ("黄金回调可能只是短期获利了结，也可能是风险偏好回暖（资金回流股市）。"
                    "如果回调幅度>5%，需确认驱动因素是否逆转。"
                    "回调到关键支撑位（如550元/克）可能是买入黄金股的机会"),
        "NEUTRAL": ("金价平稳，等待美联储政策和地缘局势变化"),
    },
    "SPDR黄金ETF持仓": {
        "BULLISH": ("全球最大黄金ETF增持=机构资金在持续买入黄金。"
                    "这通常是中长期趋势确认信号（机构不是短线资金）。"
                    "增持持续3个月以上→黄金大牛市信号。"
                    "关注：央行购金数据（如果中国央行同时增持，信号更强）"),
        "BEARISH": ("ETF减持=机构在获利了结或对黄金前景转为悲观。"
                    "如果是短期减持（<10吨），影响有限；如果连续减持>30吨，需要警惕"),
        "NEUTRAL": ("持仓稳定，机构对黄金态度中性"),
    },
    "美债10Y收益率": {
        "BULLISH": ("美债10Y下行→全球估值中枢上移→A股受益！"
                    "这是全球股市最重要的定价锚——美债收益率=全球无风险利率。"
                    "10Y下行100bp（如从5%到4%）→全球股市估值平均提升15-20%。"
                    "利好：科技成长（估值对利率最敏感）、"
                    "创新药（久期长的成长股受益最大）、"
                    "黄金（实际利率下降→黄金上涨）。"
                    "核心逻辑：资金从低风险的美债流出，寻找更高收益的风险资产（A股）"),
        "BEARISH": ("美债10Y上行→全球估值被压制，A股承压。"
                    "10Y上行=全球'资金成本'上升→外资回流美国→"
                    "新兴市场资金外流。"
                    "如果10Y突破5%，全球可能进入'高利率持久化'阶段，"
                    "对高估值成长股杀伤力最大。"
                    "防御：高股息（银行、电力、高速公路）、黄金（对冲利率风险）"),
        "NEUTRAL": ("美债收益率平稳，全球流动性环境中性"),
    },
    "美债2Y收益率": {
        "BULLISH": ("短端利率下行说明美联储降息预期增强或市场流动性改善。"
                    "2Y-10Y期限结构变化值得关注："
                    "如果2Y下行+10Y不变→曲线陡峭化→经济复苏预期增强，利好银行股"),
        "BEARISH": ("短端利率上行说明美联储可能加息或市场预期收紧。"
                    "如果2Y>10Y（收益率曲线倒挂），是经典的经济衰退预警信号！"
                    "历史上倒挂后12-18个月大概率出现衰退。"
                    "但对中国A股的影响取决于中国自身经济周期——如果中国经济在复苏，"
                    "可以独立于美国走牛"),
        "NEUTRAL": ("短端利率稳定，美联储政策路径清晰"),
    },
    "沪深300 QVIX(恐慌代理)": {
        "BEARISH": ("QVIX偏高→A股恐慌情绪升温。深层含义："
                    "市场在预期大幅下跌，波动率飙升。"
                    "QVIX>20=极度恐慌，往往是阶段性底部区域（恐慌性抛售接近尾声）。"
                    "QVIX>25=系统性风险，必须降低仓位控制回撤。"
                    "历史规律：QVIX极端高点出现后1-2周，A股大概率反弹（恐慌释放完毕）"),
        "BULLISH": ("QVIX低位→A股情绪乐观，做多环境友好。"
                    "QVIX<12=市场过于平静，可能是暴风雨前的宁静（波动率有均值回归特征）。"
                    "注意：QVIX低位+放量上涨=趋势健康；QVIX低位+缩量横盘=变盘在即"),
        "NEUTRAL": ("QVIX正常区间，市场情绪平稳"),
    },
    "中国10Y国债收益率": {
        "BULLISH": ("国债收益率下行→'资产荒'加剧→机构资金被迫向权益资产迁移。"
                    "逻辑：银行/保险找不到高收益的安全资产→被迫买股票/基金增加收益。"
                    "如果10Y跌破2.0%，'资产荒'极端化，高股息股（股息率>国债收益率）"
                    "将获得大量资金追捧。"
                    "关注：长江电力（股息率3.5%+）、中国神华（股息率5%+）、"
                    "四大行（股息率5%+）——国债越低，高股息越香"),
        "BEARISH": ("国债收益率上行→无风险利率上升→股市估值承压。"
                    "通常因经济复苏预期增强（资金从债券→股票）。"
                    "如果国债上行+股市上涨=健康复苏（经济好转驱动）；"
                    "如果国债上行+股市下跌=流动性收紧（政策收紧），需警惕"),
        "NEUTRAL": ("国债收益率稳定，货币政策空间充足"),
    },
    "中美10Y利差": {
        "BULLISH": ("中美利差转正或大幅收窄→外资回流A股的核心窗口！"
                    "利差转正意味着：投资中国国债的收益率>投资美国国债→"
                    "全球资金重新配置到中国资产。"
                    "关注：北向资金流入规模、人民币汇率、A股成交额是否同步放大。"
                    "历史上利差转正后，A股往往有一波可观的上涨行情"),
        "BEARISH": ("中美利差深度倒挂→外资持续流出压力。"
                    "倒挂>150bp=极端情况，A股资金面严重承压。"
                    "但注意：中国A股的走势最终取决于中国经济自身，"
                    "利差倒挂≠A股必跌（2022-2023年持续倒挂但A股仍有结构性机会）"),
        "NEUTRAL": ("中美利差在正常区间，外资流动处于平衡状态"),
    },
    "美国ISM制造业PMI": {
        "BULLISH": ("ISM>52=全球制造业扩张→利好中国出口链！"
                    "中国是全球制造业工厂，美国PMI↑→美国需求↑→中国出口↑。"
                    "关注：出口制造链（海尔智家、安克创新、巨星科技），"
                    "航运（中远海控），港口（上港集团）。"
                    "ISM新订单子项>52更关键——新订单领先实际出口2-3个月"),
        "BEARISH": ("ISM<48=全球制造业收缩→中国出口承压。"
                    "如果ISM连续<48超过6个月，全球经济可能进入衰退。"
                    "出口链需要回避，转向内需链（消费、基建、新能源）"),
        "NEUTRAL": ("ISM在50附近，全球制造业方向不明"),
    },
    # ─── 股指期货 ───
    "IF（沪深300）": {
        "BULLISH": ("IF升水=机构看多沪深300（期货价格>现货价格）。"
                    "升水越大，说明机构越看好后市。"
                    "关注：如果IF升水>30点+北向资金同时流入，趋势确认信号强烈"),
        "BEARISH": ("IF贴水=机构看空或量化对冲活跃。"
                    "贴水<30点=轻度看空，贴水>50点=极度悲观。"
                    "但注意：IF贴水可能是量化中性策略的自然结果（空IF对冲现货），"
                    "不一定代表真正的看空。需结合北向资金方向综合判断"),
        "NEUTRAL": ("基差接近零，市场多空分歧不大"),
    },
    "IC（中证500）": {
        "BULLISH": ("IC升水=机构看好中盘股（中小市值公司）。"
                    "IC代表中证500，成分股以制造业和中盘成长为主。"
                    "IC升水→资金流向中小盘→市场风险偏好提升。"
                    "关注：中证500ETF、中证1000ETF"),
        "BEARISH": ("IC贴水=机构看空中盘股或对冲需求增大。"
                    "中盘股流动性较差，贴水时波动更大，注意控制仓位"),
        "NEUTRAL": ("IC基差平稳"),
    },
    "IM（中证1000）": {
        "BULLISH": ("IM升水=机构看好小盘股（微盘/题材/成长）。"
                    "IM代表中证1000，成分股最小、弹性最大。"
                    "IM升水=市场情绪最激进的阶段，题材股可能活跃。"
                    "注意：小盘股波动极大，适合短线但不适合大仓位长持"),
        "BEARISH": ("IM贴水=机构看空小盘股。"
                    "小盘股在下跌趋势中流动性极差（卖不出去），"
                    "IM大幅贴水时需迅速降低小盘股仓位"),
        "NEUTRAL": ("IM基差平稳"),
    },
}


def _l2_impact(name: str, direction: str, default: str = "") -> str:
    """根据信号名称和方向，返回对A股的深层影响分析"""
    templates = _L2_IMPACT.get(name, {})
    d = direction.upper()
    return templates.get(d, default)


def get_international_signals() -> list:
    signals = []

    # ═══════ 核心数据源：中美国债收益率（一次调用获取全部） ═══════
    bond_df = None
    us10y = us2y = cn10y = cn2y = us_spread = cn_spread = None
    try:
        bond_df = ak.bond_zh_us_rate()
        if bond_df is not None and len(bond_df) > 2:
            bond_df.columns = [c.strip() for c in bond_df.columns]
            for col in ['美国国债收益率10年', '美国国债收益率2年',
                        '中国国债收益率10年', '中国国债收益率2年',
                        '美国国债收益率10年-2年', '中国国债收益率10年-2年']:
                if col in bond_df.columns:
                    bond_df[col] = pd.to_numeric(bond_df[col], errors="coerce")
            us10y = float(bond_df['美国国债收益率10年'].dropna().iloc[-1]) if '美国国债收益率10年' in bond_df.columns else None
            us2y = float(bond_df['美国国债收益率2年'].dropna().iloc[-1]) if '美国国债收益率2年' in bond_df.columns else None
            cn10y = float(bond_df['中国国债收益率10年'].dropna().iloc[-1]) if '中国国债收益率10年' in bond_df.columns else None
            cn2y = float(bond_df['中国国债收益率2年'].dropna().iloc[-1]) if '中国国债收益率2年' in bond_df.columns else None
            us_spread = float(bond_df['美国国债收益率10年-2年'].dropna().iloc[-1]) if '美国国债收益率10年-2年' in bond_df.columns else None
            cn_spread = float(bond_df['中国国债收益率10年-2年'].dropna().iloc[-1]) if '中国国债收益率10年-2年' in bond_df.columns else None
    except Exception:
        pass

    # ═══════ 1. 美元指数 DXY（东方财富全球指数） ═══════
    try:
        df = ak.index_global_hist_em(symbol="美元指数")
        if df is not None and len(df) > 5:
            df.columns = [c.strip() for c in df.columns]
            val_col = [c for c in df.columns if '收盘' in c or '最新价' in c or 'close' in c.lower()]
            if not val_col:
                val_col = [df.columns[1]] if len(df.columns) > 1 else [df.columns[0]]
            df[val_col[0]] = pd.to_numeric(df[val_col[0]], errors="coerce")
            series = df[val_col[0]].dropna()
            latest = float(series.iloc[-1])
            prev5  = float(series.iloc[-6]) if len(series) >= 6 else latest
            chg = _safe_pct(latest, prev5)
            direction = "BEARISH" if (chg and chg > 1) else ("BULLISH" if (chg and chg < -1) else "NEUTRAL")
            signals.append(IntlSignal(
                name="美元指数 DXY", value=round(latest, 2), unit="",
                change=chg, direction=direction,
                a_share_impact=_l2_impact("美元指数 DXY", direction),
                source="东方财富"
            ))
    except Exception:
        # fallback: 用 USD/CNH 汇率代理
        try:
            df = ak.forex_hist_em(symbol="USDCNH")
            if df is not None and len(df) > 5:
                df.columns = [c.strip() for c in df.columns]
                val_col = [c for c in df.columns if '收盘' in c or '最新价' in c]
                if not val_col:
                    val_col = [df.columns[1]] if len(df.columns) > 1 else [df.columns[0]]
                df[val_col[0]] = pd.to_numeric(df[val_col[0]], errors="coerce")
                series = df[val_col[0]].dropna()
                latest = float(series.iloc[-1])
                prev5  = float(series.iloc[-6]) if len(series) >= 6 else latest
                chg = _safe_pct(latest, prev5)
                direction = "BEARISH" if (chg and chg > 0.5) else ("BULLISH" if (chg and chg < -0.5) else "NEUTRAL")
                signals.append(IntlSignal(
                    name="美元/离岸人民币 USD/CNH", value=round(latest, 4), unit="",
                    change=chg, direction=direction,
                    a_share_impact=_l2_impact("美元/离岸人民币 USD/CNH", direction),
                    source="东方财富外汇"
                ))
        except Exception:
            signals.append(IntlSignal(
                name="美元指数 DXY", value=0, unit="",
                change=None, direction="NEUTRAL",
                a_share_impact="数据获取失败（网络/限流），无法判断外资流向", source="N/A"
            ))

    # ═══════ 2. 黄金（上海金交所现货 + SPDR持仓） ═══════
    try:
        df = ak.spot_golden_benchmark_sge()
        if df is not None and len(df) > 5:
            df.columns = [c.strip() for c in df.columns]
            price_col = '晚盘价' if '晚盘价' in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
            series = df[price_col].dropna()
            latest = float(series.iloc[-1])
            prev5  = float(series.iloc[-6]) if len(series) >= 6 else float(series.iloc[-2]) if len(series) >= 2 else latest
            chg = _safe_pct(latest, prev5)
            direction = "BULLISH" if (chg and chg > 1) else ("BEARISH" if (chg and chg < -1) else "NEUTRAL")
            signals.append(IntlSignal(
                name="上海金现货 Au99.99", value=round(latest, 2), unit="元/克",
                change=chg, direction=direction,
                a_share_impact=_l2_impact("上海金现货 Au99.99", direction),
                source="上海黄金交易所"
            ))
    except Exception:
        pass

    # SPDR 黄金 ETF 持仓变化（全球黄金需求代理）
    try:
        df = ak.macro_cons_gold()
        if df is not None and len(df) > 2:
            df.columns = [c.strip() for c in df.columns]
            df['增持/减持'] = pd.to_numeric(df['增持/减持'], errors="coerce")
            df['总库存'] = pd.to_numeric(df['总库存'], errors="coerce")
            latest_change = float(df['增持/减持'].dropna().iloc[-1])
            total_hold = float(df['总库存'].dropna().iloc[-1])
            direction = "BULLISH" if latest_change > 5 else ("BEARISH" if latest_change < -5 else "NEUTRAL")
            signals.append(IntlSignal(
                name="SPDR黄金ETF持仓", value=round(total_hold, 1), unit="吨",
                change=round(latest_change, 2), direction=direction,
                a_share_impact=_l2_impact("SPDR黄金ETF持仓", direction),
                source="SPDR/akshare"
            ))
    except Exception:
        pass

    # ═══════ 3. 美债10Y收益率 ═══════
    if us10y is not None:
        # 从 bond_df 计算近期变动
        chg_bp = None
        if bond_df is not None and len(bond_df) >= 6:
            series = bond_df['美国国债收益率10年'].dropna()
            if len(series) >= 6:
                prev5 = float(series.iloc[-6])
                chg_bp = round((us10y - prev5) * 100, 1)  # bp
        direction = "BEARISH" if (chg_bp and chg_bp > 20) else ("BULLISH" if (chg_bp and chg_bp < -20) else "NEUTRAL")
        signals.append(IntlSignal(
            name="美债10Y收益率", value=round(us10y, 3), unit="%",
            change=chg_bp, direction=direction,
            a_share_impact=_l2_impact("美债10Y收益率", direction),
            source="东方财富-中美国债收益率"
        ))

    # ═══════ 4. 美债2Y收益率（收益率曲线倒挂预警） ═══════
    if us2y is not None and us_spread is not None:
        chg_bp = None
        if bond_df is not None and len(bond_df) >= 6:
            series = bond_df['美国国债收益率2年'].dropna()
            if len(series) >= 6:
                prev5 = float(series.iloc[-6])
                chg_bp = round((us2y - prev5) * 100, 1)
        direction = "BEARISH" if (chg_bp and chg_bp > 20) else ("BULLISH" if (chg_bp and chg_bp < -20) else "NEUTRAL")
        curve_note = "2Y>10Y倒挂→衰退预警" if us_spread < 0 else "曲线正常"
        signals.append(IntlSignal(
            name="美债2Y收益率", value=round(us2y, 3), unit="%",
            change=chg_bp, direction=direction,
            a_share_impact=_l2_impact("美债2Y收益率", direction) + f"；{curve_note}",
            source="东方财富-中美国债收益率"
        ))

    # ═══════ 5. VIX 恐慌指数（用沪深300 QVIX 代理） ═══════
    try:
        df = ak.index_option_300etf_qvix()
        if df is not None and len(df) > 1:
            df.columns = [c.strip() for c in df.columns]
            val_col = [c for c in df.columns if 'QVIX' in c.upper() or '指数' in c or '最新' in c]
            if not val_col:
                val_col = [df.columns[-1]] if len(df.columns) > 0 else []
            if val_col:
                df[val_col[0]] = pd.to_numeric(df[val_col[0]], errors="coerce")
                series = df[val_col[0]].dropna()
                latest = float(series.iloc[-1])
                # QVIX 量化标准（参考 VIX 但阈值略低）
                if latest > 20:
                    direction = "BEARISH"
                elif latest > 15:
                    direction = "BEARISH"
                elif latest < 12:
                    direction = "BULLISH"
                else:
                    direction = "NEUTRAL"
                signals.append(IntlSignal(
                    name="沪深300 QVIX(恐慌代理)", value=round(latest, 2), unit="",
                    change=None, direction=direction,
                    a_share_impact=_l2_impact("沪深300 QVIX(恐慌代理)", direction),
                    source="东方财富期权"
                ))
    except Exception:
        pass

    # ═══════ 6. 中国10Y国债收益率 ═══════
    if cn10y is not None:
        chg_bp = None
        if bond_df is not None and len(bond_df) >= 6:
            series = bond_df['中国国债收益率10年'].dropna()
            if len(series) >= 6:
                prev5 = float(series.iloc[-6])
                chg_bp = round((cn10y - prev5) * 100, 1)
        direction = "BEARISH" if (chg_bp and chg_bp > 10) else ("BULLISH" if (chg_bp and chg_bp < -10) else "NEUTRAL")
        signals.append(IntlSignal(
            name="中国10Y国债收益率", value=round(cn10y, 3), unit="%",
            change=chg_bp, direction=direction,
            a_share_impact=_l2_impact("中国10Y国债收益率", direction),
            source="中国债券网"
        ))

    # ═══════ 7. 中美10Y利差 ═══════
    if us10y is not None and cn10y is not None:
        spread = round((cn10y - us10y) * 100, 0)  # bp
        if spread < -100:
            direction = "BEARISH"
        elif spread < -50:
            direction = "NEUTRAL"
        elif spread > 0:
            direction = "BULLISH"
        else:
            direction = "NEUTRAL"
        signals.append(IntlSignal(
            name="中美10Y利差", value=int(spread), unit="bp",
            change=None, direction=direction,
            a_share_impact=_l2_impact("中美10Y利差", direction),
            source="东方财富-中美国债收益率"
        ))

    # ═══════ 8. 美国ISM制造业PMI（全球需求代理） ═══════
    try:
        df = ak.macro_usa_ism_pmi()
        if df is not None and len(df) > 1:
            df.columns = [c.strip() for c in df.columns]
            val_col = [c for c in df.columns if 'PMI' in c.upper() or '制造业' in c]
            if not val_col:
                val_col = [df.columns[1]] if len(df.columns) > 1 else [df.columns[0]]
            df[val_col[0]] = pd.to_numeric(df[val_col[0]], errors="coerce")
            series = df[val_col[0]].dropna()
            latest = float(series.iloc[-1])
            direction = "BULLISH" if latest > 52 else ("BEARISH" if latest < 48 else "NEUTRAL")
            signals.append(IntlSignal(
                name="美国ISM制造业PMI", value=round(latest, 1), unit="",
                change=None, direction=direction,
                a_share_impact=_l2_impact("美国ISM制造业PMI", direction),
                source="ISM/akshare"
            ))
    except Exception:
        pass

    return signals


def get_domestic_futures() -> list:
    """国内股指期货基差——读懂机构判断"""
    signals = []

    # IF/IC/IM 主力合约日线（新浪数据）
    fut_map = {
        "IF（沪深300）": ("sh000300", "IF0"),
        "IC（中证500）": ("sh000905", "IC0"),
        "IM（中证1000）": ("sh000852", "IM0"),
    }

    for name, (spot_symbol, fut_symbol) in fut_map.items():
        try:
            # 现货指数
            df_spot = ak.stock_zh_index_daily(symbol=spot_symbol)
            df_spot["close"] = pd.to_numeric(df_spot["close"], errors="coerce")
            spot = float(df_spot["close"].dropna().iloc[-1])

            # 期货主力合约（新浪）
            df_fut = ak.futures_main_sina(symbol=fut_symbol)
            if df_fut is not None and len(df_fut) > 0:
                df_fut["收盘价"] = pd.to_numeric(df_fut["收盘价"], errors="coerce")
                fut_price = float(df_fut["收盘价"].dropna().iloc[-1])
                if pd.notna(fut_price) and spot > 0:
                    basis = round(fut_price - spot, 1)
                    pct   = round(basis / spot * 100, 2)
                    direction = "BEARISH" if basis < -30 else ("BULLISH" if basis > 20 else "NEUTRAL")
                    signals.append(IntlSignal(
                        name=name,
                        value=basis, unit="点",
                        change=pct,
                        direction=direction,
                        a_share_impact=_l2_impact(name, direction),
                        source="新浪期货"
                    ))
        except Exception:
            continue

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# 银行 / 信用信号 —— 商业银行业务能力 + 央行政策利率 + 银行板块走势
# ══════════════════════════════════════════════════════════════════════════════

# 银行信号深层影响模板
_L2_BANK_IMPACT = {
    "LPR 1年期": {
        "BULLISH": ("LPR下调→企业/个人贷款利率降低→刺激实体经济融资需求→利好经济复苏。"
                    "但银行净息差被压缩，短期利空银行利润（但放贷量增加可弥补）。"
                    "利好：负债率高的企业、房地产、基建。"
                    "关注：LPR连续下调2次以上→宽松周期确认→利好A股整体估值"),
        "BEARISH": ("LPR上调→融资成本上升→企业扩张意愿下降→利空实体经济。"
                    "但银行净息差扩大，利好银行利润。"
                    "如果LPR上调+经济数据走弱=滞胀风险，需警惕"),
        "NEUTRAL": ("LPR持平，货币政策稳定期"),
    },
    "LPR 5年期": {
        "BULLISH": ("5年期LPR挂钩房贷利率，下调直接利好房地产市场和居民消费。"
                    "降低居民月供负担→释放消费能力→利好消费链。"
                    "利好：房地产链、家电（海尔智家）、建材。"
                    "如果5Y LPR单独下调（1Y不动）=央行定向支持房地产"),
        "BEARISH": ("5年期LPR上调→房贷成本上升→房地产销售承压→利空地产链。"
                    "但反映央行认为经济过热需要降温"),
        "NEUTRAL": ("5年期LPR稳定，房地产融资成本平稳"),
    },
    "Shibor隔夜": {
        "BULLISH": ("Shibor低位→银行间流动性充裕→资金成本低→有利于银行同业业务和债券市场。"
                    "隔夜Shibor<1.5%=流动性极度宽松，利于股市上涨。"
                    "关注：如果Shibor极低+股市不涨=资金在金融体系空转，实体经济未受益"),
        "BEARISH": ("Shibor飙升→银行间缺钱→流动性紧张→可能传导到股市（资金抽离）。"
                    "隔夜Shibor>3%=极端紧张，历史上往往伴随A股调整。"
                    "原因：季末/年末考核、央行回笼资金、缴税期"),
        "NEUTRAL": ("Shibor正常区间，银行间流动性平稳"),
    },
    "银行板块指数": {
        "BULLISH": ("银行股走强通常意味着：①国家队/险资/社保等长线资金在护盘或建仓"
                    "②银行基本面改善（净息差企稳/资产质量好转）"
                    "③估值修复（银行PB普遍<1，安全边际高）。"
                    "银行占A股总市值~25%，银行走强→大盘底部信号。"
                    "关注：如果银行持续走强但中小盘走弱=存量博弈，大盘蓝筹风格"),
        "BEARISH": ("银行股走弱→可能是：①经济衰退预期增强（坏账率上升担忧）"
                    "②利率下行压缩净息差→银行盈利能力下降"
                    "③政策收紧（监管趋严）。"
                    "银行走弱+大盘走弱=系统性风险信号，需降低仓位"),
        "NEUTRAL": ("银行板块平稳，无明显方向"),
    },
    "社会融资规模": {
        "BULLISH": ("社融大幅超预期→信用扩张→6-12个月后经济回暖。"
                    "社融是经济的领先指标（领先GDP约2个季度）。"
                    "新增信贷>3万亿=信用脉冲强劲→历史上社融脉冲与沪深300高度正相关。"
                    "关注：社融结构（如果主要靠政府债券→政府托底；如果企业中长期贷款多→经济自发复苏）"),
        "BEARISH": ("社融大幅低于预期→信用收缩→经济下行压力加大。"
                    "连续3个月社融走弱=信用紧缩周期，A股大概率承压。"
                    "关注：M1-M2剪刀差是否扩大（如果扩大=资金在金融体系空转）"),
        "NEUTRAL": ("社融数据正常，信用环境平稳"),
    },
    "新增人民币贷款": {
        "BULLISH": ("新增贷款超预期→银行放贷积极→实体经济融资需求旺盛。"
                    "居民中长期贷款增多→房地产回暖信号。"
                    "企业中长期贷款增多→企业扩大投资→经济复苏预期增强。"
                    "贷款超预期+社融超预期=信用扩张确认，强烈利好A股"),
        "BEARISH": ("新增贷款低于预期→银行'资产荒'或企业不愿贷款→经济内生动力不足。"
                    "如果是居民中长期贷款少→房地产低迷。"
                    "如果是企业贷款少→企业对未来缺乏信心→经济底部可能尚未到来"),
        "NEUTRAL": ("新增贷款数据正常"),
    },
}


def _bank_impact(name: str, direction: str) -> str:
    templates = _L2_BANK_IMPACT.get(name, {})
    return templates.get(direction, "")


def get_bank_signals() -> list:
    """获取银行/信用相关信号"""
    signals = []

    # ═══════ 1. LPR 报价 ═══════
    try:
        df = ak.macro_china_lpr()
        if df is not None and len(df) >= 2:
            df.columns = [c.strip() for c in df.columns]
            # 找到 LPR 1Y 和 5Y 列
            lpr_1y_col = [c for c in df.columns if "1年" in c or "1Y" in c.upper()]
            lpr_5y_col = [c for c in df.columns if "5年" in c or "5Y" in c.upper()]
            if lpr_1y_col:
                series = pd.to_numeric(df[lpr_1y_col[0]], errors="coerce").dropna()
                if len(series) >= 2:
                    latest = float(series.iloc[-1])
                    prev = float(series.iloc[-2])
                    if latest < prev:
                        signals.append(IntlSignal(
                            name="LPR 1年期", value=latest, unit="%",
                            change=round(latest - prev, 3), direction="BULLISH",
                            a_share_impact=_bank_impact("LPR 1年期", "BULLISH"),
                            source="中国货币网"
                        ))
                    elif latest > prev:
                        signals.append(IntlSignal(
                            name="LPR 1年期", value=latest, unit="%",
                            change=round(latest - prev, 3), direction="BEARISH",
                            a_share_impact=_bank_impact("LPR 1年期", "BEARISH"),
                            source="中国货币网"
                        ))
                    else:
                        signals.append(IntlSignal(
                            name="LPR 1年期", value=latest, unit="%",
                            change=0, direction="NEUTRAL",
                            a_share_impact=_bank_impact("LPR 1年期", "NEUTRAL"),
                            source="中国货币网"
                        ))
            if lpr_5y_col:
                series = pd.to_numeric(df[lpr_5y_col[0]], errors="coerce").dropna()
                if len(series) >= 2:
                    latest = float(series.iloc[-1])
                    prev = float(series.iloc[-2])
                    if latest < prev:
                        signals.append(IntlSignal(
                            name="LPR 5年期", value=latest, unit="%",
                            change=round(latest - prev, 3), direction="BULLISH",
                            a_share_impact=_bank_impact("LPR 5年期", "BULLISH"),
                            source="中国货币网"
                        ))
                    elif latest > prev:
                        signals.append(IntlSignal(
                            name="LPR 5年期", value=latest, unit="%",
                            change=round(latest - prev, 3), direction="BEARISH",
                            a_share_impact=_bank_impact("LPR 5年期", "BEARISH"),
                            source="中国货币网"
                        ))
                    else:
                        signals.append(IntlSignal(
                            name="LPR 5年期", value=latest, unit="%",
                            change=0, direction="NEUTRAL",
                            a_share_impact=_bank_impact("LPR 5年期", "NEUTRAL"),
                            source="中国货币网"
                        ))
    except Exception:
        pass

    # ═══════ 2. Shibor 隔夜利率 ═══════
    try:
        df = ak.rate_interbank(market="上海银行间同业拆放利率", symbol="隔夜",
                               start_date="", end_date="")
        if df is not None and len(df) >= 5:
            df.columns = [c.strip() for c in df.columns]
            val_col = [c for c in df.columns if '利率' in c or '值' in c
                       or 'Shibor' in c or 'LPR' not in c]
            if not val_col:
                val_col = [df.columns[-1]]
            series = pd.to_numeric(df[val_col[0]], errors="coerce").dropna()
            if len(series) >= 5:
                latest = float(series.iloc[-1])
                avg5 = float(series.tail(6).iloc[:-1].mean())
                direction = "BULLISH" if latest < 1.5 else (
                    "BEARISH" if latest > 2.5 else "NEUTRAL")
                signals.append(IntlSignal(
                    name="Shibor隔夜", value=round(latest, 3), unit="%",
                    change=round(latest - avg5, 3), direction=direction,
                    a_share_impact=_bank_impact("Shibor隔夜", direction),
                    source="上海银行间同业拆借中心"
                ))
    except Exception:
        pass

    # ═══════ 3. 银行板块指数走势 ═══════
    try:
        df = ak.stock_board_industry_hist_em(symbol="银行", period="daily",
                                              start_date="20250101", adjust="qfq")
        if df is not None and len(df) >= 10:
            df.columns = [c.strip() for c in df.columns]
            close_col = [c for c in df.columns if '收盘' in c]
            if not close_col:
                close_col = [df.columns[1]]
            df[close_col[0]] = pd.to_numeric(df[close_col[0]], errors="coerce")
            series = df[close_col[0]].dropna()
            if len(series) >= 10:
                latest = float(series.iloc[-1])
                ma20 = float(series.tail(20).mean()) if len(series) >= 20 else float(series.mean())
                chg_pct = (latest - ma20) / ma20 * 100
                direction = "BULLISH" if chg_pct > 3 else (
                    "BEARISH" if chg_pct < -3 else "NEUTRAL")
                signals.append(IntlSignal(
                    name="银行板块指数", value=round(latest, 2), unit="点",
                    change=round(chg_pct, 2), direction=direction,
                    a_share_impact=_bank_impact("银行板块指数", direction),
                    source="东方财富"
                ))
    except Exception:
        pass

    # ═══════ 4. 社会融资规模（最新月度） ═══════
    try:
        df = ak.macro_china_shrzgm()
        if df is not None and len(df) >= 2:
            df.columns = [c.strip() for c in df.columns]
            # 尝试找到数值列
            val_cols = [c for c in df.columns if '社会' in c or '规模' in c
                        or '增量' in c or '亿元' in c or df.columns[-1]]
            if val_cols:
                series = pd.to_numeric(df[val_cols[0]], errors="coerce").dropna()
                if len(series) >= 2:
                    latest = float(series.iloc[-1])
                    prev = float(series.iloc[-2])
                    yoy_chg = (latest - prev) / abs(prev) * 100 if prev != 0 else 0
                    direction = "BULLISH" if yoy_chg > 10 else (
                        "BEARISH" if yoy_chg < -10 else "NEUTRAL")
                    signals.append(IntlSignal(
                        name="社会融资规模", value=round(latest / 10000, 2),
                        unit="万亿",
                        change=round(yoy_chg, 1), direction=direction,
                        a_share_impact=_bank_impact("社会融资规模", direction),
                        source="央行/akshare"
                    ))
    except Exception:
        pass

    # ═══════ 5. 新增人民币贷款 ═══════
    try:
        df = ak.macro_china_new_financial_credit()
        if df is not None and len(df) >= 2:
            df.columns = [c.strip() for c in df.columns]
            val_cols = [c for c in df.columns if '人民币' in c or '贷款' in c
                        or '亿元' in c or df.columns[-1]]
            if val_cols:
                series = pd.to_numeric(df[val_cols[0]], errors="coerce").dropna()
                if len(series) >= 2:
                    latest = float(series.iloc[-1])
                    prev = float(series.iloc[-2])
                    yoy_chg = (latest - prev) / abs(prev) * 100 if prev != 0 else 0
                    direction = "BULLISH" if yoy_chg > 15 else (
                        "BEARISH" if yoy_chg < -15 else "NEUTRAL")
                    signals.append(IntlSignal(
                        name="新增人民币贷款", value=round(latest / 10000, 2),
                        unit="万亿",
                        change=round(yoy_chg, 1), direction=direction,
                        a_share_impact=_bank_impact("新增人民币贷款", direction),
                        source="央行/akshare"
                    ))
    except Exception:
        pass

    return signals


def cross_check_l1_l2(l1_score: int, l2_signals: list) -> list:
    """Layer 1 与 Layer 2 背离检测"""
    divergence = []
    bullish  = sum(1 for s in l2_signals if s.direction == "BULLISH")
    bearish  = sum(1 for s in l2_signals if s.direction == "BEARISH")
    l2_score = int(50 + (bullish - bearish) / max(len(l2_signals), 1) * 50)

    diff = l1_score - l2_score
    if diff > 25:
        divergence.append(
            f"⚠ 背离：L1实业偏强({l1_score})但L2金融偏弱({l2_score})，"
            "资金可能掌握更多信息，降仓50%等待收敛"
        )
    elif diff < -25:
        divergence.append(
            f"⚠ 背离：L2金融信号偏强({l2_score})但L1实业偏弱({l1_score})，"
            "金融先行于实业，关注后续实业数据是否跟上"
        )
    else:
        divergence.append(
            f"✓ L1({l1_score}) 与 L2({l2_score}) 方向基本一致，信号可信度高"
        )
    return divergence


def run_layer2(l1_score: int = 50) -> Layer2Result:
    result = Layer2Result(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
    result.intl    = get_international_signals()
    result.futures = get_domestic_futures()
    result.bank    = get_bank_signals()

    all_sig = result.intl + result.futures + result.bank
    bullish = sum(1 for s in all_sig if s.direction == "BULLISH")
    bearish = sum(1 for s in all_sig if s.direction == "BEARISH")
    result.score = int(50 + (bullish - bearish) / max(len(all_sig), 1) * 50)

    result.divergence = cross_check_l1_l2(l1_score, all_sig)
    return result
