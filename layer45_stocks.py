"""
Layer 4 — 头部企业筛选（LLM分析产业链 + 自动化财务评分）
Layer 5 — 个股最终确认 + 仓位计算（A股手数）
"""
import akshare as ak
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import warnings, re, json
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 产业链 → akshare 概念板块映射（用于动态抓取成分股）
# ══════════════════════════════════════════════════════════════════════════════

CHAIN_CONCEPT_MAP = {
    "AI算力链":           ["光模块", "算力概念", "CPO概念", "液冷服务器"],
    "半导体材料链":       ["半导体", "芯片概念", "半导体材料", "靶材"],
    "新能源车链":         ["新能源汽车", "锂电池", "固态电池", "新能源车"],
    "工业自动化链":       ["工业母机", "机器人概念", "人形机器人", "减速器"],
    "航空航天材料链":     ["航空发动机", "大飞机", "商业航天", "碳纤维"],
    "新能源材料链（SiC）":["碳化硅", "第三代半导体"],
    "铜/铝/有色金属链":   ["有色金属", "铜", "铝", "黄金概念"],
    "出口制造链":         ["跨境电商", "出海概念"],
    "生猪养殖链":         ["猪肉概念", "养殖业", "鸡肉概念"],
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompanyProfile:
    code: str
    name: str
    chain: str
    node: str
    # 四道过滤器得分 (各25分，满分100)
    revenue_score: int    # 收入结构
    client_score: int     # 客户名单
    margin_score: int     # 毛利率趋势
    capex_score: int      # 资本开支方向
    # 附加项
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)
    verdict: str = ""     # REAL / WATCH / FAKE
    total_score: int = 0
    note: str = ""
    # 财务实时数据
    pe: float = 0.0
    pb: float = 0.0
    mktcap: str = ""
    gross_margin: float = 0.0
    net_margin: float = 0.0
    roe: float = 0.0
    rev_yoy: float = 0.0
    profit_yoy: float = 0.0
    net_profit_deducted_yoy: float = 0.0
    debt_ratio: float = 0.0
    cash_flow_ratio: float = 0.0
    data_source: str = "static"
    # 深度财务指标
    operating_cash_flow: str = ""
    cash_to_profit_ratio: float = 0.0   # 经营现金流/净利润
    rd_ratio: float = 0.0               # 研发费用率
    gross_margin_trend: str = ""         # 毛利率趋势
    financial_quality: str = ""          # 财务质量一句话评估
    # 技术面（由L5填充）
    tech_summary: str = ""


# 静态知识库：链/节点/静态 green/red flags 保留（结构性信息不依赖实时数据）
COMPANY_DB: Dict[str, dict] = {
    "300308": {"name": "中际旭创", "chain": "AI算力链", "node": "光模块",
               "green": ["光模块收入占比>85%", "直供微软/谷歌/亚马逊", "800G/1.6T产能提前布局"],
               "red": []},
    "300394": {"name": "天孚通信", "chain": "AI算力链", "node": "光器件",
               "green": ["无源光器件全球份额第一梯队", "直供英伟达/博通", "CPO共封装光学卡位"],
               "red": []},
    "002837": {"name": "英维克", "chain": "AI算力链", "node": "液冷散热",
               "green": ["数据中心热管理国内市占率第一", "三大运营商+头部互联网客户", "液冷收入占比快速提升"],
               "red": ["毛利率受竞争压制需持续观察"]},
    "300750": {"name": "宁德时代", "chain": "新能源车链", "node": "电芯",
               "green": ["全球动力电池市占率第一35%+", "客户：特斯拉/宝马/大众", "固态电池提前布局"],
               "red": []},
    "002812": {"name": "恩捷股份", "chain": "新能源车链", "node": "隔膜",
               "green": ["湿法隔膜全球市占率第一约40%", "客户：宁德/比亚迪/松下", "海外产能布局领先"],
               "red": ["毛利率受产能过剩压制，处于低谷期"]},
    "300124": {"name": "汇川技术", "chain": "工业自动化链", "node": "工控综合",
               "green": ["变频器/伺服国内市占率第一超越西门子", "覆盖新能源/纺织/机床全行业", "人形机器人核心零部件提前布局"],
               "red": []},
    "688017": {"name": "绿的谐波", "chain": "工业自动化链", "node": "谐波减速器",
               "green": ["谐波减速器国内唯一规模化", "毛利率55%+技术壁垒极强", "人形机器人关节核心零部件"],
               "red": ["规模偏小，客户集中度较高"]},
    "300666": {"name": "江丰电子", "chain": "半导体材料链", "node": "溅射靶材",
               "green": ["高纯靶材国内市占率第一", "客户：中芯/长江存储/华虹", "持续扩产Capex领先"],
               "red": []},
    "688019": {"name": "安集科技", "chain": "半导体材料链", "node": "CMP抛光液",
               "green": ["CMP抛光液打破日美垄断国内唯一量产", "中芯国际28nm以下验证通过", "毛利率60%+极高壁垒"],
               "red": ["规模偏小，扩产能力是关键变量"]},
    "688122": {"name": "西部超导", "chain": "航空航天材料链", "node": "钛合金",
               "green": ["航空钛合金棒丝材国内唯一军品认证", "客户：航空发动机集团/中航工业", "毛利率45%+军品定价保障稳定"],
               "red": []},
    "688295": {"name": "中复神鹰", "chain": "航空航天材料链", "node": "碳纤维",
               "green": ["T700/T800碳纤维量产达到航空级", "干喷湿纺技术全球领先成本最低"],
               "red": ["碳纤维价格下行周期短期承压"]},
    "688234": {"name": "天岳先进", "chain": "新能源材料链（SiC）", "node": "SiC衬底",
               "green": ["6英寸SiC衬底国内市占率第一", "客户：英飞凌/意法半导体海外验证", "8英寸产能布局领先"],
               "red": ["SiC价格下行毛利率承压"]},
    "603290": {"name": "斯达半导", "chain": "新能源材料链（SiC）", "node": "SiC器件",
               "green": ["IGBT国内第一，SiC模块快速放量", "客户：比亚迪/上汽/海外车厂", "毛利率38%高于行业均值"],
               "red": ["英飞凌高端市场竞争激烈"]},
    "600690": {"name": "海尔智家", "chain": "出口制造链", "node": "家电出口",
               "green": ["全球白电市占率第一海外收入>50%", "GE Appliances并购成功", "海外自有品牌溢价不靠低价"],
               "red": []},
    "300866": {"name": "安克创新", "chain": "出口制造链", "node": "消费电子出口",
               "green": ["亚马逊充电品类第一自有品牌溢价", "海外收入>90%", "毛利率40%+远超代工模式", "研发投入持续提升"],
               "red": []},
}


def _parse_pct(val) -> Optional[float]:
    """解析百分比字符串或数值为 float"""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace('%', '').replace(',', '')
    if s.lower() == 'false' or s == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_financial_data(code: str) -> dict:
    """从 akshare 拉取财务数据，返回标准化字典"""
    result = {
        "pe": None, "pb": None, "mktcap": "",
        "gross_margin": None, "net_margin": None, "roe": None,
        "rev_yoy": None, "profit_yoy": None, "net_profit_deducted_yoy": None,
        "debt_ratio": None, "cash_flow_ratio": None,
        "eps": None, "net_profit": "", "rev": "",
    }
    # 1. 基本信息（PE/PB/市值/价格）
    try:
        df = ak.stock_individual_info_em(symbol=code)
        if df is not None and len(df) > 0:
            df = df.set_index("item")
            if "市盈率-动态" in df.index:
                result["pe"] = _parse_pct(df.loc["市盈率-动态", "value"])
            if "市净率" in df.index:
                result["pb"] = _parse_pct(df.loc["市净率", "value"])
            if "总市值" in df.index:
                mktcap = df.loc["总市值", "value"]
                if isinstance(mktcap, (int, float)) and mktcap > 1e8:
                    result["mktcap"] = f"{mktcap/1e8:.0f}亿"
                else:
                    result["mktcap"] = str(mktcap)
    except Exception:
        pass

    # 2. 同花顺财务摘要（净利润/营收/毛利率/增长率/现金流等）
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if df is not None and len(df) > 0:
            # 取最近一个年报或最新报告期
            df = df.head(1).iloc[0]

            result["gross_margin"] = _parse_pct(df.get("销售毛利率"))
            result["net_margin"] = _parse_pct(df.get("销售净利率"))
            result["roe"] = _parse_pct(df.get("净资产收益率"))
            result["rev_yoy"] = _parse_pct(df.get("营业总收入同比增长率"))
            result["profit_yoy"] = _parse_pct(df.get("净利润同比增长率"))
            result["net_profit_deducted_yoy"] = _parse_pct(df.get("扣非净利润同比增长率"))
            result["debt_ratio"] = _parse_pct(df.get("资产负债率"))
            result["eps"] = _parse_pct(df.get("基本每股收益"))
            result["net_profit"] = str(df.get("净利润", ""))
            result["rev"] = str(df.get("营业总收入", ""))
    except Exception:
        pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 技术指标计算：DPO / MACD / RSI / BOLL / KDJ
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# K线数据缓存（避免同一只股票重复请求，东方财富接口不稳定）
# ══════════════════════════════════════════════════════════════════════════════

_kline_cache: Dict[str, pd.DataFrame] = {}

def _get_kline_cached(code: str) -> Optional[pd.DataFrame]:
    """获取个股日线行情（东方财富→腾讯备用，带缓存+重试）"""
    import time
    if code in _kline_cache:
        return _kline_cache[code]

    # 方案A: 东方财富（列名为中文：收盘/最高/最低/成交量）
    df = None
    for attempt in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date="20240101", adjust="qfq")
            if df is not None and len(df) > 0:
                break
        except Exception:
            if attempt < 1:
                time.sleep(2)

    # 方案B: 腾讯数据源（列名为英文：close/high/low/volume）
    if df is None or len(df) == 0:
        try:
            prefix = "sh" if code.startswith("6") else "sz"
            tx_code = prefix + code
            df = ak.stock_zh_a_hist_tx(symbol=tx_code)
            if df is not None and len(df) > 0:
                # 统一列名为中文，与东方财富一致
                col_map = {"date": "日期", "open": "开盘", "close": "收盘",
                           "high": "最高", "low": "最低", "volume": "成交量",
                           "amount": "成交额"}
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                # 腾讯无 volume 列时，用成交额近似替代（相对比例不受影响）
                if "成交量" not in df.columns and "成交额" in df.columns:
                    df["成交量"] = df["成交额"]
                # 腾讯数据无前复权，用全部数据（已足够计算技术指标）
        except Exception:
            pass

    if df is not None and len(df) > 0:
        _kline_cache[code] = df
    return df


def compute_technical_indicators(code: str) -> dict:
    """
    从日线行情数据计算5大技术指标，给出综合判断。
    返回 { overall, summary, dpo, macd_*, rsi, boll_*, kdj_* }
    """
    df = _get_kline_cached(code)
    if df is None or len(df) < 60:
        return {"overall": "NONE",
                "summary": "数据不足60日，无法计算完整技术指标"}

    try:
        df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
        df["最高"] = pd.to_numeric(df["最高"], errors="coerce")
        df["最低"] = pd.to_numeric(df["最低"], errors="coerce")
        df = df.dropna(subset=["收盘", "最高", "最低"]).reset_index(drop=True)
        close = df["收盘"]

        # ── DPO(20)：去趋势价格振荡器 ──
        n_dpo = 20
        sma20 = close.rolling(n_dpo).mean()
        dpo = close - sma20.shift(int(n_dpo / 2) + 1)
        dpo_val = float(dpo.iloc[-1])

        # ── MACD(12,26,9) ──
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = 2 * (dif - dea)
        dif_v, dea_v, hist_v = float(dif.iloc[-1]), float(dea.iloc[-1]), float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-2]) if len(hist) >= 2 else 0

        # ── RSI(14) ──
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_v = float(rsi.iloc[-1])

        # ── BOLL(20,2) ──
        boll_mid = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        boll_up = boll_mid + 2 * boll_std
        boll_dn = boll_mid - 2 * boll_std
        price = float(close.iloc[-1])
        boll_up_v = float(boll_up.iloc[-1])
        boll_mid_v = float(boll_mid.iloc[-1])
        boll_dn_v = float(boll_dn.iloc[-1])

        # ── KDJ(9,3,3) ──
        low9 = df["最低"].rolling(9).min()
        high9 = df["最高"].rolling(9).max()
        rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        k_v, d_v, j_v = float(k.iloc[-1]), float(d.iloc[-1]), float(j.iloc[-1])
        k_prev, d_prev = float(k.iloc[-2]), float(d.iloc[-2])

        # ── 信号判断 ──
        tags = []

        # DPO
        tags.append("多" if dpo_val > 0 else "空")

        # MACD
        if dif_v > dea_v and hist_v > 0 and hist_prev <= 0:
            macd_tag, macd_desc = "多", "MACD金叉"
        elif dif_v < dea_v and hist_v < 0 and hist_prev >= 0:
            macd_tag, macd_desc = "空", "MACD死叉"
        elif dif_v > dea_v:
            macd_tag, macd_desc = "多", "MACD多头排列"
        else:
            macd_tag, macd_desc = "空", "MACD空头排列"
        tags.append(macd_tag)

        # RSI
        if rsi_v > 80:
            rsi_tag, rsi_desc = "空", f"RSI={rsi_v:.0f}严重超买"
        elif rsi_v > 70:
            rsi_tag, rsi_desc = "空", f"RSI={rsi_v:.0f}超买"
        elif rsi_v < 20:
            rsi_tag, rsi_desc = "多", f"RSI={rsi_v:.0f}严重超卖"
        elif rsi_v < 30:
            rsi_tag, rsi_desc = "多", f"RSI={rsi_v:.0f}超卖"
        else:
            rsi_tag, rsi_desc = "中", f"RSI={rsi_v:.0f}中性"
        tags.append(rsi_tag)

        # BOLL
        if price >= boll_up_v:
            boll_tag, boll_desc = "空", "价格触及布林上轨"
        elif price <= boll_dn_v:
            boll_tag, boll_desc = "多", "价格触及布林下轨"
        else:
            boll_tag, boll_desc = "中", "价格在布林通道内"
        tags.append(boll_tag)

        # KDJ
        if k_v > d_v and k_prev <= d_prev:
            kdj_tag, kdj_desc = "多", "KDJ金叉"
        elif k_v < d_v and k_prev >= d_prev:
            kdj_tag, kdj_desc = "空", "KDJ死叉"
        elif k_v > 80 and d_v > 80:
            kdj_tag, kdj_desc = "空", "KDJ超买区"
        elif k_v < 20 and d_v < 20:
            kdj_tag, kdj_desc = "多", "KDJ超卖区"
        else:
            kdj_tag, kdj_desc = "中", "KDJ中性"
        tags.append(kdj_tag)

        bull = tags.count("多")
        bear = tags.count("空")
        neut = tags.count("中")

        if bull >= 3:
            overall, overall_text = "BULLISH", "技术面偏多"
        elif bear >= 3:
            overall, overall_text = "BEARISH", "技术面偏空"
        else:
            overall, overall_text = "NEUTRAL", "技术面中性"

        signals_txt = " | ".join([
            f"DPO={'偏多' if tags[0]=='多' else '偏空'}({dpo_val:.2f})",
            macd_desc, rsi_desc, boll_desc, kdj_desc,
        ])
        summary = f"{signals_txt}。{overall_text}（多{bull}/空{bear}/中{neut}）"

        return {
            "overall": overall, "summary": summary,
            "dpo": round(dpo_val, 2),
            "macd_dif": round(dif_v, 3), "macd_dea": round(dea_v, 3),
            "macd_hist": round(hist_v, 3), "macd_signal": macd_tag,
            "rsi": round(rsi_v, 1), "rsi_signal": rsi_tag,
            "boll_upper": round(boll_up_v, 2), "boll_mid": round(boll_mid_v, 2),
            "boll_lower": round(boll_dn_v, 2), "boll_signal": boll_tag,
            "kdj_k": round(k_v, 1), "kdj_d": round(d_v, 1),
            "kdj_j": round(j_v, 1), "kdj_signal": kdj_tag,
        }
    except Exception as e:
        return {"overall": "NONE", "summary": "技术指标计算异常（数据不足或网络问题）"}


# ══════════════════════════════════════════════════════════════════════════════
# 深度财务分析：现金流质量 / 研发投入 / 毛利率趋势
# ══════════════════════════════════════════════════════════════════════════════

def fetch_deep_financials(code: str) -> dict:
    """
    获取深度财务数据：现金流质量、研发费用率、毛利率趋势。
    返回 { cash_to_profit, rd_ratio, gm_trend, quality_note }
    """
    info = {"cash_to_profit": None, "rd_ratio": None,
            "gm_trend": "", "quality_note": ""}

    # ── 1. 同花顺财务摘要 → 多期毛利率趋势 ──
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if df is not None and len(df) >= 3:
            df.columns = [c.strip() for c in df.columns]
            gm_col = [c for c in df.columns if "毛利率" in c]
            if gm_col:
                gms = pd.to_numeric(df[gm_col[0]], errors="coerce").dropna()
                if len(gms) >= 3:
                    first, last = float(gms.iloc[-1]), float(gms.iloc[0])
                    if first > last:
                        info["gm_trend"] = "毛利率上升趋势↑"
                    elif first < last:
                        info["gm_trend"] = "毛利率下滑趋势↓"
                    else:
                        info["gm_trend"] = "毛利率持平→"
    except Exception:
        pass

    # ── 2. 东方财富利润表 → 净利润 / 研发费用 / 营收 ──
    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=code)
        if df is not None and len(df) >= 2:
            df.columns = [c.strip() for c in df.columns]
            np_cols = [c for c in df.columns if "净利润" in c and "合计" in c]
            rd_cols = [c for c in df.columns if "研发费用" in c]
            rev_cols = [c for c in df.columns if "营业总收入" in c]
            np_val = float(pd.to_numeric(df[np_cols[0]].iloc[0], errors="coerce")) if np_cols else None
            rd_val = float(pd.to_numeric(df[rd_cols[0]].iloc[0], errors="coerce")) if rd_cols else None
            rev_val = float(pd.to_numeric(df[rev_cols[0]].iloc[0], errors="coerce")) if rev_cols else None
            if rd_val and rev_val and rev_val != 0:
                info["rd_ratio"] = round(rd_val / rev_val * 100, 1)
            if np_val and np_val != 0:
                info["_np_val"] = np_val
            if rev_val and rev_val != 0:
                info["_rev_val"] = rev_val
    except Exception:
        pass

    # ── 3. 东方财富现金流量表 → 经营现金流 ──
    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=code)
        if df is not None and len(df) >= 2:
            df.columns = [c.strip() for c in df.columns]
            cf_cols = [c for c in df.columns
                       if "经营活动" in c and "现金流量" in c and "净额" in c]
            if cf_cols:
                cf_val = float(pd.to_numeric(df[cf_cols[0]].iloc[0], errors="coerce"))
                if pd.notna(cf_val):
                    info["_cf_val"] = cf_val
    except Exception:
        pass

    # ── 4. 汇总 ──
    np_val = info.pop("_np_val", None)
    cf_val = info.pop("_cf_val", None)
    if np_val and cf_val and np_val != 0:
        ratio = cf_val / np_val
        info["cash_to_profit"] = round(ratio, 2)
        if ratio > 1.2:
            info["quality_note"] = "经营现金流充裕，利润含金量高；"
        elif ratio > 0.8:
            info["quality_note"] = "经营现金流与净利润匹配，利润质量尚可；"
        elif ratio > 0:
            info["quality_note"] = "经营现金流低于净利润，警惕应收/存货占用；"
        else:
            info["quality_note"] = "经营现金流为负，利润可能依赖应收账款，需警惕！"

    rd = info.get("rd_ratio")
    if rd is not None:
        if rd > 10:
            info["quality_note"] += f"研发费用率{rd:.1f}%，创新投入积极。"
        elif rd > 5:
            info["quality_note"] += f"研发费用率{rd:.1f}%，研发投入中等。"
        else:
            info["quality_note"] += f"研发费用率{rd:.1f}%，研发投入偏低。"

    info.pop("_rev_val", None)
    return info


def auto_score_company(code: str, static_info: dict, fin_data: dict) -> CompanyProfile:
    """
    根据实时财务数据自动评分
    四道过滤器：收入结构(25) + 客户名单(25) + 毛利率趋势(25) + 资本开支(25)
    """
    rev_score = 0
    client_score = 0
    margin_score = 0
    capex_score = 0

    gm = fin_data.get("gross_margin")
    nm = fin_data.get("net_margin")
    roe = fin_data.get("roe")
    rev_yoy = fin_data.get("rev_yoy")
    profit_yoy = fin_data.get("profit_yoy")
    deduct_yoy = fin_data.get("net_profit_deducted_yoy")
    debt = fin_data.get("debt_ratio")
    pe = fin_data.get("pe")

    green_flags = list(static_info.get("green", []))
    red_flags = list(static_info.get("red", []))

    # ── 1. 收入结构（25分）──
    if rev_yoy is not None:
        if rev_yoy > 30:
            rev_score = 25
        elif rev_yoy > 15:
            rev_score = 22
        elif rev_yoy > 0:
            rev_score = 18
        elif rev_yoy > -10:
            rev_score = 12
        else:
            rev_score = 6
        green_flags.append(f"营收同比{rev_yoy:+.1f}%")
    else:
        rev_score = 15  # 数据缺失给中间分

    # ── 2. 客户名单（25分）──
    # 客户质量主要来自静态知识库（需要人工/行业知识判断）
    # 用净利润质量和稳定性作为代理
    has_quality_client = bool(static_info.get("green"))
    if has_quality_client:
        client_score = 25
    else:
        client_score = 15

    # 扣非净利润作为客户质量的侧面验证
    if deduct_yoy is not None:
        if deduct_yoy < -20:
            red_flags.append(f"扣非净利润同比{deduct_yoy:.1f}%，利润质量存疑")
            client_score = min(client_score, 18)

    # ── 3. 毛利率趋势（25分）──
    if gm is not None:
        if gm >= 50:
            margin_score = 25
        elif gm >= 35:
            margin_score = 23
        elif gm >= 25:
            margin_score = 20
        elif gm >= 15:
            margin_score = 15
        else:
            margin_score = 8
        green_flags.append(f"毛利率{gm:.1f}%")

    # 净利润增长率验证
    if profit_yoy is not None:
        if profit_yoy < -20:
            red_flags.append(f"净利润同比{profit_yoy:.1f}%，盈利能力下滑")
            margin_score = min(margin_score, 15)

    # ROE 加分
    if roe is not None:
        if roe >= 20:
            margin_score = min(margin_score + 2, 25)
        elif roe >= 15:
            margin_score = min(margin_score + 1, 25)

    # ── 4. 资本开支（25分）──
    # 用资产负债率+营收增长判断资本开支效率
    if debt is not None:
        if debt < 30:
            capex_score = 22
        elif debt < 50:
            capex_score = 20
        elif debt < 70:
            capex_score = 15
        else:
            capex_score = 8
            red_flags.append(f"资产负债率{debt:.1f}%偏高")
    else:
        capex_score = 18

    # 营收高增长说明资本开支有效
    if rev_yoy is not None and rev_yoy > 30:
        capex_score = min(capex_score + 3, 25)

    # ── 综合判定 ──
    total = rev_score + client_score + margin_score + capex_score
    verdict = "REAL"
    note = ""

    # 判断逻辑
    if total >= 85:
        verdict = "REAL"
        note = "财务指标优秀，产业链核心标的"
    elif total >= 70:
        verdict = "REAL"
        note = "财务指标良好，持续关注"
    elif total >= 55:
        verdict = "WATCH"
        note = "部分指标偏弱，等趋势确认"
    else:
        verdict = "WATCH"
        note = "财务指标偏弱，谨慎关注"

    # 特殊红线
    if deduct_yoy is not None and deduct_yoy < -50:
        verdict = "WATCH"
        note = "扣非净利润大幅下滑，需警惕"
    if debt is not None and debt > 80:
        red_flags.append("资产负债率过高，财务风险较大")
        verdict = "WATCH"

    # PE 合理性标注
    if pe is not None and pe > 0:
        if pe > 100:
            red_flags.append(f"PE {pe:.0f}x 估值偏高")
        elif pe < 15 and profit_yoy is not None and profit_yoy > 0:
            green_flags.append(f"PE {pe:.0f}x 估值偏低+盈利增长")

    p = CompanyProfile(
        code=code,
        name=static_info["name"],
        chain=static_info["chain"],
        node=static_info["node"],
        revenue_score=rev_score,
        client_score=client_score,
        margin_score=margin_score,
        capex_score=capex_score,
        green_flags=green_flags,
        red_flags=red_flags,
        verdict=verdict,
        total_score=total,
        note=note,
        pe=fin_data.get("pe") or 0,
        pb=fin_data.get("pb") or 0,
        mktcap=fin_data.get("mktcap") or "",
        gross_margin=fin_data.get("gross_margin") or 0,
        net_margin=fin_data.get("net_margin") or 0,
        roe=fin_data.get("roe") or 0,
        rev_yoy=fin_data.get("rev_yoy") or 0,
        profit_yoy=fin_data.get("profit_yoy") or 0,
        net_profit_deducted_yoy=fin_data.get("net_profit_deducted_yoy") or 0,
        debt_ratio=fin_data.get("debt_ratio") or 0,
        cash_flow_ratio=fin_data.get("cash_flow_ratio") or 0,
        data_source="auto",
    )
    return p


# ══════════════════════════════════════════════════════════════════════════════
# 动态成分股抓取
# ══════════════════════════════════════════════════════════════════════════════

def fetch_chain_stocks(chain_name: str, max_per_concept: int = 30) -> List[dict]:
    """
    根据产业链名称，从 akshare 概念板块动态抓取成分股。
    返回 [{"code": "000001", "name": "平安银行", "chain": "..."}, ...]
    """
    concepts = CHAIN_CONCEPT_MAP.get(chain_name, [])
    if not concepts:
        return []

    seen_codes: set = set()
    stocks: list = []

    for concept in concepts:
        try:
            df = ak.stock_board_concept_cons_em(symbol=concept)
            if df is None or len(df) == 0:
                continue
            # 标准化列名
            df.columns = [str(c).strip() for c in df.columns]
            col_code = "代码"
            col_name = "名称"
            if col_code not in df.columns or col_name not in df.columns:
                continue

            for _, row in df.head(max_per_concept).iterrows():
                code = str(row[col_code]).strip()
                name = str(row[col_name]).strip()
                # 过滤 ST、退市、北交所
                if not code or "ST" in name or "退" in name:
                    continue
                if code.startswith("8") or code.startswith("4"):
                    continue
                if code not in seen_codes:
                    seen_codes.add(code)
                    stocks.append({"code": code, "name": name, "chain": chain_name})
        except Exception:
            continue  # 概念板块不存在则跳过

    return stocks


def llm_fetch_chain_stocks(chain_name: str) -> List[dict]:
    """
    用 LLM 分析产业链的头部企业，返回 [{"code": "300308", "name": "中际旭创"}, ...]
    fallback 方案：akshare 概念板块抓取失败时使用
    """
    try:
        from llm_client import chat_json
        system = (
            "你是一位资深A股投研分析师。请分析指定产业链的A股头部企业。\n"
            "要求：\n"
            "1. 只返回该产业链核心环节的龙头/头部企业（每个环节最多2家）\n"
            "2. 代码必须是6位数字的A股代码（沪深主板+创业板+科创板）\n"
            "3. 不要包含ST、退市股\n"
            "4. 总共返回5-10家\n"
            "返回JSON格式：{\"stocks\": [{\"code\": \"300308\", \"name\": \"中际旭创\", \"node\": \"光模块\"}, ...]}"
        )
        user = f"请分析「{chain_name}」产业链的A股头部企业，列出各环节龙头公司。"
        result = chat_json(system, user, temperature=0.2)
        if result and "stocks" in result:
            stocks = []
            seen = set()
            for s in result["stocks"]:
                code = str(s.get("code", "")).strip()
                name = str(s.get("name", "")).strip()
                node = str(s.get("node", "核心")).strip()
                if not code or not name or not re.match(r'^\d{6}$', code):
                    continue
                if code in seen:
                    continue
                seen.add(code)
                stocks.append({"code": code, "name": name, "node": node, "chain": chain_name})
            if stocks:
                return stocks
    except Exception as e:
        print(f"      → LLM分析异常: {e}")
    return []


def _get_current_price(code: str) -> Optional[float]:
    """获取个股最新收盘价"""
    try:
        df = _get_kline_cached(code)
        if df is not None and len(df) > 0:
            return float(pd.to_numeric(df["收盘"].iloc[-1], errors="coerce"))
    except Exception:
        pass
    return None


def score_dynamic_company(code: str, name: str, chain: str,
                          fin_data: dict) -> Optional[CompanyProfile]:
    """
    对动态发现的股票进行财务评分（简化版，无静态 green/red flags）。
    重点看：盈利能力、成长性、估值、财务健康。
    """
    gm = fin_data.get("gross_margin")
    nm = fin_data.get("net_margin")
    roe = fin_data.get("roe")
    rev_yoy = fin_data.get("rev_yoy")
    profit_yoy = fin_data.get("profit_yoy")
    deduct_yoy = fin_data.get("net_profit_deducted_yoy")
    debt = fin_data.get("debt_ratio")
    pe = fin_data.get("pe")
    mktcap = fin_data.get("mktcap", "")

    # ── 市值门槛：低于30亿的跳过 ──
    has_mktcap = True
    try:
        cap_val = float(str(mktcap).replace("亿", "").replace(",", ""))
        if cap_val < 30:
            return None
    except (ValueError, TypeError):
        has_mktcap = False  # 无市值数据，不跳过，给默认分数

    score = 0
    green_flags: list = []
    red_flags: list = []

    # ── 盈利能力（满分30）──
    if roe is not None:
        if roe >= 20:
            score += 30
            green_flags.append(f"ROE {roe:.1f}%")
        elif roe >= 15:
            score += 25
        elif roe >= 10:
            score += 18
        elif roe >= 5:
            score += 10

    if nm is not None:
        if nm >= 25:
            score += 5
        elif nm >= 15:
            score += 3

    # ── 成长性（满分25）──
    if profit_yoy is not None:
        if profit_yoy >= 50:
            score += 25
            green_flags.append(f"净利润增速{profit_yoy:.0f}%")
        elif profit_yoy >= 30:
            score += 22
        elif profit_yoy >= 15:
            score += 18
        elif profit_yoy >= 0:
            score += 12
        else:
            red_flags.append(f"净利润下滑{profit_yoy:.1f}%")

    if rev_yoy is not None and rev_yoy > 30:
        score += 5

    # ── 估值（满分20）──
    if pe is not None and pe > 0:
        if 10 <= pe <= 30:
            score += 20
        elif 30 < pe <= 50:
            score += 15
        elif pe < 10 and profit_yoy is not None and profit_yoy > 0:
            score += 18
            green_flags.append(f"PE {pe:.0f}x 低估值+盈利增长")
        elif pe > 80:
            score += 5
            red_flags.append(f"PE {pe:.0f}x 估值偏高")
        elif pe > 50:
            score += 10

    # ── 财务健康（满分20）──
    if debt is not None:
        if debt < 40:
            score += 20
        elif debt < 60:
            score += 15
        elif debt < 70:
            score += 8
        else:
            score += 3
            red_flags.append(f"资产负债率{debt:.1f}%偏高")

    # ── 特殊红线 ──
    # 如果几乎没有财务数据（score仍为0），给基础分并标记待验证
    if score == 0:
        score = 45  # 给予基础分，不直接丢弃
        verdict = "WATCH"
        note = "LLM识别的头部企业，财务数据待补充验证"
        red_flags.append("财务数据不完整，建议人工核实")

    if score >= 70:
        verdict = "REAL"
        note = "财务指标优秀"
    elif score >= 50:
        verdict = "WATCH"
        note = "部分指标偏弱"
    else:
        verdict = "WATCH"
        note = "财务指标较弱"

    if deduct_yoy is not None and deduct_yoy < -50:
        verdict = "WATCH"
        note = "扣非净利润大幅下滑，需警惕"
        red_flags.append(f"扣非净利润同比{deduct_yoy:.1f}%")

    if debt is not None and debt > 80:
        red_flags.append("资产负债率过高，财务风险较大")

    return CompanyProfile(
        code=code,
        name=name,
        chain=chain,
        node="动态发现",
        revenue_score=0,
        client_score=0,
        margin_score=0,
        capex_score=0,
        green_flags=green_flags,
        red_flags=red_flags,
        verdict=verdict,
        total_score=min(score, 100),
        note=note,
        pe=pe or 0,
        pb=fin_data.get("pb") or 0,
        mktcap=str(mktcap),
        gross_margin=gm or 0,
        net_margin=nm or 0,
        roe=roe or 0,
        rev_yoy=rev_yoy or 0,
        profit_yoy=profit_yoy or 0,
        net_profit_deducted_yoy=deduct_yoy or 0,
        debt_ratio=debt or 0,
        cash_flow_ratio=fin_data.get("cash_flow_ratio") or 0,
        data_source="dynamic",
    )


def run_layer4(codes: List[str] = None,
               active_chains: List[str] = None) -> List[CompanyProfile]:
    """
    Layer 4: 个股筛选

    - 提供 active_chains 时：LLM分析产业链头部企业 + akshare 概念板块补充 + 静态库
    - 仅提供 codes 时：保持原逻辑，从静态 COMPANY_DB 中筛选指定代码
    - 均不提供：使用静态 COMPANY_DB 全量
    """
    results: List[CompanyProfile] = []
    processed_codes: set = set()

    # ── 1. 静态库中的标的（优先级高，有详细 green/red flags）──
    static_codes = codes if codes else list(COMPANY_DB.keys())

    if active_chains:
        # 只取属于激活产业链的静态标的
        for code in static_codes:
            if code not in COMPANY_DB:
                continue
            if COMPANY_DB[code].get("chain") not in active_chains:
                continue
            fin_data = fetch_financial_data(code)
            p = auto_score_company(code, COMPANY_DB[code], fin_data)
            results.append(p)
            processed_codes.add(code)
    else:
        # 无激活链 → 全部静态标的
        for code in static_codes:
            if code not in COMPANY_DB:
                continue
            fin_data = fetch_financial_data(code)
            p = auto_score_company(code, COMPANY_DB[code], fin_data)
            results.append(p)
            processed_codes.add(code)

    # ── 2. LLM分析激活产业链头部企业 ──
    if active_chains:
        for chain_name in active_chains:
            print(f"    ↳ LLM分析 [{chain_name}] 头部企业...")
            llm_stocks = llm_fetch_chain_stocks(chain_name)
            if llm_stocks:
                kept = 0
                for stock in llm_stocks:
                    code = stock["code"]
                    if code in processed_codes:
                        continue
                    fin_data = fetch_financial_data(code)
                    p = score_dynamic_company(
                        code, stock["name"], chain_name, fin_data
                    )
                    if p is not None:
                        # LLM 分析的标的附带节点信息
                        p.node = stock.get("node", "核心")
                        results.append(p)
                        processed_codes.add(code)
                        kept += 1
                print(f"      → LLM识别 {len(llm_stocks)} 只，通过财务筛选 {kept} 只")
            else:
                # LLM 失败，回退到 akshare 概念板块
                print(f"      → LLM未返回结果，回退到 akshare 抓取...")
                dyn_stocks = fetch_chain_stocks(chain_name)
                if not dyn_stocks:
                    print(f"      → 未找到相关概念板块，跳过")
                    continue
                kept = 0
                for stock in dyn_stocks:
                    code = stock["code"]
                    if code in processed_codes:
                        continue
                    fin_data = fetch_financial_data(code)
                    p = score_dynamic_company(
                        code, stock["name"], chain_name, fin_data
                    )
                    if p is not None:
                        results.append(p)
                        processed_codes.add(code)
                        kept += 1
                print(f"      → 抓取 {len(dyn_stocks)} 只，通过财务筛选 {kept} 只")

    # ── 为每个 profile 补充技术面和深度财务分析 ──
    for p in results:
        try:
            tech = compute_technical_indicators(p.code)
            p.tech_summary = tech.get("summary", "")
            p.tech_overall = tech.get("overall", "")
        except Exception:
            pass
        try:
            deep = fetch_deep_financials(p.code)
            df_parts = []
            if deep.get("gm_trend"):
                df_parts.append(deep["gm_trend"])
            if deep.get("quality_note"):
                df_parts.append(deep["quality_note"])
            p.deep_financial = " ".join(df_parts) if df_parts else ""
            p.gross_margin_trend = deep.get("gm_trend", "")
            p.financial_quality = deep.get("quality_note", "")
            p.cash_to_profit_ratio = deep.get("cash_to_profit", 0) or 0
            p.rd_ratio = deep.get("rd_ratio", 0) or 0
        except Exception:
            pass

        # deep_financial 为空时用 LLM 补充
        if not p.deep_financial:
            try:
                from llm_client import chat
                price = _get_current_price(p.code)
                if price and price > 0:
                    sys_prompt = "你是A股财务分析专家。根据公司已知信息给出简短（50字内）的深度财务分析，重点：现金流质量、毛利率趋势、研发投入。"
                    info_text = f"公司：{p.name}({p.code})，当前价{price}，毛利率{p.gross_margin:.1f}%，ROE{p.roe:.1f}%，PE{p.pe:.0f}，营收同比{p.rev_yoy:+.1f}%，净利同比{p.profit_yoy:+.1f}%，负债率{p.debt_ratio:.1f}%。"
                    ai_note = chat(sys_prompt, info_text, temperature=0.3)
                    if ai_note:
                        p.deep_financial = ai_note.strip()[:200]
            except Exception:
                pass

    return sorted(results, key=lambda x: x.total_score, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StockDecision:
    code: str
    name: str
    chain: str
    # 六层综合
    l0_pass: bool = True
    l1_pass: bool = False
    l2_pass: bool = False
    l3_pass: bool = False
    l4_score: int = 0
    # Layer 5 具体
    market_env: str = "SIDEWAYS"
    timing_signal: str = "NONE"
    stop_loss_pct: float = 8.0
    # 计算结果
    base_position_pct: float = 0.0
    actual_position_pct: float = 0.0
    first_batch_pct: float = 0.0
    verdict: str = "WAIT"
    reason: str = ""
    action: str = ""
    # 深度分析
    tech_summary: str = ""
    tech_overall: str = ""
    deep_financial: str = ""
    # A股实际交易数据
    current_price: float = 0.0
    first_batch_lots: int = 0       # 首批可买手数
    first_batch_cost: float = 0.0   # 首批实际金额
    stop_loss_price: float = 0.0    # 止损价位
    max_loss_amount: float = 0.0    # 最大亏损金额
    # 财务指标
    pe: float = 0.0
    cash_flow_ratio: float = 0.0
    debt_ratio: float = 0.0


def get_timing_signal(code: str) -> dict:
    """获取个股量价信号（含重试和容错）"""
    result = {"signal": "NONE", "note": ""}
    try:
        df = _get_kline_cached(code)
        if df is None or len(df) < 25:
            result["note"] = "行情数据不足25日，无法判定"
            return result

        df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
        df["成交量"] = pd.to_numeric(df["成交量"], errors="coerce")
        df = df.dropna(subset=["收盘", "成交量"]).reset_index(drop=True)

        latest_vol  = float(df["成交量"].iloc[-1])
        avg_vol_20  = float(df["成交量"].tail(21).iloc[:-1].mean())
        latest_cls  = float(df["收盘"].iloc[-1])
        high_20     = float(df["收盘"].tail(20).max())
        low_5       = float(df["收盘"].tail(5).min())
        avg_vol_5   = float(df["成交量"].tail(6).iloc[:-1].mean())

        vol_ratio = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 1

        if latest_cls >= high_20 * 0.995 and vol_ratio >= 1.5:
            result["signal"] = "BREAKOUT"
            result["note"]   = f"放量突破20日高点，成交量是均量{vol_ratio:.1f}倍，最强入场信号"
        elif vol_ratio < 0.7 and latest_cls > low_5 * 1.02:
            result["signal"] = "PULLBACK"
            result["note"]   = f"缩量回调中（成交仅均量{vol_ratio:.1f}倍），支撑有效，性价比最高入场点"
        elif avg_vol_5 > avg_vol_20 * 1.1 and latest_cls > float(df["收盘"].tail(10).mean()):
            result["signal"] = "ACCUMULATION"
            result["note"]   = f"近5日成交活跃，价格站稳均线，资金温和流入"
        else:
            result["signal"] = "NONE"
            result["note"]   = "暂无明确入场信号，继续观察"

    except Exception as e:
        result["note"] = f"行情获取异常（不影响其他层判断），建议手动确认"
    return result


def get_support_level(code: str) -> float:
    """获取最近支撑位（近20日最低点）"""
    try:
        df = _get_kline_cached(code)
        if df is not None and len(df) >= 20:
            df["最低"] = pd.to_numeric(df["最低"], errors="coerce")
            df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
            support = float(df["最低"].tail(20).min())
            latest  = float(df["收盘"].iloc[-1])
            stop_pct = (latest - support) / latest * 100
            return round(stop_pct, 1)
    except Exception:
        pass
    # 容错：返回保守默认值 8%
    return 8.0


def calculate_position(layers_passed: int, market_env: str,
                       stop_loss_pct: float, timing_signal: str) -> dict:
    env_caps = {"BULL": 10.0, "SIDEWAYS": 6.0, "BEAR": 2.0}
    total_caps = {"BULL": 60.0, "SIDEWAYS": 30.0, "BEAR": 10.0}
    single_cap = env_caps.get(market_env, 6.0)
    layer_ratios = {0: 0, 1: 0.10, 2: 0.30, 3: 0.60, 4: 0.85, 5: 1.0}
    layer_ratio  = layer_ratios.get(min(layers_passed, 5), 0)
    timing_bonus = {"BREAKOUT": 1.0, "PULLBACK": 1.0, "ACCUMULATION": 0.85, "NONE": 0.6}
    timing_ratio = timing_bonus.get(timing_signal, 0.6)
    base_pct   = single_cap * layer_ratio * timing_ratio
    actual_pct = base_pct * (5.0 / stop_loss_pct) if stop_loss_pct > 0 else base_pct
    actual_pct = min(actual_pct, single_cap)
    first_batch = actual_pct * 0.4
    return {
        "single_cap": round(single_cap, 1),
        "total_cap": total_caps.get(market_env, 30.0),
        "base_pct": round(base_pct, 2),
        "actual_pct": round(actual_pct, 2),
        "first_batch": round(first_batch, 2),
    }


def make_decision(profile: CompanyProfile,
                  l0_pass: bool, l1_score: int, l2_score: int,
                  l3_active: bool, market_env: str = "SIDEWAYS",
                  capital_total: float = 1_000_000) -> StockDecision:
    """综合六层做出最终决策"""
    # ── 技术指标分析 ──
    tech = compute_technical_indicators(profile.code)

    # ── 深度财务分析 ──
    deep = fetch_deep_financials(profile.code)

    d = StockDecision(
        code=profile.code,
        name=profile.name,
        chain=profile.chain,
        l0_pass=l0_pass,
        l1_pass=l1_score >= 55,
        l2_pass=l2_score >= 50,
        l3_pass=l3_active,
        l4_score=profile.total_score,
        market_env=market_env,
        tech_summary=tech.get("summary", ""),
        tech_overall=tech.get("overall", ""),
    )

    # 深度财务摘要
    df_parts = []
    if deep.get("gm_trend"):
        df_parts.append(deep["gm_trend"])
    if deep.get("quality_note"):
        df_parts.append(deep["quality_note"])
    d.deep_financial = " ".join(df_parts) if df_parts else ""

    # deep_financial 为空时用 LLM 补充
    if not d.deep_financial and d.current_price and d.current_price > 0:
        try:
            from llm_client import chat
            sys_prompt = "你是A股财务分析专家。根据公司已知信息给出简短（50字内）的深度财务分析，重点：现金流质量、毛利率趋势、研发投入。"
            info_text = f"公司：{profile.name}({profile.code})，当前价{d.current_price}，毛利率{profile.gross_margin:.1f}%，ROE{profile.roe:.1f}%，PE{profile.pe:.0f}，营收同比{profile.rev_yoy:+.1f}%，净利同比{profile.profit_yoy:+.1f}%，负债率{profile.debt_ratio:.1f}%。Green flags: {profile.green_flags[:3]}。Red flags: {profile.red_flags[:3]}。"
            ai_note = chat(sys_prompt, info_text, temperature=0.3)
            if ai_note:
                d.deep_financial = ai_note.strip()[:200]
        except Exception:
            pass

    if not l0_pass:
        d.verdict = "NO"
        d.reason  = "Layer 0 触发系统性风险，禁止新建仓位"
        d.action  = "空仓等待，不操作"
        return d

    if profile.verdict == "FAKE":
        d.verdict = "NO"
        d.reason  = "Layer 4 否决：概念股或不通过过滤器"
        d.action  = "移出观察名单"
        return d

    layers = sum([l0_pass, d.l1_pass, d.l2_pass, d.l3_pass,
                  profile.total_score >= 75])

    timing_info = get_timing_signal(profile.code)
    d.timing_signal = timing_info["signal"]

    d.stop_loss_pct = get_support_level(profile.code)

    pos = calculate_position(layers, market_env, d.stop_loss_pct, d.timing_signal)
    d.base_position_pct   = pos["base_pct"]
    d.actual_position_pct = pos["actual_pct"]
    d.first_batch_pct     = pos["first_batch"]

    # ── A股实际交易计算（按手数）──
    price = _get_current_price(profile.code)
    if price and price > 0:
        d.current_price = round(price, 2)
        d.pe = profile.pe or 0
        d.cash_flow_ratio = getattr(profile, "cash_to_profit_ratio", None)
        d.debt_ratio = profile.debt_ratio or 0
        # 首批金额 = 总资金 * 首批仓位比例
        budget = capital_total * d.first_batch_pct / 100
        # A股最低1手=100股，最多买多少手
        lots = int(budget / (price * 100))
        if lots < 1:
            lots = 1  # 至少1手
        d.first_batch_lots = lots
        d.first_batch_cost = round(lots * price * 100, 0)
        d.stop_loss_price = round(price * (1 - d.stop_loss_pct / 100), 2)
        d.max_loss_amount = round(lots * (price - d.stop_loss_price) * 100, 0)

    # ── 决策逻辑 ──
    has_timing = d.timing_signal in ("BREAKOUT", "PULLBACK", "ACCUMULATION")
    timing_failed = "获取异常" in timing_info.get("note", "")

    if layers >= 4 and has_timing:
        d.verdict = "GO"
        d.reason  = f"六层通过{layers}层，时机信号：{d.timing_signal}"
        d.action  = (f"建仓：第一批{d.first_batch_pct}%（占总资金），"
                     f"止损位在当前价下方{d.stop_loss_pct}%")
    elif layers >= 4 and timing_failed:
        # 通过4层但行情获取失败：降级为 WAIT 但标注"接近GO"
        d.verdict = "WAIT"
        d.reason  = f"通过{layers}层，行情数据暂时无法获取，建议手动确认后入场"
        d.action  = "手动查看K线确认时机，信号确认后可直接建仓"
    elif layers >= 3:
        d.verdict = "WAIT"
        d.reason  = f"通过{layers}层，但入场时机信号未触发（{timing_info['note']}）"
        d.action  = "加入观察名单，设价格提醒等待信号"
    else:
        d.verdict = "WAIT"
        d.reason  = f"仅通过{layers}层，信号不足"
        d.action  = "等待更多层信号确认"

    # L4 WATCH 标注
    if profile.verdict == "WATCH" and d.verdict == "GO":
        d.reason += "（注意：L4财务指标偏弱，仓位需额外谨慎）"

    return d


def run_layer5(profiles: List[CompanyProfile], l0_pass: bool,
               l1_score: int, l2_score: int,
               l3_active_chains: List[str],
               market_env: str = "SIDEWAYS",
               capital_total: float = 1_000_000) -> List[StockDecision]:
    decisions = []
    for p in profiles:
        if p.verdict == "FAKE":
            continue
        l3_active = any(p.chain in c for c in l3_active_chains)
        d = make_decision(p, l0_pass, l1_score, l2_score, l3_active, market_env, capital_total)
        decisions.append(d)
    return sorted(decisions, key=lambda x: (x.verdict == "GO", x.actual_position_pct), reverse=True)
