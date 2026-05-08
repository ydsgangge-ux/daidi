#!/usr/bin/env python3
"""
六层市场分析决策系统
安装：pip install akshare pandas rich
运行：python main.py
     python main.py --capital 500000 --env SIDEWAYS
"""

# ══════════════════════════════════════════════════════════════════════════════
# 依赖
# ══════════════════════════════════════════════════════════════════════════════
import sys, argparse, warnings
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
warnings.filterwarnings("ignore")

try:
    import akshare as ak
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError as e:
    print(f"缺少依赖库：{e}\n请运行：pip install akshare pandas rich")
    sys.exit(1)

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Indicator:
    name: str; value: float; unit: str
    yoy: Optional[float]; zscore: Optional[float]
    signal: str; note: str; source: str

@dataclass
class RiskStatus:
    level: str; score: int; triggered: list
    max_position: float; note: str

@dataclass
class IntlSignal:
    name: str; value: float; unit: str
    change: Optional[float]; direction: str
    a_share_impact: str; source: str

@dataclass
class CompanyProfile:
    code: str; name: str; chain: str; node: str
    revenue_score: int; client_score: int
    margin_score: int; capex_score: int
    green_flags: List[str] = field(default_factory=list)
    red_flags:   List[str] = field(default_factory=list)
    verdict: str = "REAL"; total_score: int = 0; note: str = ""

@dataclass
class StockDecision:
    code: str; name: str; chain: str
    l0_pass: bool = True; l1_pass: bool = False
    l2_pass: bool = False; l3_pass: bool = False
    l4_score: int = 0; market_env: str = "SIDEWAYS"
    timing_signal: str = "NONE"; stop_loss_pct: float = 8.0
    actual_position_pct: float = 0.0; first_batch_pct: float = 0.0
    verdict: str = "WAIT"; reason: str = ""; action: str = ""
    # 产业链周期
    cycle_phase: str = ""
    cycle_maturity: int = 50
    cycle_remaining_months: float = 0
    cycle_discount: float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0 · 风险管理
# ══════════════════════════════════════════════════════════════════════════════

def run_layer0() -> tuple:
    env = {}
    # 上证指数
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        s = df["close"].dropna()
        latest = float(s.iloc[-1]); prev5 = float(s.iloc[-6]) if len(s)>=6 else latest
        df["pct"] = df["close"].pct_change()*100
        env.update({"index_latest": round(latest,2),
                    "index_5d_chg": round((latest-prev5)/prev5*100,2),
                    "max_1d_drop":  round(float(df["pct"].min()),2)})
    except Exception as e:
        env.update({"index_latest":0,"index_5d_chg":0,"max_1d_drop":0,"err_idx":str(e)[:50]})

    # 北向资金（沪股通+深股通，近日净买入）
    try:
        df = ak.stock_hsgt_north_acc_flow_in_em(symbol="沪深港通")
        df.columns = [c.strip() for c in df.columns]
        val_col = [c for c in df.columns if "净" in c or "flow" in c.lower()]
        if val_col:
            env["north_net"] = round(float(pd.to_numeric(df[val_col[0]].iloc[-1], errors="coerce")),2)
    except Exception:
        env["north_net"] = None

    # A股波动率指数（50ETF期权QVIX）
    try:
        df = ak.index_option_50etf_qvix()
        df.columns = [c.strip() for c in df.columns]
        val_col = [c for c in df.columns if "收盘" in c or "close" in c.lower()]
        if not val_col: val_col = [df.columns[1]]
        env["qvix"] = round(float(pd.to_numeric(df[val_col[0]].iloc[-1], errors="coerce")),2)
    except Exception:
        env["qvix"] = None

    # 人民币汇率
    try:
        df = ak.currency_boc_sina(symbol="美元人民币")
        df["中间价"] = pd.to_numeric(df["中间价"], errors="coerce")
        df = df.dropna(subset=["中间价"])
        lv = float(df["中间价"].iloc[-1]); pv = float(df["中间价"].iloc[-6]) if len(df)>=6 else lv
        env.update({"usdcny": round(lv,4), "usdcny_5d_chg": round((lv-pv)/pv*100,3)})
    except Exception:
        env.update({"usdcny":None,"usdcny_5d_chg":None})

    # 评估
    triggered = []; ded = 0
    if env.get("max_1d_drop",0) < -3.0:
        triggered.append(f"单日最大跌幅{env['max_1d_drop']}%，触发红线"); ded+=40
    if env.get("index_5d_chg",0) < -5.0:
        triggered.append(f"5日累计跌幅{env['index_5d_chg']}%，偏弱"); ded+=25
    qvix = env.get("qvix")
    if qvix and qvix > 30:
        triggered.append(f"QVIX={qvix}，国内期权恐慌指数过高"); ded+=30
    elif qvix and qvix > 22:
        triggered.append(f"QVIX={qvix}，波动率偏高，注意风险"); ded+=15
    fx = env.get("usdcny_5d_chg")
    if fx and fx > 1.0:
        triggered.append(f"人民币5日贬值{fx}%，超1%阈值"); ded+=20

    score = max(0, 100-ded)
    if score >= 70:
        chg = env.get("index_5d_chg",0)
        level, max_pos = "GREEN", (0.60 if chg>2 else 0.30)
        note = "做多环境最大仓位60%" if chg>2 else "震荡环境最大仓位30%"
    elif score >= 40:
        level, max_pos, note = "YELLOW", 0.10, "黄色预警，最大仓位10%，不新开仓"
    else:
        level, max_pos, note = "RED", 0.0, "红色警报，空仓等待"

    return RiskStatus(level=level, score=score, triggered=triggered,
                      max_position=max_pos, note=note), env


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 · 实业数据
# ══════════════════════════════════════════════════════════════════════════════

def _safe(name, fn, *args, unit="", note="", source="", invert=False, **kw):
    """通用安全包装：执行数据获取，自动处理异常"""
    try:
        df = fn(*args, **kw)
        if df is None or len(df)==0:
            raise ValueError("空数据")
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        # 取最后一个数值列
        num_cols = [c for c in df.columns if pd.to_numeric(df[c].iloc[-1], errors="coerce") == pd.to_numeric(df[c].iloc[-1], errors="coerce")]
        if not num_cols: raise ValueError("无数值列")
        col = num_cols[-1]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        series = df[col].dropna()
        latest = float(series.iloc[-1])
        # 同比
        yoy = None
        if len(series)>12:
            prev = float(series.iloc[-13])
            yoy = round((latest-prev)/abs(prev)*100,1) if prev!=0 else None
        # Z-score（同月历史）
        hist = [float(series.iloc[i]) for i in range(len(series)-1)
                if (len(series)-1-i)%12==0 and i!=len(series)-1]
        z = None
        if len(hist)>=2:
            mu,sigma = np.mean(hist), np.std(hist)
            z = round((latest-mu)/sigma,2) if sigma>0 else 0.0
        # 信号
        if z is not None:
            zv = -z if invert else z
            sig = ("ALERT_UP" if zv>2 else "UP" if zv>1.5 else
                   "ALERT_DOWN" if zv<-2 else "DOWN" if zv<-1.5 else "NEUTRAL")
        elif yoy is not None:
            sig = "UP" if yoy>5 else ("DOWN" if yoy<-5 else "NEUTRAL")
        else:
            sig = "NEUTRAL"
        return Indicator(name=name, value=round(latest,2), unit=unit,
                         yoy=yoy, zscore=z, signal=sig, note=note, source=source)
    except Exception as e:
        return Indicator(name=name, value=0, unit=unit, yoy=None, zscore=None,
                         signal="NEUTRAL", note=f"获取失败:{str(e)[:40]}", source=source)


def run_layer1() -> dict:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    groups = {
        "食品": [
            _safe("生鲜乳收购价（全国均价）",
                  ak.get_spot_price_of_agricultural_products_by_product_and_date,
                  "生鲜乳","20220101", unit="元/kg",
                  note="供给过剩→价格低位，养殖端承压", source="农业农村部"),
            _safe("猪肉批发价",
                  ak.get_spot_price_of_agricultural_products_by_product_and_date,
                  "猪肉","20230101", unit="元/kg",
                  note="消费景气代理指标", source="农业农村部"),
        ],
        "大宗": [
            _safe("沪铜主力", ak.futures_zh_realtime, "铜",
                  unit="元/吨", note="制造业晴雨表，LME联动", source="上期所"),
            _safe("螺纹钢主力", ak.futures_zh_realtime, "螺纹钢",
                  unit="元/吨", note="基建/地产需求代理", source="上期所"),
            _safe("沪金主力", ak.futures_zh_realtime, "黄金",
                  unit="元/克", note="避险+去美元化双驱动", source="上期所"),
        ],
        "宏观": [
            _safe("M2同比增速", ak.macro_china_money_supply,
                  unit="%", note="货币供给水位", source="央行"),
            _safe("社融增速", ak.macro_china_shrzgm,
                  unit="%", note="信用扩张领先股市6-9月", source="央行"),
            _safe("制造业PMI", ak.macro_china_pmi_yearly,
                  unit="", note="50荣枯线，新订单子项更有预测力", source="统计局"),
        ],
        "基建": [
            _safe("房地产开发投资增速", ak.macro_china_real_estate,
                  unit="%", note="地产下行拖累螺纹/水泥需求", source="统计局", invert=True),
        ],
        "用电": [
            _safe("全国用电量", ak.macro_china_electricity_statistics,
                  unit="亿度", note="实业温度计，Z-score消除季节性", source="国家能源局"),
        ],
    }
    alerts = []
    all_inds = [i for grp in groups.values() for i in grp]
    for i in all_inds:
        if "ALERT" in i.signal:
            alerts.append(i)
    up   = sum(1 for i in all_inds if "UP" in i.signal)
    down = sum(1 for i in all_inds if "DOWN" in i.signal)
    score = int(50 + (up-down)/max(len(all_inds),1)*50)
    return {"groups": groups, "alerts": alerts, "score": score, "ts": ts}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 · 金融期货 & 国际信号
# ══════════════════════════════════════════════════════════════════════════════

def _intl(name, fn, unit="", impact_bull="", impact_bear="", *args, **kw):
    try:
        df = fn(*args, **kw)
        if df is None or len(df)<5: raise ValueError("数据不足")
        df.columns = [c.strip() for c in df.columns]
        num_cols = [c for c in df.columns
                    if pd.to_numeric(df[c].dropna().iloc[-1], errors="coerce") is not None]
        col = num_cols[-1]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        series = df[col].dropna()
        latest = float(series.iloc[-1])
        prev   = float(series.iloc[-6]) if len(series)>=6 else latest
        chg    = round((latest-prev)/abs(prev)*100, 2) if prev!=0 else 0
        direction = "BULLISH" if chg < -1 else ("BEARISH" if chg > 1 else "NEUTRAL")
        impact = impact_bull if direction=="BULLISH" else (impact_bear if direction=="BEARISH" else "中性")
        return IntlSignal(name=name, value=round(latest,3), unit=unit,
                          change=chg, direction=direction,
                          a_share_impact=impact, source="akshare")
    except Exception as e:
        return IntlSignal(name=name, value=0, unit=unit, change=None,
                          direction="NEUTRAL", a_share_impact=f"获取失败:{str(e)[:40]}",
                          source="err")


def run_layer2(l1_score=50) -> dict:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    intl = [
        _intl("美元指数 DXY", ak.macro_usa_dxy,
              impact_bull="美元走弱→北向资金流入+大宗商品受益",
              impact_bear="美元走强→新兴市场资金承压"),
        _intl("中国国债10Y收益率", ak.bond_china_yield,
              unit="%",
              impact_bull="中国国债收益率↓→资产荒加剧，机构资金向权益迁移",
              impact_bear="国债收益率↑→无风险利率上行，高估值板块承压",
              start_date="20240101",
              end_date=datetime.now().strftime("%Y%m%d")),
    ]
    # 国内QVIX
    try:
        df = ak.index_option_50etf_qvix()
        df.columns = [c.strip() for c in df.columns]
        col = [c for c in df.columns if "收盘" in c or "close" in c.lower()]
        if not col: col = [df.columns[1]]
        df[col[0]] = pd.to_numeric(df[col[0]], errors="coerce")
        v = float(df[col[0]].dropna().iloc[-1])
        dir_ = "BEARISH" if v>25 else ("BEARISH" if v>20 else ("BULLISH" if v<13 else "NEUTRAL"))
        impact = ("QVIX过高！Layer 0警戒，暂停新建仓" if v>25 else
                  "波动率偏高，控制仓位" if v>20 else
                  "QVIX低位，做多环境友好" if v<13 else "正常区间")
        intl.append(IntlSignal("A股QVIX（50ETF）",v,"",None,dir_,impact,"akshare"))
    except Exception:
        pass

    # 股指期货基差
    futures = []
    for name, spot_sym, fut_sym in [
        ("IF（沪深300）","sh000300","IF"),
        ("IC（中证500）","sh000905","IC"),
        ("IM（中证1000）","sh000852","IM"),
    ]:
        try:
            df_s = ak.stock_zh_index_daily(symbol=spot_sym)
            df_s["close"] = pd.to_numeric(df_s["close"], errors="coerce")
            spot = float(df_s["close"].dropna().iloc[-1])
            df_f = ak.futures_zh_realtime(symbol=fut_sym)
            fp   = float(pd.to_numeric(df_f["最新价"].iloc[0], errors="coerce"))
            basis = round(fp - spot, 1)
            dir_ = "BULLISH" if basis > 20 else ("BEARISH" if basis < -30 else "NEUTRAL")
            futures.append(IntlSignal(
                name, basis, "点", round(basis/spot*100,2), dir_,
                f"基差{basis:+.0f}点({'贴水量化对冲' if basis<0 else '升水机构看多'})",
                "上期所"
            ))
        except Exception:
            continue

    all_sig = [s for s in intl+futures if s.value != 0]
    bull = sum(1 for s in all_sig if s.direction=="BULLISH")
    bear = sum(1 for s in all_sig if s.direction=="BEARISH")
    l2_score = int(50+(bull-bear)/max(len(all_sig),1)*50)

    diff = l1_score - l2_score
    if abs(diff) > 25:
        divergence = (f"⚠ 背离：L1={l1_score} L2={l2_score}，差异{diff}，降仓50%等待收敛")
    else:
        divergence = f"✓ L1({l1_score}) L2({l2_score}) 方向一致，信号可信"

    return {"intl": intl, "futures": futures, "score": l2_score,
            "divergence": divergence, "ts": ts}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 · 产业链（静态知识库）
# ══════════════════════════════════════════════════════════════════════════════

CHAINS = [
    {"name":"AI算力链",       "strength":85, "node":"光模块→液冷散热",
     "window":"液冷散热当前最佳；光模块等回调；高压电源4-8周后"},
    {"name":"半导体材料链",   "strength":70, "node":"光刻胶/靶材验证中",
     "window":"中芯/华虹季报采购比例上升是最强入场信号"},
    {"name":"航空航天材料链", "strength":60, "node":"上游原材料持续受益",
     "window":"C919订单+军机交付加速，西部超导/中复神鹰长期配置"},
    {"name":"出口制造链",     "strength":55, "node":"运价回升→港口验证中",
     "window":"等美国库存比下行+SCFI吞吐量双确认"},
    {"name":"SiC材料链",      "strength":55, "node":"等待价格筑底",
     "window":"8英寸量产确认+天岳海外客户验证通过"},
    {"name":"新能源车链",     "strength":35, "node":"等待锂价筑底",
     "window":"矿山减产公告+库存4周连降+期货远月升水收窄"},
    {"name":"工业自动化链",   "strength":45, "node":"国产替代独立行情",
     "window":"汇川/绿的谐波国产替代不依赖宏观景气"},
]

def run_layer3() -> dict:
    active   = [c for c in CHAINS if c["strength"]>=50]
    inactive = [c for c in CHAINS if c["strength"]<50]
    top      = sorted(active, key=lambda x: x["strength"], reverse=True)[:3]
    return {"active": active, "inactive": inactive, "top": top}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 · 头部企业知识库
# ══════════════════════════════════════════════════════════════════════════════

COMPANY_DB = {
    "300308": CompanyProfile("300308","中际旭创","AI算力链","光模块",25,25,22,22,
        ["光模块收入>85%","直供微软/谷歌/亚马逊","毛利35%+全球最高","800G/1.6T提前布局"],[],
        "REAL","AI算力核心，操作窗口仍在"),
    "300394": CompanyProfile("300394","天孚通信","AI算力链","光器件",25,25,25,22,
        ["无源光器件全球第一","直供英伟达/博通","毛利50%+","CPO卡位"],[],
        "REAL","壁垒最深，毛利率护城河最强"),
    "002837": CompanyProfile("002837","英维克","AI算力链","液冷散热",22,20,15,20,
        ["数据中心热管理国内第一","三大运营商客户","液冷占比快速提升"],
        ["毛利率受竞争压制"],"REAL","液冷赛道当前最佳窗口"),
    "300750": CompanyProfile("300750","宁德时代","新能源车链","电芯",25,25,25,25,
        ["全球动力电池第一35%+","特斯拉/宝马/大众","毛利22%+行业最高","固态电池布局"],[],
        "REAL","行业绝对龙头，等锂价筑底"),
    "300124": CompanyProfile("300124","汇川技术","工业自动化链","工控综合",25,25,23,23,
        ["变频器/伺服国内第一超越西门子","覆盖全行业","毛利43%+","人形机器人布局"],[],
        "REAL","工控中综合质量最高"),
    "688017": CompanyProfile("688017","绿的谐波","工业自动化链","谐波减速器",22,20,25,22,
        ["谐波减速器国内唯一规模化","毛利55%+极强","人形机器人关节核心"],
        ["客户集中度较高"],"REAL","国产替代独立逻辑"),
    "300666": CompanyProfile("300666","江丰电子","半导体材料链","溅射靶材",23,23,22,22,
        ["高纯靶材国内第一","中芯/长江存储/华虹客户","毛利35%+","持续扩产"],[],
        "REAL","半导体材料验证最完整"),
    "688019": CompanyProfile("688019","安集科技","半导体材料链","CMP抛光液",20,22,25,18,
        ["打破日美垄断唯一量产","中芯28nm以下验证通过","毛利60%+"],
        ["规模偏小"],"REAL","壁垒最深但规模最小"),
    "688122": CompanyProfile("688122","西部超导","航空航天材料链","钛合金",23,25,23,22,
        ["航空钛合金唯一军品认证","航空发动机集团/中航客户","毛利45%+稳定"],[],
        "REAL","军品属性强，军机交付受益"),
    "600690": CompanyProfile("600690","海尔智家","出口制造链","家电出口",25,25,22,22,
        ["全球白电第一海外收入>50%","GE Appliances并购成功","自有品牌溢价"],[],
        "REAL","品牌化出口典范"),
    "300866": CompanyProfile("300866","安克创新","出口制造链","消费电子出口",25,25,25,22,
        ["亚马逊充电第一","海外收入>90%","毛利40%+远超代工","研发持续提升"],[],
        "REAL","出口链质量最高"),
    "688234": CompanyProfile("688234","天岳先进","SiC材料链","SiC衬底",22,22,15,22,
        ["6英寸SiC国内第一","英飞凌/意法半导体验证","8英寸布局领先"],
        ["价格下行毛利承压"],"WATCH","等8英寸量产确认"),
    "603290": CompanyProfile("603290","斯达半导","SiC材料链","SiC器件",22,22,20,20,
        ["IGBT国内第一，SiC快速放量","比亚迪/上汽客户","毛利38%高于均值"],
        ["英飞凌高端竞争激烈"],"REAL","SiC器件国产化最快落地"),
    "002812": CompanyProfile("002812","恩捷股份","新能源车链","隔膜",25,22,12,20,
        ["湿法隔膜全球第一约40%","宁德/比亚迪/松下客户","海外产能领先"],
        ["毛利率受产能过剩压制"],"WATCH","等行业出清毛利修复"),
    "688295": CompanyProfile("688295","中复神鹰","航空航天材料链","碳纤维",22,22,15,20,
        ["T700/T800量产达到航空级","干喷湿纺技术全球领先成本最低"],
        ["碳纤维价格下行期"],"WATCH","等价格企稳后建仓"),
}


def run_layer4(codes=None, active_chains=None):
    """Layer 4: 使用 layer45_stocks 的动态抓取版本"""
    from layer45_stocks import run_layer4 as _dyn_layer4
    if active_chains:
        return _dyn_layer4(active_chains=active_chains)
    if codes:
        return _dyn_layer4(codes=codes)
    return _dyn_layer4()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 · 个股最终确认 + 仓位计算
# ══════════════════════════════════════════════════════════════════════════════

def get_timing(code: str) -> dict:
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date="20240101", adjust="qfq")
        if df is None or len(df)<25: return {"signal":"NONE","note":"数据不足"}
        df["收盘"] = pd.to_numeric(df["收盘"],errors="coerce")
        df["成交量"] = pd.to_numeric(df["成交量"],errors="coerce")
        df = df.dropna(subset=["收盘","成交量"]).reset_index(drop=True)
        v_now = float(df["成交量"].iloc[-1])
        v_avg20 = float(df["成交量"].tail(21).iloc[:-1].mean())
        c_now = float(df["收盘"].iloc[-1])
        c_hi20 = float(df["收盘"].tail(20).max())
        c_lo5  = float(df["收盘"].tail(5).min())
        vr = v_now/v_avg20 if v_avg20>0 else 1
        if c_now >= c_hi20*0.995 and vr>=1.5:
            return {"signal":"BREAKOUT","note":f"放量突破20日高点，量比{vr:.1f}倍"}
        elif vr<0.7 and c_now > c_lo5*1.02:
            return {"signal":"PULLBACK","note":f"缩量回调({vr:.1f}倍均量)支撑有效，最佳入场点"}
        elif float(df["成交量"].tail(6).iloc[:-1].mean())>v_avg20*1.1 and c_now>float(df["收盘"].tail(10).mean()):
            return {"signal":"ACCUMULATION","note":"近5日成交活跃，价格站稳均线"}
        else:
            return {"signal":"NONE","note":"暂无明确信号，继续观察"}
    except Exception as e:
        return {"signal":"NONE","note":f"行情获取失败:{str(e)[:30]}"}


def get_stop_pct(code: str) -> float:
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date="20240101", adjust="qfq")
        if df is not None and len(df)>=20:
            df["最低"] = pd.to_numeric(df["最低"],errors="coerce")
            df["收盘"] = pd.to_numeric(df["收盘"],errors="coerce")
            support = float(df["最低"].tail(20).min())
            latest  = float(df["收盘"].iloc[-1])
            return round((latest-support)/latest*100,1)
    except Exception:
        pass
    return 8.0


def calc_position(layers: int, env: str, stop_pct: float, timing: str,
                  cycle_maturity: int = 50) -> dict:
    caps = {"BULL":10.0,"SIDEWAYS":6.0,"BEAR":2.0}
    ratios = {0:0,1:0.10,2:0.30,3:0.60,4:0.85,5:1.0}
    t_bonus = {"BREAKOUT":1.0,"PULLBACK":1.0,"ACCUMULATION":0.85,"NONE":0.6}
    single_cap = caps.get(env,6.0)

    # 周期成熟度折扣
    if cycle_maturity <= 40:
        cycle_r = 1.0
    elif cycle_maturity <= 60:
        cycle_r = 0.85
    elif cycle_maturity <= 80:
        cycle_r = 0.65
    else:
        cycle_r = 0.4

    base = single_cap * ratios.get(min(layers,5),0) * t_bonus.get(timing,0.6) * cycle_r
    actual = min(base*(5.0/stop_pct) if stop_pct>0 else base, single_cap)
    return {"actual":round(actual,2),"first":round(actual*0.4,2),
            "cap":single_cap, "total_cap":{"BULL":60,"SIDEWAYS":30,"BEAR":10}.get(env,30),
            "cycle_ratio":round(cycle_r,2)}


def run_layer5(profiles, l0_pass, l1_score, l2_score,
               active_chains, market_env="SIDEWAYS",
               lifecycle_data=None) -> List[StockDecision]:
    decisions = []
    for p in profiles:
        if p.verdict == "FAKE": continue
        d = StockDecision(code=p.code, name=p.name, chain=p.chain,
                          l0_pass=l0_pass, l4_score=p.total_score,
                          market_env=market_env)
        if not l0_pass:
            d.verdict="NO"; d.reason="Layer 0 系统性风险"
            d.action="空仓等待"; decisions.append(d); continue
        d.l1_pass = l1_score>=55
        d.l2_pass = l2_score>=50
        d.l3_pass = any(p.chain in c["name"] for c in active_chains)
        layers = sum([l0_pass, d.l1_pass, d.l2_pass, d.l3_pass, p.total_score>=75])
        timing_info = get_timing(p.code)
        d.timing_signal = timing_info["signal"]
        d.stop_loss_pct = get_stop_pct(p.code)

        # ── 产业链周期折扣 ──
        cycle_maturity = 50
        if lifecycle_data and p.chain in lifecycle_data:
            lc = lifecycle_data[p.chain]
            d.cycle_phase = lc.get("phase", "")
            d.cycle_maturity = lc.get("maturity", 50)
            d.cycle_remaining_months = lc.get("remaining_months", 0)
            cycle_maturity = d.cycle_maturity

        pos = calc_position(layers, market_env, d.stop_loss_pct, d.timing_signal,
                           cycle_maturity=cycle_maturity)
        d.actual_position_pct = pos["actual"]
        d.first_batch_pct     = pos["first"]
        d.cycle_discount = pos.get("cycle_ratio", 1.0)

        # ── 决策 ──
        cycle_note = ""
        if lifecycle_data and p.chain in lifecycle_data:
            lc = lifecycle_data[p.chain]
            if d.cycle_maturity >= 70:
                cycle_note = f"（周期成熟度{d.cycle_maturity}/100，仓位{d.cycle_discount:.0%}折）"
        if layers>=4 and d.timing_signal in ("BREAKOUT","PULLBACK","ACCUMULATION"):
            d.verdict = "GO"
            d.reason  = f"六层通过{layers}层，{timing_info['note']}{cycle_note}"
            d.action  = f"首批{d.first_batch_pct}%，止损-{d.stop_loss_pct}%无条件执行"
        elif layers>=3:
            d.verdict = "WAIT"
            d.reason  = f"通过{layers}层，{timing_info['note']}"
            d.action  = "加入观察名单，设价格提醒"
        else:
            d.verdict = "WAIT"
            d.reason  = f"通过{layers}层，信号不足"
            d.action  = "继续等待更多层确认"

        if lifecycle_data and p.chain in lifecycle_data:
            if d.verdict == "GO" and d.cycle_maturity >= 75:
                d.reason += " ⚠周期接近尾声，严格止损"

        decisions.append(d)
    return sorted(decisions,
                  key=lambda x:(x.verdict=="GO", x.actual_position_pct), reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 输出 & 报告
# ══════════════════════════════════════════════════════════════════════════════

SCOL = {"UP":"green","DOWN":"red","NEUTRAL":"dim","ALERT_UP":"bold green","ALERT_DOWN":"bold red"}
DCOL = {"BULLISH":"green","BEARISH":"red","NEUTRAL":"dim"}
VCOL = {"GO":"bold green","WAIT":"yellow","NO":"bold red"}
L0COL = {"GREEN":"green","YELLOW":"yellow","RED":"bold red"}

def print_header():
    console.print()
    console.print(Panel.fit(
        "[bold]六层市场分析决策系统 v1.0[/bold]\n"
        "[dim]L0风险 → L1实业 → L2期货 → L3产业链 → L4头部企业 → L5个股[/dim]",
        border_style="dim"))
    console.print(f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

def pprint_l0(st, env):
    c = L0COL.get(st.level,"white")
    body = (f"[{c}]● {st.level}[/{c}]  评分 {st.score}/100  "
            f"最大仓位 [bold]{st.max_position*100:.0f}%[/bold]\n"
            f"[dim]{st.note}[/dim]")
    if st.triggered:
        body += "\n" + "\n".join(f"  [yellow]⚠ {t}[/yellow]" for t in st.triggered)
    else:
        body += "\n  [green]✓ 无风险规则触发[/green]"
    console.print(Panel(body, title="[bold]L0 · 风险管理[/bold]", border_style=c))
    t = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
    t.add_column(style="dim",width=16); t.add_column(justify="right")
    for k,v in [("上证5日涨跌",f"{env.get('index_5d_chg','?')}%"),
                ("单日最大跌幅",f"{env.get('max_1d_drop','?')}%"),
                ("A股QVIX",str(env.get('qvix','N/A'))),
                ("美元/人民币",str(env.get('usdcny','N/A')))]:
        t.add_row(k,v)
    console.print(t)

def pprint_l1(r):
    all_inds = [i for grp in r["groups"].values() for i in grp]
    console.print(Panel(
        f"综合评分 [bold]{r['score']}/100[/bold]  "
        f"异动 [bold yellow]{len(r['alerts'])}[/bold yellow] 条  [dim]{r['ts']}[/dim]",
        title="[bold]L1 · 实业数据[/bold]", border_style="blue"))
    t = Table(box=box.SIMPLE_HEAVY, padding=(0,1))
    t.add_column("指标",width=20); t.add_column("数值",justify="right",width=14)
    t.add_column("同比",justify="right",width=8); t.add_column("偏离σ",justify="right",width=7)
    t.add_column("信号",width=13); t.add_column("来源",style="dim",width=12)
    for ind in all_inds:
        sc = SCOL.get(ind.signal,"white")
        t.add_row(ind.name, f"{ind.value} {ind.unit}",
                  f"{ind.yoy:+.1f}%" if ind.yoy else "-",
                  f"{ind.zscore:+.2f}" if ind.zscore is not None else "-",
                  f"[{sc}]{ind.signal}[/{sc}]", ind.source)
    console.print(t)
    for a in r["alerts"]:
        console.print(f"  [bold yellow]⚡ 异动：{a.name} → {a.note}[/bold yellow]")
    console.print()

def pprint_l2(r):
    console.print(Panel(
        f"综合评分 [bold]{r['score']}/100[/bold]  [dim]{r['ts']}[/dim]\n"
        f"[{'yellow' if '背离' in r['divergence'] else 'green'}]{r['divergence']}[/{'yellow' if '背离' in r['divergence'] else 'green'}]",
        title="[bold]L2 · 金融期货 & 国际信号[/bold]", border_style="magenta"))
    t = Table(box=box.SIMPLE_HEAVY, padding=(0,1))
    t.add_column("指标",width=20); t.add_column("数值",justify="right",width=12)
    t.add_column("变动",justify="right",width=9); t.add_column("方向",width=10)
    t.add_column("A股传导含义",width=36)
    for s in r["intl"]+r["futures"]:
        dc = DCOL.get(s.direction,"white")
        chg = f"{s.change:+.2f}" if s.change is not None else "-"
        t.add_row(s.name,f"{s.value} {s.unit}",chg,
                  f"[{dc}]{s.direction}[/{dc}]",s.a_share_impact[:36])
    console.print(t); console.print()

def pprint_l3(r):
    console.print(Panel(
        f"激活 [bold green]{len(r['active'])}[/bold green] 条  "
        f"待激活 [dim]{len(r['inactive'])}[/dim] 条",
        title="[bold]L3 · 产业链图谱[/bold]", border_style="cyan"))
    for c in r["top"]:
        col = "green" if c["strength"]>=70 else "yellow"
        console.print(f"  [{col}]▸ {c['name']}[/{col}]  节点：[bold]{c['node']}[/bold]  强度：{c['strength']}")
        console.print(f"    [dim]{c['window']}[/dim]")
    console.print()

def pprint_l45(decisions, capital):
    console.print(Panel("四道过滤器 + A股时机信号 + 仓位计算",
        title="[bold]L4+5 · 头部企业 & 个股决策[/bold]", border_style="green"))
    t = Table(box=box.SIMPLE_HEAVY, padding=(0,1), show_lines=True)
    t.add_column("代码",width=8); t.add_column("名称",width=10)
    t.add_column("产业链",width=12); t.add_column("L4",justify="center",width=5)
    t.add_column("时机",width=13); t.add_column("止损",justify="right",width=6)
    t.add_column("仓位%",justify="right",width=7); t.add_column("首批金额",justify="right",width=11)
    t.add_column("决策",width=6); t.add_column("行动建议",width=26)
    for d in decisions:
        vc = VCOL.get(d.verdict,"white")
        tc = {"BREAKOUT":"green","PULLBACK":"cyan","ACCUMULATION":"yellow","NONE":"dim"}.get(d.timing_signal,"dim")
        amt = capital*d.first_batch_pct/100
        t.add_row(d.code,d.name,d.chain[:12],str(d.l4_score),
                  f"[{tc}]{d.timing_signal}[/{tc}]",
                  f"{d.stop_loss_pct}%",f"{d.actual_position_pct:.1f}%",
                  f"¥{amt:,.0f}",f"[{vc}]{d.verdict}[/{vc}]",d.action[:26])
    console.print(t)
    go_list = [d for d in decisions if d.verdict=="GO"]
    if go_list:
        console.print("\n[bold green]✅ 可入场标的：[/bold green]")
        for d in go_list:
            amt = capital*d.first_batch_pct/100
            console.print(Panel(
                f"[bold]{d.name}[/bold]（{d.code}）· {d.chain}\n"
                f"原因：{d.reason}\n"
                f"[green]执行：{d.action}[/green]\n"
                f"首批金额：¥{amt:,.0f}（总资金{d.first_batch_pct:.1f}%）\n"
                f"止损：当前价下方{d.stop_loss_pct}%，触发无条件执行",
                border_style="green"))
    else:
        console.print("\n[yellow]当前无 GO 标的，所有标的处于观察状态。[/yellow]")


def save_txt(l0, l1, l2, l3, decisions, capital):
    import os
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"report_{ts}.txt")
    with open(path,"w",encoding="utf-8") as f:
        f.write(f"六层决策报告 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*60}\n")
        f.write(f"L0: {l0.level} 评分{l0.score} 最大仓位{l0.max_position*100:.0f}%\n")
        f.write(f"L1实业评分: {l1['score']}\n")
        f.write(f"L2期货评分: {l2['score']}\n{l2['divergence']}\n")
        f.write(f"激活产业链: {len(l3['active'])} 条\n\n个股决策:\n")
        for d in decisions:
            f.write(f"  [{d.verdict}] {d.name}({d.code}) "
                    f"仓位{d.actual_position_pct:.1f}% 首批¥{capital*d.first_batch_pct/100:,.0f}\n"
                    f"    {d.reason}\n    {d.action}\n")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="六层市场分析决策系统")
    parser.add_argument("--capital", type=float, default=1_000_000, help="总资金（元）")
    parser.add_argument("--env",     type=str,   default="AUTO",
                        choices=["AUTO","BULL","SIDEWAYS","BEAR"])
    parser.add_argument("--codes",   nargs="*",  help="指定股票代码，默认全库")
    args = parser.parse_args()

    print_header()

    console.print("[dim]L0 获取风险数据...[/dim]")
    l0_status, env_data = run_layer0()
    pprint_l0(l0_status, env_data)

    console.print("[dim]L1 获取实业数据（约30-60秒）...[/dim]")
    l1 = run_layer1()
    pprint_l1(l1)

    console.print("[dim]L2 获取金融期货数据...[/dim]")
    l2 = run_layer2(l1["score"])
    pprint_l2(l2)

    console.print("[dim]L3 匹配产业链...[/dim]")
    l3 = run_layer3()
    pprint_l3(l3)

    # ── 产业链周期分析 ──
    console.print("[dim]分析产业链周期...[/dim]")
    lifecycle_data = {}
    try:
        from chain_lifecycle import analyze_all_chains, cycle_to_dict
        active_names = [c["name"] for c in l3["active"]]
        strengths = {c["name"]: c["strength"] for c in l3["active"]}
        lifecycle = analyze_all_chains(active_names, strengths)
        lifecycle_data = {k: cycle_to_dict(v) for k, v in lifecycle.items()}
        for name, lc in lifecycle_data.items():
            console.print(f"  {name}: {lc['phase_name']}，进度{lc['progress_pct']:.0f}%，"
                          f"剩余约{lc['remaining_months']:.0f}个月，成熟度{lc['maturity']}/100")
    except Exception as e:
        console.print(f"  [yellow]周期分析失败: {e}[/yellow]")

    console.print("[dim]L4/L5 个股分析（获取行情数据）...[/dim]")
    l3_active_names = [c["name"] for c in l3["active"]]
    profiles = run_layer4(active_chains=l3_active_names)

    if args.env == "AUTO":
        market_env = ("BULL"    if l0_status.score>=75 and l1["score"]>=60 else
                      "BEAR"    if l0_status.level=="RED" else "SIDEWAYS")
    else:
        market_env = args.env

    decisions = run_layer5(
        profiles, l0_status.level!="RED",
        l1["score"], l2["score"], l3["active"], market_env,
        lifecycle_data=lifecycle_data,
    )
    pprint_l45(decisions, args.capital)

    path = save_txt(l0_status, l1, l2, l3, decisions, args.capital)
    console.print(f"\n[dim]报告已保存：{path}[/dim]")
    console.print("[bold]分析完成。[/bold]\n")


if __name__ == "__main__":
    main()
