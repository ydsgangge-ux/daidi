"""
ETF 分析模块 — 面向小投资者的 ETF 筛选与趋势分析

覆盖范围：
  - 宽基ETF（沪深300、中证500、创业板等）
  - 行业/主题ETF（半导体、新能源、医药、消费等）
  - 跨境ETF（纳斯达克、标普500、恒生科技等）
  - 商品ETF（黄金、原油等）

数据来源：akshare（东方财富）
"""
# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import warnings, time, sys, io, logging
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

_log = logging.getLogger(__name__)

from data_cache import load_kline as _load_kline_disk, save_kline as _save_kline_disk

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


ETF_WATCHLIST = {
    "宽基ETF": [
        {"code": "510300", "name": "沪深300ETF", "index": "沪深300", "desc": "A股核心资产基准"},
        {"code": "510500", "name": "中证500ETF", "index": "中证500", "desc": "中盘成长代表"},
        {"code": "159915", "name": "创业板ETF", "index": "创业板指", "desc": "高成长科技板块"},
        {"code": "588000", "name": "科创50ETF", "index": "科创50", "desc": "硬科技核心标的"},
        {"code": "512100", "name": "中证1000ETF", "index": "中证1000", "desc": "小盘股代表"},
        {"code": "159919", "name": "沪深300ETF(嘉实)", "index": "沪深300", "desc": "沪深300备选"},
    ],
    "行业/主题ETF": [
        {"code": "512480", "name": "半导体ETF", "index": "中华半导体", "desc": "国产替代+AI算力"},
        {"code": "515030", "name": "新能源车ETF", "index": "CS新能车", "desc": "电动车产业链"},
        {"code": "516160", "name": "新能源ETF", "index": "中证新能源", "desc": "风光储一体化"},
        {"code": "512010", "name": "医药ETF", "index": "中证医药", "desc": "创新药+医疗器械"},
        {"code": "159928", "name": "消费ETF", "index": "中证主要消费", "desc": "必选消费龙头"},
        {"code": "515880", "name": "通信ETF", "index": "中证全指通信", "desc": "5G+光通信"},
        {"code": "512660", "name": "军工ETF", "index": "中证军工", "desc": "国防+商业航天"},
        {"code": "159766", "name": "旅游ETF", "index": "中证旅游", "desc": "出行消费复苏"},
        {"code": "512200", "name": "房地产ETF", "index": "中证房地产", "desc": "地产开发+物业"},
        {"code": "515790", "name": "光伏ETF", "index": "中证光伏产业", "desc": "光伏组件+逆变器"},
        {"code": "159995", "name": "芯片ETF", "index": "国证芯片", "desc": "芯片设计+封测"},
        {"code": "562500", "name": "机器人ETF", "index": "中证机器人", "desc": "人形机器人+工业自动化"},
    ],
    "跨境ETF": [
        {"code": "513100", "name": "纳指ETF", "index": "纳斯达克100", "desc": "美股科技龙头"},
        {"code": "513500", "name": "标普500ETF", "index": "标普500", "desc": "美股大盘基准"},
        {"code": "513180", "name": "恒生科技ETF", "index": "恒生科技", "desc": "港股互联网龙头"},
        {"code": "513060", "name": "恒生医疗ETF", "index": "恒生医疗保健", "desc": "港股创新药"},
        {"code": "159920", "name": "恒生ETF", "index": "恒生指数", "desc": "港股核心资产"},
        {"code": "513030", "name": "德国ETF", "index": "德国DAX", "desc": "欧洲工业龙头"},
        {"code": "513520", "name": "日经ETF", "index": "日经225", "desc": "日本股市基准"},
    ],
    "商品ETF": [
        {"code": "518880", "name": "黄金ETF", "index": "Au99.99", "desc": "避险+抗通胀"},
        {"code": "518860", "name": "黄金基金ETF", "index": "上海金", "desc": "黄金备选"},
        {"code": "159985", "name": "豆粕ETF", "index": "大商所豆粕", "desc": "农产品对冲"},
    ],
}


@dataclass
class ETFInfo:
    code: str
    name: str
    category: str
    index_name: str
    desc: str
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    turnover_rate: float = 0.0
    pe: Optional[float] = None
    pb: Optional[float] = None
    total_mv: Optional[float] = None
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    trend_signal: str = "NEUTRAL"
    trend_score: int = 50
    rsi_14: Optional[float] = None
    vol_ratio: Optional[float] = None
    recent_5d_chg: Optional[float] = None
    recent_20d_chg: Optional[float] = None
    recent_60d_chg: Optional[float] = None
    advice: str = ""
    source: str = ""


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _fetch_etf_spot() -> Optional[pd.DataFrame]:
    for attempt in range(3):
        try:
            df = ak.fund_etf_spot_em()
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            _log.warning(f"ETF实时行情抓取失败（东方财富，第{attempt+1}次）: {e}")
    return None


def _fetch_etf_kline(code: str, days: int = 120) -> tuple:
    """获取ETF日K线，返回 (df, source)。优先当日磁盘缓存，东方财富→新浪备用。"""
    # 当日磁盘缓存（data_cache.load_kline 返回 (source, df)，注意顺序）
    csource, cdf = _load_kline_disk("etf", code)
    if cdf is not None:
        return cdf, (csource or "cache")

    # 方案A: 东方财富（列名为中文：日期/开盘/收盘/成交量...）
    for attempt in range(2):
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                     start_date="20240101", adjust="qfq")
            if df is not None and len(df) > 0:
                _save_kline_disk("etf", code, "em", df)
                return df, "em"
        except Exception as e:
            if attempt < 1:
                time.sleep(1)
            _log.warning(f"ETF日K抓取失败（东方财富 {code}，第{attempt+1}次）: {e}")

    # 方案B: 新浪备用源（列名为英文，统一改为中文与东财一致）
    try:
        prefix = "sh" if code.startswith("5") else "sz"
        df = ak.fund_etf_hist_sina(symbol=prefix + code)
        if df is not None and len(df) > 0:
            col_map = {"date": "日期", "open": "开盘", "close": "收盘",
                       "high": "最高", "low": "最低", "volume": "成交量",
                       "amount": "成交额"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "日期" in df.columns:
                df = df.sort_values("日期").reset_index(drop=True)
            _save_kline_disk("etf", code, "sina", df)
            return df, "sina"
    except Exception as e:
        _log.warning(f"ETF日K抓取失败（新浪备用 {code}）: {e}")

    return None, ""


def _calc_ma(series: pd.Series, n: int) -> Optional[float]:
    if len(series) < n:
        return None
    return round(float(series.tail(n).mean()), 4)


def _calc_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    if len(series) < period + 1:
        return None
    delta = series.diff().tail(period + 1)
    gain = delta.where(delta > 0, 0.0).mean()
    loss = (-delta.where(delta < 0, 0.0)).mean()
    if loss == 0:
        return 100.0
    rs = gain / loss
    return round(100 - 100 / (1 + rs), 1)


def _calc_trend(close: pd.Series) -> tuple:
    if len(close) < 20:
        return "NEUTRAL", 50

    ma5 = close.tail(5).mean()
    ma10 = close.tail(10).mean()
    ma20 = close.tail(20).mean()
    latest = float(close.iloc[-1])

    score = 50

    if latest > ma5:
        score += 8
    if latest > ma10:
        score += 8
    if latest > ma20:
        score += 8
    if ma5 > ma10:
        score += 6
    if ma10 > ma20:
        score += 6

    if len(close) >= 60:
        ma60 = close.tail(60).mean()
        if latest > ma60:
            score += 10
        if ma20 > ma60:
            score += 4

    rsi = _calc_rsi(close)
    if rsi is not None:
        if rsi > 70:
            score -= 10
        elif rsi > 60:
            score += 5
        elif rsi < 30:
            score += 10
        elif rsi < 40:
            score -= 5

    vol = None
    if len(close) >= 6:
        chg = close.pct_change().tail(5)
        vol = float(chg.std() * 100)
        if vol > 3:
            score -= 5

    score = max(0, min(100, score))

    if score >= 70:
        signal = "BULLISH"
    elif score >= 55:
        signal = "LEAN_BULL"
    elif score >= 45:
        signal = "NEUTRAL"
    elif score >= 30:
        signal = "LEAN_BEAR"
    else:
        signal = "BEARISH"

    return signal, score


def _generate_advice(etf: ETFInfo) -> str:
    parts = []
    if etf.trend_signal == "BULLISH":
        parts.append("趋势向上，可考虑分批建仓")
    elif etf.trend_signal == "LEAN_BULL":
        parts.append("偏多震荡，可小仓位布局")
    elif etf.trend_signal == "NEUTRAL":
        parts.append("方向不明，建议观望等待")
    elif etf.trend_signal == "LEAN_BEAR":
        parts.append("偏弱震荡，暂不介入")
    else:
        parts.append("趋势向下，回避或减仓")

    if etf.rsi_14 is not None:
        if etf.rsi_14 > 70:
            parts.append("RSI超买，注意回调风险")
        elif etf.rsi_14 < 30:
            parts.append("RSI超卖，可能存在反弹机会")

    if etf.recent_5d_chg is not None:
        if abs(etf.recent_5d_chg) > 5:
            parts.append(f"5日{'大涨' if etf.recent_5d_chg > 0 else '大跌'}{abs(etf.recent_5d_chg):.1f}%，注意波动风险")

    return "；".join(parts)


def analyze_etf(code: str, name: str, category: str, index_name: str,
                desc: str, spot_df: Optional[pd.DataFrame] = None) -> ETFInfo:
    etf = ETFInfo(
        code=code, name=name, category=category,
        index_name=index_name, desc=desc
    )

    if spot_df is not None and len(spot_df) > 0:
        try:
            spot_df.columns = [str(c).strip() for c in spot_df.columns]
            row = spot_df[spot_df["代码"] == code]
            if row is not None and len(row) > 0:
                r = row.iloc[0]
                etf.price = _safe_float(r.get("最新价", 0))
                etf.change_pct = _safe_float(r.get("涨跌幅", 0))
                etf.volume = _safe_float(r.get("成交量", 0))
                etf.amount = _safe_float(r.get("成交额", 0))
                etf.turnover_rate = _safe_float(r.get("换手率", 0))
        except Exception:
            pass

    kline_df, kline_source = _fetch_etf_kline(code)
    if kline_source:
        etf.source = kline_source
    if kline_df is not None and len(kline_df) > 5:
        try:
            kline_df.columns = [str(c).strip() for c in kline_df.columns]
            close_col = "收盘" if "收盘" in kline_df.columns else "close"
            close = pd.to_numeric(kline_df[close_col], errors="coerce").dropna()

            if len(close) >= 5:
                etf.ma5 = _calc_ma(close, 5)
                etf.ma10 = _calc_ma(close, 10) if len(close) >= 10 else None
                etf.ma20 = _calc_ma(close, 20) if len(close) >= 20 else None
                etf.ma60 = _calc_ma(close, 60) if len(close) >= 60 else None
                etf.rsi_14 = _calc_rsi(close)

                if len(close) >= 6:
                    vol_col = "成交量" if "成交量" in kline_df.columns else "volume"
                    vol = pd.to_numeric(kline_df[vol_col], errors="coerce").dropna()
                    if len(vol) >= 6:
                        avg5 = vol.tail(5).mean()
                        prev5 = vol.iloc[-6:-1].mean()
                        if prev5 and prev5 > 0:
                            etf.vol_ratio = round(float(avg5 / prev5), 2)

                if etf.price == 0 and len(close) > 0:
                    etf.price = round(float(close.iloc[-1]), 4)

                # 东财实时行情不可用时，用日K末两位收盘计算涨跌幅兜底
                if etf.change_pct == 0 and len(close) >= 2:
                    etf.change_pct = round(
                        (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100, 2)

                if len(close) >= 6:
                    etf.recent_5d_chg = round(
                        (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100, 2)
                if len(close) >= 21:
                    etf.recent_20d_chg = round(
                        (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100, 2)
                if len(close) >= 61:
                    etf.recent_60d_chg = round(
                        (float(close.iloc[-1]) / float(close.iloc[-61]) - 1) * 100, 2)

                etf.trend_signal, etf.trend_score = _calc_trend(close)
        except Exception:
            pass

    etf.advice = _generate_advice(etf)
    return etf


def run_etf_analysis() -> Dict:
    print("  ETF 实时行情...")
    spot_df = _fetch_etf_spot()

    all_etfs: List[ETFInfo] = []
    categories: Dict[str, List[dict]] = {}

    # 并行抓取各ETF日K（相互独立），小并发避免被数据源限流
    tasks = [(cat, item) for cat, lst in ETF_WATCHLIST.items() for item in lst]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda t: (t[0], analyze_etf(
                code=t[1]["code"], name=t[1]["name"], category=t[0],
                index_name=t[1]["index"], desc=t[1]["desc"], spot_df=spot_df
            )),
            tasks
        ))

    for cat_name, etf in results:
        all_etfs.append(etf)
        categories.setdefault(cat_name, []).append(_etf_to_dict(etf))

    bullish = [e for e in all_etfs if e.trend_signal in ("BULLISH", "LEAN_BULL")]
    bearish = [e for e in all_etfs if e.trend_signal in ("BEARISH", "LEAN_BEAR")]
    neutral = [e for e in all_etfs if e.trend_signal == "NEUTRAL"]

    summary = {
        "total": len(all_etfs),
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "top_bullish": sorted(bullish, key=lambda e: e.trend_score, reverse=True)[:5],
        "top_bearish": sorted(bearish, key=lambda e: e.trend_score)[:5],
    }

    return {
        "categories": categories,
        "all": [_etf_to_dict(e) for e in all_etfs],
        "summary": {
            "total": summary["total"],
            "bullish_count": summary["bullish_count"],
            "bearish_count": summary["bearish_count"],
            "neutral_count": summary["neutral_count"],
            "top_bullish": [_etf_to_dict(e) for e in summary["top_bullish"]],
            "top_bearish": [_etf_to_dict(e) for e in summary["top_bearish"]],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def _etf_to_dict(etf: ETFInfo) -> dict:
    return {
        "code": etf.code,
        "name": etf.name,
        "category": etf.category,
        "index_name": etf.index_name,
        "desc": etf.desc,
        "price": etf.price,
        "change_pct": round(etf.change_pct, 2),
        "volume": etf.volume,
        "amount": etf.amount,
        "turnover_rate": round(etf.turnover_rate, 2) if etf.turnover_rate else None,
        "pe": etf.pe,
        "pb": etf.pb,
        "total_mv": etf.total_mv,
        "ma5": etf.ma5,
        "ma10": etf.ma10,
        "ma20": etf.ma20,
        "ma60": etf.ma60,
        "trend_signal": etf.trend_signal,
        "trend_score": etf.trend_score,
        "rsi_14": etf.rsi_14,
        "vol_ratio": etf.vol_ratio,
        "recent_5d_chg": etf.recent_5d_chg,
        "recent_20d_chg": etf.recent_20d_chg,
        "recent_60d_chg": etf.recent_60d_chg,
        "advice": etf.advice,
        "source": etf.source,
    }
