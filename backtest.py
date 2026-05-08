"""
回测引擎 —— 验证六层系统的实际收益表现

核心能力：
  - 基于历史 K 线 + 财务数据模拟交易决策
  - 计算胜率/收益曲线/最大回撤/夏普率
  - 与沪深 300 基准对比
  - 生成可交互的 Web 报告

使用方法：
  python backtest.py --stock 300394 --start 2024-01-01 --end 2025-12-31 --capital 1000000
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json, os, sys
from pathlib import Path

# ── 导入现有分析模块 ──
from layer45_stocks import (
    _get_kline_cached, _parse_pct, fetch_financial_data,
    _get_current_price, get_support_level, calculate_position,
    score_dynamic_company,
)
from strategy_config import (
    KLINE_START_DATE, DEFAULT_CAPITAL, ENV_SINGLE_CAP,
    L4_PASS_THRESHOLD, L1_PASS_THRESHOLD, L2_PASS_THRESHOLD,
)


# ═════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """一笔完整交易记录"""
    code: str
    name: str
    entry_date: str
    entry_price: float
    entry_shares: int          # 手数
    entry_amount: float        # 买入金额
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_amount: Optional[float] = None
    stop_loss_price: float = 0
    position_pct: float = 0    # 仓位 %
    return_pct: Optional[float] = None  # 收益率
    pnl: Optional[float] = None
    reason: str = ""


@dataclass
class BacktestMetrics:
    """回测绩效指标"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    win_trades: int = 0
    sharpe_ratio: float = 0.0
    benchmark_return: float = 0.0
    benchmark_annual: float = 0.0
    alpha: Optional[float] = None
    beta: Optional[float] = None
    avg_hold_days: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 技术信号模拟（用于回测中替代 LLM 判定）
# ═════════════════════════════════════════════════════════════════════════════

def calc_technical_signal(df: pd.DataFrame, idx: int) -> str:
    """
    基于历史 K 线模拟时序信号（简化版，用于回测）。
    与 layer45 中的技术指标逻辑保持一致。
    """
    if idx < 30:
        return "NONE"
    try:
        close = df["收盘"].values
        high = df["最高"].values
        low = df["最低"].values

        # MACD
        ema12 = pd.Series(close[:idx+1]).ewm(span=12).mean().values
        ema26 = pd.Series(close[:idx+1]).ewm(span=26).mean().values
        macd = ema12[-1] - ema26[-1]
        signal = pd.Series(macd).ewm(span=9).mean().values[-1] if idx > 33 else 0
        macd_hist = macd - signal

        # RSI(14)
        if idx >= 14:
            deltas = np.diff(close[idx-14:idx+1])
            gains = deltas[deltas > 0].sum() / 14 if len(deltas[deltas > 0]) > 0 else 0
            losses = -deltas[deltas < 0].sum() / 14 if len(deltas[deltas < 0]) > 0 else 0.001
            rsi = 100 - 100 / (1 + gains / losses)
        else:
            rsi = 50

        # BOLL 突破
        ma20 = np.mean(close[idx-19:idx+1]) if idx >= 19 else close[idx]
        std20 = np.std(close[idx-19:idx+1]) if idx >= 19 else 0
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20

        # 量能
        if idx >= 5:
            vol = df["成交量"].values
            vol_ratio = vol[idx] / max(np.mean(vol[idx-5:idx]), 0.1)
        else:
            vol_ratio = 1

        # ── 信号判定 ──
        # BREAKOUT: 价格突破 BOLL 上轨 + 放量 + MACD > 0
        if (close[idx] > upper and vol_ratio > 1.3
                and macd_hist > 0 and 20 < rsi < 80):
            return "BREAKOUT"

        # PULLBACK: 价格跌到 BOLL 下轨附近 + RSI 超卖 + 缩量
        if (close[idx] < ma20 * 0.95 and rsi < 40
                and vol_ratio < 1.0 and macd_hist > -2):
            return "PULLBACK"

        # ACCUMULATION: MACD 水下金叉 + 温和放量
        if (macd_hist > 0 and macd_hist < 2
                and 1.0 < vol_ratio < 1.5 and rsi < 60 and rsi > 30):
            return "ACCUMULATION"

        return "NONE"
    except Exception:
        return "NONE"


def calc_index_risk(date_str: str) -> tuple:
    """
    获取历史日期的 L0 风险等级。
    返回 (level, score)，level 取 "GREEN"/"YELLOW"/"RED"
    """
    try:
        end = datetime.strptime(date_str, "%Y-%m-%d")
        start = end - timedelta(days=30)
        df = ak.stock_zh_index_hist_em(symbol="000001", start_date=start.strftime("%Y%m%d"),
                                       end_date=end.strftime("%Y%m%d"))
        if df is not None and len(df) > 5:
            closes = df["收盘"].values
            close_t = closes[-1]
            # 5日跌幅
            chg_5d = (close_t - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
            # 单日最大
            max_drop = min(np.diff(closes) / closes[:-1] * 100) if len(closes) > 1 else 0

            score = 100
            triggered = []
            if max_drop < -3:
                score -= 40
                triggered.append("max_drop")
            if chg_5d < -5:
                score -= 25
                triggered.append("chg_5d")
            if score >= 70:
                level = "GREEN"
                max_pos = 0.60 if chg_5d > 2 else 0.30
            elif score >= 40:
                level = "YELLOW"
                max_pos = 0.10
            else:
                level = "RED"
                max_pos = 0.0
            return level, max_pos, score
    except Exception:
        pass
    return "SIDEWAYS", 0.30, 50


def load_csi300_benchmark(start: str, end: str) -> pd.DataFrame:
    """获取沪深 300 历史走势作为基准"""
    try:
        df = ak.stock_zh_index_hist_em(symbol="000300",
                                       start_date=start.replace("-", ""),
                                       end_date=end.replace("-", ""))
        if df is not None and len(df) > 0:
            df = df.rename(columns={"日期": "date", "收盘": "close"})
            df["date"] = pd.to_datetime(df["date"])
            df["return"] = df["close"].pct_change()
            df["cum_return"] = (1 + df["return"]).cumprod()
            return df
    except Exception:
        pass
    return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# 回测引擎
# ═════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    六层回测引擎

    在历史数据上模拟交易决策，计算绩效指标。
    由于 L1/L2/L3 需要实时宏观数据，回测中使用简化信号替代：
      - L0: 基于历史指数数据计算风险等级
      - L1: 默认通过（回测模式简化）
      - L2: 默认通过
      - L3: 基于技术信号判断产业链活跃度（BREAKOUT/PULLBACK=活跃，ACCUMULATION=关注，NONE=不活跃）
      - L4: 基于历史财务数据评分（按季度更新）
      - L5: 完整的仓位计算
    """

    def __init__(self, capital: float = 1_000_000,
                 start_date: str = "2024-01-01",
                 end_date: str = None,
                 max_hold_days: int = 60,
                 stop_loss_atr: float = 2.0):
        self.initial_capital = capital
        self.capital = capital
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.max_hold_days = max_hold_days
        self.stop_loss_atr = stop_loss_atr

        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []
        self.current_position: Optional[Trade] = None
        self.entry_date: Optional[str] = None
        self.entry_price: Optional[float] = None

        # 缓存财务评分（按季度刷新）
        self._fin_score_cache: Dict[str, tuple] = {}

    def _get_financial_score(self, code: str, current_date: str) -> tuple:
        """
        获取历史日期的财务评分（按季度缓存）
        返回 (score_passed: bool, score: float)
        """
        # 季度缓存键
        q_key = current_date[:7]  # "2024-01"
        cache_key = f"{code}_{q_key}"

        if cache_key in self._fin_score_cache:
            return self._fin_score_cache[cache_key]

        try:
            # 用同花顺财务摘要获取当时可用的最新一期数据
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
            if df is not None and len(df) > 0:
                # 取当前日期之前的最近一期
                df["报告期"] = pd.to_datetime(df["报告期"])
                df = df[df["报告期"] <= pd.to_datetime(current_date)]
                if len(df) > 0:
                    row = df.sort_values("报告期", ascending=False).iloc[0]

                    # 简化的财务评分（参考 score_dynamic_company）
                    gross_margin = _parse_pct(row.get("销售毛利率")) or 0
                    net_margin = _parse_pct(row.get("销售净利率")) or 0
                    roe = _parse_pct(row.get("净资产收益率")) or 0
                    rev_yoy = _parse_pct(row.get("营业总收入同比增长率")) or 0
                    profit_yoy = _parse_pct(row.get("净利润同比增长率")) or 0
                    debt_ratio = _parse_pct(row.get("资产负债率")) or 0

                    score = 0
                    # 毛利率评分
                    if gross_margin >= 50: score += 25
                    elif gross_margin >= 35: score += 23
                    elif gross_margin >= 25: score += 20
                    elif gross_margin >= 15: score += 15
                    else: score += 8

                    # 营收增长
                    if rev_yoy > 30: score += 25
                    elif rev_yoy > 15: score += 22
                    elif rev_yoy > 0: score += 18
                    elif rev_yoy > -10: score += 12
                    else: score += 6

                    # ROE
                    if roe >= 20: score += 2
                    elif roe >= 15: score += 1

                    # 负债率
                    if debt_ratio < 30: score += 22
                    elif debt_ratio < 50: score += 20
                    elif debt_ratio < 70: score += 15
                    else: score += 8

                    # 净利增长
                    if profit_yoy > 30: score += 25
                    elif profit_yoy > 15: score += 23
                    elif profit_yoy > 0: score += 20
                    elif profit_yoy > -20: score += 15
                    else: score += 10

                    passed = score >= L4_PASS_THRESHOLD
                    result = (passed, score)
                    self._fin_score_cache[cache_key] = result
                    return result
        except Exception:
            pass

        result = (False, 0)
        self._fin_score_cache[cache_key] = result
        return result

    def run_stock(self, code: str, name: str = "") -> BacktestMetrics:
        """
        对单只股票运行回测
        """
        print(f"  回测 {code} {name} ({self.start_date} ~ {self.end_date})")

        # 1. 获取历史 K 线
        df = _get_kline_cached(code)
        if df is None or len(df) == 0:
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                        start_date=self.start_date.replace("-", ""),
                                        end_date=self.end_date.replace("-", ""),
                                        adjust="qfq")
            except Exception:
                print(f"    [WARN] 无法获取 {code} 历史数据")
                return BacktestMetrics()

        if df is None or len(df) == 0:
            return BacktestMetrics()

        df = df.reset_index(drop=True)
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期").reset_index(drop=True)

        # 2. 加载基准
        benchmark = load_csi300_benchmark(self.start_date, self.end_date)
        bench_start = benchmark["close"].iloc[0] if len(benchmark) > 0 else 1
        bench_end = benchmark["close"].iloc[-1] if len(benchmark) > 0 else 1

        # 3. 逐日模拟
        daily_values = []
        cash = self.initial_capital
        shares = 0
        entry_price = 0
        entry_date = None
        position_pct = 0
        stop_loss = 0
        current_drawdown = 0
        peak = self.initial_capital

        for i in range(len(df)):
            row = df.iloc[i]
            date_str = row["日期"].strftime("%Y-%m-%d") if hasattr(row["日期"], "strftime") else str(row["日期"])
            close = float(pd.to_numeric(row["收盘"], errors="coerce"))
            if pd.isna(close) or close <= 0:
                continue

            # ── 开仓检查（未持仓时） ──
            if shares == 0:
                # 计算风险等级
                level, max_pos_ratio, risk_score = calc_index_risk(date_str)

                # 计算技术信号
                tech_signal = calc_technical_signal(df, i)

                # 财务评分（按季度）
                fin_passed, fin_score = self._get_financial_score(code, date_str)

                # L3 信号（技术信号作为链活跃度代理）
                l3_active = tech_signal in ("BREAKOUT", "PULLBACK", "ACCUMULATION")

                # 简化版层数计算
                l0_pass = level != "RED"
                l1_pass = True   # 回测简化
                l2_pass = True   # 回测简化
                l3_pass = l3_active
                l4_pass = fin_passed
                layers_passed = sum([l0_pass, l1_pass, l2_pass, l3_pass, l4_pass])

                # 市场环境
                market_env = "BULL" if level == "GREEN" and risk_score >= 80 else (
                    "BEAR" if level == "RED" else "SIDEWAYS")

                # 计算止损
                try:
                    support = close * 0.95  # 简化止损：5%
                    sl_pct = round((close - support) / close * 100, 1)
                except Exception:
                    sl_pct = 8.0

                # 周期成熟度（简化：取中间值）
                cycle_mat = 50

                # 计算仓位
                pos = calculate_position(layers_passed, market_env, sl_pct,
                                         tech_signal, cycle_mat)
                target_pct = pos["actual_pct"]

                # 判定是否入场：L0 通过 + L3/L4 至少一个通过 + 有技术信号
                should_enter = (layers_passed >= 3
                                and tech_signal != "NONE"
                                and level != "RED"
                                and fin_passed)

                if should_enter:
                    # 计算手数
                    amt = cash * target_pct / 100
                    lots = int(amt / (close * 100))
                    if lots >= 1:
                        buy_amt = lots * 100 * close
                        cash -= buy_amt
                        shares = lots * 100
                        entry_price = close
                        entry_date = date_str
                        stop_loss = close * (1 - sl_pct / 100)
                        position_pct = target_pct

                        self.trades.append(Trade(
                            code=code, name=name,
                            entry_date=date_str, entry_price=close,
                            entry_shares=lots, entry_amount=buy_amt,
                            stop_loss_price=stop_loss,
                            position_pct=target_pct,
                            reason=f"Layers={layers_passed}, Signal={tech_signal}"
                        ))
                        self.current_position = self.trades[-1]

            # ── 持仓管理 ──
            else:
                # 检查止损
                if close <= stop_loss:
                    exit_amt = shares * close
                    pnl = exit_amt - (shares * entry_price)
                    ret = pnl / (shares * entry_price) * 100
                    cash += exit_amt
                    shares = 0

                    if self.current_position:
                        self.current_position.exit_date = date_str
                        self.current_position.exit_price = close
                        self.current_position.exit_amount = exit_amt
                        self.current_position.return_pct = round(ret, 2)
                        self.current_position.pnl = round(pnl, 2)
                        self.current_position.reason += " | 止损"
                    continue

                # 检查最大持有天数
                if entry_date:
                    hold_days = (pd.to_datetime(date_str) - pd.to_datetime(entry_date)).days
                    if hold_days > self.max_hold_days:
                        exit_amt = shares * close
                        pnl = exit_amt - (shares * entry_price)
                        ret = pnl / (shares * entry_price) * 100
                        cash += exit_amt
                        shares = 0

                        if self.current_position:
                            self.current_position.exit_date = date_str
                            self.current_position.exit_price = close
                            self.current_position.exit_amount = exit_amt
                            self.current_position.return_pct = round(ret, 2)
                            self.current_position.pnl = round(pnl, 2)
                            self.current_position.reason += f" | 持仓{hold_days}天自动平仓"
                        continue

                # 反转向下信号出场
                tech_signal = calc_technical_signal(df, i)
                if tech_signal == "NONE" and close < entry_price * 0.98:
                    exit_amt = shares * close
                    pnl = exit_amt - (shares * entry_price)
                    ret = pnl / (shares * entry_price) * 100
                    cash += exit_amt
                    shares = 0

                    if self.current_position:
                        self.current_position.exit_date = date_str
                        self.current_position.exit_price = close
                        self.current_position.exit_amount = exit_amt
                        self.current_position.return_pct = round(ret, 2)
                        self.current_position.pnl = round(pnl, 2)
                        self.current_position.reason += " | 信号消失"
                    continue

            # ── 每日净值 ──
            nav = cash + shares * close
            if nav > peak:
                peak = nav
            dd = (nav - peak) / peak * 100 if peak > 0 else 0

            # 基准净值
            bench_nav = None
            if len(benchmark) > 0:
                bench_row = benchmark[benchmark["date"] == row["日期"]]
                if len(bench_row) > 0:
                    bench_nav = self.initial_capital * bench_row["cum_return"].values[0]

            daily_values.append({
                "date": date_str,
                "nav": round(nav, 2),
                "cash": round(cash, 2),
                "position": round(shares * close, 2),
                "drawdown": round(dd, 2),
                "benchmark": round(bench_nav, 2) if bench_nav else None,
            })

        # 最后一笔：如果仍在持仓，以收盘价平仓
        if shares > 0 and len(df) > 0:
            last_close = float(pd.to_numeric(df["收盘"].iloc[-1], errors="coerce"))
            exit_amt = shares * last_close
            pnl = exit_amt - (shares * entry_price)
            ret = pnl / (shares * entry_price) * 100
            cash += exit_amt

            if self.current_position:
                last_date = df["日期"].iloc[-1].strftime("%Y-%m-%d") if hasattr(df["日期"].iloc[-1], "strftime") else ""
                self.current_position.exit_date = last_date
                self.current_position.exit_price = last_close
                self.current_position.exit_amount = exit_amt
                self.current_position.return_pct = round(ret, 2)
                self.current_position.pnl = round(pnl, 2)
                self.current_position.reason += " | 回测期末平仓"

            nav = cash
            daily_values[-1]["nav"] = round(nav, 2)
            daily_values[-1]["cash"] = round(cash, 2)
            daily_values[-1]["position"] = 0

        self.equity_curve = daily_values

        # 4. 计算绩效指标
        return self._calc_metrics(daily_values, benchmark)

    def _calc_metrics(self, daily_values: List[dict],
                      benchmark: pd.DataFrame) -> BacktestMetrics:
        """计算绩效指标"""
        metrics = BacktestMetrics()

        if len(daily_values) < 2:
            return metrics

        start_nav = daily_values[0]["nav"]
        end_nav = daily_values[-1]["nav"]
        total_days = (pd.to_datetime(daily_values[-1]["date"]) -
                      pd.to_datetime(daily_values[0]["date"])).days
        years = max(total_days / 365, 0.1)

        # 总收益
        metrics.total_return = round((end_nav / start_nav - 1) * 100, 2)
        metrics.annual_return = round(((end_nav / start_nav) ** (1 / years) - 1) * 100, 2)

        # 最大回撤
        peak = start_nav
        for d in daily_values:
            if d["nav"] > peak:
                peak = d["nav"]
            dd = (d["nav"] - peak) / peak * 100
            if dd < metrics.max_drawdown:
                metrics.max_drawdown = round(dd, 2)

        # 交易统计
        closed = [t for t in self.trades if t.exit_date is not None]
        metrics.total_trades = len(closed)
        metrics.win_trades = sum(1 for t in closed if t.return_pct and t.return_pct > 0)
        metrics.win_rate = round(metrics.win_trades / metrics.total_trades * 100, 1) if metrics.total_trades > 0 else 0

        wins = [t.return_pct for t in closed if t.return_pct and t.return_pct > 0]
        losses = [t.return_pct for t in closed if t.return_pct and t.return_pct <= 0]
        metrics.avg_win_pct = round(np.mean(wins), 2) if wins else 0
        metrics.avg_loss_pct = round(np.mean(losses), 2) if losses else 0

        # 平均持仓天数
        hold_days = []
        for t in closed:
            if t.entry_date and t.exit_date:
                d = (pd.to_datetime(t.exit_date) - pd.to_datetime(t.entry_date)).days
                hold_days.append(d)
        metrics.avg_hold_days = round(np.mean(hold_days), 1) if hold_days else 0

        # 夏普比率（年化）
        daily_returns = []
        for i in range(1, len(daily_values)):
            if daily_values[i - 1]["nav"] > 0:
                r = (daily_values[i]["nav"] / daily_values[i - 1]["nav"]) - 1
                daily_returns.append(r)
        if len(daily_returns) > 10:
            avg_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            if std_ret > 0:
                rf_daily = 0.02 / 252  # 2% 无风险利率
                metrics.sharpe_ratio = round(
                    (avg_ret - rf_daily) / std_ret * np.sqrt(252), 2
                )

        # 基准收益
        if len(benchmark) > 1:
            bench_start = benchmark["close"].iloc[0]
            bench_end = benchmark["close"].iloc[-1]
            metrics.benchmark_return = round((bench_end / bench_start - 1) * 100, 2)
            metrics.benchmark_annual = round(
                ((bench_end / bench_start) ** (1 / years) - 1) * 100, 2
            )

        return metrics

    def export_report(self, stock_code: str, stock_name: str) -> dict:
        """导出回测报告（JSON 格式）"""
        metrics = self._calc_metrics(self.equity_curve,
                                     load_csi300_benchmark(self.start_date, self.end_date))

        return {
            "meta": {
                "stock": f"{stock_code} {stock_name}",
                "period": f"{self.start_date} ~ {self.end_date}",
                "initial_capital": self.initial_capital,
            },
            "metrics": {
                "total_return": metrics.total_return,
                "annual_return": metrics.annual_return,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "sharpe_ratio": metrics.sharpe_ratio,
                "total_trades": metrics.total_trades,
                "win_trades": metrics.win_trades,
                "avg_hold_days": metrics.avg_hold_days,
                "avg_win_pct": metrics.avg_win_pct,
                "avg_loss_pct": metrics.avg_loss_pct,
                "benchmark_return": metrics.benchmark_return,
                "benchmark_annual": metrics.benchmark_annual,
                "excess_return": round(metrics.total_return - metrics.benchmark_return, 2),
            },
            "trades": [
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date or "N/A",
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price or 0,
                    "shares": t.entry_shares,
                    "amount": round(t.entry_amount, 0),
                    "return_pct": t.return_pct,
                    "pnl": t.pnl,
                    "reason": t.reason,
                }
                for t in self.trades
            ],
            "equity_curve": self.equity_curve,
        }


# ═════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="待敌回测引擎")
    parser.add_argument("--stock", default="300394", help="股票代码")
    parser.add_argument("--name", default="", help="股票名称")
    parser.add_argument("--start", default="2024-01-01", help="回测开始日期")
    parser.add_argument("--end", default=None, help="回测结束日期(默认今天)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--output", type=str, default=None, help="输出JSON路径")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  待敌量化回测")
    print(f"{'='*60}")
    print(f"  股票: {args.stock} {args.name or '(自动获取)'}")
    print(f"  区间: {args.start} ~ {args.end or '今天'}")
    print(f"  本金: ¥{args.capital:,.0f}")
    print(f"{'='*60}\n")

    engine = BacktestEngine(capital=args.capital, start_date=args.start, end_date=args.end)
    metrics = engine.run_stock(args.stock, args.name)

    report = engine.export_report(args.stock, args.name)

    print(f"\n{'='*60}")
    print(f"  回测结果")
    print(f"{'='*60}")
    m = report["metrics"]
    print(f"  总收益率:       {m['total_return']:+.2f}%")
    print(f"  年化收益率:     {m['annual_return']:+.2f}%")
    print(f"  最大回撤:       {m['max_drawdown']:.2f}%")
    print(f"  夏普比率:       {m['sharpe_ratio']}")
    print(f"  胜率:           {m['win_rate']}% ({m['win_trades']}/{m['total_trades']})")
    print(f"  平均持仓天数:   {m['avg_hold_days']}天")
    print(f"  平均盈利:       {m['avg_win_pct']:+.2f}%")
    print(f"  平均亏损:       {m['avg_loss_pct']:.2f}%")
    print(f"  ─────────────────────────────")
    print(f"  沪深300收益:    {m['benchmark_return']:+.2f}%")
    print(f"  超额收益:       {m['excess_return']:+.2f}%")
    print(f"{'='*60}")

    # 输出 JSON
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(__file__), "web", "backtest_report.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {output_path}")
    print(f"  交易记录: {m['total_trades']} 笔")


if __name__ == "__main__":
    main()
