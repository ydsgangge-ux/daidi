"""
JSON 导出器
把六层分析结果序列化成 dashboard.json，供前端面板读取
用法：python export_json.py  → 生成 web/dashboard.json
"""
import json, sys, os, io, warnings, traceback, logging
from datetime import datetime
warnings.filterwarnings("ignore")

# 初始化日志（异常详情写入日志文件，不输出到终端）
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_log_dir, "error.log"),
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 使用脚本所在目录，不硬编码路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer0_risk     import run_layer0
from layer1_industry import run_layer1
from layer2_futures  import run_layer2
from layer3_chains   import run_layer3
from layer45_stocks  import run_layer4, run_layer5
from trend_judge     import llm_trend_judge, trend_to_dict
from llm_client      import is_available
from international_signals import run_intl_signals_collection


def ind_to_dict(ind) -> dict:
    return {
        "name":   ind.name,
        "value":  ind.value,
        "unit":   ind.unit,
        "yoy":    ind.yoy,
        "zscore": ind.zscore,
        "signal": ind.signal,
        "note":   ind.note,
        "source": ind.source,
    }


def sig_to_dict(s) -> dict:
    return {
        "name":      s.name,
        "value":     s.value,
        "unit":      s.unit,
        "change":    s.change,
        "direction": s.direction,
        "impact":    s.a_share_impact,
        "source":    s.source,
    }


def chain_to_dict(c) -> dict:
    return {
        "name":       c.name,
        "desc":       c.description,
        "strength":   c.signal_strength,
        "node":       c.current_node,
        "window":     c.window,
        "reasoning":  getattr(c, 'reasoning', ''),
        "nodes": [
            {
                "stage":      n.stage,
                "name":       n.name,
                "detail":     n.detail,
                "lead":       n.lead_weeks,
                "stocks":     n.a_stocks,
                "active":     n.active,
            }
            for n in c.nodes
        ],
    }


def _extract_kline(code: str, days: int = 90) -> list:
    """提取最近N日K线数据（供前端画图）"""
    kline = []
    try:
        from layer45_stocks import _get_kline_cached
        df = _get_kline_cached(code)
        if df is not None and len(df) > 0:
            tail = df.tail(days)
            for _, row in tail.iterrows():
                kline.append({
                    "date": str(row.get("日期", "")),
                    "open": round(float(pd.to_numeric(row.get("开盘", 0), errors="coerce") or 0), 2),
                    "close": round(float(pd.to_numeric(row.get("收盘", 0), errors="coerce") or 0), 2),
                    "high": round(float(pd.to_numeric(row.get("最高", 0), errors="coerce") or 0), 2),
                    "low": round(float(pd.to_numeric(row.get("最低", 0), errors="coerce") or 0), 2),
                    "volume": round(float(pd.to_numeric(row.get("成交量", 0), errors="coerce") or 0), 0),
                })
    except Exception:
        pass
    return kline


def profile_to_dict(p) -> dict:
    return {
        "code":    p.code,
        "name":    p.name,
        "chain":   p.chain,
        "node":    p.node,
        "scores": {
            "revenue": p.revenue_score,
            "client":  p.client_score,
            "margin":  p.margin_score,
            "capex":   p.capex_score,
            "total":   p.total_score,
        },
        "green_flags": p.green_flags,
        "red_flags":   p.red_flags,
        "verdict":     p.verdict,
        "note":        p.note,
        "financials": {
            "pe":        p.pe,
            "pb":        p.pb,
            "mktcap":    p.mktcap,
            "gross_margin": p.gross_margin,
            "net_margin": p.net_margin,
            "roe":       p.roe,
            "rev_yoy":   p.rev_yoy,
            "profit_yoy": p.profit_yoy,
            "deduct_yoy": p.net_profit_deducted_yoy,
            "debt_ratio": p.debt_ratio,
        },
        "tech_summary":    getattr(p, "tech_summary", ""),
        "tech_overall":    getattr(p, "tech_overall", ""),
        "deep_financial":  getattr(p, "deep_financial", ""),
        "data_source": p.data_source,
        "kline":       _extract_kline(p.code),
    }


def decision_to_dict(d, capital: float) -> dict:
    return {
        "code":    d.code,
        "name":    d.name,
        "chain":   d.chain,
        "layers": {
            "l0": d.l0_pass,
            "l1": d.l1_pass,
            "l2": d.l2_pass,
            "l3": d.l3_pass,
            "l4": d.l4_score,
        },
        "timing":          d.timing_signal,
        "stop_loss_pct":   d.stop_loss_pct,
        "position_pct":    d.actual_position_pct,
        "first_batch_pct": d.first_batch_pct,
        "first_batch_amt": round(capital * d.first_batch_pct / 100, 0),
        "verdict":         d.verdict,
        "reason":          d.reason,
        "action":          d.action,
        "tech_summary":    getattr(d, "tech_summary", ""),
        "tech_overall":    getattr(d, "tech_overall", ""),
        "deep_financial":  getattr(d, "deep_financial", ""),
        "kline":           _extract_kline(d.code),
    }


def run_export(capital: float = 1_000_000,
               market_env: str = "AUTO",
               output_path: str = None):
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "web", "dashboard.json")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始六层分析...")

    if not is_available():
        print("  [WARN] DeepSeek API Key 未配置，趋势判定和产业链分析将使用规则兜底")
        print("         配置方法：在项目根目录创建 .env 文件，写入 DEEPSEEK_API_KEY=sk-xxx")
    else:
        print("  DeepSeek API 已就绪")

    try:
        # L0
        print("  Layer 0 风险管理...")
        l0_status, env_data = run_layer0()
    except Exception as e:
        print(f"  [ERROR] Layer 0 失败: {e}（详情见 logs/error.log）")
        _log.exception("Layer 0 失败")
        l0_status = type('RiskStatus', (), {'level':'RED','score':0,'triggered':['系统错误'],'max_position':0.0,'note':str(e)})()
        env_data = {}

    try:
        # L1
        print("  Layer 1 实业数据（约30-60秒）...")
        l1 = run_layer1()
    except Exception as e:
        print(f"  [ERROR] Layer 1 失败: {e}（详情见 logs/error.log）")
        _log.exception("Layer 1 失败")
        from layer1_industry import Layer1Result
        l1 = Layer1Result(score=50, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))

    try:
        # L2
        print("  Layer 2 金融期货...")
        l2 = run_layer2(l1_score=l1.score)
    except Exception as e:
        print(f"  [ERROR] Layer 2 失败: {e}（详情见 logs/error.log）")
        _log.exception("Layer 2 失败")
        from layer2_futures import Layer2Result
        l2 = Layer2Result(score=50, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"), divergence=["数据获取失败"])

    # ── 趋势判定（L0+L1+L2 综合分析） ──
    try:
        print("  趋势判定（L0+L1+L2 综合分析）...")
        trend = llm_trend_judge(l0_status, l1, l2, env_data)
        print(f"  → 趋势判定：{trend.verdict} | 方向：{trend.trend} | 置信度：{trend.confidence}% | 来源：{trend.source}")
    except Exception as e:
        print(f"  [ERROR] 趋势判定失败: {e}（详情见 logs/error.log）")
        _log.exception("趋势判定失败")
        from trend_judge import TrendJudgment
        trend = TrendJudgment(
            verdict="WAIT", trend="SIDEWAYS", confidence=50, l0_verdict="PASS",
            reasoning="趋势判定异常，默认WAIT", key_signals=[], risks=[str(e)],
            opportunities=[], action="等待数据恢复", score=50, source="ERROR",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

    # ── 国际信号源采集 ──
    try:
        print("  国际信号源采集（新闻+费城半导体指数）...")
        intl_sig_result = run_intl_signals_collection(days=14)
        intl_signals = intl_sig_result.signals
        print(f"  → {intl_sig_result.raw_news_count} 条新闻，"
              f"{intl_sig_result.llm_extracted} 条结构化信号")
    except Exception as e:
        print(f"  [WARN] 国际信号采集失败: {e}")
        intl_signals = []
        intl_sig_result = type('R', (), {
            'signals': [], 'sox_index': None,
            'raw_news_count': 0, 'llm_extracted': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        })()

    # ── L3（LLM 动态产业链匹配） ──

    try:
        # L3
        print("  Layer 3 产业链动态匹配...")
        l3 = run_layer3(l1, l2, intl_signals)
        print(f"  → 激活产业链 {len(l3.active_chains)} 条 | 来源：{l3.source}")
    except Exception as e:
        print(f"  [ERROR] Layer 3 失败: {e}（详情见 logs/error.log）")
        _log.exception("Layer 3 失败")
        from layer3_chains import Layer3Result
        l3 = Layer3Result()

    l3_active_names = [c.name for c in l3.active_chains]

    try:
        # L4 + L5（将激活产业链传入，动态抓取成分股）
        print("  Layer 4/5 个股分析...")
        profiles = run_layer4(active_chains=l3_active_names)

        if market_env == "AUTO":
            env = ("BULL"    if l0_status.score >= 75 and l1.score >= 60 else
                   "BEAR"    if l0_status.level == "RED" else "SIDEWAYS")
        else:
            env = market_env

        decisions = run_layer5(
            profiles, l0_status.level != "RED",
            l1.score, l2.score, l3_active_names, env
        )
    except Exception as e:
        print(f"  [ERROR] Layer 4/5 失败: {e}（详情见 logs/error.log）")
        _log.exception("Layer 4/5 失败")
        profiles = []
        decisions = []

    # ── 组装 JSON ─────────────────────────────────────────────────────────
    all_l1 = (l1.food + l1.bulk + l1.macro + l1.infra + l1.electricity + l1.logistics)

    payload = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capital":      capital,
            "market_env":   env,
        },
        "layer0": {
            "level":        l0_status.level,
            "score":        l0_status.score,
            "max_position": l0_status.max_position,
            "note":         l0_status.note,
            "triggered":    l0_status.triggered,
            "env": {
                "index_5d_chg": env_data.get("index_5d_chg"),
                "max_1d_drop":  env_data.get("max_1d_drop"),
                "vix":          env_data.get("vix") or env_data.get("qvix"),
                "usdcny":       env_data.get("usdcny"),
            },
        },
        "layer1": {
            "score":   l1.score,
            "ts":      l1.timestamp,
            "alerts":  [ind_to_dict(a) for a in l1.alerts],
            "groups": {
                "food":        [ind_to_dict(i) for i in l1.food],
                "bulk":        [ind_to_dict(i) for i in l1.bulk],
                "macro":       [ind_to_dict(i) for i in l1.macro],
                "infra":       [ind_to_dict(i) for i in l1.infra],
                "electricity": [ind_to_dict(i) for i in l1.electricity],
                "logistics":   [ind_to_dict(i) for i in l1.logistics],
            },
            "all": [ind_to_dict(i) for i in all_l1],
        },
        "layer2": {
            "score":      l2.score,
            "ts":         l2.timestamp,
            "divergence": l2.divergence,
            "intl":       [sig_to_dict(s) for s in l2.intl],
            "futures":    [sig_to_dict(s) for s in l2.futures],
            "bank":       [sig_to_dict(s) for s in l2.bank],
        },
        "layer3": {
            "active":          [chain_to_dict(c) for c in l3.active_chains],
            "inactive":        [chain_to_dict(c) for c in l3.inactive_chains],
            "top_nodes":       l3.top_nodes,
            "market_narrative": l3.market_narrative,
            "cross_signals":   l3.cross_chain_signals,
            "source":          l3.source,
        },
        "intl_signals": {
            "sox_index":    intl_sig_result.sox_index,
            "raw_news":     intl_sig_result.raw_news_count,
            "extracted":    intl_sig_result.llm_extracted,
            "timestamp":    intl_sig_result.timestamp,
            "signals": [
                {
                    "category":  s.category,
                    "name":      s.name,
                    "value":     s.value,
                    "direction": s.direction,
                    "detail":    s.detail,
                    "source":    s.source,
                    "chains":    s.relevance_chains,
                }
                for s in intl_signals
            ],
        },
        "trend": trend_to_dict(trend),
        "layer4": [profile_to_dict(p) for p in profiles],
        "layer5": [decision_to_dict(d, capital) for d in decisions],
        "summary": {
            "l0_level":      l0_status.level,
            "l1_score":      l1.score,
            "l2_score":      l2.score,
            "trend_verdict": trend.verdict,
            "trend_direction": trend.trend,
            "trend_confidence": trend.confidence,
            "trend_source":  trend.source,
            "active_chains": len(l3.active_chains),
            "go_count":      sum(1 for d in decisions if d.verdict == "GO"),
            "wait_count":    sum(1 for d in decisions if d.verdict == "WAIT"),
            "no_count":      sum(1 for d in decisions if d.verdict == "NO"),
            "market_env":    env,
        },
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 同时生成一个自带数据的 HTML（可直接双击打开，无需 HTTP 服务器）
    standalone_path = os.path.join(out_dir or ".", "dashboard_standalone.html")
    html_path = os.path.join(out_dir or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        # 在 </body> 前注入内嵌数据
        json_str = json.dumps(payload, ensure_ascii=False)
        embed_tag = f'<script id="embedded-data" type="application/json">{json_str}</script>\n'
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', embed_tag + '</body>')
        with open(standalone_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    print(f"  JSON 已写入：{output_path}")
    if os.path.exists(standalone_path):
        print(f"  独立HTML已写入：{standalone_path}")
    print(f"  GO标的：{payload['summary']['go_count']} 个")
    print(f"  WAIT标的：{payload['summary']['wait_count']} 个")
    print(f"  分析完成！")
    print()
    print("  打开面板方式：")
    print("    方式1：双击 web/dashboard_standalone.html（推荐，无需服务器）")
    print("    方式2：cd web && python -m http.server 8080，访问 http://localhost:8080")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--env",     type=str,   default="AUTO")
    parser.add_argument("--output",  type=str,   default=None)
    args = parser.parse_args()
    try:
        run_export(args.capital, args.env, args.output)
    except Exception as e:
        print(f"[FATAL] 程序异常: {e}（详情见 logs/error.log）")
        _log.exception("程序致命异常")
        input("\n按回车键退出...")  # 防止闪退
