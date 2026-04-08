"""
主控程序 — 串联六层，生成决策报告
用法：python main.py [--capital 1000000] [--env SIDEWAYS]
"""
import sys
import os
import io
import argparse
import traceback
import logging
from datetime import datetime

# 日志配置（异常写入日志文件）
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

import warnings
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.text import Text

from layer0_risk     import run_layer0
from layer1_industry import run_layer1
from layer2_futures  import run_layer2
from layer3_chains   import run_layer3
from layer45_stocks  import run_layer4, run_layer5

console = Console(force_terminal=True)

SIGNAL_COLOR = {
    "UP": "green", "DOWN": "red", "NEUTRAL": "dim",
    "ALERT_UP": "bold green", "ALERT_DOWN": "bold red"
}
DIRECTION_COLOR = {
    "BULLISH": "green", "BEARISH": "red", "NEUTRAL": "dim"
}
VERDICT_COLOR = {"GO": "bold green", "WAIT": "yellow", "NO": "bold red"}
L0_COLOR = {"GREEN": "green", "YELLOW": "yellow", "RED": "bold red"}


def print_header():
    console.print()
    console.print(Panel.fit(
        "[bold]「待敌」市场分析系统[/bold]\n"
        "[dim]昔之善战者，先为不可胜，以待敌之可胜。 ——《孙子兵法·形篇》[/dim]\n"
        "[dim]L0风险 → L1实业 → L2金融 → L3产业链 → L4头部企业 → L5个股确认[/dim]",
        border_style="dim"
    ))
    console.print(f"[dim]运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")


def print_layer0(status, env):
    color = L0_COLOR.get(status.level, "white")
    console.print(Panel(
        f"[{color}]● {status.level}[/{color}]  评分：{status.score}/100  "
        f"最大仓位：[bold]{status.max_position*100:.0f}%[/bold]\n"
        f"[dim]{status.note}[/dim]\n" +
        ("\n".join(f"  [yellow]⚠ {t}[/yellow]" for t in status.triggered)
         if status.triggered else "  [green]✓ 无风险规则触发[/green]"),
        title="[bold]Layer 0 · 风险管理[/bold]",
        border_style=color
    ))

    # 环境数据小表
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim", width=18)
    t.add_column(justify="right")
    t.add_row("上证5日涨跌", f"{env.get('index_5d_chg', 'N/A')}%")
    t.add_row("单日最大跌幅", f"{env.get('max_1d_drop', 'N/A')}%")
    t.add_row("VIX恐慌指数", str(env.get('vix', 'N/A')))
    t.add_row("美元/人民币", str(env.get('usdcny', 'N/A')))
    console.print(t)


def print_layer1(result):
    console.print(Panel(
        f"综合评分：[bold]{result.score}/100[/bold]  "
        f"异动指标：[bold yellow]{len(result.alerts)}[/bold yellow] 个  "
        f"[dim]更新：{result.timestamp}[/dim]",
        title="[bold]Layer 1 · 实业数据[/bold]",
        border_style="blue"
    ))

    # 各板块汇总表
    t = Table(box=box.SIMPLE_HEAVY, padding=(0, 1))
    t.add_column("指标", width=22)
    t.add_column("数值", justify="right", width=12)
    t.add_column("同比", justify="right", width=8)
    t.add_column("偏离σ", justify="right", width=7)
    t.add_column("信号", width=12)
    t.add_column("来源", style="dim", width=14)

    all_inds = (result.food + result.bulk + result.macro +
                result.infra + result.electricity)
    for ind in all_inds:
        sig_color = SIGNAL_COLOR.get(ind.signal, "white")
        yoy_str   = f"{ind.yoy:+.1f}%" if ind.yoy is not None else "-"
        z_str     = f"{ind.zscore:+.2f}" if ind.zscore is not None else "-"
        t.add_row(
            ind.name,
            f"{ind.value} {ind.unit}",
            yoy_str,
            z_str,
            f"[{sig_color}]{ind.signal}[/{sig_color}]",
            ind.source
        )
    console.print(t)

    if result.alerts:
        console.print("[bold yellow]⚡ 异动提醒:[/bold yellow]")
        for a in result.alerts:
            console.print(f"  [yellow]▸ {a.name}: {a.note}[/yellow]")
    console.print()


def print_layer2(result):
    console.print(Panel(
        f"综合评分：[bold]{result.score}/100[/bold]  "
        f"[dim]更新：{result.timestamp}[/dim]",
        title="[bold]Layer 2 · 金融期货 & 国际信号 & 银行信用[/bold]",
        border_style="magenta"
    ))

    # 国际信号 + 股指期货
    t = Table(box=box.SIMPLE_HEAVY, padding=(0, 1))
    t.add_column("指标", width=20)
    t.add_column("数值", justify="right", width=12)
    t.add_column("变动", justify="right", width=10)
    t.add_column("方向", width=10)
    t.add_column("A股传导含义", width=36)

    for s in result.intl + result.futures:
        dc = DIRECTION_COLOR.get(s.direction, "white")
        chg_str = f"{s.change:+.2f}" if s.change is not None else "-"
        t.add_row(
            s.name,
            f"{s.value} {s.unit}",
            chg_str,
            f"[{dc}]{s.direction}[/{dc}]",
            s.a_share_impact[:36]
        )
    console.print(t)

    # 银行/信用信号区块
    if result.bank:
        console.print(Panel(
            f"[bold]银行/信用信号（{len(result.bank)}项）[/bold]",
            border_style="blue"
        ))
        tb = Table(box=box.SIMPLE_HEAVY, padding=(0, 1))
        tb.add_column("指标", width=16)
        tb.add_column("数值", justify="right", width=10)
        tb.add_column("变动", justify="right", width=8)
        tb.add_column("方向", width=10)
        tb.add_column("深层含义", width=40)
        for s in result.bank:
            dc = DIRECTION_COLOR.get(s.direction, "white")
            chg_str = f"{s.change:+.2f}" if s.change is not None else "-"
            tb.add_row(
                s.name,
                f"{s.value} {s.unit}",
                chg_str,
                f"[{dc}]{s.direction}[/{dc}]",
                s.a_share_impact[:40],
            )
        console.print(tb)

    for d in result.divergence:
        style = "yellow" if "背离" in d else "green"
        console.print(f"  [{style}]{d}[/{style}]")
    console.print()


def print_layer3(result):
    console.print(Panel(
        f"激活产业链：[bold green]{len(result.active_chains)}[/bold green]  "
        f"待激活：[dim]{len(result.inactive_chains)}[/dim]",
        title="[bold]Layer 3 · 产业链传导图谱[/bold]",
        border_style="cyan"
    ))

    if result.top_nodes:
        console.print("[bold]当前最佳操作节点：[/bold]")
        for node in result.top_nodes:
            color = "green" if node["strength"] >= 70 else "yellow"
            console.print(
                f"  [{color}]▸ {node['chain']}[/{color}]  "
                f"当前节点：[bold]{node['node']}[/bold]  "
                f"信号强度：{node['strength']}  "
                f"L2一致：{'✓' if node['l2_aligned'] else '✗'}"
            )
            console.print(f"    [dim]{node['window']}[/dim]")
    console.print()


def print_layer45(decisions, capital: float):
    console.print(Panel(
        "四道过滤器 + A股时机信号 + 仓位计算",
        title="[bold]Layer 4+5 · 头部企业 & 个股决策[/bold]",
        border_style="green"
    ))

    t = Table(box=box.SIMPLE_HEAVY, padding=(0, 1), show_lines=True)
    t.add_column("代码", width=8)
    t.add_column("名称", width=10)
    t.add_column("产业链", width=12)
    t.add_column("L4评分", justify="center", width=8)
    t.add_column("时机信号", width=12)
    t.add_column("止损", justify="right", width=6)
    t.add_column("建议仓位%", justify="right", width=9)
    t.add_column("首批金额", justify="right", width=10)
    t.add_column("决策", width=6)
    t.add_column("行动", width=28)

    for d in decisions:
        vc = VERDICT_COLOR.get(d.verdict, "white")
        amount = capital * d.first_batch_pct / 100
        timing_color = {"BREAKOUT": "green", "PULLBACK": "cyan",
                        "ACCUMULATION": "yellow", "NONE": "dim"}.get(d.timing_signal, "dim")
        t.add_row(
            d.code,
            d.name,
            d.chain[:12],
            str(d.l4_score),
            f"[{timing_color}]{d.timing_signal}[/{timing_color}]",
            f"{d.stop_loss_pct}%",
            f"{d.actual_position_pct:.1f}%",
            f"¥{amount:,.0f}",
            f"[{vc}]{d.verdict}[/{vc}]",
            d.action[:28]
        )
    console.print(t)

    # GO标的详情
    go_list = [d for d in decisions if d.verdict == "GO"]
    if go_list:
        console.print("\n[bold green]✅ 可入场标的详情：[/bold green]")
        for d in go_list:
            amount = capital * d.first_batch_pct / 100
            console.print(Panel(
                f"[bold]{d.name}[/bold] ({d.code}) · {d.chain}\n"
                f"决策依据：{d.reason}\n"
                f"[green]执行：{d.action}[/green]\n"
                f"第一批金额：¥{amount:,.0f}（总资金{d.first_batch_pct:.1f}%）\n"
                f"止损触发：当前价下方 {d.stop_loss_pct}% 无条件执行",
                border_style="green"
            ))
    else:
        console.print("\n[yellow]当前无 GO 标的，所有标的处于观察等待状态。[/yellow]")


def save_report(status, l1, l2, l3, decisions, capital: float):
    """保存文本报告"""
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)
    fname = os.path.join(report_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"六层决策报告 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Layer 0: {status.level} (评分{status.score}) 最大仓位{status.max_position*100:.0f}%\n")
        f.write(f"Layer 1 实业评分: {l1.score}\n")
        f.write(f"Layer 2 期货评分: {l2.score}\n")
        f.write(f"激活产业链: {len(l3.active_chains)} 条\n\n")
        f.write("个股决策:\n")
        for d in decisions:
            f.write(f"  [{d.verdict}] {d.name}({d.code}): "
                    f"仓位{d.actual_position_pct:.1f}% 首批¥{capital*d.first_batch_pct/100:,.0f}\n"
                    f"    {d.reason}\n"
                    f"    {d.action}\n")
    return fname


def main():
    parser = argparse.ArgumentParser(description="六层市场分析决策系统")
    parser.add_argument("--capital", type=float, default=1_000_000, help="总资金（元）")
    parser.add_argument("--env", type=str, default="SIDEWAYS",
                        choices=["BULL", "SIDEWAYS", "BEAR"], help="市场环境手动覆盖")
    parser.add_argument("--codes", nargs="*", help="指定分析的股票代码，默认全库")
    args = parser.parse_args()

    print_header()

    # ── Layer 0 ──────────────────────────────────────────────────────────────
    console.print("[dim]正在获取 Layer 0 风险数据...[/dim]")
    l0_status, env_data = run_layer0()
    print_layer0(l0_status, env_data)

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    console.print("[dim]正在获取 Layer 1 实业数据（需要30-60秒）...[/dim]")
    l1 = run_layer1()
    print_layer1(l1)

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    console.print("[dim]正在获取 Layer 2 金融期货数据...[/dim]")
    l2 = run_layer2(l1_score=l1.score)
    print_layer2(l2)

    # ── Layer 3 ──────────────────────────────────────────────────────────────
    console.print("[dim]正在匹配 Layer 3 产业链...[/dim]")
    l3 = run_layer3(l1, l2)
    print_layer3(l3)

    # ── Layer 4 + 5 ──────────────────────────────────────────────────────────
    console.print("[dim]正在运行 Layer 4/5 个股分析（获取行情数据）...[/dim]")
    l3_active_names = [c.name for c in l3.active_chains]
    profiles = run_layer4(active_chains=l3_active_names)

    # 市场环境判断（L0 score为基础，可手动覆盖）
    if args.env != "SIDEWAYS":
        market_env = args.env
    else:
        market_env = ("BULL" if l0_status.score >= 75 and l1.score >= 60
                      else ("BEAR" if l0_status.level == "RED" else "SIDEWAYS"))

    decisions = run_layer5(
        profiles, l0_status.level != "RED",
        l1.score, l2.score, l3_active_names, market_env
    )
    print_layer45(decisions, args.capital)

    # ── 保存报告 ──────────────────────────────────────────────────────────────
    report_path = save_report(l0_status, l1, l2, l3, decisions, args.capital)
    console.print(f"\n[dim]报告已保存：{report_path}[/dim]")
    console.print("[bold]分析完成。[/bold]\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"\n[bold red]程序异常: {e}[/bold red]")
        _log.exception("主程序异常")
        input("\n按回车键退出...")  # 防止闪退
