"""
快速验证脚本——每层独立测试，确认数据可获取
"""
import sys
import os
import io
import traceback
import logging

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 使用脚本所在目录，不硬编码路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

# 日志配置
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_log_dir, "error.log"),
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)

from rich.console import Console
console = Console(force_terminal=True)

def test_layer0():
    console.print("\n[bold blue]── Layer 0 测试 ──[/bold blue]")
    from layer0_risk import run_layer0
    status, env = run_layer0()
    console.print(f"  Level: [bold]{status.level}[/bold]  Score: {status.score}")
    console.print(f"  最大仓位: {status.max_position*100:.0f}%")
    console.print(f"  VIX: {env.get('vix', 'N/A')}")
    console.print(f"  上证5日: {env.get('index_5d_chg', 'N/A')}%")
    console.print(f"  触发规则: {len(status.triggered)} 条")
    console.print(f"  [green]✓ Layer 0 正常[/green]")
    return status, env

def test_layer1():
    console.print("\n[bold blue]── Layer 1 测试 ──[/bold blue]")
    from layer1_industry import run_layer1
    r = run_layer1()
    console.print(f"  综合评分: {r.score}")
    console.print(f"  食品指标: {len(r.food)} 条")
    console.print(f"  大宗商品: {len(r.bulk)} 条")
    console.print(f"  宏观货币: {len(r.macro)} 条")
    console.print(f"  基建需求: {len(r.infra)} 条")
    console.print(f"  用电数据: {len(r.electricity)} 条")
    console.print(f"  异动提醒: {len(r.alerts)} 条")
    for i in r.electricity:
        console.print(f"  用电量: {i.value} {i.unit}  偏离σ={i.zscore}  信号={i.signal}")
    console.print(f"  [green]✓ Layer 1 正常[/green]")
    return r

def test_layer2(l1_score=50):
    console.print("\n[bold blue]── Layer 2 测试 ──[/bold blue]")
    from layer2_futures import run_layer2
    r = run_layer2(l1_score)
    console.print(f"  综合评分: {r.score}")
    console.print(f"  国际信号: {len(r.intl)} 条")
    console.print(f"  股指期货: {len(r.futures)} 条")
    for s in r.intl:
        console.print(f"  {s.name}: {s.value} [{s.direction}]")
    for d in r.divergence:
        console.print(f"  {d}")
    console.print(f"  [green]✓ Layer 2 正常[/green]")
    return r

def test_layer3(l1, l2):
    console.print("\n[bold blue]── Layer 3 测试 ──[/bold blue]")
    from layer3_chains import run_layer3
    r = run_layer3(l1, l2)
    console.print(f"  激活产业链: {len(r.active_chains)} 条")
    console.print(f"  最佳节点: {len(r.top_nodes)} 个")
    for node in r.top_nodes:
        console.print(f"  ▸ {node['chain']}  {node['node']}  强度{node['strength']}")
    console.print(f"  [green]✓ Layer 3 正常[/green]")
    return r

def test_layer45(l1, l2, l3, l0_pass=True):
    console.print("\n[bold blue]── Layer 4+5 测试 ──[/bold blue]")
    from layer45_stocks import run_layer4, run_layer5
    l3_names = [c.name for c in l3.active_chains]
    profiles = run_layer4(active_chains=l3_names)
    console.print(f"  分析标的数: {len(profiles)}")
    decisions = run_layer5(profiles, l0_pass, l1.score, l2.score, l3_names, "SIDEWAYS")
    go  = [d for d in decisions if d.verdict == "GO"]
    wait = [d for d in decisions if d.verdict == "WAIT"]
    no   = [d for d in decisions if d.verdict == "NO"]
    console.print(f"  GO: {len(go)}  WAIT: {len(wait)}  NO: {len(no)}")
    for d in go:
        console.print(f"  [green]✓ GO[/green] {d.name}({d.code}) 仓位{d.actual_position_pct:.1f}%  {d.timing_signal}")
    console.print(f"  [green]✓ Layer 4+5 正常[/green]")
    return decisions

if __name__ == "__main__":
    console.print("[bold]开始六层系统测试...[/bold]")
    try:
        l0_status, env = test_layer0()
        l1 = test_layer1()
        l2 = test_layer2(l1.score)
        l3 = test_layer3(l1, l2)
        decisions = test_layer45(l1, l2, l3, l0_status.level != "RED")
        console.print("\n[bold green]✅ 所有层测试完成[/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]测试失败: {e}[/bold red]")
        _log.exception("测试失败")
    input("\n按回车键退出...")  # 防止闪退
