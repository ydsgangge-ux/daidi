#!/bin/bash
# 「待敌」六层市场分析系统 — macOS/Linux 一键启动
# 使用方法：双击 start.command，或在终端运行：chmod +x start.command && ./start.command

cd "$(dirname "$0")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║     ${BOLD}「待敌」六层市场分析系统 一键启动${NC}       ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# ── 1. 检查 Python ──
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo -e "  ${RED}[X] 未检测到 Python！${NC}"
    echo ""
    echo "  请先安装 Python 3.9+："
    echo "  macOS: brew install python3"
    echo "  或下载: https://www.python.org/downloads/"
    echo ""
    read -p "按回车键退出..."
    exit 1
fi

PY_VERSION=$($PY --version 2>&1 | awk '{print $2}')
echo -e "  ${GREEN}[OK]${NC} Python ${PY_VERSION}"

# ── 2. 安装依赖 ──
echo ""
echo -e "  [*] 正在检查依赖..."
$PY -m pip install -r requirements.txt -q 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "  ${YELLOW}[!] 部分依赖安装失败，尝试使用清华镜像源...${NC}"
    $PY -m pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null
fi
echo -e "  ${GREEN}[OK]${NC} 依赖检查完成"

# ── 3. 检查大模型配置（可选） ──
if [ ! -f llm_config.json ] && [ ! -f .env ]; then
    echo ""
    echo -e "  ${YELLOW}[!] 未检测到大模型 API Key 配置${NC}"
    echo "      系统将使用规则兜底运行（趋势判定和产业链分析精度会降低）"
    echo ""
    echo "      如需配置大模型，运行以下任一命令："
    echo "        python setup_llm.py        （图形化配置，推荐）"
    echo "        或手动创建 .env 文件写入 DEEPSEEK_API_KEY=sk-xxx"
    echo ""
else
    echo -e "  ${GREEN}[OK]${NC} 大模型已配置"
fi

# ── 4. 运行分析 ──
echo ""
echo -e "  [*] 开始六层市场分析（需要1-3分钟）..."
echo ""
$PY export_json.py
echo ""

# ── 5. 启动 Web 服务器并打开浏览器 ──
PORT=8080
echo -e "  [*] 启动 Web 服务器..."

# 检查端口是否被占用
if lsof -i :$PORT &>/dev/null; then
    echo -e "  ${YELLOW}[!] 端口 $PORT 已被占用，尝试使用 8081...${NC}"
    PORT=8081
fi

# 启动服务器（后台运行）
$PY -m http.server $PORT --directory web &>/dev/null &
SERVER_PID=$!
sleep 1

# 检查服务器是否启动成功
if kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} 服务器已启动 (PID: $SERVER_PID)"
else
    echo -e "  ${RED}[X] 服务器启动失败${NC}"
    echo "  请手动运行：python -m http.server $PORT --directory web"
    read -p "按回车键退出..."
    exit 1
fi

echo ""
echo "  ╔════════════════════════════════════════════════════╗"
echo "  ║  浏览器将自动打开，地址：http://localhost:$PORT        ║"
echo "  ║  查看完成后关闭本窗口 / 按 Ctrl+C 停止服务器        ║"
echo "  ╚════════════════════════════════════════════════════╝"
echo ""

# 打开浏览器
if command -v open &>/dev/null; then
    open "http://localhost:$PORT"
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:$PORT" 2>/dev/null
fi

# 等待用户中断
trap "kill $SERVER_PID 2>/dev/null; echo ''; echo '  服务器已停止。'; exit 0" INT TERM
wait $SERVER_PID 2>/dev/null
