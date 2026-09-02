#!/bin/bash
# 待敌市场分析系统 — macOS/Linux 一键运行脚本
# 用法：chmod +x run_daily.sh && ./run_daily.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始六层市场分析..."
python3 export_json.py
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Analysis failed!"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 分析完成，开始回测（非关键，失败可跳过）..."
python3 backtest.py || echo "[!] Backtest skipped (not critical)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done"
