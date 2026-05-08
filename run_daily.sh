#!/bin/bash
# 待敌市场分析系统 — macOS/Linux 一键运行脚本
# 用法：chmod +x run_daily.sh && ./run_daily.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始六层市场分析..."
python3 export_json.py
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 分析完成"
