@echo off
chcp 65001 >nul
echo [%date% %time%] 开始六层市场分析...
cd /d "%~dp0"
python export_json.py
echo [%date% %time%] 分析完成
