@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo [%date% %time%] 开始六层市场分析...
python export_json.py
if %errorlevel% neq 0 (
    echo [%date% %time%] [ERROR] Analysis failed!
    pause
    exit /b 1
)
echo [%date% %time%] Analysis complete, running backtest...
python backtest.py
if %errorlevel% neq 0 (
    echo [!] Backtest skipped (not critical)
)
echo [%date% %time%] All done
