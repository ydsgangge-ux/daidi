@echo off
cd /d "%~dp0"
echo [%date% %time%] Starting 6-layer market analysis...
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
