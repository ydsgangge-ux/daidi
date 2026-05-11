@echo off
title Daidi - 6-Layer Market Analysis System
cd /d "%~dp0"

echo.
echo  ========================================================
echo         Daidi 6-Layer Market Analysis System
echo  ========================================================
echo.

REM ── 1. Check Python ──
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [X] Python not found!
    echo.
    echo  Please install Python 3.9+ first:
    echo  https://www.python.org/downloads/
    echo.
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  [OK] Python %%v

REM ── 2. Install dependencies (core only, chromadb is optional) ──
echo.
echo  [*] Installing core dependencies...
echo.
python -m pip install --upgrade pip -q
python -m pip install akshare pandas numpy rich openai -q
if %errorlevel% neq 0 (
    echo.
    echo  [!] Some dependencies failed, retrying with mirror...
    echo.
    python -m pip install akshare pandas numpy rich openai -q -i https://pypi.tuna.tsinghua.edu.cn/simple
)
if %errorlevel% neq 0 (
    echo.
    echo  [X] Core dependency installation failed!
    echo      Please check your network connection and try again.
    echo.
    pause
    exit /b 1
)
echo  [OK] Core dependencies ready

REM ── 2b. Try installing chromadb (optional, for long-term memory) ──
echo.
echo  [*] Trying optional chromadb (memory store, skip if fails)...
python -m pip install chromadb -q >nul 2>&1
if %errorlevel% equ 0 (
    echo  [OK] ChromaDB installed (long-term memory enabled)
) else (
    echo  [!] ChromaDB skipped (memory feature will be unavailable)
)

REM ── 3. Check LLM config (optional) ──
if not exist llm_config.json (
    if not exist .env (
        echo.
        echo  [!] No LLM API Key detected.
        echo      System will run in rule-based mode.
        echo      To configure LLM, double-click setup_llm.bat
        echo.
    )
) else (
    echo  [OK] LLM configured
)

REM ── 4. Run analysis ──
echo.
echo  [*] Starting 6-layer market analysis (takes 1-3 minutes)...
echo.
python export_json.py
if %errorlevel% neq 0 (
    echo.
    echo  [X] Market analysis failed! See logs\error.log for details.
    echo.
    pause
    exit /b 1
)

REM ── 4b. Run backtest (optional, skip if fails) ──
echo.
echo  [*] Running backtest (optional, default: 300394 天孚通信)...
echo.
python backtest.py
if %errorlevel% neq 0 (
    echo  [!] Backtest skipped (not critical)
)

REM ── 5. Start Web server and open browser ──
echo.
echo  [*] Starting Web server...
start /b python -m http.server 8080 --directory web >nul 2>&1
timeout /t 1 /nobreak >nul

echo  [OK] Server started
echo.
echo  ========================================================
echo    Browser will open at: http://localhost:8080
echo    Close this window to stop the server.
echo  ========================================================
echo.

start http://localhost:8080

echo  Press any key to stop the server and exit...
pause >nul
taskkill /f /im python.exe /fi "WINDOWTITLE eq *http.server*" >nul 2>&1
