@echo off
title LLM Configurator
cd /d "%~dp0"

echo.
echo  ================================================
echo         LLM Configurator
echo  ================================================
echo.

::: ── Check Python ──
where python >nul 2>&1
if %errorlevel% neq 0 (
    where python3 >nul 2>&1
    if %errorlevel% neq 0 (
        echo  [X] Python not found!
        echo.
        echo  Please install Python 3.9+ first:
        echo  https://www.python.org/downloads/
        echo  Make sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
    set PY=python3
) else (
    set PY=python
)

for /f "tokens=2" %%v in ('%PY% --version 2^>^&1') do echo  [OK] Python %%v

::: ── Install dependencies ──
echo  [*] Checking dependencies...
%PY% -c "import openai" >nul 2>&1
if %errorlevel% neq 0 (
    pip install openai -q 2>nul
)
%PY% -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    pip install flask -q 2>nul
)

echo  [OK] Starting configuration page...
echo.
echo  Your browser will open automatically.
echo  Close this window when done.
echo.

%PY% setup_llm.py
pause
