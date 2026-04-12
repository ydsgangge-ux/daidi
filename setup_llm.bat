@echo off
chcp 65001 >nul
title 大模型配置器
cd /d "%~dp0"

echo.
echo  ═══════════════════════════════════════════
echo         大模型配置器
echo  ═══════════════════════════════════════════
echo.

:: ── 检查 Python ──
where python >nul 2>&1
if %errorlevel% neq 0 (
    where python3 >nul 2>&1
    if %errorlevel% neq 0 (
        echo  [X] 未检测到 Python！
        echo.
        echo  请先安装 Python 3.9+：
        echo  https://www.python.org/downloads/
        echo  安装时务必勾选 "Add Python to PATH"
        echo.
        pause
        exit /b 1
    )
    set PY=python3
) else (
    set PY=python
)

for /f "tokens=2" %%v in ('%PY% --version 2^>^&1') do echo  [OK] Python %%v

:: ── 安装依赖 ──
echo  [*] 检查依赖...
%PY% -c "import openai" >nul 2>&1
if %errorlevel% neq 0 (
    pip install openai -q 2>nul
)
%PY% -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    pip install flask -q 2>nul
)

echo  [OK] 启动配置页面...
echo.
echo  浏览器将自动打开配置页面
echo  配置完成后关闭本窗口即可
echo.

%PY% setup_llm.py
pause
