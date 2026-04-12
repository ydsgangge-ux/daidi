@echo off
chcp 65001 >nul
title 待敌 - 六层市场分析系统
cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║       「待敌」六层市场分析系统 一键启动       ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ── 1. 检查 Python ──
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [X] 未检测到 Python！
    echo.
    echo  请先安装 Python 3.9+：
    echo  https://www.python.org/downloads/
    echo.
    echo  安装时务必勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:: 显示 Python 版本
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  [OK] Python %%v

:: ── 2. 安装依赖 ──
echo.
echo  [*] 正在检查依赖...
pip install -r requirements.txt -q 2>nul
if %errorlevel% neq 0 (
    echo  [!] 部分依赖安装失败，尝试使用镜像源重新安装...
    pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>nul
)
echo  [OK] 依赖检查完成

:: ── 3. 检查大模型配置（可选） ──
if not exist llm_config.json (
    if not exist .env (
        echo.
        echo  [!] 未检测到大模型 API Key 配置
        echo      系统将使用规则兜底运行（趋势判定和产业链分析精度会降低）
        echo.
        echo      如需配置大模型，双击 setup_llm.bat 即可配置
        echo      或手动创建 .env 文件写入 DEEPSEEK_API_KEY=sk-xxx
        echo.
    )
) else (
    echo  [OK] 大模型已配置
)

:: ── 4. 运行分析 ──
echo.
echo  [*] 开始六层市场分析（需要1-3分钟）...
echo.
python export_json.py
echo.

:: ── 5. 启动 Web 服务器并打开浏览器 ──
echo  [*] 启动 Web 服务器...
start /b python -m http.server 8080 --directory web >nul 2>&1
timeout /t 1 /nobreak >nul

echo  [OK] 服务器已启动
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║  浏览器将自动打开，地址：http://localhost:8080  ║
echo  ║  查看完成后关闭本窗口即可停止服务器              ║
echo  ╚══════════════════════════════════════════════╝
echo.

start http://localhost:8080

echo  按任意键停止服务器并退出...
pause >nul
taskkill /f /im python.exe /fi "WINDOWTITLE eq *http.server*" >nul 2>&1
