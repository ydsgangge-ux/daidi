@echo off
echo [%date% %time%] Starting 6-layer market analysis...
cd /d "%~dp0"
python export_json.py
echo [%date% %time%] Analysis complete
