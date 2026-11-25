@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 일별 대수 (logs\count.db)
.venv\Scripts\python app\db_tool.py summary
pause
