@echo off
chcp 65001 >nul
cd /d "%~dp0.."
.venv\Scripts\python app\count_live.py %*
if errorlevel 1 pause
