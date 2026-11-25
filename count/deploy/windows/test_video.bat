@echo off
chcp 65001 >nul
rem 영상 파일을 이 bat 위에 끌어다 놓으면 그 영상으로 시험 실행
cd /d "%~dp0.."
if "%~1"=="" (echo 영상 파일을 이 파일 위에 끌어다 놓으세요 & pause & exit /b 1)
.venv\Scripts\python app\count_live.py --source "%~1" --test
pause
