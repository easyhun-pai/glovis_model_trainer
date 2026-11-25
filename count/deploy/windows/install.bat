@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo [1/3] Python 3.11 가상환경 만들기 (.venv)
py -3.11 -m venv .venv || (echo Python 3.11 이 필요합니다: https://www.python.org/downloads/release/python-3119/ & pause & exit /b 1)
echo [2/3] PyTorch (CUDA 12.4) 설치
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 || (echo PyTorch 설치 실패 & pause & exit /b 1)
echo [3/3] 나머지 패키지 설치
.venv\Scripts\python -m pip install -r app\requirements.txt || (echo 패키지 설치 실패 & pause & exit /b 1)
echo [4/4] TensorRT (optional)
.venv\Scripts\python -m pip install --extra-index-url https://pypi.nvidia.com -r app\requirements-trt.txt || echo TensorRT install failed - runs with PyTorch
.venv\Scripts\python -c "import torch;print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"없음 (CPU 로 실행됨)\")"
echo 설치 완료. config.json 에 카메라 주소를 넣고 run.bat 을 실행하세요.
pause
