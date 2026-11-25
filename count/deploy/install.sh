#!/usr/bin/env bash
# COUNT 설치 — Ubuntu 22.04 / 24.04, NVIDIA 드라이버가 설치된 PC. 처음 한 번만 실행.
#   ./install.sh
# Python 은 시스템 것을 쓰지 않고 uv 로 3.11 을 따로 받는다
# (22.04 기본 3.10 은 일부 패키지 미지원, 24.04 는 시스템 Python 에 pip 설치가 막혀 있음).
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/6] 시스템 패키지 (한글 폰트, OpenCV 화면 라이브러리) — sudo 비밀번호를 물을 수 있음"
sudo apt-get update -y
sudo apt-get install -y curl ca-certificates fonts-nanum fontconfig libgl1 libglib2.0-0

echo "[2/6] NVIDIA 드라이버 확인"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else echo "  경고: nvidia-smi 가 없습니다. NVIDIA 드라이버(535 이상 권장)를 먼저 설치하세요. 없으면 CPU 로 돌아 실시간이 어렵습니다."; fi

echo "[3/6] uv 설치 + Python 3.11 가상환경(.venv)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv

echo "[4/6] PyTorch (CUDA 12.4)"
uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "[5/6] 나머지 패키지"
uv pip install --python .venv/bin/python -r app/requirements.txt

echo "[6/6] TensorRT (선택, 약 2 GB) — 실패해도 설치는 계속되고 PyTorch 로 실행된다"
if uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com -r app/requirements-trt.txt; then
  echo "  TensorRT 설치됨 (처음 ./run.sh 때 엔진 생성, 약 10분)"
else
  echo "  경고: TensorRT 설치 실패 → PyTorch 로 실행 (조금 느림). 나중에 이 단계만 다시: "
  echo "    uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com -r app/requirements-trt.txt"
fi

chmod +x ./*.sh linux/*.sh 2>/dev/null || true
.venv/bin/python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음 (CPU 로 실행됨)')"
.venv/bin/python -c "import sys; sys.path.insert(0, 'app'); import line_setup as L; print('한글 폰트:', L._font_path() or '없음 (화면 한글이 깨짐)')"
echo "설치 완료. config.json 에 카메라 주소(source)를 넣고 ./run.sh 를 실행하세요."
