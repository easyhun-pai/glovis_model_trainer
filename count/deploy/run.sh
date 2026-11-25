#!/usr/bin/env bash
# 실행. 시작할 때 기준선 확인 화면이 뜬다 (15초 뒤 자동 시작). 옵션: --no-show (화면 없이), --setup (기준선 새로)
cd "$(dirname "$0")"
exec .venv/bin/python app/count_live.py "$@"
