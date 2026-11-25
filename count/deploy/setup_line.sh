#!/usr/bin/env bash
# 기준선을 새로 긋고 실행 (카메라 화각이 바뀌었을 때)
cd "$(dirname "$0")"
exec .venv/bin/python app/count_live.py --setup "$@"
