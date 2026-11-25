#!/usr/bin/env bash
# 녹화 영상으로 시험 실행 (저장된 기준선·기록에 영향 없음, 기록은 logs_test/)
#   ./test_video.sh /경로/영상.mp4
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "사용: ./test_video.sh 영상파일.mp4"; exit 1; }
exec .venv/bin/python app/count_live.py --source "$(realpath "$1")" --test
