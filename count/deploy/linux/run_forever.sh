#!/usr/bin/env bash
# 프로그램이 오류로 꺼지면 10초 뒤 다시 켠다. 사람이 Q/Esc 로 끄면(정상 종료) 그대로 끝.
cd "$(dirname "$0")/.."
while true; do
  ./run.sh "$@"; code=$?
  [ $code -eq 0 ] && break
  echo "$(date '+%F %T') 비정상 종료(코드 $code) — 10초 뒤 다시 시작" | tee -a logs/restart.log
  sleep 10
done
