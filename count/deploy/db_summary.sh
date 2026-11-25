#!/usr/bin/env bash
# 일별 대수 (logs/count.db)
cd "$(dirname "$0")"
exec .venv/bin/python app/db_tool.py summary "$@"
