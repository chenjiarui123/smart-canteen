#!/bin/sh
set -eu
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "请先安装 Python 3.10 至 3.12，推荐 Python 3.12。"
  exit 1
fi
exec python3 scripts/bootstrap.py "$@"
