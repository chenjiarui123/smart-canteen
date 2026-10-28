#!/bin/sh
# Reuse the Python environment verified on this Mac; no installation or training.
set -eu
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.targeted-venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "本机已验证的 Python 环境不存在，请参阅交付验收报告的环境说明。"
  exit 1
fi
exec "$PYTHON_BIN" "$PROJECT_DIR/smart_canteen_web/run_app.py" "$@"
