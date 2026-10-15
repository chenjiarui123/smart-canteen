#!/usr/bin/env python3
"""Cross-platform production launcher for the local smart-canteen system."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "dist" / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not FRONTEND_INDEX.exists():
        raise SystemExit(
            "前端生产文件不存在。请先按 README 安装 Node.js 并在 frontend 目录执行 npm run build。"
        )
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "缺少后端依赖。请先执行：python -m pip install -r backend/requirements.txt"
        ) from exc
    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"智慧食堂系统已启动：{url}")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        app_dir=str(BACKEND_ROOT),
    )


if __name__ == "__main__":
    main()
