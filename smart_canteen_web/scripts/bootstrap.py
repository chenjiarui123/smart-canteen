#!/usr/bin/env python3
"""Create a portable local environment; retry interrupted dependency installs."""
import hashlib
import subprocess
import sys
import venv
from pathlib import Path


def main():
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise SystemExit("请使用 Python 3.10–3.12，推荐 Python 3.12。")
    root = Path(__file__).resolve().parents[1]
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        return subprocess.call([sys.executable, str(root / "run_app.py"), "--help"], cwd=root)
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        print("正在创建本地运行环境...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    version_check = subprocess.run([str(python), "-c", "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"])
    if version_check.returncode:
        raise SystemExit("现有 .venv 的 Python 版本不兼容。请重命名该目录后，用 Python 3.10–3.12 重新启动。")
    requirements = root / "backend/requirements.txt"
    stamp = environment / ".installed-requirements-sha256"
    expected = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if not stamp.exists() or stamp.read_text(encoding="ascii").strip() != expected:
        print("正在安装运行组件，请保持网络连接；安装失败后可重新运行本脚本。", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True, cwd=root)
        stamp.write_text(expected + "\n", encoding="ascii")
    return subprocess.call([str(python), str(root / "run_app.py"), *sys.argv[1:]], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
