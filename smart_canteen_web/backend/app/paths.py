from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
FRONTEND_DIST = FRONTEND_ROOT / "dist"
MODEL_ROOT = PROJECT_ROOT / "model"
DATA_ROOT = PROJECT_ROOT / "data"
FEATURE_ROOT = DATA_ROOT / "features"
STAGING_ROOT = DATA_ROOT / "staging"
BACKUP_ROOT = DATA_ROOT / "backups"
LOCK_ROOT = DATA_ROOT / ".locks"
RECOMMENDATION_ROOT = PROJECT_ROOT / "temp_recommendations"


def ensure_runtime_directories() -> None:
    for path in [
        STAGING_ROOT,
        BACKUP_ROOT,
        LOCK_ROOT,
        RECOMMENDATION_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)
