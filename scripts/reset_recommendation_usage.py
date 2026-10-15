"""Back up and reset recommendation usage only; preserve sales and model data."""
from pathlib import Path
from datetime import datetime
import sys
import json
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "smart_canteen_web"
sys.path.insert(0, str(WEB / "backend"))
import pandas as pd
from app.dataset_repository import DatasetRepository, RECOMMENDATION_HISTORY_COLUMNS
from app.history_store import HistoryStore


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repository = DatasetRepository()
    store = HistoryStore(repository)
    histories = [repository.dataset_path(ds, "recommendation_history.csv")
                 for ds in ("a_hupan", "f_jichuyuan")]
    protected = [p for folder in (WEB / "data/datasets", WEB / "data/features", WEB / "model")
                 for p in folder.rglob("*") if p.is_file() and p not in histories
                 and "__pycache__" not in p.parts]
    protected += [p for p in (WEB / "data").glob("*.json") if p.is_file()]
    backup = ROOT / "outputs" / ("recommendation_usage_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    with repository.write_lock():
        before = {str(p.relative_to(WEB)): digest(p) for p in protected}
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copytree(store.recommendation_root, backup / "temp_recommendations")
        counts = {}
        for ds, path in zip(("a_hupan", "f_jichuyuan"), histories):
            counts[ds] = len(repository.read_recommendation_history(ds))
            destination = backup / "datasets" / ds
            destination.mkdir(parents=True)
            shutil.copy2(path, destination / path.name)
        saved_files = [p for p in store.recommendation_root.rglob("*")
                       if p.is_file() and p.name != ".gitkeep"]
        # Backups must match before any active records are cleared.
        for path in saved_files:
            assert digest(path) == digest(backup / "temp_recommendations" / path.relative_to(store.recommendation_root))
        for ds, path in zip(("a_hupan", "f_jichuyuan"), histories):
            assert digest(path) == digest(backup / "datasets" / ds / path.name)
        for path in histories:
            repository._atomic_write_csv_unlocked(
                pd.DataFrame(columns=RECOMMENDATION_HISTORY_COLUMNS), path, create_backup=False)
        for path in saved_files:
            path.unlink()
        repository._atomic_write_json_unlocked({"recommendations": {}}, store.index_path, False)
        for ds, name in (("a_hupan", "A湖畔智慧餐厅"), ("f_jichuyuan", "F基础园智慧餐厅")):
            assert repository.read_recommendation_history(ds).empty
            for meal in ("早餐", "午餐", "晚餐"):
                assert store.initial_menu_history(ds, name, meal, pd.Timestamp("2100-01-01")) == {}
        assert store._read_index() == {"recommendations": {}}
        assert not list((store.recommendation_root / "automatic_high_frequency").rglob("*.json"))
        after = {str(p.relative_to(WEB)): digest(p) for p in protected}
        assert before == after, "Protected data changed"
        result = {"backup": str(backup), "cleared_history_rows": counts,
                  "cleared_recommendation_files": len(saved_files),
                  "protected_files_unchanged": len(before), "empty_history_read_verified": True,
                  "protected_sha256": before}
        repository._atomic_write_json_unlocked(result, backup / "reset_verification.json", False)
    print(json.dumps({k: v for k, v in result.items() if k != "protected_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
