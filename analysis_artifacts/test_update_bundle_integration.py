from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "smart_canteen_web" / "backend"
PRODUCTION_DATA_ROOT = PROJECT_ROOT / "smart_canteen_web" / "data"
WORKBOOK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "data_update_test_20260824"
    / "smart_canteen_update_test_20260611_20260824.xlsx"
)
sys.path.insert(0, str(BACKEND_ROOT))

from app.dataset_repository import DatasetRepository, read_csv_utf8  # noqa: E402
from app.updater import DataUpdateService  # noqa: E402


TRACKED_PRODUCTION_FILES = [
    PRODUCTION_DATA_ROOT / "features" / "weather.csv",
    PRODUCTION_DATA_ROOT / "features" / "calendar.csv",
    PRODUCTION_DATA_ROOT / "datasets" / "a_hupan" / "runtime_actual_sales.csv",
    PRODUCTION_DATA_ROOT
    / "datasets"
    / "f_jichuyuan"
    / "runtime_actual_sales.csv",
    PRODUCTION_DATA_ROOT / "manifest.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_isolated_data(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for filename in ["dataset_registry.json", "ui_options.json", "manifest.json"]:
        shutil.copy2(PRODUCTION_DATA_ROOT / filename, target / filename)
    (target / "features").mkdir(parents=True, exist_ok=True)
    for filename in ["weather.csv", "calendar.csv"]:
        shutil.copy2(
            PRODUCTION_DATA_ROOT / "features" / filename,
            target / "features" / filename,
        )
    for dataset_id in ["a_hupan", "f_jichuyuan"]:
        source = PRODUCTION_DATA_ROOT / "datasets" / dataset_id
        destination = target / "datasets" / dataset_id
        destination.mkdir(parents=True, exist_ok=True)
        for filename in [
            "sales.csv",
            "runtime_actual_sales.csv",
            "dish_library.csv",
            "recommendation_history.csv",
        ]:
            shutil.copy2(source / filename, destination / filename)


def main() -> None:
    before_hashes = {str(path): sha256(path) for path in TRACKED_PRODUCTION_FILES}
    isolated_root = Path(
        tempfile.mkdtemp(prefix="smart_canteen_update_integration_20260824_")
    )
    isolated_data = isolated_root / "data"
    copy_isolated_data(isolated_data)

    repository = DatasetRepository(isolated_data)
    service = DataUpdateService(repository)
    content = WORKBOOK_PATH.read_bytes()
    filename = WORKBOOK_PATH.name
    results: dict[str, object] = {
        "isolated_data_root": str(isolated_data),
        "workbook": str(WORKBOOK_PATH),
        "checks": {},
    }
    checks = results["checks"]

    selection = service.preview(
        kind="weather", filename=filename, content=content
    )
    assert selection["requires_sheet_selection"] is True
    assert selection["can_commit"] is False
    assert selection["sheet_names"] == [
        "使用说明",
        "天气",
        "校历",
        "A湖畔销售",
        "F基础园销售",
    ]
    checks["multi_sheet_selection"] = {
        "passed": True,
        "sheet_names": selection["sheet_names"],
    }

    blocked_sales = service.preview(
        kind="sales",
        filename=filename,
        content=content,
        dataset_id="a_hupan",
        sheet_name="A湖畔销售",
    )
    assert blocked_sales["can_commit"] is False
    assert len(blocked_sales["summary"]["missing_weather_dates"]) == 75
    assert len(blocked_sales["summary"]["missing_calendar_dates"]) == 75
    checks["sales_blocked_before_features"] = {
        "passed": True,
        "missing_weather_dates": 75,
        "missing_calendar_dates": 75,
    }

    weather_preview = service.preview(
        kind="weather",
        filename=filename,
        content=content,
        sheet_name="天气",
    )
    assert weather_preview["can_commit"] is True
    assert weather_preview["summary"]["valid_rows"] == 75
    assert weather_preview["summary"]["abnormal_rows"] == 0
    weather_commit = service.commit(weather_preview["preview_token"], None)
    assert weather_commit["added_rows"] == 75
    checks["weather_update"] = {
        "passed": True,
        "preview": weather_preview["summary"],
        "commit": weather_commit,
    }

    calendar_preview = service.preview(
        kind="calendar",
        filename=filename,
        content=content,
        sheet_name="校历",
    )
    assert calendar_preview["can_commit"] is True
    assert calendar_preview["summary"]["valid_rows"] == 75
    assert calendar_preview["summary"]["abnormal_rows"] == 0
    calendar_commit = service.commit(calendar_preview["preview_token"], None)
    assert calendar_commit["added_rows"] == 75
    checks["calendar_update"] = {
        "passed": True,
        "preview": calendar_preview["summary"],
        "commit": calendar_commit,
    }

    expected_rows = {"a_hupan": 22799, "f_jichuyuan": 24011}
    sheet_names = {"a_hupan": "A湖畔销售", "f_jichuyuan": "F基础园销售"}
    for dataset_id in ["a_hupan", "f_jichuyuan"]:
        sales_preview = service.preview(
            kind="sales",
            filename=filename,
            content=content,
            dataset_id=dataset_id,
            sheet_name=sheet_names[dataset_id],
        )
        assert sales_preview["can_commit"] is True
        assert sales_preview["summary"]["valid_rows"] == expected_rows[dataset_id]
        assert sales_preview["summary"]["duplicate_rows"] == 0
        assert sales_preview["summary"]["abnormal_rows"] == 0
        assert sales_preview["summary"]["missing_weather_dates"] == []
        assert sales_preview["summary"]["missing_calendar_dates"] == []

        mismatch_rejected = False
        try:
            service.commit(
                sales_preview["preview_token"],
                "f_jichuyuan" if dataset_id == "a_hupan" else "a_hupan",
            )
        except ValueError as exc:
            mismatch_rejected = "预览不一致" in str(exc)
        assert mismatch_rejected is True

        sales_commit = service.commit(
            sales_preview["preview_token"], dataset_id
        )
        assert sales_commit["added_rows"] == expected_rows[dataset_id]
        runtime = read_csv_utf8(
            repository.dataset_path(dataset_id, "runtime_actual_sales.csv")
        )
        runtime_dates = pd.to_datetime(
            {"year": runtime["年"], "month": runtime["月"], "day": runtime["日"]}
        )
        assert len(runtime) == expected_rows[dataset_id]
        assert runtime_dates.min().strftime("%Y-%m-%d") == "2026-06-11"
        assert runtime_dates.max().strftime("%Y-%m-%d") == "2026-08-24"

        duplicate_preview = service.preview(
            kind="sales",
            filename=filename,
            content=content,
            dataset_id=dataset_id,
            sheet_name=sheet_names[dataset_id],
        )
        assert duplicate_preview["can_commit"] is False
        assert duplicate_preview["summary"]["valid_rows"] == 0
        assert (
            duplicate_preview["summary"]["duplicate_rows"]
            == expected_rows[dataset_id]
        )
        checks[f"{dataset_id}_sales_update"] = {
            "passed": True,
            "preview_summary": sales_preview["summary"],
            "dataset_mismatch_rejected": mismatch_rejected,
            "commit": sales_commit,
            "runtime_rows": int(len(runtime)),
            "runtime_date_min": runtime_dates.min().strftime("%Y-%m-%d"),
            "runtime_date_max": runtime_dates.max().strftime("%Y-%m-%d"),
            "duplicate_reupload": duplicate_preview["summary"],
        }

    weather = repository.read_weather()
    calendar = repository.read_calendar()
    assert weather["日期"].max() == "2026-08-24"
    assert calendar["日期"].max() == "2026-08-24"
    assert len(weather.loc[weather["日期"].between("2026-06-11", "2026-08-24")]) == 75
    assert len(calendar.loc[calendar["日期"].between("2026-06-11", "2026-08-24")]) == 75
    checks["feature_date_coverage"] = {
        "passed": True,
        "weather_max_date": weather["日期"].max(),
        "calendar_max_date": calendar["日期"].max(),
    }

    combined_max_dates: dict[str, str] = {}
    for dataset_id in ["a_hupan", "f_jichuyuan"]:
        combined = repository.combined_sales_for_model(dataset_id)
        combined_max_dates[dataset_id] = pd.to_datetime(combined["date"]).max().strftime(
            "%Y-%m-%d"
        )
        assert combined_max_dates[dataset_id] == "2026-08-24"
    checks["model_history_refresh"] = {
        "passed": True,
        "combined_max_dates": combined_max_dates,
    }

    after_hashes = {str(path): sha256(path) for path in TRACKED_PRODUCTION_FILES}
    assert before_hashes == after_hashes
    checks["production_data_unchanged"] = {
        "passed": True,
        "tracked_file_count": len(TRACKED_PRODUCTION_FILES),
    }
    results["all_checks_passed"] = True
    report_path = isolated_root / "integration_test_report.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
