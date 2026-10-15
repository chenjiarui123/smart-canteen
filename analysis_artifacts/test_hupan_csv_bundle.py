from __future__ import annotations

import json
import tempfile
from pathlib import Path

from test_update_bundle_integration import (
    PRODUCTION_DATA_ROOT,
    TRACKED_PRODUCTION_FILES,
    copy_isolated_data,
    sha256,
)

from app.dataset_repository import DatasetRepository, read_csv_utf8
from app.updater import DataUpdateService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_ROOT = PROJECT_ROOT / "outputs" / "data_update_hupan_csv_20260824"


def main() -> None:
    before = {str(path): sha256(path) for path in TRACKED_PRODUCTION_FILES}
    isolated_root = Path(tempfile.mkdtemp(prefix="hupan_csv_update_test_"))
    isolated_data = isolated_root / "data"
    copy_isolated_data(isolated_data)
    repository = DatasetRepository(isolated_data)
    service = DataUpdateService(repository)

    report: dict[str, object] = {}
    for kind, filename in [
        ("weather", "weather_20260611_20260824.csv"),
        ("calendar", "calendar_20260611_20260824.csv"),
    ]:
        path = CSV_ROOT / filename
        preview = service.preview(
            kind=kind,
            filename=path.name,
            content=path.read_bytes(),
        )
        assert preview["requires_sheet_selection"] is False
        assert preview["can_commit"] is True
        assert preview["summary"]["valid_rows"] == 75
        assert preview["summary"]["abnormal_rows"] == 0
        committed = service.commit(preview["preview_token"], None)
        assert committed["added_rows"] == 75
        report[kind] = {"preview": preview["summary"], "commit": committed}

    sales_path = CSV_ROOT / "a_hupan_sales_20260611_20260824.csv"
    sales_preview = service.preview(
        kind="sales",
        filename=sales_path.name,
        content=sales_path.read_bytes(),
        dataset_id="a_hupan",
    )
    assert sales_preview["requires_sheet_selection"] is False
    assert sales_preview["can_commit"] is True
    assert sales_preview["summary"]["valid_rows"] == 22799
    assert sales_preview["summary"]["abnormal_rows"] == 0
    assert sales_preview["summary"]["unmatched_dish_count"] == 0
    sales_commit = service.commit(sales_preview["preview_token"], "a_hupan")
    assert sales_commit["added_rows"] == 22799
    runtime = read_csv_utf8(
        repository.dataset_path("a_hupan", "runtime_actual_sales.csv")
    )
    assert len(runtime) == 22799
    assert int(runtime["年"].max()) == 2026
    assert int(runtime.loc[runtime["年"].eq(2026), "月"].max()) == 8
    assert int(runtime.loc[runtime["月"].eq(8), "日"].max()) == 24
    report["sales"] = {
        "preview": sales_preview["summary"],
        "commit": sales_commit,
    }

    after = {str(path): sha256(path) for path in TRACKED_PRODUCTION_FILES}
    assert before == after
    report["production_data_unchanged"] = True
    report["all_checks_passed"] = True
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
