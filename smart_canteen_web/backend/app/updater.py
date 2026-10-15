from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .dataset_repository import DatasetRepository
from .preprocessor import (
    PreviewArtifact,
    parse_upload,
    preprocess_calendar,
    preprocess_sales,
    preprocess_weather,
)


class DataUpdateService:
    def __init__(self, repository: DatasetRepository):
        self.repository = repository

    def preview(
        self,
        *,
        kind: str,
        filename: str,
        content: bytes,
        dataset_id: str | None = None,
        sheet_name: str | None = None,
    ) -> dict[str, Any]:
        if len(content) > 30 * 1024 * 1024:
            raise ValueError("上传文件不能超过30MB")
        parsed = parse_upload(filename, content, sheet_name)
        if parsed.requires_sheet_selection:
            return {
                "preview_token": None,
                "kind": kind,
                "dataset_id": dataset_id,
                "requires_sheet_selection": True,
                "sheet_names": parsed.sheet_names,
                "summary": {},
                "rows": [],
                "issues": [],
                "can_commit": False,
            }
        if parsed.frame is None:
            raise ValueError("上传文件没有可处理数据")
        if kind == "sales":
            if not dataset_id:
                raise ValueError("销售数据预览必须选择逻辑食堂")
            self.repository.dataset(dataset_id)
            artifact = preprocess_sales(
                parsed.frame,
                dataset_id,
                parsed.safe_source_name,
                self.repository,
            )
        elif kind == "weather":
            artifact = preprocess_weather(
                parsed.frame, parsed.safe_source_name, self.repository
            )
        elif kind == "calendar":
            artifact = preprocess_calendar(
                parsed.frame, parsed.safe_source_name, self.repository
            )
        else:
            raise ValueError("未知更新类型")
        token = self._stage(artifact)
        return {
            "preview_token": token,
            "kind": artifact.kind,
            "dataset_id": artifact.dataset_id,
            "requires_sheet_selection": False,
            "sheet_names": parsed.sheet_names,
            "summary": artifact.summary,
            "rows": artifact.rows,
            "issues": artifact.issues,
            "can_commit": artifact.can_commit,
        }

    def _stage(self, artifact: PreviewArtifact) -> str:
        token = uuid.uuid4().hex
        meta_path, csv_path = self.repository.stage_paths(token)
        metadata = {
            "token": token,
            "kind": artifact.kind,
            "dataset_id": artifact.dataset_id,
            "source_file": artifact.source_file,
            "can_commit": artifact.can_commit,
            "created_at": datetime.now().astimezone().isoformat(),
            "expires_at": (
                datetime.now().astimezone() + timedelta(hours=2)
            ).isoformat(),
            "summary": artifact.summary,
        }
        with self.repository.write_lock():
            self.repository._atomic_write_csv_unlocked(
                artifact.processed, csv_path, create_backup=False
            )
            self.repository._atomic_write_json_unlocked(
                metadata, meta_path, create_backup=False
            )
        return token

    def commit(
        self, preview_token: str, dataset_id: str | None
    ) -> dict[str, Any]:
        meta_path, csv_path = self.repository.stage_paths(preview_token)
        if not meta_path.exists() or not csv_path.exists():
            raise ValueError("预览已失效，请重新上传")
        metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if datetime.fromisoformat(metadata["expires_at"]) < datetime.now().astimezone():
            raise ValueError("预览已过期，请重新上传")
        if not metadata.get("can_commit"):
            raise ValueError("该预览存在阻止写入的问题")
        kind = metadata["kind"]
        staged_dataset = metadata.get("dataset_id")
        if kind == "sales":
            if dataset_id != staged_dataset:
                raise ValueError("确认写入的数据集与预览不一致")
        elif dataset_id not in {None, ""}:
            raise ValueError("公共特征更新不接受逻辑食堂参数")
        rows = pd.read_csv(csv_path, encoding="utf-8-sig")
        if kind == "sales":
            added, ignored, backup = self.repository.append_runtime_sales(
                staged_dataset, rows, metadata["source_file"]
            )
        else:
            added, backup = self.repository.commit_feature_rows(
                kind, rows, metadata["source_file"]
            )
            ignored = int(len(rows) - added)
        with self.repository.write_lock():
            if meta_path.exists():
                meta_path.unlink()
            if csv_path.exists():
                csv_path.unlink()
        return {
            "status": "success",
            "kind": kind,
            "added_rows": int(added),
            "ignored_duplicate_rows": int(ignored),
            "backup_created": bool(backup),
            "message": (
                f"写入完成：新增{added}行，忽略重复{ignored}行"
            ),
        }
