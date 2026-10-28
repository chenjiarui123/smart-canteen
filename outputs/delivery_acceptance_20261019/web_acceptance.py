"""Isolated real-data HTTP acceptance; no training or production-data writes.

Runs FastAPI through TestClient, not a TCP socket. Temporary application data
and generated history are discarded; only aggregate evidence and two private
local CSV/Markdown examples remain in this acceptance folder.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "smart_canteen_web"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB / "backend"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(folder):
    return {str(p.relative_to(folder)): digest(p) for p in folder.rglob("*") if p.is_file()}


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Audit:
    def __init__(self):
        self.result = {"date": "2026-10-19", "transport": "FastAPI TestClient ASGI (no TCP)", "scope": "isolated copies of current delivery data; no training", "checks": [], "inference_scenarios": [], "uploads": {}, "errors": []}

    def check(self, name, passed, **evidence):
        self.result["checks"].append({"check": name, "passed": bool(passed), **evidence})
        if not passed:
            print(json.dumps({"check_failed": name, **evidence}, ensure_ascii=False), flush=True)
        self.save()

    def save(self):
        self.result["passed_checks"] = sum(x["passed"] for x in self.result["checks"])
        self.result["failed_checks"] = sum(not x["passed"] for x in self.result["checks"])
        (OUT / "web_acceptance.json").write_text(json.dumps(self.result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def report(self):
        r = self.result
        lines = ["# Web 交付链路验收（2026-10-19）", "", f"通过 {r['passed_checks']} 项；失败 {r['failed_checks']} 项。", "", "本次使用当前真实交付数据的临时副本，通过 FastAPI TestClient 执行 HTTP/ASGI 请求；没有占用真实 TCP 端口，没有训练或搜索参数。日期为 2026-06-11，天气为现有 UI 允许选项；这些是验证输入，不声称是当天实测天气。所有销售写入均为标记的合成验收数据，仅写副本。", "", "## 双食堂三餐", "", "| 数据集 | 餐次 | 状态 | 高频数 | 潜力数 | 耗时秒 |", "| --- | --- | --- | ---: | ---: | ---: |"]
        for item in r["inference_scenarios"]:
            lines.append(f"| {item['dataset']} | {item['meal']} | {item['status_code']} | {item.get('high_count', 0)} | {item.get('low_count', 0)} | {item['seconds']:.3f} |")
        lines += ["", "## 验收范围", "", "- 当前冻结模型与双食堂三餐推理、逐菜有限非负数量、历史日期隔离。", "- 同请求重复输出、10 次连续重复、4 路并发请求。", "- 不合法日期、目标日前无历史、跨食堂请求、缺必要天气字段拒绝。", "- 正式确认、CSV/Markdown 下载、克×50换算、原始模型数量历史持久化。", "- 重建服务对象后历史与下载恢复。", "- 销售／天气／校历预览及提交、重复提交不新增、缺上下文和跨数据集提交拒绝。", "- 原始 data、temp_recommendations 及 model 文件前后内容哈希比较。"]
        if "repeat_probe" in r:
            s = r["repeat_probe"]
            lines += ["", f"连续重复：{s['count']} 次，平均 {s['mean_seconds']:.3f} 秒，最大 {s['max_seconds']:.3f} 秒。"]
        if "concurrent_probe" in r:
            s = r["concurrent_probe"]
            lines += [f"并发：{s['count']} 路，总墙钟 {s['wall_seconds']:.3f} 秒，平均请求 {s['mean_seconds']:.3f} 秒。"]
        lines += ["", "## 逐项结果", "", "| 检查 | 结果 |", "| --- | --- |"]
        for c in r["checks"]:
            lines.append(f"| {c['check']} | {'通过' if c['passed'] else '失败'} |")
        if r["errors"]:
            lines += ["", "## 执行异常", "", *[f"- {e}" for e in r["errors"]]]
        lines += ["", "## 证据与边界", "", "聚合机器证据：`web_acceptance.json`。私有本地报告样例：`exports/`。临时销售副本已自动清理。", "", "本轮是有界烟测与短时重复／并发验收，不等于长时间压测或所有操作系统认证。数据仍截止 2026-06-10，当前业务精度与历史单位变化见 `data_audit.md`。"]
        (OUT / "web_acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(audit, temp_root):
    data = temp_root / "data"
    recommendations = temp_root / "recommendations"
    shutil.copytree(WEB / "data", data, ignore=shutil.ignore_patterns("backups", "staging", ".locks", "*.lock"))
    shutil.copytree(WEB / "temp_recommendations", recommendations, ignore=shutil.ignore_patterns("backups", "staging", "*.lock"))
    # Patch defaults before importing routes, so import-time service construction
    # and ensure_runtime_directories also operate entirely inside the copy.
    import app.paths as paths
    paths.DATA_ROOT = data
    paths.FEATURE_ROOT = data / "features"
    paths.STAGING_ROOT = data / "staging"
    paths.BACKUP_ROOT = data / "backups"
    paths.LOCK_ROOT = data / ".locks"
    paths.RECOMMENDATION_ROOT = recommendations
    from app.api import routes
    from app.dataset_repository import DatasetRepository
    from app.history_store import HistoryStore
    from app.recommender import RecommenderService
    from app.updater import DataUpdateService
    from app.main import create_app
    from fastapi.testclient import TestClient

    def rebuild():
        routes.repository = DatasetRepository(data)
        routes.history_store = HistoryStore(routes.repository, recommendations)
        routes.recommender = RecommenderService(routes.repository, routes.history_store)
        routes.updater = DataUpdateService(routes.repository)

    rebuild()
    client = TestClient(create_app(), raise_server_exceptions=False)
    def request(method, url, **kwargs):
        start = time.perf_counter()
        response = client.request(method, url, **kwargs)
        return response, time.perf_counter() - start

    health = client.get("/api/health")
    audit.check("health_current_data_and_model_integrity", health.status_code == 200 and health.json()["status"] == "ok" and health.json()["model"]["integrity_ok"])
    options = client.get("/api/options").json()
    ui = routes.repository.ui_options()
    weather = {}
    desired = {"temperature_c": 20, "humidity_pct": 60, "precipitation_mm": 0, "wind_speed_kmh": 5}
    for key, value in desired.items():
        weather[key] = min(options["weather"][key]["values"], key=lambda x: abs(x - value))
    audit.result["synthetic_context_weather"] = weather
    base = {"prediction_date": "2026-06-11", "school_break": False, "statutory_holiday": False, "adjusted_workday": False, **weather}
    requests, drafts = {}, {}
    for dataset in options["datasets"]:
        for meal in options["meals"]:
            key = (dataset["id"], meal)
            payload = dict(base, dataset_id=dataset["id"], canteen=dataset["canonical_canteen"], meal=meal)
            requests[key] = payload
            response, elapsed = request("POST", "/api/recommend", json=payload)
            row = {"dataset": dataset["id"], "meal": meal, "status_code": response.status_code, "seconds": round(elapsed, 6)}
            if response.status_code != 200:
                audit.check(f"inference_{key[0]}_{meal}", False, status_code=response.status_code, detail=response.json().get("detail"))
            else:
                obj = response.json()
                drafts[key] = obj
                row.update(high_count=len(obj["high_frequency_menu"]), low_count=len(obj["low_frequency_opportunities"]), low_frequency_status=obj["low_frequency_status"], history_end=obj["summary"]["history_end"], output_digest=payload_digest([obj["high_frequency_menu"], obj["low_frequency_opportunities"]]))
                nums_ok = all(math.isfinite(item[k]) and item[k] >= 0 for item in obj["menu"] for k in ["pred_qty", "pred_orders", "pred_rev"])
                names = [item["dish_name"] for item in obj["menu"]]
                audit.check(f"inference_{key[0]}_{meal}", obj["status"] == "draft" and row["high_count"] > 0 and obj["summary"]["history_end"] < base["prediction_date"] and nums_ok and len(names) == len(set(names)) and obj["summary"]["canteen"] == payload["canteen"] and obj["low_frequency_status"] == "ok")
            audit.result["inference_scenarios"].append(row)
            audit.save()
            print(json.dumps({"inference": key, "status": response.status_code, "seconds": round(elapsed, 3)}, ensure_ascii=False), flush=True)

    anchor = ("a_hupan", "午餐")
    anchor_payload = requests[anchor]
    original = drafts[anchor]
    repeat_times, stable = [], []
    for i in range(10):
        response, elapsed = request("POST", "/api/recommend", json=anchor_payload)
        repeat_times.append(elapsed)
        obj = response.json()
        stable.append(response.status_code == 200 and obj.get("high_frequency_menu") == original["high_frequency_menu"] and obj.get("low_frequency_opportunities") == original["low_frequency_opportunities"])
        if i in (0, 4, 9):
            print(json.dumps({"repeat_completed": i + 1, "seconds": round(elapsed, 3), "stable": stable[-1]}), flush=True)
    audit.result["repeat_probe"] = {"count": len(repeat_times), "successful_stable": sum(stable), "seconds": repeat_times, "mean_seconds": statistics.mean(repeat_times), "max_seconds": max(repeat_times)}
    audit.check("ten_identical_requests_are_stable", all(stable), count=10)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda _: request("POST", "/api/recommend", json=anchor_payload), range(4)))
    concurrent_times = [t for _, t in concurrent]
    concurrent_ok = [r.status_code == 200 and r.json().get("high_frequency_menu") == original["high_frequency_menu"] and r.json().get("low_frequency_opportunities") == original["low_frequency_opportunities"] for r, _ in concurrent]
    audit.result["concurrent_probe"] = {"count": 4, "wall_seconds": time.perf_counter() - start, "mean_seconds": statistics.mean(concurrent_times), "max_seconds": max(concurrent_times), "successful_stable": sum(concurrent_ok), "status_codes": [r.status_code for r, _ in concurrent]}
    audit.check("four_concurrent_inferences_are_stable", all(concurrent_ok))
    print(json.dumps({"concurrency_finished": audit.result["concurrent_probe"]}), flush=True)

    bad_requests = {"invalid_date_rejected": dict(anchor_payload, prediction_date="2026-02-30"), "cross_canteen_request_rejected": dict(anchor_payload, canteen="F基础园智慧餐厅"), "unknown_dataset_rejected": dict(anchor_payload, dataset_id="not_a_dataset"), "no_prior_history_rejected": dict(anchor_payload, prediction_date="2020-01-01")}
    missing = dict(anchor_payload)
    missing.pop("humidity_pct")
    bad_requests["missing_weather_input_rejected"] = missing
    for name, payload in bad_requests.items():
        response = client.post("/api/recommend", json=payload)
        audit.check(name, response.status_code in (400, 422), status_code=response.status_code)

    exports = OUT / "exports"
    exports.mkdir(exist_ok=True)
    finalized = {}
    for key, draft in drafts.items():
        final_request = {"proposal_id": draft["proposal_id"], "selected_high_frequency": [x["dish_name"] for x in draft["high_frequency_menu"]], "selected_low_frequency": [], "redistribute_to_total_demand": False}
        response = client.post("/api/recommend/finalize", json=final_request)
        audit.check(f"finalize_{key[0]}_{key[1]}", response.status_code == 200 and response.json().get("status") == "finalized", status_code=response.status_code)
        if response.status_code != 200:
            continue
        obj = response.json()
        finalized[key] = obj
        csv_response = client.get(obj["csv_download_url"])
        md_response = client.get(obj["markdown_download_url"])
        audit.check(f"downloads_{key[0]}_{key[1]}", csv_response.status_code == md_response.status_code == 200)
        csv_data = pd.read_csv(io.BytesIO(csv_response.content), encoding="utf-8-sig")
        md_rows = [[s.strip() for s in line.split("|")[1:-1]] for line in md_response.text.splitlines() if line.startswith(("| 高频菜单 |", "| 低频潜力菜 |"))]
        md_lookup = {row[1]: (row[2], float(row[3])) for row in md_rows}
        csv_lookup = csv_data.set_index("菜品名")["菜品销量"].to_dict()
        unit_ok = len(md_lookup) == len(obj["menu"]) == len(csv_data)
        gram_count = 0
        for item in obj["menu"]:
            gram = item["sale_unit"].strip() == "克"
            gram_count += int(gram)
            expected = item["pred_qty"] * (50 if gram else 1)
            unit_ok &= math.isclose(csv_lookup[item["dish_name"]], expected, rel_tol=1e-12, abs_tol=1e-8)
            unit_ok &= abs(md_lookup[item["dish_name"]][1] - expected) <= 0.0050001
            unit_ok &= md_lookup[item["dish_name"]][0] == item["sale_unit"].strip()
        audit.check(f"csv_markdown_business_units_{key[0]}_{key[1]}", unit_ok and gram_count > 0, gram_rows=gram_count)
        audit.check(f"final_menu_constraints_{key[0]}_{key[1]}", obj["combination_check"]["total_constraint_violation_count"] == 0)
        history = routes.repository.read_recommendation_history(key[0])
        saved = history.loc[history["request_id"].eq(obj["recommendation_id"])]
        saved_qty = saved.set_index("菜品名称")["pred_qty"].to_dict()
        history_ok = len(saved) == len(obj["menu"]) and all(math.isclose(saved_qty[item["dish_name"]], item["pred_qty"], rel_tol=1e-12, abs_tol=1e-8) for item in obj["menu"])
        audit.check(f"persisted_model_quantities_{key[0]}_{key[1]}", history_ok)
        if key[1] == "午餐":
            (exports / f"{key[0]}_20260611_lunch.csv").write_bytes(csv_response.content)
            (exports / f"{key[0]}_20260611_lunch.md").write_text(md_response.text, encoding="utf-8")
        obj["_download_digests"] = [hashlib.sha256(csv_response.content).hexdigest(), hashlib.sha256(md_response.content).hexdigest()]
    history_rows = {key: len(routes.repository.read_recommendation_history(key)) for key in ("a_hupan", "f_jichuyuan")}
    rebuild()
    client.close()
    client = TestClient(create_app(), raise_server_exceptions=False)
    restored = True
    for obj in finalized.values():
        responses = [client.get(obj["csv_download_url"]), client.get(obj["markdown_download_url"])]
        restored &= all(r.status_code == 200 for r in responses)
        restored &= [hashlib.sha256(r.content).hexdigest() for r in responses] == obj["_download_digests"]
    audit.check("downloads_survive_service_reconstruction", restored, count=len(finalized) * 2)
    audit.check("history_survives_service_reconstruction", history_rows == {key: len(routes.repository.read_recommendation_history(key)) for key in history_rows}, rows=history_rows)

    def preview(kind, frame, dataset=None):
        kwargs = {"files": {"file": ("synthetic_acceptance.csv", frame.to_csv(index=False).encode("utf-8-sig"), "text/csv")}}
        if dataset:
            kwargs["data"] = {"dataset_id": dataset}
        return client.post(f"/api/{kind}/preview", **kwargs)

    synthetic_sale = pd.DataFrame([{"日期": "2026-06-11", "餐次": "午餐", "菜品名称": "验收合成菜_不可用于经营", "菜品总销量": 1.0, "菜品售卖单位": "份", "订单数量": 1.0, "菜品单价": "1元/份", "销售额": 1.0, "平均取菜量": 1.0}])
    response = preview("data", synthetic_sale, "a_hupan")
    obj = response.json()
    issues = {x["type"] for x in obj["issues"]}
    audit.check("sales_missing_stored_context_rejected", response.status_code == 200 and not obj["can_commit"] and {"missing_weather", "missing_calendar"} <= issues)
    bad_commit = client.post("/api/data/commit", json={"preview_token": obj["preview_token"], "dataset_id": "a_hupan"})
    audit.check("blocked_preview_cannot_commit", bad_commit.status_code == 400)
    # Feature uploader validates historical raw options, unlike rounded UI options.
    raw_weather = {key: min(ui["weather"][key]["values"], key=lambda x: abs(x - desired[key])) for key in desired}
    weather_frame = pd.DataFrame([{"日期": "2026-06-11", "气温（℃）": raw_weather["temperature_c"], "相对湿度（%）": raw_weather["humidity_pct"], "降水量（毫米）": raw_weather["precipitation_mm"], "10米风速（公里/小时）": raw_weather["wind_speed_kmh"]}])
    calendar_frame = pd.DataFrame([{"日期": "2026-06-11", "星期": "星期四", "是否校历寒暑假": 0, "校历假期类型": "非寒暑假", "是否法定节假日假期": 0, "法定节假日名称": "非节假日", "是否调休工作日": 0}])
    before_f_sales = digest(data / "datasets/f_jichuyuan/runtime_actual_sales.csv")
    for kind, frame, dataset in [("weather", weather_frame, None), ("calendar", calendar_frame, None), ("data", synthetic_sale, "a_hupan")]:
        first, second = preview(kind, frame, dataset), preview(kind, frame, dataset)
        a, b = first.json(), second.json()
        preview_ok = first.status_code == second.status_code == 200 and a["can_commit"] and b["can_commit"]
        audit.check(f"{kind}_preview_accepts_valid_row", preview_ok)
        if not preview_ok:
            audit.result["uploads"][kind] = {"preview_failed": True, "issue_types": [x["type"] for x in a.get("issues", [])]}
            continue
        if dataset:
            cross = client.post("/api/data/commit", json={"preview_token": a["preview_token"], "dataset_id": "f_jichuyuan"})
            audit.check("sales_cross_dataset_commit_rejected", cross.status_code == 400)
        commit_payload = {"preview_token": a["preview_token"]}
        if dataset:
            commit_payload["dataset_id"] = dataset
        committed = client.post(f"/api/{kind}/commit", json=commit_payload)
        commit_payload["preview_token"] = b["preview_token"]
        repeated = client.post(f"/api/{kind}/commit", json=commit_payload)
        repeated_preview = preview(kind, frame, dataset)
        audit.check(f"{kind}_commit_and_duplicate_are_idempotent", committed.status_code == repeated.status_code == 200 and committed.json()["added_rows"] == 1 and repeated.json()["added_rows"] == 0 and not repeated_preview.json()["can_commit"])
        audit.result["uploads"][kind] = {"commit": committed.json(), "duplicate_commit": repeated.json(), "repeat_preview_can_commit": repeated_preview.json()["can_commit"]}
    audit.check("sales_update_does_not_change_other_canteen", before_f_sales == digest(data / "datasets/f_jichuyuan/runtime_actual_sales.csv"))
    audit.check("synthetic_sales_written_once", len(csv := pd.read_csv(data / "datasets/a_hupan/runtime_actual_sales.csv", encoding="utf-8-sig")) == 1 and set(csv["食堂名称"]) == {"A湖畔智慧餐厅"})
    context = client.get("/api/context/2026-06-11").json()
    audit.check("committed_weather_calendar_are_readable", context["found_weather"] and context["found_calendar"])
    audit.check("health_after_updates", client.get("/api/health").json()["status"] == "ok")
    client.close()


def main():
    audit = Audit()
    protected = {name: snapshot(WEB / name) for name in ("data", "temp_recommendations", "model")}
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="canteen_acceptance_") as temp:
            run(audit, Path(temp))
    except Exception as exc:
        audit.result["errors"].append(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        for name, baseline in protected.items():
            current = snapshot(WEB / name)
            changed = sorted(set(baseline) ^ set(current) | {k for k in baseline.keys() & current.keys() if baseline[k] != current[k]})
            audit.check(f"original_{name}_byte_identical", baseline == current, checked_files=len(baseline), changed_files=changed)
        audit.result["total_seconds"] = time.perf_counter() - start
        audit.result["completed"] = not audit.result["errors"]
        audit.save()
        audit.report()
        print(json.dumps({"completed": audit.result["completed"], "passed": audit.result["passed_checks"], "failed": audit.result["failed_checks"], "seconds": audit.result["total_seconds"]}), flush=True)
    return 0 if not audit.result["errors"] and not audit.result["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
