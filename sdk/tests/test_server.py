import sqlite3
import time

from fastapi.testclient import TestClient

from smart_canteen_sdk.server import create_app


def test_http_flow_restart_and_idempotency(payload, tmp_path):
    path = tmp_path / "proposals.sqlite3"
    with TestClient(create_app(path)) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.post("/v1/data/validate", json=payload["data"]).status_code == 200
        response = client.post("/v1/recommendations", json=payload)
        assert response.status_code == 200, response.text
        proposal = response.json()
        assert "state" not in proposal and "state_sha256" not in proposal
    body = {"proposal_id": proposal["proposal_id"], "selection": {
        "high_frequency_dish_ids": [i["dish_id"] for i in proposal["high_frequency_menu"]]}}
    with TestClient(create_app(path)) as client:
        first = client.post("/v1/recommendations/finalize", json=body)
        assert first.status_code == 200, first.text
        second = client.post("/v1/recommendations/finalize", json=body)
        assert second.json() == first.json()
        body["selection"]["high_frequency_dish_ids"].pop()
        assert client.post("/v1/recommendations/finalize", json=body).status_code == 409
        body["proposal_id"] = "a" * 32
        assert client.post("/v1/recommendations/finalize", json=body).status_code == 404


def test_auth_expiry_and_invalid_requests(payload, tmp_path):
    path = tmp_path / "proposals.sqlite3"
    with TestClient(create_app(path, api_key="test-key")) as client:
        assert client.get("/v1/model-info").status_code == 401
        client.headers["X-API-Key"] = "test-key"
        assert client.get("/v1/model-info").status_code == 200
        assert client.post("/v1/recommendations", json={}).status_code == 422
        proposal = client.post("/v1/recommendations", json=payload).json()
        with sqlite3.connect(path) as db:
            db.execute("UPDATE proposals SET expires=?", (time.time() - 10,))
        body = {"proposal_id": proposal["proposal_id"], "selection": {"high_frequency_dish_ids": []}}
        assert client.post("/v1/recommendations/finalize", json=body).status_code == 410
        payload["data"]["sales_history"][0]["canteen_id"] = "other"
        response = client.post("/v1/recommendations", json=payload)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "CANTEEN_MISMATCH"
