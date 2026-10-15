"""Exercise a running server over a real TCP port using synthetic data only."""
import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from demo import demo_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--output")
    args = parser.parse_args()
    results = []
    origin = args.base_url.rstrip("/")
    parsed = urlparse(origin)
    with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=5):
        results.append({"check": "tcp_connect", "status": "passed"})

    def call(method, path, data=None, expected=200):
        headers = {"Content-Type": "application/json"}
        if os.getenv("CANTEEN_API_KEY"):
            headers["X-API-Key"] = os.environ["CANTEEN_API_KEY"]
        body = json.dumps(data).encode() if data is not None else None
        start = time.monotonic()
        try:
            response = urlopen(Request(origin + path, data=body, headers=headers, method=method), timeout=60)
        except HTTPError as exc:
            response = exc
        with response:
            status, content = response.status, response.read()
        assert status == expected, (path, status, content[:2000])
        results.append({"method": method, "path": path, "status_code": status,
                        "elapsed_ms": round((time.monotonic() - start) * 1000, 2)})
        return json.loads(content) if content.startswith(b"{") else content.decode()

    health = call("GET", "/health")
    info = call("GET", "/v1/model-info")
    call("GET", "/docs")
    schema = call("GET", "/openapi.json")
    assert "/v1/recommendations" in schema["paths"]
    payload = demo_payload()
    call("POST", "/v1/data/validate", payload["data"])
    proposal = call("POST", "/v1/recommendations", payload)
    assert proposal["high_frequency_menu"] and "state" not in proposal
    body = {"proposal_id": proposal["proposal_id"], "selection": {
        "high_frequency_dish_ids": [i["dish_id"] for i in proposal["high_frequency_menu"]],
        "low_frequency_dish_ids": [], "redistribute_to_total_demand": False}}
    final = call("POST", "/v1/recommendations/finalize", body)
    assert final["audit"]["constraint_violation_count"] == 0
    assert call("POST", "/v1/recommendations/finalize", body) == final
    call("POST", "/v1/recommendations", {}, expected=422)
    body["proposal_id"] = "0" * 32
    call("POST", "/v1/recommendations/finalize", body, expected=404)
    report = {"tested_at": datetime.now(timezone.utc).isoformat(), "base_url": origin,
              "status": "passed", "health": health, "model": info,
              "menu_count": len(final["menu"]), "checks": results}
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
