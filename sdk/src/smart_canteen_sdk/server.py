"""Optional HTTP adapter. Persistence and HTTP concerns stay outside the engine."""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import Field

from .engine import CanteenEngine, digest
from .errors import CanteenError
from .schemas import Contract, DataSnapshot, FinalResult, Proposal, PublicProposal, RecommendationRequest, Selection


class RecommendBody(Contract):
    request: RecommendationRequest
    data: DataSnapshot


class FinalizeBody(Contract):
    proposal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    selection: Selection


class ProposalRepository:
    """SQLite-backed proposals, restart-safe and idempotent across workers."""

    def __init__(self, path, ttl_seconds):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, expires REAL NOT NULL, payload TEXT NOT NULL, selection_hash TEXT, result TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, proposal):
        with self.connect() as db:
            db.execute("DELETE FROM proposals WHERE expires < ?", (time.time(),))
            db.execute("INSERT INTO proposals(id, expires, payload) VALUES(?,?,?)",
                       (proposal.proposal_id, time.time() + self.ttl_seconds, proposal.model_dump_json()))

    def finalize(self, engine, body):
        selection_hash = digest(body.selection.model_dump(mode="json"))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT expires, payload, selection_hash, result FROM proposals WHERE id=?", (body.proposal_id,)).fetchone()
            if row is None:
                raise HTTPException(404, detail={"code": "PROPOSAL_NOT_FOUND", "message": "候选方案不存在或已清理"})
            if row[0] < time.time():
                raise HTTPException(410, detail={"code": "PROPOSAL_EXPIRED", "message": "候选方案已过期"})
            if row[3] is not None:
                if row[2] != selection_hash:
                    raise HTTPException(409, detail={"code": "ALREADY_FINALIZED", "message": "该方案已按其他选择确认，请生成新方案"})
                return FinalResult.model_validate_json(row[3])
            result = engine.finalize(Proposal.model_validate_json(row[1]), body.selection)
            db.execute("UPDATE proposals SET selection_hash=?, result=? WHERE id=?",
                       (selection_hash, result.model_dump_json(), body.proposal_id))
            return result


def create_app(db_path=None, api_key=None, proposal_ttl_seconds=86400):
    """Create a server; callers supply persistent storage path and optional API key."""
    if proposal_ttl_seconds <= 0:
        raise ValueError("proposal_ttl_seconds must be positive")
    engine = CanteenEngine()
    repository = ProposalRepository(db_path or os.getenv("CANTEEN_PROPOSAL_DB", "canteen_sdk_state.sqlite3"), proposal_ttl_seconds)
    api_key = api_key if api_key is not None else os.getenv("CANTEEN_API_KEY")
    app = FastAPI(title="Smart Canteen SDK API", version="0.1.0", description="业务数量使用实际售卖单位；菜单确认结果由业务后端持久化。")
    app.state.engine, app.state.repository = engine, repository

    def authorize(x_api_key: str | None = Header(default=None)):
        if api_key and (x_api_key is None or not hmac.compare_digest(x_api_key, api_key)):
            raise HTTPException(401, detail={"code": "UNAUTHORIZED", "message": "API key required"})

    @app.exception_handler(CanteenError)
    async def canteen_error(_request, exc):
        return JSONResponse(status_code=422, content={"detail": {"code": exc.code, "message": str(exc)}})

    @app.get("/health")
    def health():
        with repository.connect() as db:
            db.execute("SELECT 1 FROM proposals LIMIT 1").fetchone()
        return {"status": "ok", "sdk_version": "0.1.0", "strategy_version": engine.get_model_info()["model_version"], "proposal_store": "sqlite"}

    @app.get("/v1/model-info", dependencies=[Depends(authorize)])
    def model_info():
        return engine.get_model_info()

    @app.post("/v1/data/validate", dependencies=[Depends(authorize)])
    def validate_data(data: DataSnapshot):
        return engine.validate_data(data)

    @app.post("/v1/recommendations", response_model=PublicProposal, dependencies=[Depends(authorize)])
    def recommend(body: RecommendBody):
        proposal = engine.recommend(body.request, body.data)
        repository.save(proposal)
        return proposal.public_dict()

    @app.post("/v1/recommendations/finalize", response_model=FinalResult, dependencies=[Depends(authorize)])
    def finalize(body: FinalizeBody):
        return repository.finalize(engine, body)

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="启动智慧食堂 SDK HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--proposal-db", default="canteen_sdk_state.sqlite3")
    parser.add_argument("--proposal-ttl-seconds", type=int, default=86400)
    args = parser.parse_args()
    uvicorn.run(create_app(args.proposal_db, proposal_ttl_seconds=args.proposal_ttl_seconds), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
