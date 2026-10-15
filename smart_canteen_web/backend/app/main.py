from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .paths import FRONTEND_DIST, ensure_runtime_directories


LOGGER = logging.getLogger("smart_canteen")


def create_app(frontend_dist: Path = FRONTEND_DIST) -> FastAPI:
    ensure_runtime_directories()
    app = FastAPI(
        title="智慧食堂菜单推荐系统",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        messages = []
        for error in exc.errors():
            location = ".".join(map(str, error.get("loc", [])[1:]))
            messages.append(
                f"{location or '请求'}：{error.get('msg', '输入不合法')}"
            )
        return JSONResponse(
            status_code=422,
            content={"detail": "；".join(messages)},
        )

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc.detail)},
        )

    @app.exception_handler(Exception)
    async def unhandled_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        LOGGER.exception("内部错误：%s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "系统处理失败，请检查数据后重试"},
        )

    app.include_router(router)
    index = frontend_dist / "index.html"
    assets = frontend_dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    async def frontend_root():
        if index.exists():
            return FileResponse(index)
        return JSONResponse(
            status_code=503,
            content={"detail": "前端尚未构建，请先执行前端生产构建"},
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_fallback(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="接口不存在")
        if index.exists():
            return FileResponse(index)
        return JSONResponse(
            status_code=503,
            content={"detail": "前端尚未构建，请先执行前端生产构建"},
        )
    return app


app = create_app()
