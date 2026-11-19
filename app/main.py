"""FastAPI application with bounded request bodies and privacy-safe errors."""

import logging
import math
import re
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.exceptions import HTTPException

from app.api.routes import MAX_UPLOAD_BYTES, router
from app.services.llm_client import GeminiRequestError, GeminiResponseError
from app.services.profile_service import ResumeExtractionError

logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    """Pure ASGI middleware; no sync database operations on the event loop."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = (
            supplied if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", supplied) else uuid4().hex
        )
        scope.setdefault("state", {})["request_id"] = request_id

        async def identified(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ] + [(b"x-request-id", request_id.encode("ascii"))]
            await send(message)

        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES + 65536:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": "Request body is too large",
                        "request_id": request_id,
                    },
                )
                await response(scope, receive, identified)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": b"".join(chunks),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay, identified)


def create_app(*, search_timeout_seconds=120.0):
    if (
        isinstance(search_timeout_seconds, bool)
        or not isinstance(search_timeout_seconds, (int, float))
        or not math.isfinite(search_timeout_seconds)
        or not 0 < search_timeout_seconds <= 120
    ):
        raise ValueError("Search timeout must be finite and between 0 and 120 seconds")
    app = FastAPI(title="CareerMatch", version="0.1.0")
    app.state.search_timeout_seconds = search_timeout_seconds
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)

    def response(request, status, detail):
        request_id = getattr(request.state, "request_id", uuid4().hex)
        return JSONResponse(
            status_code=status,
            content={"detail": detail, "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # Pydantic's input/ctx may contain the complete résumé or credentials.
        return response(request, 422, "Invalid request")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        return response(request, error.status_code, error.detail)

    @app.exception_handler(ResumeExtractionError)
    async def extraction_error(request: Request, error: ResumeExtractionError):
        return response(
            request,
            422,
            "Resume could not be extracted; provide readable text or an unlocked PDF",
        )

    @app.exception_handler(GeminiRequestError)
    @app.exception_handler(GeminiResponseError)
    async def provider_error(request: Request, error: Exception):
        return response(
            request, 502, "Model service failed; no fabricated fallback was used"
        )

    @app.exception_handler(IntegrityError)
    async def integrity_error(request: Request, error: IntegrityError):
        return response(request, 409, "The update conflicts with existing data")

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, error: SQLAlchemyError):
        return response(request, 503, "Database is unavailable")

    @app.exception_handler(OverflowError)
    async def version_overflow(request: Request, error: OverflowError):
        return response(request, 409, "Version limit reached")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        logger.error(
            "Request failed request_id=%s error_type=%s",
            request.state.request_id,
            type(error).__name__,
        )
        return response(request, 500, "Internal server error")

    return app


app = create_app()
