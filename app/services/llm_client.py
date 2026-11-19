"""Synchronous Gemini provider with bounded retries and explicit offline provenance.

Rate limits and duplicate-request locks are shared across threads in this process.
Each SQLite operation owns its connection. Separate worker processes must coordinate
their quota externally. Cache hits represent zero new API tokens or cost.
"""

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable
from contextlib import closing, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self, TypeVar
from urllib.parse import quote, quote_plus
from uuid import uuid4

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import Settings, settings
from app.schemas.contracts import LLMUsageMeta

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
_LOGGER = logging.getLogger(__name__)
# The SDK also sends this as X-Server-Timeout, so a short HTTP timeout can
# prematurely end model reasoning on the server, not just the local connection.
_REQUEST_TIMEOUT_MS = 120_000
_FLIGHT_LOCKS = tuple(threading.Lock() for _ in range(64))
_LIMITERS_LOCK = threading.Lock()
_LIMITERS: dict[str, "_RateLimiter"] = {}
_TASK_TYPES = frozenset(
    {
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
        "CODE_RETRIEVAL_QUERY",
    }
)


class GeminiRequestError(RuntimeError):
    """Public message omits provider content; diagnostic is for synthetic tests only.

    The optional diagnostic has credentials redacted but can contain provider-
    quoted input. Do not include it in application logs or resume trace endpoints.
    """

    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


def _provider_diagnostic(error: errors.APIError, api_key: str) -> str:
    message = (
        error.message
        if isinstance(error.message, str)
        else "No provider message available"
    )
    secrets = {api_key, api_key.strip(), quote(api_key, safe=""), quote_plus(api_key)}
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(
        r"(?:AIza[A-Za-z0-9_-]+|AQ\.[A-Za-z0-9._~-]+)", "[REDACTED]", message
    )
    return " ".join(message.split())[:2000]


class GeminiResponseError(RuntimeError):
    """A response or cache record violates the provider's output contract."""


class CostBudgetExceeded(Exception):
    """No paid request was started because its maximum charge would not fit."""


class CostAccountingError(Exception):
    """Spending is uncertain; retain the reservation and stop paid requests."""


class RequestAccounting:
    """Opt-in, thread-safe ledger for physical requests, including failed output.

    Reserve the provider's full advertised input limit and a capped output limit.
    This deliberately stops early when the remaining budget cannot cover a whole
    request. Unknown charges retain their reservation, never become zero dollars.
    Callbacks receive only cost/token/provenance metadata, never input or output.
    """

    def __init__(self, limit_usd: float, *, max_output_tokens=8192, on_change=None):
        if not math.isfinite(limit_usd) or limit_usd < 0:
            raise ValueError("Cost limit must be finite and nonnegative")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("Output token cap must be positive")
        self.limit = Decimal(str(limit_usd))
        self.max_output_tokens = max_output_tokens
        self.records = []
        self._lock = threading.RLock()
        self._on_change = on_change

    def snapshot(self):
        with self._lock:
            return deepcopy(self.records)

    def _notify(self):
        if self._on_change is not None:
            self._on_change(self.snapshot())

    def reserve(self, amount: float, model_id: str) -> int:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Reservation must be finite and nonnegative")
        with self._lock:
            if any(r["usage"] is None for r in self.records):
                raise CostAccountingError("Unsettled request; refusing more spending")
            spent = sum(
                (Decimal(str(r["reserved_usd"])) for r in self.records), Decimal(0)
            )
            if spent + Decimal(str(amount)) > self.limit or spent >= self.limit:
                raise CostBudgetExceeded(
                    "Request reservation exceeds remaining evaluation budget"
                )
            self.records.append(
                {
                    "model_id": model_id,
                    "reserved_usd": amount,
                    "usage": None,
                    "physical_request": True,
                }
            )
            self._notify()
            return len(self.records) - 1

    def settle(self, ticket: int, usage: LLMUsageMeta) -> None:
        with self._lock:
            record = self.records[ticket]
            if record["usage"] is not None:
                raise CostAccountingError("Request already settled")
            over = usage.estimated_cost_usd > record["reserved_usd"] + 1e-12
            record["usage"] = usage.model_dump(mode="json")
            record["reserved_usd"] = usage.estimated_cost_usd
            self._notify()
            if over:
                raise CostAccountingError(
                    "Provider usage exceeded its advertised request bound"
                )

    def record_free(self, usage: LLMUsageMeta) -> None:
        if usage.estimated_cost_usd or usage.total_tokens:
            raise ValueError("Free receipt must have zero new tokens and cost")
        with self._lock:
            self.records.append(
                {
                    "model_id": usage.model_id,
                    "reserved_usd": 0.0,
                    "usage": usage.model_dump(mode="json"),
                    "physical_request": False,
                }
            )
            self._notify()

    def note_failure(
        self, ticket: int, *, category: str, http_status=None, api_status=None
    ):
        """Persist allowlisted diagnostics, never provider messages or requests.

        Keep the unsettled reservation intact. A diagnostic is not evidence that
        the request was free, and must not enable another paid attempt.
        """
        if category not in {
            "http_error",
            "timeout",
            "transport_error",
            "connection_error",
            "invalid_usage",
            "missing_usage",
        }:
            raise ValueError("Unknown provider failure category")
        safe_http = (
            http_status
            if type(http_status) is int and 100 <= http_status <= 599
            else None
        )
        allowed_statuses = {
            "INVALID_ARGUMENT",
            "FAILED_PRECONDITION",
            "UNAUTHENTICATED",
            "PERMISSION_DENIED",
            "NOT_FOUND",
            "RESOURCE_EXHAUSTED",
            "INTERNAL",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "CANCELLED",
        }
        safe_api = (
            api_status
            if isinstance(api_status, str) and api_status in allowed_statuses
            else None
        )
        with self._lock:
            self.records[ticket]["failure"] = {
                "category": category,
                "http_status": safe_http,
                "api_status": safe_api,
            }
            self._notify()


_ACCOUNTING: ContextVar[RequestAccounting | None] = ContextVar(
    "gemini_accounting", default=None
)


@contextmanager
def account_requests(ledger: RequestAccounting):
    """Apply one ledger to all synchronous services in this execution context."""
    token = _ACCOUNTING.set(ledger)
    try:
        yield ledger
    finally:
        _ACCOUNTING.reset(token)


class _RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_start: float | None = None
        self._interval = 0.0

    def wait(self, interval: float) -> None:
        with self._lock:
            self._interval = max(self._interval, interval)
            if self._last_start is not None:
                delay = self._last_start + self._interval - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self._last_start = time.monotonic()


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))


def _chat_rates(model_id: str, today: date | None = None) -> tuple[float, float]:
    """Standard paid USD per million tokens; output includes thinking.

    Source checked 2026-11-19: https://ai.google.dev/gemini-api/docs/pricing
    Flash introductory prices expire 2026-12-31. These are estimates even for
    free-tier users; billing-tier discounts are not inferred from an API key.
    """
    if model_id == "gemini-3.8-flash":
        return (
            (0.75, 3.75)
            if (today or datetime.now(UTC).date()) < date(2027, 1, 1)
            else (1.50, 7.50)
        )
    if model_id == "gemini-3.5-flash-lite":
        return 0.30, 2.50
    raise ValueError("No verified pricing configured for the requested chat model")


def _image_mime(data: bytes) -> str:
    if not isinstance(data, bytes) or not data:
        raise ValueError("Each image must be nonempty bytes")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Images must be PNG, JPEG, or WebP")


def _validate_schema_depth(root: dict) -> None:
    def visit(node: dict, depth: int, refs: frozenset[str]) -> None:
        if "$ref" in node:
            reference = node["$ref"]
            if reference in refs or not reference.startswith("#/$defs/"):
                raise ValueError(
                    "Recursive or external response schemas are unsupported"
                )
            visit(
                root["$defs"][reference.rsplit("/", 1)[-1]], depth, refs | {reference}
            )
            return
        if node.get("type") == "object":
            depth += 1
            if depth > 3:
                raise ValueError(
                    "Response schema exceeds Batch -> Item -> FlatSubItem depth"
                )
            for child in node.get("properties", {}).values():
                visit(child, depth, refs)
        if "items" in node:
            visit(node["items"], depth, refs)
        for keyword in ("anyOf", "oneOf", "allOf"):
            for child in node.get(keyword, []):
                visit(child, depth, refs)

    visit(root, 0, frozenset())


def _gemini_response_schema(schema: dict) -> dict:
    """Adapt strict Pydantic JSON Schema to Gemini's response_schema subset.

    The SDK serializes additionalProperties=False as additional_properties,
    which this endpoint rejects. Only omit that wire-level keyword; validation
    always uses the original Pydantic class, including its extra='forbid' rules.
    """
    result = deepcopy(schema)

    def clean(node: dict) -> None:
        if "additionalProperties" in node:
            if node["additionalProperties"] is not False:
                raise ValueError(
                    "Gemini response_schema requires named object fields, not arbitrary dictionaries"
                )
            del node["additionalProperties"]
        for keyword in ("properties", "$defs", "definitions", "patternProperties"):
            for child in node.get(keyword, {}).values():
                clean(child)
        if isinstance(node.get("items"), dict):
            clean(node["items"])
        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for child in node.get(keyword, []):
                clean(child)

    clean(result)
    return result


def _mock_value(schema: dict, root: dict, seed: str, depth: int = 0) -> Any:
    """Synthesize schema-valid fixtures, never inferred candidate/job evidence.

    Collections are empty unless their schema requires items. Unsupported custom
    constraints fail model validation rather than returning an invalid fixture.
    """
    if depth > 12:
        raise ValueError("Recursive/deep schemas are unsupported in offline mode")
    if "$ref" in schema:
        return _mock_value(
            root["$defs"][schema["$ref"].rsplit("/", 1)[-1]], root, seed, depth + 1
        )
    if "default" in schema:
        return schema["default"]
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        option = next(
            (s for s in schema["anyOf"] if s.get("type") == "null"), schema["anyOf"][0]
        )
        return _mock_value(option, root, seed, depth + 1)
    kind = schema.get("type")
    if kind == "object":
        return {
            name: _mock_value(value, root, seed, depth + 1)
            for name, value in schema.get("properties", {}).items()
            if name in schema.get("required", [])
        }
    if kind == "array":
        return [
            _mock_value(schema["items"], root, seed, depth + 1)
            for _ in range(schema.get("minItems", 0))
        ]
    if kind == "string":
        value = f"offline_mock_{seed[:16]}"
        value = value.ljust(schema.get("minLength", 0), "_")
        return value[: schema.get("maxLength", len(value))]
    if kind in {"integer", "number"}:
        value = schema.get("minimum", 0)
        if "exclusiveMinimum" in schema:
            value = schema["exclusiveMinimum"] + 1
        if "maximum" in schema:
            value = min(value, schema["maximum"])
        if "exclusiveMaximum" in schema:
            value = min(value, schema["exclusiveMaximum"] - 1)
        return int(value) if kind == "integer" else float(value)
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    raise ValueError("Unsupported offline response schema")


def _offline_embedding(text: str) -> list[float]:
    """Signed, SHA-256 keyword hashing; independent of Python hash/random seeds."""
    tokens = Counter(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))
    if not tokens:
        tokens = Counter({"<empty>": 1})
    vector = [0.0] * 768
    for token, count in sorted(tokens.items()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # Multiple coordinates per word reduce collisions in small fixtures.
        for offset in range(0, 32, 4):
            index = int.from_bytes(digest[offset : offset + 2], "big") % 768
            vector[index] += count * (1 if digest[offset + 2] & 1 else -1)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        norm = 1.0
    return [value / norm for value in vector]


class GeminiService:
    """Use as a context manager or call close() when its SDK client is no longer used.

    embed_texts retains the blueprint's vector-only return signature. Inspect
    used_mock_fallback and last_embedding_usage on this service for provenance.
    last_embedding_usage is thread-local and None if live token usage is unavailable.
    """

    def __init__(self, *, config: Settings | None = None, client: Any = None) -> None:
        self._config = (config or settings).model_copy(deep=True)
        if self._config.EMBEDDING_DIM != 768:
            raise ValueError("CareerMatch requires EMBEDDING_DIM=768")
        self._offline = not bool(self._config.GEMINI_API_KEY.strip())
        if self.used_mock_fallback and client is not None:
            raise ValueError("Cannot inject a live client when the API key is empty")
        self._local = threading.local()
        self._closed = False
        self._owns_client = client is None
        self._client = client
        self._transport: httpx.Client | None = None
        if self._config.LLM_CACHE_PATH == ":memory:":
            raise ValueError("LLM_CACHE_PATH must be a persistent SQLite file")
        self._cache_path = Path(self._config.LLM_CACHE_PATH)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._cache_path, timeout=30)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS structured_cache "
                "(cache_key TEXT PRIMARY KEY, output_json TEXT NOT NULL, usage_json TEXT NOT NULL)"
            )
        fingerprint = hashlib.sha256(self._config.GEMINI_API_KEY.encode()).hexdigest()
        with _LIMITERS_LOCK:
            self._limiter = _LIMITERS.setdefault(fingerprint, _RateLimiter())
        if not self.used_mock_fallback and self._client is None:
            # Explicit httpx transport keeps transport exceptions predictable.
            self._transport = httpx.Client(timeout=_REQUEST_TIMEOUT_MS / 1000)
            try:
                self._client = genai.Client(
                    api_key=self._config.GEMINI_API_KEY,
                    vertexai=False,
                    http_options=types.HttpOptions(
                        timeout=_REQUEST_TIMEOUT_MS,
                        retry_options=types.HttpRetryOptions(attempts=1),
                        httpx_client=self._transport,
                    ),
                )
            except (ValueError, TypeError, ImportError):
                self._transport.close()
                raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                if self._owns_client and self._client is not None:
                    self._client.close()
            finally:
                if self._transport is not None:
                    self._transport.close()

    @property
    def used_mock_fallback(self) -> bool:
        return self._offline

    @property
    def last_embedding_usage(self) -> LLMUsageMeta | None:
        usage = getattr(self._local, "embedding_usage", None)
        return usage.model_copy(deep=True) if usage is not None else None

    @property
    def last_generation_usage(self) -> LLMUsageMeta | None:
        """Usage survives malformed/truncated structured output; thread-local."""
        usage = getattr(self._local, "generation_usage", None)
        return usage.model_copy(deep=True) if usage is not None else None

    def _accounted_request(
        self, call, *, model_id, operation, usage_of, input_count=1, config=None
    ):
        ledger = _ACCOUNTING.get()
        if ledger is None:
            response = self._request(call, operation=operation)
            return response, usage_of(response)
        # Model metadata is a non-generating request. A missing/invalid limit
        # blocks evaluation rather than substituting a guessed token estimate.
        model = self._request(
            lambda: self._client.models.get(model=model_id), operation="model_limits"
        )
        input_limit = model.input_token_limit
        if type(input_limit) is not int or input_limit <= 0:
            raise CostAccountingError("Provider omitted model input limit")
        if config is not None:
            output_limit = model.output_token_limit
            if type(output_limit) is not int or output_limit <= 0:
                raise CostAccountingError("Provider omitted model output limit")
            config.max_output_tokens = min(ledger.max_output_tokens, output_limit)
            input_rate, output_rate = _chat_rates(model_id)
            maximum = (
                input_limit * input_rate + config.max_output_tokens * output_rate
            ) / 1_000_000
        else:
            maximum = input_count * input_limit * 0.20 / 1_000_000
        ticket = ledger.reserve(maximum, model_id)

        def once():
            try:
                return call()
            except errors.APIError as error:
                ledger.note_failure(
                    ticket,
                    category="http_error",
                    http_status=error.code,
                    api_status=error.status,
                )
                raise CostAccountingError(
                    "Provider HTTP request failed without a usage receipt"
                ) from None
            except (
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
            ) as error:
                category = (
                    "timeout"
                    if isinstance(error, (httpx.TimeoutException, TimeoutError))
                    else "connection_error"
                    if isinstance(error, ConnectionError)
                    else "transport_error"
                )
                ledger.note_failure(ticket, category=category)
                # No usage receipt means no safe retry: the first attempt may
                # already have been billed. Keep its conservative reservation.
                raise CostAccountingError(
                    "Provider attempt failed without a usage receipt"
                ) from None

        response = self._request(once, operation=operation)
        try:
            usage = usage_of(response)
        except GeminiResponseError:
            ledger.note_failure(ticket, category="invalid_usage")
            raise
        if usage is None:
            ledger.note_failure(ticket, category="missing_usage")
            raise CostAccountingError("Provider omitted billable usage")
        ledger.settle(ticket, usage)
        return response, usage

    def _request(self, call: Callable[[], R], *, operation: str) -> R:
        # Correlate concurrent requests without logging keys, inputs, responses,
        # or exception text. Timings are telemetry, not deterministic model output.
        request_id = uuid4().hex
        request_started = time.monotonic()
        attempt = 0

        def limited_call() -> R:
            nonlocal attempt
            attempt += 1
            waiting_started = time.monotonic()
            self._limiter.wait(60.0 / self._config.LLM_RPM_LIMIT)
            started = time.monotonic()
            _LOGGER.info(
                "Gemini attempt start request_id=%s operation=%s attempt=%d/3 "
                "rate_wait_s=%.3f timeout_s=%.1f",
                request_id,
                operation,
                attempt,
                started - waiting_started,
                _REQUEST_TIMEOUT_MS / 1000,
            )
            try:
                result = call()
            except (
                errors.APIError,
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
            ) as error:
                outcome = (
                    f"http_{error.code}"
                    if isinstance(error, errors.APIError)
                    else type(error).__name__
                )
                _LOGGER.info(
                    "Gemini attempt end request_id=%s operation=%s attempt=%d/3 "
                    "outcome=%s elapsed_s=%.3f request_elapsed_s=%.3f",
                    request_id,
                    operation,
                    attempt,
                    outcome,
                    time.monotonic() - started,
                    time.monotonic() - request_started,
                )
                raise
            _LOGGER.info(
                "Gemini attempt end request_id=%s operation=%s attempt=%d/3 "
                "outcome=success elapsed_s=%.3f request_elapsed_s=%.3f",
                request_id,
                operation,
                attempt,
                time.monotonic() - started,
                time.monotonic() - request_started,
            )
            return result

        try:
            return Retrying(
                retry=retry_if_exception(_is_transient),
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=8, jitter=1),
                reraise=True,
            )(limited_call)
        except errors.APIError as exc:
            raise GeminiRequestError(
                f"Gemini {operation} failed (HTTP {exc.code}); no offline fallback used",
                diagnostic=_provider_diagnostic(exc, self._config.GEMINI_API_KEY),
            ) from None
        except (httpx.TransportError, TimeoutError, ConnectionError):
            raise GeminiRequestError(
                f"Gemini {operation} transport failed after at most 3 attempts; no offline fallback used"
            ) from None

    def _cache_key(
        self,
        model_id: str,
        thinking_level: str,
        prompt: str,
        system_instruction: str,
        schema: dict,
        images: list[bytes],
    ) -> str:
        # Framed JSON avoids concatenation collisions and includes both prompts.
        components = [
            "structured-v1",
            self.used_mock_fallback,
            model_id,
            thinking_level,
            system_instruction,
            prompt,
            schema,
            [hashlib.sha256(data).hexdigest() for data in images],
        ]
        if _ACCOUNTING.get() is not None:
            components.append(
                {"max_output_tokens": _ACCOUNTING.get().max_output_tokens}
            )
        return hashlib.sha256(
            json.dumps(
                components, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()

    def _read_cache(
        self, key: str, response_schema: type[T]
    ) -> tuple[T, LLMUsageMeta] | None:
        with closing(sqlite3.connect(self._cache_path, timeout=30)) as db:
            row = db.execute(
                "SELECT output_json, usage_json FROM structured_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            output = response_schema.model_validate_json(row[0])
            usage = LLMUsageMeta.model_validate_json(row[1])
        except ValidationError:
            raise GeminiResponseError(
                "Invalid structured cache record; remove the damaged cache entry"
            ) from None
        if usage.used_mock_fallback != self.used_mock_fallback:
            raise GeminiResponseError("Cache provenance mismatch")
        return output, LLMUsageMeta(
            model_id=usage.model_id,
            cached=True,
            used_mock_fallback=usage.used_mock_fallback,
        )

    def _write_cache(self, key: str, output: BaseModel, usage: LLMUsageMeta) -> None:
        with closing(sqlite3.connect(self._cache_path, timeout=30)) as db, db:
            db.execute(
                "INSERT INTO structured_cache VALUES (?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET output_json=excluded.output_json, "
                "usage_json=excluded.usage_json",
                (key, output.model_dump_json(), usage.model_dump_json()),
            )

    def _usage(
        self, response: types.GenerateContentResponse, model_id: str
    ) -> LLMUsageMeta:
        metadata = response.usage_metadata
        if metadata is None or metadata.prompt_token_count is None:
            raise GeminiResponseError(
                "Gemini omitted token usage; cannot report an honest cost"
            )
        raw_counts = (
            metadata.prompt_token_count,
            metadata.candidates_token_count,
            metadata.thoughts_token_count,
            metadata.total_token_count,
            metadata.cached_content_token_count,
        )
        if any(value is not None and value < 0 for value in raw_counts):
            raise GeminiResponseError("Gemini returned negative token usage")
        if (
            metadata.total_token_count is None
            and metadata.candidates_token_count is None
        ):
            raise GeminiResponseError("Gemini omitted output token usage")
        prompt = metadata.prompt_token_count
        completion = (metadata.candidates_token_count or 0) + (
            metadata.thoughts_token_count or 0
        )
        total = (
            metadata.total_token_count
            if metadata.total_token_count is not None
            else prompt + completion
        )
        completion = max(completion, total - prompt)
        cached_prompt = metadata.cached_content_token_count or 0
        if (
            min(prompt, completion, total, cached_prompt) < 0
            or cached_prompt > prompt
            or total < prompt + completion
        ):
            raise GeminiResponseError("Gemini returned inconsistent token usage")
        input_rate, output_rate = _chat_rates(model_id)
        cost = (
            (prompt - cached_prompt) * input_rate
            + cached_prompt * input_rate * 0.1
            + completion * output_rate
        ) / 1_000_000
        return LLMUsageMeta(
            model_id=model_id,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            estimated_cost_usd=cost,
            used_mock_fallback=False,
        )

    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: type[T],
        model_id: str | None = None,
        thinking_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM",
        image_bytes_list: list[bytes] | None = None,
    ) -> tuple[T, LLMUsageMeta]:
        self._local.generation_usage = None
        if self._closed:
            raise RuntimeError("GeminiService is closed")
        if not isinstance(prompt, str) or not isinstance(system_instruction, str):
            raise TypeError("Prompt and system instruction must be strings")
        if not isinstance(response_schema, type) or not issubclass(
            response_schema, BaseModel
        ):
            raise TypeError("response_schema must be a Pydantic model class")
        selected_model = self._config.CHAT_MODEL_ID if model_id is None else model_id
        _chat_rates(selected_model)  # Reject unknown pricing before spending tokens.
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if selected_model == "gemini-3.5-flash-lite":
            allowed.add("MINIMAL")
        if thinking_level not in allowed:
            raise ValueError("Unsupported thinking_level for the selected model")
        if image_bytes_list is not None and not isinstance(image_bytes_list, list):
            raise TypeError("image_bytes_list must be a list of image bytes")
        images = list(image_bytes_list or [])
        mimes = [_image_mime(data) for data in images]
        if not prompt.strip() and not images:
            raise ValueError("A nonempty prompt or at least one image is required")
        schema = response_schema.model_json_schema()
        _validate_schema_depth(schema)
        key = self._cache_key(
            selected_model, thinking_level, prompt, system_instruction, schema, images
        )
        with _FLIGHT_LOCKS[int(key[:8], 16) % len(_FLIGHT_LOCKS)]:
            cached = self._read_cache(key, response_schema)
            if cached is not None:
                self._local.generation_usage = cached[1]
                if _ACCOUNTING.get() is not None:
                    _ACCOUNTING.get().record_free(cached[1])
                return cached
            if self.used_mock_fallback:
                output = response_schema.model_validate(
                    _mock_value(schema, schema, key)
                )
                usage = LLMUsageMeta(model_id=selected_model, used_mock_fallback=True)
                if _ACCOUNTING.get() is not None:
                    _ACCOUNTING.get().record_free(usage)
            else:
                parts = [types.Part.from_text(text=prompt)] if prompt else []
                parts.extend(
                    types.Part.from_bytes(data=data, mime_type=mime)
                    for data, mime in zip(images, mimes, strict=True)
                )
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                    response_mime_type="application/json",
                    response_schema=_gemini_response_schema(schema),
                )

                def capture_usage(response):
                    usage = self._usage(response, selected_model)
                    self._local.generation_usage = usage
                    return usage

                response, usage = self._accounted_request(
                    lambda: self._client.models.generate_content(
                        model=selected_model,
                        contents=[types.Content(role="user", parts=parts)],
                        config=config,
                    ),
                    operation=f"generate_content/{selected_model}",
                    model_id=selected_model,
                    usage_of=capture_usage,
                    config=config,
                )
                candidates = response.candidates or []
                if candidates and candidates[0].finish_reason not in (
                    None,
                    types.FinishReason.STOP,
                ):
                    raise GeminiResponseError(
                        "Gemini response was blocked, truncated, or unfinished"
                    )
                try:
                    if response.parsed is not None:
                        raw = (
                            response.parsed.model_dump()
                            if isinstance(response.parsed, BaseModel)
                            else response.parsed
                        )
                        output = response_schema.model_validate(raw)
                    elif response.text:
                        output = response_schema.model_validate_json(response.text)
                    else:
                        raise GeminiResponseError(
                            "Gemini returned no structured content"
                        )
                except ValidationError:
                    raise GeminiResponseError(
                        "Gemini output did not match response_schema"
                    ) from None
            self._local.generation_usage = usage
            self._write_cache(key, output, usage)
            return output, usage

    def embed_texts(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        if self._closed:
            raise RuntimeError("GeminiService is closed")
        self._local.embedding_usage = None
        if not isinstance(texts, list) or any(
            not isinstance(value, str) for value in texts
        ):
            raise TypeError("texts must be a list of strings")
        if task_type not in _TASK_TYPES:
            raise ValueError("Unsupported embedding task_type")
        model_id = self._config.EMBEDDING_MODEL_ID
        if model_id != "gemini-embedding-2":
            raise ValueError("CareerMatch requires gemini-embedding-2")
        if self.used_mock_fallback or not texts:
            self._local.embedding_usage = LLMUsageMeta(
                model_id=model_id, used_mock_fallback=self.used_mock_fallback
            )
            if _ACCOUNTING.get() is not None:
                _ACCOUNTING.get().record_free(self._local.embedding_usage)
            return [_offline_embedding(value) for value in texts]
        if any(not value.strip() for value in texts):
            raise ValueError("Live embedding inputs must be nonempty strings")
        results: list[list[float]] = []
        prompt_tokens = 0
        usage_known = True
        for start in range(0, len(texts), 50):
            chunk = [
                types.Content(parts=[types.Part.from_text(text=value)])
                for value in texts[start : start + 50]
            ]

            def capture_usage(response):
                nonlocal prompt_tokens, usage_known
                metadata = response.usage_metadata
                if metadata is None or metadata.prompt_token_count is None:
                    usage_known = False
                    self._local.embedding_usage = None
                    return None
                count = metadata.prompt_token_count
                if count < 0:
                    raise GeminiResponseError(
                        "Gemini returned invalid embedding token usage"
                    )
                prompt_tokens += count
                usage = LLMUsageMeta(
                    model_id=model_id,
                    prompt_tokens=count,
                    total_tokens=count,
                    estimated_cost_usd=count * 0.20 / 1_000_000,
                )
                self._local.embedding_usage = (
                    LLMUsageMeta(
                        model_id=model_id,
                        prompt_tokens=prompt_tokens,
                        total_tokens=prompt_tokens,
                        estimated_cost_usd=prompt_tokens * 0.20 / 1_000_000,
                    )
                    if usage_known
                    else None
                )
                return usage

            response, _usage = self._accounted_request(
                lambda chunk=chunk: self._client.models.embed_content(
                    model=model_id,
                    contents=chunk,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=768
                    ),
                ),
                operation=f"embed_content/{model_id}",
                model_id=model_id,
                usage_of=capture_usage,
                input_count=len(chunk),
            )
            embeddings = response.embeddings
            if (
                embeddings is None
                or len(embeddings) != len(chunk)
                or response.truncated
            ):
                raise GeminiResponseError(
                    "Gemini returned missing, truncated, or mismatched embeddings"
                )
            for item in embeddings:
                values = item.values
                if item.statistics is not None and item.statistics.truncated:
                    raise GeminiResponseError("Gemini truncated an embedding input")
                if (
                    values is None
                    or len(values) != 768
                    or not all(math.isfinite(v) for v in values)
                ):
                    raise GeminiResponseError(
                        "Gemini returned an invalid 768-dimensional embedding"
                    )
                if math.hypot(*values) == 0:
                    raise GeminiResponseError("Gemini returned a zero embedding")
                results.append([float(value) for value in values])
        return results
