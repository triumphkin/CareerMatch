"""Unit tests are isolated from credentials; the final gate requires live Gemini."""

import base64
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import date
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field
from tenacity import wait_none

from app.config import settings
from app.schemas.contracts import MatchingBatchOutput
from app.services import llm_client
from app.services.llm_client import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiService,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII="
)


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


class OtherReply(Reply):
    """Different schema identity for cache invalidation checks."""


def generation_reply(**changes):
    values = {
        "parsed": Reply(message="unit test response"),
        "text": None,
        "candidates": [],
        "usage_metadata": types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=10,
            total_token_count=130,
            cached_content_token_count=10,
        ),
    }
    values.update(changes)
    return SimpleNamespace(**values)


class FakeModels:
    def __init__(self):
        self.generations = []
        self.embeddings = []
        self.responses = []
        self.embedding_response = None

    def generate_content(self, **kwargs):
        self.generations.append(kwargs)
        response = self.responses.pop(0) if self.responses else generation_reply()
        if isinstance(response, Exception):
            raise response
        return response

    def embed_content(self, **kwargs):
        self.embeddings.append(kwargs)
        if self.embedding_response is not None:
            return self.embedding_response
        contents = kwargs["contents"]
        return types.EmbedContentResponse(
            embeddings=[
                types.ContentEmbedding(
                    values=[float(content.parts[0].text) + 1] + [0.0] * 767
                )
                for content in contents
            ],
            usage_metadata=types.UsageMetadata(
                prompt_token_count=len(contents), total_token_count=len(contents)
            ),
        )


@pytest.fixture
def config(tmp_path):
    return settings.model_copy(
        update={
            "GEMINI_API_KEY": f"synthetic-key-{tmp_path.name}",
            "LLM_CACHE_PATH": str(tmp_path / "cache.sqlite"),
            "LLM_RPM_LIMIT": 60_000_000,
        }
    )


@pytest.fixture
def fake():
    return SimpleNamespace(models=FakeModels())


@pytest.fixture
def isolated_mock_transport(monkeypatch):
    """No-network transport tests must not depend on the host's proxy packages."""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cache_persists_and_reports_no_new_usage(config, fake):
    with GeminiService(config=config, client=fake) as service:
        first, usage = service.generate_structured("test", "system", Reply)
    with GeminiService(config=config, client=fake) as service:
        second, cached_usage = service.generate_structured("test", "system", Reply)
    assert first == second and len(fake.models.generations) == 1
    assert (
        usage.prompt_tokens == 100
        and usage.completion_tokens == 30
        and usage.total_tokens == 130
    )
    input_rate, output_rate = llm_client._chat_rates(config.CHAT_MODEL_ID)
    assert usage.estimated_cost_usd == pytest.approx(
        (91 * input_rate + 30 * output_rate) / 1_000_000
    )
    assert not usage.cached and not usage.used_mock_fallback
    assert cached_usage.cached and not cached_usage.used_mock_fallback
    assert cached_usage.total_tokens == cached_usage.estimated_cost_usd == 0


def test_cache_key_includes_every_input(config, fake):
    with GeminiService(config=config, client=fake) as service:
        base = {
            "prompt": "test",
            "system_instruction": "system",
            "response_schema": Reply,
        }
        variants = [
            {},
            {"prompt": "other"},
            {"system_instruction": "other"},
            {"thinking_level": "LOW"},
            {"response_schema": OtherReply},
            {"model_id": "gemini-3.5-flash-lite"},
            {"image_bytes_list": [PNG]},
            {"image_bytes_list": [PNG + b"different"]},
        ]
        for changes in variants:
            service.generate_structured(**(base | changes))
        assert len(fake.models.generations) == len(variants)
        service.generate_structured(**base)
        assert len(fake.models.generations) == len(variants)
        key1 = service._cache_key("ab", "c", "d", "", {}, [])
        key2 = service._cache_key("a", "bc", "d", "", {}, [])
        assert key1 != key2


def test_cache_rejects_corrupt_record(config, fake):
    with GeminiService(config=config, client=fake) as service:
        service.generate_structured("test", "system", Reply)
        with sqlite3.connect(config.LLM_CACHE_PATH) as db:
            db.execute("UPDATE structured_cache SET output_json = ?", ("not json",))
        with pytest.raises(GeminiResponseError, match="Invalid structured cache"):
            service.generate_structured("test", "system", Reply)
        assert len(fake.models.generations) == 1


def test_offline_and_live_cache_entries_never_mix(config, fake):
    offline_config = config.model_copy(update={"GEMINI_API_KEY": ""})
    with GeminiService(config=offline_config) as offline:
        output, meta = offline.generate_structured("test", "system", Reply)
        assert meta.used_mock_fallback and meta.total_tokens == 0
        assert "offline_mock" in output.message
        assert offline.generate_structured("test", "system", Reply)[
            1
        ].used_mock_fallback
        assert (
            offline.generate_structured("test", "system", MatchingBatchOutput)[
                0
            ].assessments
            == []
        )
    with GeminiService(config=config, client=fake) as live:
        output, meta = live.generate_structured("test", "system", Reply)
        assert not meta.used_mock_fallback and not meta.cached
        assert output.message == "unit test response"
    with GeminiService(config=offline_config) as offline:
        assert offline.generate_structured("test", "system", Reply)[
            1
        ].used_mock_fallback


def test_offline_embeddings_empty_duplicates_and_keyword_similarity(config):
    with GeminiService(
        config=config.model_copy(update={"GEMINI_API_KEY": ""})
    ) as service:
        assert service.embed_texts([]) == []
        assert service.last_embedding_usage.used_mock_fallback
        vectors = service.embed_texts(
            ["Python SQL", "python sql", "Python", "", "!!!", "Python SQL"]
        )
        assert vectors[0] == vectors[1] == vectors[-1]
        assert vectors[3] == vectors[4]
        assert all(
            len(vector) == 768 and math.hypot(*vector) == pytest.approx(1)
            for vector in vectors
        )
        assert sum(a * b for a, b in zip(vectors[0], vectors[2], strict=True)) > 0.5
        assert (
            service.used_mock_fallback
            and service.last_embedding_usage.estimated_cost_usd == 0
        )


def test_offline_results_stable_across_process_hash_seeds(tmp_path):
    script = """
import hashlib, json
from app.config import Settings
from app.services.llm_client import GeminiService
from app.schemas.contracts import MatchingBatchOutput
with GeminiService(config=Settings()) as service:
    result, usage = service.generate_structured('fixture', '', MatchingBatchOutput)
    vectors = service.embed_texts(['Python SQL', 'नमस्ते', ''])
    assert usage.used_mock_fallback
    print(hashlib.sha256(json.dumps([result.model_dump(), vectors], sort_keys=True).encode()).hexdigest())
"""
    outputs = []
    for seed in ("1", "999"):
        env = os.environ.copy()
        env.update(
            {
                "PYTHONHASHSEED": seed,
                "GEMINI_API_KEY": "",
                "LLM_CACHE_PATH": str(tmp_path / f"process-{seed}.sqlite"),
            }
        )
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    assert outputs[0] == outputs[1] and len(outputs[0]) == 64


@pytest.mark.parametrize("size", [0, 1, 49, 50, 51, 100, 101])
def test_embedding_batch_boundaries_and_order(config, fake, size):
    with GeminiService(config=config, client=fake) as service:
        vectors = service.embed_texts([str(index) for index in range(size)])
        assert len(vectors) == size and all(len(v) == 768 for v in vectors)
        assert [v[0] for v in vectors] == [index + 1.0 for index in range(size)]
        assert [len(call["contents"]) for call in fake.models.embeddings] == [
            min(50, size - start) for start in range(0, size, 50)
        ]
        for call in fake.models.embeddings:
            assert call["model"] == "gemini-embedding-2"
            assert call["config"].output_dimensionality == 768
            assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
        assert not service.used_mock_fallback
        assert service.last_embedding_usage.prompt_tokens == size
        assert service.last_embedding_usage.estimated_cost_usd == pytest.approx(
            size * 0.20 / 1_000_000
        )


@pytest.mark.parametrize(
    "values",
    [None, [], [0.0] * 767, [0.0] * 768, [float("nan")] * 768, [float("inf")] * 768],
)
def test_invalid_embedding_response_not_replaced_with_mock(config, fake, values):
    fake.models.embedding_response = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=values)]
    )
    with (
        GeminiService(config=config, client=fake) as service,
        pytest.raises(GeminiResponseError),
    ):
        service.embed_texts(["0"])
    assert len(fake.models.embeddings) == 1


@pytest.mark.parametrize(
    "response",
    [
        types.EmbedContentResponse(embeddings=[]),
        types.EmbedContentResponse(embeddings=None),
        types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[1.0] * 768)], truncated=True
        ),
        types.EmbedContentResponse(
            embeddings=[
                types.ContentEmbedding(
                    values=[1.0] * 768,
                    statistics=types.ContentEmbeddingStatistics(truncated=True),
                )
            ]
        ),
    ],
)
def test_embedding_count_and_truncation_failures(config, fake, response):
    fake.models.embedding_response = response
    with (
        GeminiService(config=config, client=fake) as service,
        pytest.raises(GeminiResponseError),
    ):
        service.embed_texts(["0"])


def test_missing_embedding_usage_is_explicit(config, fake):
    fake.models.embedding_response = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[1.0] + [0.0] * 767)]
    )
    with GeminiService(config=config, client=fake) as service:
        assert len(service.embed_texts(["0"])) == 1
        assert service.last_embedding_usage is None
        assert not service.used_mock_fallback


@pytest.mark.parametrize(
    "failure",
    [
        errors.APIError(429, {"error": {"message": "rate limit"}}),
        errors.APIError(503, {"error": {"message": "unavailable"}}),
        httpx.ReadTimeout("timeout"),
    ],
)
def test_transient_failures_retry_at_most_twice(config, fake, monkeypatch, failure):
    monkeypatch.setattr(
        llm_client, "wait_exponential_jitter", lambda **kwargs: wait_none()
    )
    fake.models.responses = [failure, failure, generation_reply()]
    with GeminiService(config=config, client=fake) as service:
        assert not service.generate_structured("test", "", Reply)[1].used_mock_fallback
    assert len(fake.models.generations) == 3


def test_attempt_logs_separate_rate_wait_duration_and_redact_payloads(
    config, fake, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="app.services.llm_client")
    monkeypatch.setattr(
        llm_client, "wait_exponential_jitter", lambda **kwargs: wait_none()
    )
    now = [0.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: now[0])
    durations = iter([110.0, 70.0, 0.5])

    def wait(interval):
        now[0] += 0.25

    def request():
        duration = next(durations)
        now[0] += duration
        if duration > 1:
            raise errors.APIError(
                504, {"error": {"message": f"PRIVATE PROMPT {config.GEMINI_API_KEY}"}}
            )
        return "ok"

    with GeminiService(config=config, client=fake) as service:
        monkeypatch.setattr(service._limiter, "wait", wait)
        assert (
            service._request(request, operation="generate_content/gemini-3.8-flash")
            == "ok"
        )
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.services.llm_client"
    ]
    assert len(messages) == 6
    assert all("rate_wait_s=0.250 timeout_s=120.0" in line for line in messages[::2])
    assert (
        "attempt=1/3 outcome=http_504 elapsed_s=110.000 request_elapsed_s=110.250"
        in messages[1]
    )
    assert (
        "attempt=2/3 outcome=http_504 elapsed_s=70.000 request_elapsed_s=180.500"
        in messages[3]
    )
    assert (
        "attempt=3/3 outcome=success elapsed_s=0.500 request_elapsed_s=181.250"
        in messages[5]
    )
    request_ids = {line.split("request_id=", 1)[1].split()[0] for line in messages}
    assert len(request_ids) == 1
    assert (
        "PRIVATE PROMPT" not in caplog.text and config.GEMINI_API_KEY not in caplog.text
    )


@pytest.mark.parametrize("code,attempts", [(504, 3), (401, 1), (None, 3)])
def test_failed_attempt_logs_preserve_retry_limits_without_exception_text(
    config, fake, monkeypatch, caplog, code, attempts
):
    caplog.set_level(logging.INFO, logger="app.services.llm_client")
    monkeypatch.setattr(
        llm_client, "wait_exponential_jitter", lambda **kwargs: wait_none()
    )
    message = f"PRIVATE RESUME {config.GEMINI_API_KEY}"
    failure = (
        errors.APIError(code, {"error": {"message": message}})
        if code
        else httpx.ReadTimeout(message)
    )
    fake.models.responses = [failure] * 3
    with (
        GeminiService(config=config, client=fake) as service,
        pytest.raises(GeminiRequestError),
    ):
        service.generate_structured("PRIVATE PROMPT", "PRIVATE SYSTEM", Reply)
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.services.llm_client"
    ]
    assert len(messages) == attempts * 2
    assert "attempt=4/3" not in caplog.text
    outcome = f"http_{code}" if code else "ReadTimeout"
    assert all(f"outcome={outcome}" in line for line in messages[1::2])
    assert "PRIVATE" not in caplog.text and config.GEMINI_API_KEY not in caplog.text


def test_cached_and_offline_calls_do_not_emit_request_attempts(config, fake, caplog):
    caplog.set_level(logging.INFO, logger="app.services.llm_client")
    with GeminiService(config=config, client=fake) as service:
        service.generate_structured("test", "", Reply)
        assert len(caplog.records) == 2
        caplog.clear()
        assert service.generate_structured("test", "", Reply)[1].cached
        assert not caplog.records
    offline = config.model_copy(update={"GEMINI_API_KEY": ""})
    with GeminiService(config=offline) as service:
        assert service.generate_structured("offline", "", Reply)[1].used_mock_fallback
    assert not caplog.records


@pytest.mark.parametrize(
    ("failure", "attempts"),
    [
        (
            errors.APIError(
                400, {"error": {"message": "secret payload must not appear"}}
            ),
            1,
        ),
        (errors.APIError(401, {"error": {"message": "bad key"}}), 1),
        (errors.APIError(429, {"error": {"message": "rate limit"}}), 3),
        (httpx.ConnectError("secret payload must not appear"), 3),
    ],
)
def test_permanent_or_exhausted_errors_propagate_without_fallback(
    config, fake, monkeypatch, failure, attempts
):
    monkeypatch.setattr(
        llm_client, "wait_exponential_jitter", lambda **kwargs: wait_none()
    )
    fake.models.responses = [failure] * 3
    with GeminiService(config=config, client=fake) as service:
        with pytest.raises(GeminiRequestError) as error:
            service.generate_structured("test", "", Reply)
        assert "secret payload" not in str(error.value)
        assert not service.used_mock_fallback
    assert len(fake.models.generations) == attempts


@pytest.mark.parametrize("encoding", ["raw", "url"])
def test_api_diagnostic_retains_validation_reason_but_redacts_key(
    config, fake, encoding
):
    key = "AQ.synthetic/key+with=characters"
    config = config.model_copy(update={"GEMINI_API_KEY": key})
    echoed = key if encoding == "raw" else quote(key, safe="")
    fake.models.responses = [
        errors.APIError(
            400,
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": f"Invalid response_schema field; credential={echoed}",
                }
            },
        )
    ]
    with GeminiService(config=config, client=fake) as service:
        with pytest.raises(GeminiRequestError) as error:
            service.generate_structured("synthetic", "", Reply)
        assert "generate_content/gemini-3.8-flash" in str(error.value)
        assert "Invalid response_schema field" in error.value.diagnostic
        assert (
            key not in error.value.diagnostic and echoed not in error.value.diagnostic
        )
        assert "[REDACTED]" in error.value.diagnostic
        assert "credential=" not in str(error.value)
    assert len(fake.models.generations) == 1


@contextmanager
def _live_api_diagnostics():
    """Only the synthetic live fixture opts into provider validation details."""
    try:
        yield
    except GeminiRequestError as error:
        detail = error.diagnostic or "No provider diagnostic available"
        pytest.fail(
            f"{error}\nProvider diagnostic (credentials redacted): {detail}",
            pytrace=False,
        )


@pytest.mark.parametrize(
    "response",
    [
        generation_reply(parsed=None, text='{"message": 123}'),
        generation_reply(parsed=None, text="not json"),
        generation_reply(parsed=None, text=None),
        generation_reply(usage_metadata=None),
        generation_reply(
            candidates=[types.Candidate(finish_reason=types.FinishReason.MAX_TOKENS)]
        ),
    ],
)
def test_invalid_structured_responses_fail_without_retry_or_cache(
    config, fake, response
):
    fake.models.responses = [response]
    with GeminiService(config=config, client=fake) as service:
        with pytest.raises(GeminiResponseError):
            service.generate_structured("test", "", Reply)
        assert not service.generate_structured("test", "", Reply)[1].cached
    assert len(fake.models.generations) == 2


def test_json_text_response_and_image_parts(config, fake):
    fake.models.responses = [generation_reply(parsed=None, text='{"message": "ok"}')]
    with GeminiService(config=config, client=fake) as service:
        assert (
            service.generate_structured(
                "read image", "system", Reply, image_bytes_list=[PNG]
            )[0].message
            == "ok"
        )
    call = fake.models.generations[0]
    assert call["contents"][0].parts[0].text == "read image"
    image = call["contents"][0].parts[1].inline_data
    assert image.mime_type == "image/png" and image.data == PNG
    config_values = call["config"].model_dump(exclude_none=True)
    assert config_values["thinking_config"]["thinking_level"] == "MEDIUM"
    assert config_values["response_mime_type"] == "application/json"
    assert (
        not {
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "frequency_penalty",
            "presence_penalty",
        }
        & config_values.keys()
    )


def test_minimal_only_allowed_for_flash_lite(config, fake):
    with GeminiService(config=config, client=fake) as service:
        with pytest.raises(ValueError, match="thinking_level"):
            service.generate_structured("test", "", Reply, thinking_level="MINIMAL")
        service.generate_structured(
            "test",
            "",
            Reply,
            model_id="gemini-3.5-flash-lite",
            thinking_level="MINIMAL",
        )
    assert len(fake.models.generations) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt": None},
        {"prompt": ""},
        {"system_instruction": None},
        {"response_schema": dict},
        {"model_id": "unknown"},
        {"thinking_level": "bad"},
        {"image_bytes_list": [b""]},
        {"image_bytes_list": [b"not an image"]},
        {"image_bytes_list": PNG},
    ],
)
def test_invalid_generation_inputs_fail_before_api(config, fake, changes):
    with (
        GeminiService(config=config, client=fake) as service,
        pytest.raises((TypeError, ValueError)),
    ):
        service.generate_structured(
            **(
                {"prompt": "test", "system_instruction": "", "response_schema": Reply}
                | changes
            )
        )
    assert fake.models.generations == []


@pytest.mark.parametrize("texts", [None, "text", [None], ["ok", ""], [" "]])
def test_invalid_embedding_inputs_fail_before_api(config, fake, texts):
    with (
        GeminiService(config=config, client=fake) as service,
        pytest.raises((TypeError, ValueError)),
    ):
        service.embed_texts(texts)
    assert fake.models.embeddings == []


def test_thread_safe_rate_limiter_uses_spacing(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        llm_client.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay)
    )
    limiter = llm_client._RateLimiter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: limiter.wait(4.0), range(8)))
    assert now[0] == 28.0


def test_concurrent_cache_misses_issue_one_call_across_instances(config, fake):
    with (
        GeminiService(config=config, client=fake) as first,
        GeminiService(config=config, client=fake) as second,
    ):
        assert first._limiter is second._limiter
        barrier = threading.Barrier(8)

        def request(index):
            barrier.wait(timeout=5)
            return (first if index % 2 else second).generate_structured(
                "same", "", Reply
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(request, range(8)))
    assert len(fake.models.generations) == 1
    assert sum(meta.cached for _, meta in results) == 7


def test_embedding_usage_is_thread_local(config, fake):
    with GeminiService(config=config, client=fake) as service:
        barrier = threading.Barrier(2)

        def request(size):
            service.embed_texts(["0"] * size)
            barrier.wait(timeout=5)
            return service.last_embedding_usage.prompt_tokens

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(request, [1, 2])) == [1, 2]
        assert service.last_embedding_usage is None


def test_pricing_dates_and_no_silent_unknown_model():
    assert llm_client._chat_rates("gemini-3.8-flash", date(2026, 12, 31)) == (
        0.75,
        3.75,
    )
    assert llm_client._chat_rates("gemini-3.8-flash", date(2027, 1, 1)) == (1.50, 7.50)
    assert llm_client._chat_rates("gemini-3.5-flash-lite") == (0.30, 2.50)
    with pytest.raises(ValueError):
        llm_client._chat_rates("unknown")


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt_token_count": -1},
        {"candidates_token_count": -1},
        {"thoughts_token_count": -1},
        {"total_token_count": -1},
        {"cached_content_token_count": 101},
        {"total_token_count": 119},
        {"total_token_count": None, "candidates_token_count": None},
    ],
)
def test_invalid_usage_never_becomes_a_zero_cost_success(config, fake, changes):
    metadata = generation_reply().usage_metadata.model_copy(update=changes)
    fake.models.responses = [generation_reply(usage_metadata=metadata)]
    with (
        GeminiService(config=config, client=fake) as service,
        pytest.raises(GeminiResponseError),
    ):
        service.generate_structured("test", "", Reply)
    assert len(fake.models.generations) == 1


def test_lifecycle_and_fixed_dimension(config, fake):
    service = GeminiService(config=config, client=fake)
    service.close()
    service.close()
    with pytest.raises(RuntimeError, match="closed"):
        service.embed_texts([])
    with pytest.raises(RuntimeError, match="closed"):
        service.generate_structured("test", "", Reply)
    with pytest.raises(ValueError, match="768"):
        GeminiService(
            config=config.model_copy(update={"EMBEDDING_DIM": 512}), client=fake
        )


def test_offline_schema_defaults_and_bounds(config):
    class Fixture(BaseModel):
        count: int = Field(ge=1)
        label: str = Field(min_length=3, max_length=8)
        choices: list[str]
        enabled: bool
        optional: str | None

    with GeminiService(
        config=config.model_copy(update={"GEMINI_API_KEY": ""})
    ) as service:
        result, meta = service.generate_structured("test", "", Fixture)
    assert result.count == 1 and len(result.label) == 8
    assert result.choices == [] and not result.enabled and result.optional is None
    assert meta.used_mock_fallback


def test_deep_schema_rejected_before_call(config, fake):
    class Leaf(BaseModel):
        value: str

    class LevelThree(BaseModel):
        child: Leaf

    class LevelTwo(BaseModel):
        child: LevelThree

    class Root(BaseModel):
        child: LevelTwo

    with (
        GeminiService(config=config, client=fake) as service,
        pytest.raises(ValueError, match="depth"),
    ):
        service.generate_structured("test", "", Root)
    assert fake.models.generations == []


def test_sdk_retry_policy_is_one_attempt_and_owned_client_closes(
    config, monkeypatch, isolated_mock_transport
):
    arguments = {}
    closed = []

    def make_client(**kwargs):
        arguments.update(kwargs)
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr(llm_client.genai, "Client", make_client)
    with GeminiService(config=config):
        assert arguments["http_options"].retry_options.attempts == 1
        assert arguments["http_options"].timeout == 120_000
        assert arguments["http_options"].httpx_client.timeout.read == 120
        assert arguments["vertexai"] is False
    assert closed == [True]


def test_real_sdk_wire_schema_and_embedding_batching_without_network(
    config, isolated_mock_transport
):
    """Exercise SDK serialization with a mock transport, NOT a live API gate."""
    requests = []

    def handler(request):
        assert request.headers["X-Server-Timeout"] == "120"
        assert request.extensions["timeout"]["read"] == 120
        payload = json.loads(request.content)
        requests.append(payload)
        if "batchEmbedContents" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        {"values": [1.0] + [0.0] * 767} for _ in payload["requests"]
                    ]
                },
            )
        assert_no_additional_properties(payload["generationConfig"]["responseSchema"])
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": '{"assessments": []}'}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as transport,
        genai.Client(
            api_key="synthetic-key",
            vertexai=False,
            http_options=types.HttpOptions(
                httpx_client=transport,
                retry_options=types.HttpRetryOptions(attempts=1),
                timeout=llm_client._REQUEST_TIMEOUT_MS,
            ),
        ) as client,
        GeminiService(config=config, client=client) as service,
    ):
        service.generate_structured(
            "schema test", "test", MatchingBatchOutput, thinking_level="LOW"
        )
        assert len(service.embed_texts(["one", "two"])) == 2
    generation = requests[0]["generationConfig"]
    assert (
        types.ThinkingConfig.model_validate(generation["thinkingConfig"]).thinking_level
        == "LOW"
    )
    assert generation["responseMimeType"] == "application/json"
    assert "responseSchema" in generation or "responseJsonSchema" in generation
    assert (
        not {
            "temperature",
            "topP",
            "topK",
            "candidateCount",
            "frequencyPenalty",
            "presencePenalty",
        }
        & generation.keys()
    )
    assert len(requests[1]["requests"]) == 2
    assert [
        item["content"]["parts"][0]["text"] for item in requests[1]["requests"]
    ] == ["one", "two"]


def assert_no_additional_properties(node):
    if isinstance(node, dict):
        assert "additionalProperties" not in node
        assert "additional_properties" not in node
        for value in node.values():
            assert_no_additional_properties(value)
    elif isinstance(node, list):
        for item in node:
            assert_no_additional_properties(item)


def test_wire_schema_adapter_preserves_contract_and_nested_fields():
    original = MatchingBatchOutput.model_json_schema()
    before = deepcopy(original)
    wire = llm_client._gemini_response_schema(original)
    assert original == before
    assert original["additionalProperties"] is False
    assert_no_additional_properties(wire)
    assert wire["required"] == original["required"]
    for name, definition in original["$defs"].items():
        assert wire["$defs"][name].get("required") == definition.get("required")
        assert (
            wire["$defs"][name].get("properties")
            == {key: value for key, value in definition.get("properties", {}).items()}
            if "properties" in definition
            else wire["$defs"][name] == definition
        )
    assert wire["$defs"]["DecisionType"]["enum"] == [
        "accept",
        "needs_review",
        "reject",
        "revise",
    ]


def test_schema_adapter_preserves_property_names_and_enum_values():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "additionalProperties": {
                "type": "string",
                "enum": ["additional_properties"],
            }
        },
    }
    adapted = llm_client._gemini_response_schema(schema)
    assert adapted["properties"] == schema["properties"]
    assert "additionalProperties" not in adapted


def test_nested_extra_field_still_rejected_after_wire_adaptation(config, fake):
    fake.models.responses = [
        generation_reply(
            parsed={
                "assessments": [
                    {
                        "job_id": "job_001",
                        "title": "test",
                        "company": "test",
                        "source_url": "test",
                        "captured_at": "test",
                        "decision": "needs_review",
                        "gaps": [],
                        "unknowns": [],
                        "supported_matches": [],
                        "constraint_checks": [
                            {
                                "field": "student_eligible",
                                "result": "unknown",
                                "reason": "test",
                                "source_excerpt_ids": [],
                                "unexpected": "reject me",
                            }
                        ],
                    }
                ]
            }
        )
    ]
    with (
        GeminiService(config=config, client=fake) as service,
        pytest.raises(GeminiResponseError, match="response_schema"),
    ):
        service.generate_structured("synthetic fixture", "", MatchingBatchOutput)
    assert_no_additional_properties(
        fake.models.generations[0]["config"].response_schema
    )


def test_live_gemini_tiny_json_same_model(tmp_path, caplog):
    """Independent diagnostic: same chat model/thinking, much smaller schema.

    This does not replace or skip the full MatchingBatchOutput live gate.
    """
    if not settings.GEMINI_API_KEY.strip():
        pytest.fail(
            "GEMINI_API_KEY is not set: tiny live diagnostic cannot run", pytrace=False
        )
    caplog.set_level(logging.INFO, logger="app.services.llm_client")
    config = settings.model_copy(
        update={"LLM_CACHE_PATH": str(tmp_path / "fresh-tiny-live.sqlite")}
    )
    llm_client._LOGGER.info("Live probe=tiny_json thinking=LOW fresh_cache=True")
    with _live_api_diagnostics(), GeminiService(config=config) as service:
        assert not service.used_mock_fallback
        output, usage = service.generate_structured(
            'Return {"message":"ok"}.',
            "Return only the requested JSON.",
            Reply,
            model_id=config.CHAT_MODEL_ID,
            thinking_level="LOW",
        )
        assert output.message == "ok"
        assert not usage.cached and not usage.used_mock_fallback
        assert usage.total_tokens > 0


def test_live_gemini_schema_thinking_embeddings_and_cache(tmp_path, caplog):
    """Requires a real key/network; a missing prerequisite is a failed gate."""
    if not settings.GEMINI_API_KEY.strip():
        pytest.fail(
            "GEMINI_API_KEY is not set: live Step 1.4(c)/(d) cannot be verified",
            pytrace=False,
        )
    config = settings.model_copy(
        update={"LLM_CACHE_PATH": str(tmp_path / "fresh-live.sqlite")}
    )
    caplog.set_level(logging.INFO, logger="app.services.llm_client")
    llm_client._LOGGER.info("Live probe=matching_batch thinking=LOW fresh_cache=True")
    expected = {
        "assessments": [
            {
                "job_id": f"job_{index:03d}",
                "title": "Python internship",
                "company": "Synthetic fixture",
                "source_url": "https://example.test/job",
                "captured_at": "2026-11-19T00:00:00Z",
                "decision": "needs_review",
                "supported_matches": [
                    {
                        "requirement": "Python",
                        "candidate_evidence_ids": ["ev_001"],
                        "job_excerpt_ids": [f"job_{index:03d}_ex_01"],
                        "explanation": "Self-reported Python.",
                        "is_self_reported_only": True,
                    }
                ],
                "gaps": [],
                "unknowns": ["Student eligibility"],
                "constraint_checks": [
                    {
                        "field": "student_eligible",
                        "result": "unknown",
                        "reason": "Not stated.",
                        "source_excerpt_ids": [],
                    }
                ],
                "internal_rank_score": 0.0,
            }
            for index in (1, 2)
        ]
    }
    prompt = (
        "Return this synthetic JSON fixture exactly, preserving both assessments and all fields: "
        + json.dumps(expected)
    )
    with _live_api_diagnostics(), GeminiService(config=config) as service:
        assert not service.used_mock_fallback
        output, usage = service.generate_structured(
            prompt,
            "You serialize supplied synthetic test data as JSON.",
            MatchingBatchOutput,
            thinking_level="LOW",
        )
        assert output == MatchingBatchOutput.model_validate(expected)
        assert (
            not usage.used_mock_fallback and not usage.cached and usage.total_tokens > 0
        )
        assert usage.estimated_cost_usd > 0
        llm_client._LOGGER.info("Live probe=matching_batch_cache expected_cached=True")
        again, cached = service.generate_structured(
            prompt,
            "You serialize supplied synthetic test data as JSON.",
            MatchingBatchOutput,
            thinking_level="LOW",
        )
        assert again == output and cached.cached and not cached.used_mock_fallback
        assert cached.estimated_cost_usd == 0
        llm_client._LOGGER.info("Live probe=two_embeddings")
        vectors = service.embed_texts(["Python internship", "SQL internship"])
        assert len(vectors) == 2
        assert all(
            len(vector) == 768 and math.hypot(*vector) == pytest.approx(1, abs=0.02)
            for vector in vectors
        )
        assert vectors[0] != vectors[1]
        llm_client._LOGGER.info("Live probe=fast_model_json thinking=LOW")
        lite, lite_usage = service.generate_structured(
            'Return {"message":"ok"}.',
            "Return only the requested JSON.",
            Reply,
            model_id=config.CHAT_MODEL_ID_FAST,
            thinking_level="LOW",
        )
        assert lite.message == "ok" and not lite_usage.used_mock_fallback
