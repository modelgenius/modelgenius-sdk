"""Tests for modelgenius_sdk.capture.litellm_callback — litellm success callback."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from modelgenius_sdk.capture import core, litellm_callback
from modelgenius_sdk.capture.core import FORMAT_TAG, CaptureConfig
from modelgenius_sdk.capture.litellm_callback import (
    capture_success_callback,
    make_litellm_callback,
)

START_AWARE = datetime(2026, 7, 5, 12, 0, 0, tzinfo=timezone.utc)
END_AWARE = START_AWARE + timedelta(seconds=1.25)
START_NAIVE = datetime(2026, 7, 5, 12, 0, 0)
END_NAIVE = START_NAIVE + timedelta(milliseconds=340)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Isolate module-global state (writer cache, warn-once registry, env, default cb)."""
    monkeypatch.setattr(core, "_writers", {})
    monkeypatch.setattr(core, "_warned_keys", set())
    monkeypatch.setattr(litellm_callback, "_default_callback", None)
    for env_var in (core.ENV_DIR, core.ENV_SAMPLE_RATE, core.ENV_DISABLE, core.ENV_DEBUG):
        monkeypatch.delenv(env_var, raising=False)


def read_spool_records(spool_dir: Path) -> list:
    records = []
    for path in sorted(spool_dir.glob("snapshots-*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Fakes — NO litellm import anywhere. Attribute-flavored fakes mimic litellm's
# pydantic-ish ModelResponse; dict-flavored fakes mimic plain-dict responses.
# ---------------------------------------------------------------------------


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = None  # unused; model_dump is authoritative
        self._name = name
        self._arguments = arguments

    def model_dump(self, mode=None):
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self._name, "arguments": self._arguments},
        }


class FakeCompletionTokensDetails:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens, reasoning_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.completion_tokens_details = (
            FakeCompletionTokensDetails(reasoning_tokens) if reasoning_tokens is not None else None
        )


class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class FakeModelResponse:
    """Attribute-flavored ModelResponse look-alike (pydantic-ish model_dump)."""

    def __init__(self, content="Here is the summary.", tool_calls=None, hidden_cost=None):
        self.id = "chatcmpl-attr-123"
        self.model = "gpt-4o-mini-2024-07-18"
        self.choices = [FakeChoice(FakeMessage(content, tool_calls))]
        self.usage = FakeUsage(120, 45, 165, reasoning_tokens=7)
        self._hidden_params = {}
        if hidden_cost is not None:
            self._hidden_params["response_cost"] = hidden_cost

    def model_dump(self, mode=None):
        return {
            "id": self.id,
            "object": "chat.completion",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": choice.finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": choice.message.content,
                        "tool_calls": [
                            tc.model_dump() for tc in (choice.message.tool_calls or [])
                        ]
                        or None,
                    },
                }
                for choice in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


def make_dict_response(content="Dict-flavored reply.", hidden_cost=None):
    response = {
        "id": "chatcmpl-dict-456",
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup_order", "arguments": '{"order_id": "A-9"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    }
    if hidden_cost is not None:
        response["_hidden_params"] = {"response_cost": hidden_cost}
    return response


def make_kwargs(**overrides):
    """Plausible litellm success-callback kwargs (top-level + nested params)."""
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Summarize this support ticket"}],
        "optional_params": {"temperature": 0.3, "max_tokens": 500},
        "litellm_params": {"custom_llm_provider": "openai"},
        "api_key": "sk-secret-should-never-appear",
        "litellm_call_id": "11112222-3333",
        "response_cost": 0.00021,
    }
    kwargs.update(overrides)
    return kwargs


def run_callback(tmp_path, kwargs, response, start=START_AWARE, end=END_AWARE, **config_overrides):
    callback = make_litellm_callback(spool_dir=tmp_path, **config_overrides)
    callback(kwargs, response, start, end)
    return read_spool_records(tmp_path)


# ---------------------------------------------------------------------------
# Request part
# ---------------------------------------------------------------------------


class TestRequestPart:
    def test_canonical_intersection_copied_and_junk_dropped(self, tmp_path):
        records = run_callback(tmp_path, make_kwargs(), FakeModelResponse())
        assert len(records) == 1
        record = records[0]
        assert record["model"] == "gpt-4o-mini"
        assert record["messages"] == [{"role": "user", "content": "Summarize this support ticket"}]
        # litellm plumbing keys never land in the record.
        for junk in ("api_key", "litellm_call_id", "optional_params", "litellm_params", "response_cost"):
            assert junk not in record

    def test_optional_params_fallback_and_top_level_wins(self, tmp_path):
        kwargs = make_kwargs(temperature=0.9)  # top-level beats optional_params' 0.3
        records = run_callback(tmp_path, kwargs, FakeModelResponse())
        record = records[0]
        assert record["temperature"] == 0.9
        assert record["max_tokens"] == 500  # filled from optional_params
        assert "custom_llm_provider" not in record  # non-canonical nested key ignored

    def test_none_values_skipped(self, tmp_path):
        kwargs = make_kwargs(temperature=None, top_p=None, stop=None)
        del kwargs["optional_params"]
        records = run_callback(tmp_path, kwargs, FakeModelResponse())
        record = records[0]
        assert "top_p" not in record
        assert "stop" not in record
        assert "temperature" not in record

    def test_capture_only_params_pass_through(self, tmp_path):
        kwargs = make_kwargs(seed=42, user="tenant-17")
        records = run_callback(tmp_path, kwargs, FakeModelResponse())
        assert records[0]["seed"] == 42
        assert records[0]["user"] == "tenant-17"


# ---------------------------------------------------------------------------
# Cost extraction
# ---------------------------------------------------------------------------


class TestCostExtraction:
    def test_from_kwargs_response_cost(self, tmp_path):
        records = run_callback(tmp_path, make_kwargs(response_cost=0.00021), FakeModelResponse())
        assert records[0]["captured_output"]["cost_usd"] == pytest.approx(0.00021)

    def test_from_hidden_params_attribute_style(self, tmp_path):
        kwargs = make_kwargs()
        del kwargs["response_cost"]
        records = run_callback(tmp_path, kwargs, FakeModelResponse(hidden_cost=0.00042))
        assert records[0]["captured_output"]["cost_usd"] == pytest.approx(0.00042)

    def test_from_hidden_params_dict_style(self, tmp_path):
        kwargs = make_kwargs()
        del kwargs["response_cost"]
        records = run_callback(tmp_path, kwargs, make_dict_response(hidden_cost=0.00099))
        assert records[0]["captured_output"]["cost_usd"] == pytest.approx(0.00099)

    def test_kwargs_wins_over_hidden_params(self, tmp_path):
        records = run_callback(
            tmp_path, make_kwargs(response_cost=0.001), FakeModelResponse(hidden_cost=0.9)
        )
        assert records[0]["captured_output"]["cost_usd"] == pytest.approx(0.001)

    def test_absent_everywhere_is_null(self, tmp_path):
        kwargs = make_kwargs()
        del kwargs["response_cost"]
        records = run_callback(tmp_path, kwargs, FakeModelResponse())  # no hidden cost
        assert records[0]["captured_output"]["cost_usd"] is None

    def test_garbage_cost_is_null(self, tmp_path):
        records = run_callback(tmp_path, make_kwargs(response_cost="not-a-number"), FakeModelResponse())
        assert records[0]["captured_output"]["cost_usd"] is None


# ---------------------------------------------------------------------------
# Timestamps / latency
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_aware_datetimes(self, tmp_path):
        records = run_callback(tmp_path, make_kwargs(), FakeModelResponse())
        record = records[0]
        captured = record["captured_output"]
        assert record["timestamp"] == START_AWARE.isoformat()
        assert captured["request_timestamp"] == START_AWARE.isoformat()
        assert captured["response_timestamp"] == END_AWARE.isoformat()
        assert captured["latency_ms"] == pytest.approx(1250.0)

    def test_naive_datetimes_treated_as_local_and_converted_to_utc(self, tmp_path):
        # litellm passes datetime.now() — naive LOCAL wall-clock. Relabeling it
        # as UTC would skew every timestamp by the host's UTC offset, so the
        # callback converts via astimezone (a no-op on UTC hosts).
        records = run_callback(
            tmp_path, make_kwargs(), FakeModelResponse(), start=START_NAIVE, end=END_NAIVE
        )
        captured = records[0]["captured_output"]
        assert captured["request_timestamp"] == START_NAIVE.astimezone(timezone.utc).isoformat()
        assert captured["response_timestamp"] == END_NAIVE.astimezone(timezone.utc).isoformat()
        assert captured["latency_ms"] == pytest.approx(340.0)
        # Every timestamp parses as tz-aware ISO-8601 pinned to UTC.
        for key in ("request_timestamp", "response_timestamp"):
            parsed = datetime.fromisoformat(captured[key])
            assert parsed.utcoffset() == timedelta(0)

    def test_epoch_float_times_accepted(self, tmp_path):
        start = START_AWARE.timestamp()
        records = run_callback(
            tmp_path, make_kwargs(), FakeModelResponse(), start=start, end=start + 0.5
        )
        captured = records[0]["captured_output"]
        assert datetime.fromisoformat(captured["request_timestamp"]) == START_AWARE
        assert captured["latency_ms"] == pytest.approx(500.0)

    def test_garbage_times_fall_back_to_now(self, tmp_path):
        records = run_callback(
            tmp_path, make_kwargs(), FakeModelResponse(), start="not-a-time", end=None
        )
        record = records[0]
        captured = record["captured_output"]
        assert captured["latency_ms"] is None
        for key in ("request_timestamp", "response_timestamp"):
            assert datetime.fromisoformat(captured[key]).tzinfo is not None
        assert record["timestamp"] == captured["request_timestamp"]


# ---------------------------------------------------------------------------
# captured_output mapping (attribute AND dict flavored responses)
# ---------------------------------------------------------------------------


class TestCapturedOutputMapping:
    def test_attribute_flavored_response(self, tmp_path):
        tool_call = FakeToolCall("call_9", "create_ticket", '{"priority": "high"}')
        response = FakeModelResponse(content="Escalating now.", tool_calls=[tool_call])
        records = run_callback(tmp_path, make_kwargs(), response)
        captured = records[0]["captured_output"]

        assert captured["output_text"] == "Escalating now."
        assert captured["tool_calls"] == [
            {
                "id": "call_9",
                "type": "function",
                "function": {"name": "create_ticket", "arguments": '{"priority": "high"}'},
            }
        ]
        assert captured["model"] == "gpt-4o-mini-2024-07-18"
        assert captured["api_surface"] == "chat"
        assert captured["usage"] == {
            "input_tokens": 120,
            "output_tokens": 45,
            "thinking_tokens": 7,
            "total_tokens": 165,
        }
        assert captured["provider_metadata"]["finish_reason"] == "stop"
        assert captured["provider_metadata"]["request_id"] == "chatcmpl-attr-123"
        assert isinstance(captured["raw_response"], dict)
        assert captured["raw_response"]["id"] == "chatcmpl-attr-123"

    def test_dict_flavored_response(self, tmp_path):
        records = run_callback(tmp_path, make_kwargs(), make_dict_response())
        captured = records[0]["captured_output"]

        assert captured["output_text"] == "Dict-flavored reply."
        assert captured["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_order", "arguments": '{"order_id": "A-9"}'},
            }
        ]
        assert captured["model"] == "gpt-4o-mini-2024-07-18"
        assert captured["usage"] == {
            "input_tokens": 80,
            "output_tokens": 20,
            "thinking_tokens": None,
            "total_tokens": 100,
        }
        assert captured["provider_metadata"]["finish_reason"] == "tool_calls"
        assert captured["provider_metadata"]["request_id"] == "chatcmpl-dict-456"
        assert captured["raw_response"]["choices"][0]["finish_reason"] == "tool_calls"

    def test_null_content_yields_empty_output_text(self, tmp_path):
        records = run_callback(tmp_path, make_kwargs(), FakeModelResponse(content=None))
        captured = records[0]["captured_output"]
        assert captured["output_text"] == ""
        assert captured["tool_calls"] is None


# ---------------------------------------------------------------------------
# Envelope shape + workflow metadata + validator agreement
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_canonical_envelope_with_workflow_metadata(self, tmp_path):
        callback = make_litellm_callback(
            spool_dir=tmp_path,
            workflow_name="support-bot",
            call_name="draft_reply",
            workflow_id="wf-support-1",
        )
        callback(make_kwargs(), FakeModelResponse(), START_AWARE, END_AWARE)

        records = read_spool_records(tmp_path)
        assert len(records) == 1
        record = records[0]
        assert record["format"] == FORMAT_TAG
        assert len(record["snapshot_id"]) == 32
        assert record["api_surface"] == "chat"
        assert record["workflow_id"] == "wf-support-1"
        assert record["metadata"]["workflow_name"] == "support-bot"
        assert record["metadata"]["call_name"] == "draft_reply"
        assert record["metadata"]["capture_sdk"]["name"] == "modelgenius-sdk"

    def test_config_object_plus_overrides(self, tmp_path):
        base = CaptureConfig(workflow_name="base-flow")
        callback = make_litellm_callback(base, spool_dir=tmp_path, call_name="step-2")
        callback(make_kwargs(), FakeModelResponse(), START_AWARE, END_AWARE)
        record = read_spool_records(tmp_path)[0]
        assert record["metadata"]["workflow_name"] == "base-flow"
        assert record["metadata"]["call_name"] == "step-2"
        assert base.spool_dir is None  # original config not mutated

    def test_record_passes_public_validator(self, tmp_path):
        validator = pytest.importorskip("modelgenius_sdk.validate.validator")
        records = run_callback(
            tmp_path, make_kwargs(), FakeModelResponse(), workflow_name="support-bot"
        )
        issues = validator.validate_record(records[0])
        errors = [issue for issue in issues if issue.severity == "error"]
        assert errors == []

    def test_lazy_export_is_this_factory(self):
        import modelgenius_sdk

        assert modelgenius_sdk.make_litellm_callback is make_litellm_callback


# ---------------------------------------------------------------------------
# Fail-open behaviour + capture gating
# ---------------------------------------------------------------------------


class Hostile:
    """Object whose every attribute access raises (worst-case duck typing)."""

    def __getattr__(self, name):
        raise RuntimeError(f"boom on {name}")


class TestFailOpen:
    def test_never_raises_on_garbage(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="modelgenius_sdk.capture")
        callback = make_litellm_callback(spool_dir=tmp_path)
        # None response, None kwargs, None times: no messages -> gated out.
        callback(None, None, None, None)
        # Non-dict kwargs, attribute-less response, garbage times: gated out.
        callback("not-a-dict", object(), "later", 10**20)
        # Hostile response object (but real chat kwargs): captured, fail-open.
        callback(make_kwargs(), Hostile(), START_AWARE, END_AWARE)
        # Response with garbage choices: captured, fail-open.
        callback(make_kwargs(), {"choices": "nope"}, START_AWARE, END_AWARE)

        # Never fatal; message-less garbage is skipped by the chat gate rather
        # than spooled as an unusable record.
        records = read_spool_records(tmp_path)
        assert len(records) == 2
        for record in records:
            assert record["format"] == FORMAT_TAG
            assert record["messages"]
            assert "request_timestamp" in record["captured_output"]
            assert "response_timestamp" in record["captured_output"]

    def test_none_response_produces_empty_captured_output(self, tmp_path):
        records = run_callback(tmp_path, make_kwargs(), None)
        captured = records[0]["captured_output"]
        assert captured["output_text"] == ""
        assert captured["tool_calls"] is None
        assert captured["raw_response"] == {}
        assert captured["model"] is None
        assert captured["usage"] is None

    def test_spool_failure_swallowed(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        callback = make_litellm_callback(spool_dir=blocker / "sub")
        callback(make_kwargs(), FakeModelResponse(), START_AWARE, END_AWARE)  # must not raise

    def test_debug_env_reraises(self, tmp_path, monkeypatch):
        def boom(record):
            raise ValueError("redact exploded")

        callback = make_litellm_callback(spool_dir=tmp_path, redact=boom)
        callback(make_kwargs(), FakeModelResponse(), START_AWARE, END_AWARE)  # swallowed
        monkeypatch.setenv(core.ENV_DEBUG, "1")
        with pytest.raises(ValueError):
            callback(make_kwargs(), FakeModelResponse(), START_AWARE, END_AWARE)

    def test_env_kill_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv(core.ENV_DISABLE, "0")
        records = run_callback(tmp_path, make_kwargs(), FakeModelResponse())
        assert records == []

    def test_sample_rate_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core.random, "random", lambda: 0.99)
        records = run_callback(
            tmp_path, make_kwargs(), FakeModelResponse(), sample_rate=0.5
        )
        assert records == []

    def test_request_filter_skips(self, tmp_path):
        records = run_callback(
            tmp_path,
            make_kwargs(),
            FakeModelResponse(),
            request_filter=lambda kwargs: kwargs.get("model") == "some-other-model",
        )
        assert records == []


# ---------------------------------------------------------------------------
# Module-level zero-config callback
# ---------------------------------------------------------------------------


class TestCaptureSuccessCallback:
    def test_built_lazily_from_env_on_first_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv(core.ENV_DIR, str(tmp_path))
        assert litellm_callback._default_callback is None  # nothing built at import
        capture_success_callback(make_kwargs(), FakeModelResponse(), START_AWARE, END_AWARE)
        assert litellm_callback._default_callback is not None
        built = litellm_callback._default_callback
        capture_success_callback(make_kwargs(), FakeModelResponse(), START_AWARE, END_AWARE)
        assert litellm_callback._default_callback is built  # reused, not rebuilt

        records = read_spool_records(tmp_path)
        assert len(records) == 2
        for record in records:
            assert record["format"] == FORMAT_TAG
            assert record["captured_output"]["api_surface"] == "chat"

    def test_zero_config_never_raises_on_garbage(self, tmp_path, monkeypatch):
        monkeypatch.setenv(core.ENV_DIR, str(tmp_path))
        capture_success_callback(None, None, None, None)  # must not raise


# ---------------------------------------------------------------------------
# Regressions: chat-only gate (litellm fires success_callback for ALL call types)
# ---------------------------------------------------------------------------


class TestChatGate:
    def test_embedding_call_gated_out(self, tmp_path):
        """litellm synthesizes chat-shaped `messages` for embedding calls; the
        callback must not spool them as fake chat snapshots."""
        kwargs = make_kwargs(call_type="embedding", model="text-embedding-3-small")
        records = run_callback(tmp_path, kwargs, FakeModelResponse())
        assert records == []

    def test_other_non_chat_call_types_gated_out(self, tmp_path):
        for call_type in ("image_generation", "transcription", "text_completion"):
            kwargs = make_kwargs(call_type=call_type)
            run_callback(tmp_path, kwargs, FakeModelResponse())
        assert read_spool_records(tmp_path) == []

    def test_chat_call_types_captured(self, tmp_path):
        for call_type in ("completion", "acompletion"):
            kwargs = make_kwargs(call_type=call_type)
            run_callback(tmp_path, kwargs, FakeModelResponse())
        # Absent call_type (older litellm) is also treated as chat.
        run_callback(tmp_path, make_kwargs(), FakeModelResponse())
        assert len(read_spool_records(tmp_path)) == 3
