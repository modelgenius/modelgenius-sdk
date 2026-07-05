"""Tests for modelgenius_sdk.capture.core — the frozen capture core API."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import threading
from datetime import timezone
from pathlib import Path

import pytest

import modelgenius_sdk
from modelgenius_sdk.capture import core
from modelgenius_sdk.capture.core import (
    ENV_DEBUG,
    ENV_DIR,
    ENV_DISABLE,
    ENV_SAMPLE_RATE,
    FORMAT_TAG,
    CaptureConfig,
    SpoolWriter,
    build_envelope,
    emit,
    get_writer,
    sanitize_request_kwargs,
    should_capture,
    to_jsonable,
    utcnow_iso,
    warn_once,
)

REQUEST_TS = "2026-07-05T12:00:00+00:00"


@pytest.fixture(autouse=True)
def _reset_core_state(monkeypatch):
    """Isolate module-global state (writer cache, warn-once registry, env)."""
    monkeypatch.setattr(core, "_writers", {})
    monkeypatch.setattr(core, "_warned_keys", set())
    for env_var in (ENV_DIR, ENV_SAMPLE_RATE, ENV_DISABLE, ENV_DEBUG):
        monkeypatch.delenv(env_var, raising=False)


def read_spool_records(spool_dir: Path) -> list:
    records = []
    for path in sorted(spool_dir.glob("snapshots-*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# build_envelope
# ---------------------------------------------------------------------------


class TestBuildEnvelope:
    def test_envelope_shape(self):
        config = CaptureConfig(
            workflow_id="wf-123",
            workflow_name="grading",
            call_name="essay_feedback",
            metadata={"env": "prod"},
        )
        request_kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        }
        captured = {"output_text": "hello", "request_timestamp": REQUEST_TS}
        record = build_envelope(
            config,
            request_kwargs=request_kwargs,
            api_surface="chat",
            request_ts=REQUEST_TS,
            captured_output=captured,
        )

        assert record["format"] == FORMAT_TAG
        assert re.fullmatch(r"[0-9a-f]{32}", record["snapshot_id"])
        assert record["timestamp"] == REQUEST_TS  # passthrough, not regenerated
        assert record["api_surface"] == "chat"
        assert record["model"] == "gpt-4o"
        assert record["messages"] == [{"role": "user", "content": "hi"}]
        assert record["temperature"] == 0.2
        assert record["workflow_id"] == "wf-123"
        assert record["captured_output"] == captured

        metadata = record["metadata"]
        assert metadata["env"] == "prod"
        assert metadata["workflow_name"] == "grading"
        assert metadata["call_name"] == "essay_feedback"
        assert metadata["capture_sdk"] == {
            "name": "modelgenius-sdk",
            "version": modelgenius_sdk.__version__,
        }

    def test_snapshot_ids_unique(self):
        config = CaptureConfig()
        ids = {
            build_envelope(
                config,
                request_kwargs={},
                api_surface="chat",
                request_ts=REQUEST_TS,
                captured_output=None,
            )["snapshot_id"]
            for _ in range(5)
        }
        assert len(ids) == 5

    def test_captured_output_none_omitted(self):
        record = build_envelope(
            CaptureConfig(),
            request_kwargs={"model": "m"},
            api_surface="chat",
            request_ts=REQUEST_TS,
            captured_output=None,
        )
        assert "captured_output" not in record

    def test_api_surface_falsy_omitted(self):
        # Anthropic-shaped records omit top-level api_surface (server sets it).
        record = build_envelope(
            CaptureConfig(),
            request_kwargs={"model": "m"},
            api_surface=None,
            request_ts=REQUEST_TS,
            captured_output=None,
        )
        assert "api_surface" not in record

    def test_workflow_fields_omitted_when_unset(self):
        record = build_envelope(
            CaptureConfig(),
            request_kwargs={"model": "m"},
            api_surface="chat",
            request_ts=REQUEST_TS,
            captured_output=None,
        )
        assert "workflow_id" not in record
        assert "workflow_name" not in record["metadata"]
        assert "call_name" not in record["metadata"]
        assert "capture_sdk" in record["metadata"]  # provenance always present

    def test_envelope_keys_win_over_request_kwargs(self):
        config = CaptureConfig(workflow_id="cfg-wf-id", workflow_name="cfg-wf")
        request_kwargs = {
            "model": "gpt-4o",
            "format": "bogus-format",
            "snapshot_id": "bogus-id",
            "timestamp": "1999-01-01T00:00:00+00:00",
            "workflow_id": "kwarg-wf-id",
            "metadata": {"workflow_name": "kwarg-wf", "custom": 7},
        }
        record = build_envelope(
            config,
            request_kwargs=request_kwargs,
            api_surface="chat",
            request_ts=REQUEST_TS,
            captured_output=None,
        )
        assert record["format"] == FORMAT_TAG
        assert record["snapshot_id"] != "bogus-id"
        assert record["timestamp"] == REQUEST_TS
        assert record["workflow_id"] == "cfg-wf-id"
        # Request-kwarg metadata is merged in but config-level keys win.
        assert record["metadata"]["custom"] == 7
        assert record["metadata"]["workflow_name"] == "cfg-wf"
        assert record["metadata"]["capture_sdk"]["name"] == "modelgenius-sdk"
        # Non-colliding kwargs pass through untouched.
        assert record["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# sanitize_request_kwargs / to_jsonable
# ---------------------------------------------------------------------------


class ModelDumpObj:
    def model_dump(self, mode=None):
        return {"kind": "model_dump", "mode": mode}


class ToDictObj:
    def to_dict(self):
        return {"kind": "to_dict"}


@dataclasses.dataclass
class Point:
    x: int
    label: str
    tags: tuple


class PlainObj:
    def __str__(self):
        return "plain-obj"


class TestSanitizeRequestKwargs:
    def test_drops_transport_keys_without_mutating_input(self):
        stream_cls = object()
        original = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_headers": {"authorization": "Bearer secret"},
            "extra_query": {"api-version": "1"},
            "timeout": 30,
            "stream_cls": stream_cls,
        }
        snapshot_of_input = dict(original)

        result = sanitize_request_kwargs(original)

        assert set(result) == {"model", "messages"}
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        # Input not mutated: same keys, same values/identities.
        assert original == snapshot_of_input
        assert original["stream_cls"] is stream_cls

    def test_values_pass_through_to_jsonable(self):
        result = sanitize_request_kwargs({"response_format": ModelDumpObj()})
        assert result["response_format"] == {"kind": "model_dump", "mode": "json"}


class TestToJsonable:
    def test_primitives_pass_through(self):
        for value in (None, "s", 3, 2.5, True):
            assert to_jsonable(value) == value

    def test_model_dump_object(self):
        assert to_jsonable(ModelDumpObj()) == {"kind": "model_dump", "mode": "json"}

    def test_to_dict_object(self):
        assert to_jsonable(ToDictObj()) == {"kind": "to_dict"}

    def test_dataclass_instance(self):
        assert to_jsonable(Point(x=1, label="a", tags=("t1", "t2"))) == {
            "x": 1,
            "label": "a",
            "tags": ["t1", "t2"],
        }

    def test_plain_object_falls_back_to_str(self):
        assert to_jsonable(PlainObj()) == "plain-obj"

    def test_nested_containers(self):
        nested = {"a": [ModelDumpObj(), (1, 2)], "b": {"c": ToDictObj()}}
        assert to_jsonable(nested) == {
            "a": [{"kind": "model_dump", "mode": "json"}, [1, 2]],
            "b": {"c": {"kind": "to_dict"}},
        }


# ---------------------------------------------------------------------------
# SpoolWriter / get_writer
# ---------------------------------------------------------------------------


class TestSpoolWriter:
    def test_lazy_dir_creation_and_filename_pattern(self, tmp_path):
        spool = tmp_path / "spool"
        writer = SpoolWriter(CaptureConfig(spool_dir=spool))
        assert not spool.exists()  # nothing until the first write
        writer.write({"a": 1})
        files = list(spool.glob("*.jsonl"))
        assert len(files) == 1
        assert re.fullmatch(r"snapshots-\d{8}-\d+-000\.jsonl", files[0].name)
        writer.close()

    def test_rotation_at_max_file_bytes(self, tmp_path):
        # Each record serializes to ~110 bytes; cap at 150 so every file
        # takes exactly two records before rotation triggers.
        record = {"x": "a" * 100}
        line_len = len(json.dumps(record, ensure_ascii=False)) + 1
        assert 75 < line_len < 150

        writer = SpoolWriter(CaptureConfig(spool_dir=tmp_path, max_file_bytes=150))
        for _ in range(4):
            writer.write(record)
        writer.close()

        files = sorted(tmp_path.glob("snapshots-*.jsonl"))
        assert len(files) == 2
        assert files[0].name.endswith("-000.jsonl")
        assert files[1].name.endswith("-001.jsonl")
        for path in files:
            lines = path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 2
            for line in lines:
                assert json.loads(line) == record

    def test_write_after_close_reopens(self, tmp_path):
        writer = SpoolWriter(CaptureConfig(spool_dir=tmp_path))
        writer.write({"n": 1})
        writer.close()
        writer.write({"n": 2})
        writer.close()
        assert [r["n"] for r in read_spool_records(tmp_path)] == [1, 2]

    def test_thread_safety_smoke(self, tmp_path):
        writer = get_writer(CaptureConfig(spool_dir=tmp_path))
        n_threads, n_writes = 8, 25

        def worker(thread_idx):
            for write_idx in range(n_writes):
                writer.write({"t": thread_idx, "j": write_idx, "pad": "z" * 50})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        writer.close()

        records = read_spool_records(tmp_path)  # every line must parse as JSON
        assert len(records) == n_threads * n_writes
        assert {(r["t"], r["j"]) for r in records} == {
            (t, j) for t in range(n_threads) for j in range(n_writes)
        }

    def test_unwritable_spool_dir_does_not_raise(self, tmp_path, caplog):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        writer = SpoolWriter(CaptureConfig(spool_dir=blocker / "sub"))
        caplog.set_level(logging.WARNING, logger="modelgenius_sdk.capture")
        writer.write({"a": 1})  # must not raise
        writer.write({"a": 2})  # same error class → no second warning
        warnings = [r for r in caplog.records if r.name == "modelgenius_sdk.capture"]
        assert len(warnings) == 1

    def test_unwritable_spool_dir_raises_under_debug(self, tmp_path, monkeypatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        writer = SpoolWriter(CaptureConfig(spool_dir=blocker / "sub"))
        monkeypatch.setenv(ENV_DEBUG, "1")
        with pytest.raises(OSError):
            writer.write({"a": 1})


class TestGetWriter:
    def test_singleton_per_resolved_dir(self, tmp_path):
        dir_a = tmp_path / "a"
        writer_1 = get_writer(CaptureConfig(spool_dir=dir_a))
        writer_2 = get_writer(CaptureConfig(spool_dir=str(dir_a)))  # str vs Path
        writer_3 = get_writer(CaptureConfig(spool_dir=tmp_path / "b"))
        assert writer_1 is writer_2
        assert writer_1 is not writer_3


# ---------------------------------------------------------------------------
# CaptureConfig resolution + should_capture
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_spool_dir_explicit_env_default(self, monkeypatch, tmp_path):
        assert CaptureConfig().resolved_spool_dir() == Path("./modelgenius_snapshots")
        monkeypatch.setenv(ENV_DIR, str(tmp_path / "from-env"))
        assert CaptureConfig().resolved_spool_dir() == tmp_path / "from-env"
        explicit = tmp_path / "explicit"
        assert CaptureConfig(spool_dir=explicit).resolved_spool_dir() == explicit

    def test_sample_rate_explicit_env_default(self, monkeypatch):
        assert CaptureConfig().resolved_sample_rate() == 1.0
        monkeypatch.setenv(ENV_SAMPLE_RATE, "0.25")
        assert CaptureConfig().resolved_sample_rate() == 0.25
        assert CaptureConfig(sample_rate=0.5).resolved_sample_rate() == 0.5
        monkeypatch.setenv(ENV_SAMPLE_RATE, "not-a-float")
        assert CaptureConfig().resolved_sample_rate() == 1.0

    def test_enabled_explicit_env_default(self, monkeypatch):
        assert CaptureConfig().resolved_enabled() is True
        for value in ("0", "false", "No", "FALSE"):
            monkeypatch.setenv(ENV_DISABLE, value)
            assert CaptureConfig().resolved_enabled() is False
        monkeypatch.setenv(ENV_DISABLE, "1")
        assert CaptureConfig().resolved_enabled() is True
        monkeypatch.setenv(ENV_DISABLE, "0")
        assert CaptureConfig(enabled=True).resolved_enabled() is True  # explicit wins


class TestShouldCapture:
    def test_kill_switch_env(self, monkeypatch):
        assert should_capture(CaptureConfig(), {}) is True
        monkeypatch.setenv(ENV_DISABLE, "0")
        assert should_capture(CaptureConfig(), {}) is False
        monkeypatch.setenv(ENV_DISABLE, "false")
        assert should_capture(CaptureConfig(), {}) is False
        # Explicit config beats the env kill switch.
        assert should_capture(CaptureConfig(enabled=True), {}) is True

    def test_disabled_via_config(self):
        assert should_capture(CaptureConfig(enabled=False), {}) is False

    def test_sample_rate_monkeypatched_random(self, monkeypatch):
        config = CaptureConfig(sample_rate=0.5)
        monkeypatch.setattr(core.random, "random", lambda: 0.49)
        assert should_capture(config, {}) is True
        monkeypatch.setattr(core.random, "random", lambda: 0.5)
        assert should_capture(config, {}) is False  # random() >= rate → skip
        monkeypatch.setattr(core.random, "random", lambda: 0.99)
        assert should_capture(config, {}) is False

    def test_sample_rate_from_env(self, monkeypatch):
        monkeypatch.setenv(ENV_SAMPLE_RATE, "0.6")
        config = CaptureConfig()
        monkeypatch.setattr(core.random, "random", lambda: 0.59)
        assert should_capture(config, {}) is True
        monkeypatch.setattr(core.random, "random", lambda: 0.75)
        assert should_capture(config, {}) is False

    def test_request_filter(self):
        config = CaptureConfig(request_filter=lambda kwargs: kwargs.get("model") == "gpt-4o")
        assert should_capture(config, {"model": "gpt-4o"}) is True
        assert should_capture(config, {"model": "o3"}) is False

    def test_request_filter_raising_is_fail_open(self, monkeypatch):
        def boom(kwargs):
            raise RuntimeError("filter exploded")

        config = CaptureConfig(request_filter=boom)
        assert should_capture(config, {}) is False  # swallowed, capture skipped
        monkeypatch.setenv(ENV_DEBUG, "1")
        with pytest.raises(RuntimeError):
            should_capture(config, {})


# ---------------------------------------------------------------------------
# emit / redact
# ---------------------------------------------------------------------------


class TestEmit:
    def test_writes_record(self, tmp_path):
        config = CaptureConfig(spool_dir=tmp_path)
        emit(config, {"model": "m", "timestamp": REQUEST_TS})
        assert read_spool_records(tmp_path) == [{"model": "m", "timestamp": REQUEST_TS}]

    def test_redact_returning_none_drops_record(self, tmp_path):
        spool = tmp_path / "spool"
        config = CaptureConfig(spool_dir=spool, redact=lambda record: None)
        emit(config, {"model": "m"})
        assert not spool.exists()  # dropped before the writer ever touched disk

    def test_redact_can_rewrite_record(self, tmp_path):
        def redact(record):
            clean = dict(record)
            clean["messages"] = "[REDACTED]"
            return clean

        config = CaptureConfig(spool_dir=tmp_path, redact=redact)
        emit(config, {"model": "m", "messages": [{"role": "user", "content": "pii"}]})
        assert read_spool_records(tmp_path) == [{"model": "m", "messages": "[REDACTED]"}]

    def test_redact_raising_is_fail_open(self, tmp_path, monkeypatch):
        def boom(record):
            raise ValueError("redact exploded")

        spool = tmp_path / "spool"
        config = CaptureConfig(spool_dir=spool, redact=boom)
        emit(config, {"model": "m"})  # must not raise
        assert not spool.exists()
        monkeypatch.setenv(ENV_DEBUG, "1")
        with pytest.raises(ValueError):
            emit(config, {"model": "m"})


# ---------------------------------------------------------------------------
# warn_once / utcnow_iso / lazy exports
# ---------------------------------------------------------------------------


class TestMisc:
    def test_warn_once_per_key(self, caplog):
        caplog.set_level(logging.WARNING, logger="modelgenius_sdk.capture")
        warn_once("key-1", "message one")
        warn_once("key-1", "message repeated")
        warn_once("key-2", "message two")
        messages = [
            r.getMessage() for r in caplog.records if r.name == "modelgenius_sdk.capture"
        ]
        assert messages == ["message one", "message two"]

    def test_utcnow_iso_is_tz_aware_utc(self):
        from datetime import datetime

        parsed = datetime.fromisoformat(utcnow_iso())
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timezone.utc.utcoffset(None)

    def test_lazy_exports(self):
        assert modelgenius_sdk.__version__ == "0.1.0"
        assert modelgenius_sdk.CaptureConfig is CaptureConfig

        import modelgenius_sdk.capture as capture_pkg

        assert capture_pkg.CaptureConfig is CaptureConfig
        assert capture_pkg.SpoolWriter is SpoolWriter
        assert capture_pkg.get_writer is get_writer

        with pytest.raises(AttributeError):
            _ = modelgenius_sdk.definitely_not_an_export
        with pytest.raises(AttributeError):
            _ = capture_pkg.definitely_not_an_export

        # Sibling wrapper modules are authored in parallel; lazy access must
        # simply attempt the import — tolerate their absence here.
        for lazy_name in ("wrap_openai", "wrap_anthropic", "make_litellm_callback"):
            try:
                exported = getattr(modelgenius_sdk, lazy_name)
            except (ImportError, AttributeError):
                continue
            assert callable(exported)
            assert getattr(capture_pkg, lazy_name) is exported


# ---------------------------------------------------------------------------
# Request-message normalization (upload-path acceptance guarantees)
# ---------------------------------------------------------------------------


class TestMessageNormalization:
    """Shapes routine in production OpenAI traffic that the upload path rejects
    verbatim: the tool-loop assistant message (content: null + tool_calls) and
    the o-series "developer" role. sanitize_request_kwargs must normalize both
    so every spooled record is uploadable."""

    def test_tool_loop_assistant_null_content_coerced_to_empty_string(self):
        kwargs = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 21}'},
            ],
        }
        sanitized = sanitize_request_kwargs(kwargs)
        assistant = sanitized["messages"][1]
        assert assistant["content"] == ""
        assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        # Original input never mutated.
        assert kwargs["messages"][1]["content"] is None

    def test_developer_role_mapped_to_system(self):
        kwargs = {"messages": [{"role": "developer", "content": "Be terse."}]}
        sanitized = sanitize_request_kwargs(kwargs)
        assert sanitized["messages"][0]["role"] == "system"
        assert kwargs["messages"][0]["role"] == "developer"  # input untouched

    def test_non_dict_message_entries_pass_through(self):
        kwargs = {"messages": ["not-a-dict", None]}
        sanitized = sanitize_request_kwargs(kwargs)
        assert sanitized["messages"] == ["not-a-dict", None]
