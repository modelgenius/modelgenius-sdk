"""Tests for modelgenius_sdk.capture.openai — the OpenAI chat-completions wrapper.

All provider objects are hand-rolled fakes: a pydantic-like variant exposing
``model_dump(mode="json")`` plus attribute access, and a plain-attrs variant
WITHOUT ``model_dump`` to prove the duck-typing fallbacks. The ``openai``
package is never imported.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelgenius_sdk.capture import core
from modelgenius_sdk.capture.core import (
    ENV_DEBUG,
    ENV_DIR,
    ENV_DISABLE,
    ENV_SAMPLE_RATE,
    FORMAT_TAG,
    CaptureConfig,
)
from modelgenius_sdk.capture.openai import wrap_openai

MESSAGES = [{"role": "user", "content": "What's the weather in Paris?"}]


@pytest.fixture(autouse=True)
def _reset_core_state(monkeypatch):
    """Isolate module-global state (writer cache, warn-once registry, env)."""
    monkeypatch.setattr(core, "_writers", {})
    monkeypatch.setattr(core, "_warned_keys", set())
    for env_var in (ENV_DIR, ENV_SAMPLE_RATE, ENV_DISABLE, ENV_DEBUG):
        monkeypatch.delenv(env_var, raising=False)


def read_spool(spool_dir: Path) -> list:
    records = []
    for path in sorted(Path(spool_dir).glob("snapshots-*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
    return records


def assert_isoformat(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _dump(value):
    if isinstance(value, Obj):
        return _dump(vars(value))
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


class Obj:
    """Pydantic-like fake: attribute access plus model_dump(mode="json")."""

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)

    def model_dump(self, mode="python"):
        return _dump(vars(self))


class Plain:
    """Plain-attrs fake WITHOUT model_dump — proves duck-typing fallbacks."""

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


def make_response(cls=Obj, content="The grade is B+.", reasoning_tokens=7):
    return cls(
        id="chatcmpl-test-1",
        model="gpt-4o-2024-08-06",
        choices=[
            cls(
                index=0,
                finish_reason="stop",
                message=cls(role="assistant", content=content, tool_calls=None),
            )
        ],
        usage=cls(
            prompt_tokens=12,
            completion_tokens=34,
            total_tokens=46,
            completion_tokens_details=cls(reasoning_tokens=reasoning_tokens, audio_tokens=0),
        ),
    )


def text_chunk(text, finish=None):
    return Obj(
        id="chatcmpl-stream-1",
        model="gpt-4o-mini",
        choices=[
            Obj(index=0, delta=Obj(content=text, tool_calls=None), finish_reason=finish)
        ],
        usage=None,
    )


def tool_chunk(fragment, tool_index=0, call_id=None, name=None):
    call = Obj(
        index=tool_index,
        id=call_id,
        type="function" if call_id else None,
        function=Obj(name=name, arguments=fragment),
    )
    return Obj(
        id="chatcmpl-stream-1",
        model="gpt-4o-mini",
        choices=[Obj(index=0, delta=Obj(content=None, tool_calls=[call]), finish_reason=None)],
        usage=None,
    )


def usage_chunk():
    """Usage-bearing final chunk with EMPTY choices (stream_options include_usage)."""
    return Obj(
        id="chatcmpl-stream-1",
        model="gpt-4o-mini",
        choices=[],
        usage=Obj(
            prompt_tokens=9,
            completion_tokens=12,
            total_tokens=21,
            completion_tokens_details=Obj(reasoning_tokens=3),
        ),
    )


class FakeCompletions:
    def __init__(self, result=None, error=None, delay=0.001):
        self.calls = []
        self.result = result
        self.error = error
        self.delay = delay

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.result


class FakeAsyncCompletions:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result
        self.error = error

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0.001)
        if self.error is not None:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class FakeStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self._pos = 0
        self.closed = False
        self.entered = False
        self.exited = False
        self.custom_marker = "inner-stream"

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= len(self.chunks):
            raise StopIteration
        chunk = self.chunks[self._pos]
        self._pos += 1
        return chunk

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        self.close()
        return False

    def close(self):
        self.closed = True


class FakeAsyncStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self._pos = 0
        self.closed = False
        self.entered = False
        self.exited = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pos >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self._pos]
        self._pos += 1
        return chunk

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        await self.aclose()
        return False

    async def aclose(self):
        self.closed = True


class ExplodingChunk:
    """Chunk whose choices access raises — simulates accumulation failure."""

    id = "chatcmpl-stream-1"
    model = "gpt-4o-mini"
    usage = None

    @property
    def choices(self):
        raise RuntimeError("accumulator boom")


# ---------------------------------------------------------------------------
# Sync non-stream
# ---------------------------------------------------------------------------


class TestSyncNonStream:
    def test_capture_happy_path(self, tmp_path):
        response = make_response()
        completions = FakeCompletions(result=response)
        client = FakeClient(completions)
        returned = wrap_openai(
            client,
            CaptureConfig(spool_dir=tmp_path),
            workflow_name="grading",
            call_name="essay_feedback",
        )
        assert returned is client

        result = client.chat.completions.create(
            model="gpt-4o", messages=MESSAGES, temperature=0.2, seed=42
        )
        assert result is response
        assert completions.calls == [
            {"model": "gpt-4o", "messages": MESSAGES, "temperature": 0.2, "seed": 42}
        ]

        records = read_spool(tmp_path)
        assert len(records) == 1
        record = records[0]

        # Envelope
        assert record["format"] == FORMAT_TAG
        assert record["api_surface"] == "chat"
        assert record["model"] == "gpt-4o"
        assert record["messages"] == MESSAGES
        assert record["temperature"] == 0.2
        assert record["seed"] == 42  # capture-only param passes through faithfully
        assert record["snapshot_id"]
        assert record["metadata"]["workflow_name"] == "grading"
        assert record["metadata"]["call_name"] == "essay_feedback"
        assert record["metadata"]["capture_sdk"]["name"] == "modelgenius-sdk"
        assert_isoformat(record["timestamp"])

        # captured_output
        co = record["captured_output"]
        assert co["output_text"] == "The grade is B+."
        assert co["tool_calls"] is None
        assert co["model"] == "gpt-4o-2024-08-06"
        assert co["api_surface"] == "chat"
        assert co["cost_usd"] is None
        assert co["usage"] == {
            "input_tokens": 12,
            "output_tokens": 34,
            "thinking_tokens": 7,
            "total_tokens": 46,
        }
        assert co["provider_metadata"]["finish_reason"] == "stop"
        assert co["provider_metadata"]["request_id"] == "chatcmpl-test-1"
        assert isinstance(co["raw_response"], dict)
        assert co["raw_response"]["model"] == "gpt-4o-2024-08-06"
        assert_isoformat(co["request_timestamp"])
        assert_isoformat(co["response_timestamp"])
        assert record["timestamp"] == co["request_timestamp"]
        assert co["latency_ms"] > 0

    def test_plain_attrs_duck_typing(self, tmp_path):
        """Responses without model_dump still capture via attribute access."""
        response = make_response(cls=Plain)
        client = FakeClient(FakeCompletions(result=response))
        wrap_openai(client, spool_dir=tmp_path)

        result = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert result is response

        records = read_spool(tmp_path)
        assert len(records) == 1
        co = records[0]["captured_output"]
        assert co["output_text"] == "The grade is B+."
        assert co["model"] == "gpt-4o-2024-08-06"
        assert co["usage"] == {
            "input_tokens": 12,
            "output_tokens": 34,
            "thinking_tokens": 7,
            "total_tokens": 46,
        }
        assert co["provider_metadata"]["finish_reason"] == "stop"
        # No model_dump/to_dict → raw_response degrades to an (empty) dict.
        assert isinstance(co["raw_response"], dict)


# ---------------------------------------------------------------------------
# Async non-stream
# ---------------------------------------------------------------------------


class TestAsyncNonStream:
    def test_async_capture(self, tmp_path):
        response = make_response()
        completions = FakeAsyncCompletions(result=response)
        client = FakeClient(completions)
        wrap_openai(client, spool_dir=tmp_path)

        result = asyncio.run(client.chat.completions.create(model="gpt-4o", messages=MESSAGES))
        assert result is response
        assert completions.calls == [{"model": "gpt-4o", "messages": MESSAGES}]

        records = read_spool(tmp_path)
        assert len(records) == 1
        record = records[0]
        assert record["api_surface"] == "chat"
        co = record["captured_output"]
        assert co["output_text"] == "The grade is B+."
        assert co["usage"]["thinking_tokens"] == 7
        assert co["latency_ms"] > 0
        assert_isoformat(co["request_timestamp"])
        assert_isoformat(co["response_timestamp"])


# ---------------------------------------------------------------------------
# Sync streaming
# ---------------------------------------------------------------------------


def stream_chunks():
    return [
        text_chunk("Checking "),
        text_chunk("the weather."),
        tool_chunk('{"city": ', tool_index=0, call_id="call_w1", name="get_weather"),
        tool_chunk('"Paris"}', tool_index=0),
        tool_chunk('{"tz": "CET"}', tool_index=1, call_id="call_w2", name="get_time"),
        text_chunk(None, finish="tool_calls"),
        usage_chunk(),
    ]


class TestSyncStream:
    def test_full_iteration_capture(self, tmp_path):
        chunks = stream_chunks()
        stream = FakeStream(chunks)
        client = FakeClient(FakeCompletions(result=stream))
        wrap_openai(client, spool_dir=tmp_path)

        proxy = client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        assert proxy.custom_marker == "inner-stream"  # __getattr__ delegation

        received = []
        with proxy as s:
            for chunk in s:
                received.append(chunk)
        assert received == chunks  # identical objects, order preserved
        assert stream.entered and stream.exited and stream.closed

        records = read_spool(tmp_path)
        assert len(records) == 1  # emit-once: StopIteration then CM exit
        record = records[0]
        assert record["stream"] is True  # request kwargs pass through faithfully
        co = record["captured_output"]
        assert co["output_text"] == "Checking the weather."
        assert co["tool_calls"] == [
            {
                "id": "call_w1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            },
            {
                "id": "call_w2",
                "type": "function",
                "function": {"name": "get_time", "arguments": '{"tz": "CET"}'},
            },
        ]
        assert co["model"] == "gpt-4o-mini"
        assert co["usage"] == {
            "input_tokens": 9,
            "output_tokens": 12,
            "thinking_tokens": 3,
            "total_tokens": 21,
        }
        assert co["provider_metadata"]["finish_reason"] == "tool_calls"
        assert co["provider_metadata"]["request_id"] == "chatcmpl-stream-1"
        assert "truncated_stream" not in co["provider_metadata"]
        assert co["raw_response"]["stream"] is True
        assert co["raw_response"]["chunk_count"] == 7
        reconstructed = co["raw_response"]["reconstructed"]
        assert reconstructed["choices"][0]["message"]["content"] == "Checking the weather."
        assert reconstructed["choices"][0]["finish_reason"] == "tool_calls"
        assert co["latency_ms"] > 0
        assert_isoformat(co["request_timestamp"])
        assert_isoformat(co["response_timestamp"])

    def test_plain_iteration_without_cm(self, tmp_path):
        stream = FakeStream([text_chunk("Hel"), text_chunk("lo"), text_chunk(None, finish="stop")])
        client = FakeClient(FakeCompletions(result=stream))
        wrap_openai(client, spool_dir=tmp_path)

        proxy = client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        received = list(proxy)
        assert len(received) == 3

        records = read_spool(tmp_path)
        assert len(records) == 1
        co = records[0]["captured_output"]
        assert co["output_text"] == "Hello"
        assert co["provider_metadata"]["finish_reason"] == "stop"
        assert "truncated_stream" not in co["provider_metadata"]

    def test_abandoned_stream_marks_truncated(self, tmp_path):
        stream = FakeStream(
            [text_chunk("Hel"), text_chunk("lo"), text_chunk(" world"), text_chunk(None, finish="stop")]
        )
        client = FakeClient(FakeCompletions(result=stream))
        wrap_openai(client, spool_dir=tmp_path)

        proxy = client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        with proxy as s:
            next(s)
            next(s)
            # abandon mid-iteration; CM exit should still emit
        records = read_spool(tmp_path)
        assert len(records) == 1
        co = records[0]["captured_output"]
        assert co["output_text"] == "Hello"
        assert co["provider_metadata"]["truncated_stream"] is True
        assert co["raw_response"]["chunk_count"] == 2

    def test_accumulation_failure_never_breaks_iteration(self, tmp_path):
        chunks = [text_chunk("Hi"), ExplodingChunk(), text_chunk(" there")]
        stream = FakeStream(chunks)
        client = FakeClient(FakeCompletions(result=stream))
        wrap_openai(client, spool_dir=tmp_path)

        proxy = client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        received = list(proxy)  # must NOT raise
        assert received == chunks
        # Capture stopped fail-open: nothing (possibly corrupt) is emitted.
        assert read_spool(tmp_path) == []


# ---------------------------------------------------------------------------
# Async streaming
# ---------------------------------------------------------------------------


class TestAsyncStream:
    def test_async_full_iteration_capture(self, tmp_path):
        chunks = stream_chunks()
        stream = FakeAsyncStream(chunks)
        client = FakeClient(FakeAsyncCompletions(result=stream))
        wrap_openai(client, spool_dir=tmp_path)

        async def consume():
            proxy = await client.chat.completions.create(
                model="gpt-4o-mini", messages=MESSAGES, stream=True
            )
            received = []
            async with proxy as s:
                async for chunk in s:
                    received.append(chunk)
            return received

        received = asyncio.run(consume())
        assert received == chunks
        assert stream.entered and stream.exited and stream.closed

        records = read_spool(tmp_path)
        assert len(records) == 1
        co = records[0]["captured_output"]
        assert co["output_text"] == "Checking the weather."
        assert co["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
        assert co["usage"]["total_tokens"] == 21
        assert co["provider_metadata"]["finish_reason"] == "tool_calls"
        assert "truncated_stream" not in co["provider_metadata"]

    def test_async_abandoned_stream_marks_truncated(self, tmp_path):
        stream = FakeAsyncStream([text_chunk("par"), text_chunk("tial"), text_chunk("!")])
        client = FakeClient(FakeAsyncCompletions(result=stream))
        wrap_openai(client, spool_dir=tmp_path)

        async def consume_partially():
            proxy = await client.chat.completions.create(
                model="gpt-4o-mini", messages=MESSAGES, stream=True
            )
            async with proxy as s:
                await s.__anext__()
                await s.__anext__()

        asyncio.run(consume_partially())
        records = read_spool(tmp_path)
        assert len(records) == 1
        co = records[0]["captured_output"]
        assert co["output_text"] == "partial"
        assert co["provider_metadata"]["truncated_stream"] is True


# ---------------------------------------------------------------------------
# Failure modes & wrapping semantics
# ---------------------------------------------------------------------------


class TestFailureModesAndWrapping:
    def test_provider_exception_propagates_and_nothing_written(self, tmp_path):
        client = FakeClient(FakeCompletions(error=RuntimeError("rate limited")))
        wrap_openai(client, spool_dir=tmp_path)
        with pytest.raises(RuntimeError, match="rate limited"):
            client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert read_spool(tmp_path) == []

    def test_async_provider_exception_propagates(self, tmp_path):
        client = FakeClient(FakeAsyncCompletions(error=ValueError("bad request")))
        wrap_openai(client, spool_dir=tmp_path)
        with pytest.raises(ValueError, match="bad request"):
            asyncio.run(client.chat.completions.create(model="gpt-4o", messages=MESSAGES))
        assert read_spool(tmp_path) == []

    def test_double_wrap_is_noop(self, tmp_path):
        response = make_response()
        client = FakeClient(FakeCompletions(result=response))
        wrap_openai(client, spool_dir=tmp_path)
        wrapped_create = client.chat.completions.create
        wrap_openai(client, spool_dir=tmp_path)
        assert client.chat.completions.create is wrapped_create  # not re-wrapped

        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert len(read_spool(tmp_path)) == 1  # captured exactly once

    def test_unwritable_spool_dir_never_breaks_call(self, tmp_path):
        blocker = tmp_path / "spool-as-file"
        blocker.write_text("not a directory")
        response = make_response()
        client = FakeClient(FakeCompletions(result=response))
        wrap_openai(client, spool_dir=blocker)

        result = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert result is response  # capture failure swallowed, call unaffected

    def test_redact_drop_writes_nothing(self, tmp_path):
        response = make_response()
        client = FakeClient(FakeCompletions(result=response))
        wrap_openai(client, spool_dir=tmp_path, redact=lambda record: None)
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert read_spool(tmp_path) == []

    def test_env_kill_switch_disables_capture(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_DISABLE, "0")
        response = make_response()
        client = FakeClient(FakeCompletions(result=response))
        wrap_openai(client, spool_dir=tmp_path)
        result = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert result is response
        assert read_spool(tmp_path) == []


# ---------------------------------------------------------------------------
# Regressions: real-SDK async detection + tool-loop capture
# ---------------------------------------------------------------------------


class DecoratedAsyncCompletions:
    """Mimics the real AsyncOpenAI: ``create`` is a SYNC ``functools.wraps``
    wrapper around an async def (openai's ``@required_args`` pattern), so
    plain ``inspect.iscoroutinefunction(create)`` is False and only the
    ``inspect.unwrap`` chain reveals the coroutine function."""

    def __init__(self, result):
        import functools

        self.calls = []

        async def _create(**kwargs):
            self.calls.append(kwargs)
            await asyncio.sleep(0.001)
            return result

        @functools.wraps(_create)
        def create(**kwargs):
            return _create(**kwargs)

        self.create = create


class TestDecoratedAsyncClient:
    def test_sync_decorated_async_create_captured_correctly(self, tmp_path):
        """An AsyncOpenAI-style client must be detected as async: the capture
        must contain the real response (not a duck-typed coroutine husk)."""
        response = make_response()
        completions = DecoratedAsyncCompletions(response)
        client = FakeClient(completions)
        wrap_openai(client, spool_dir=tmp_path)

        result = asyncio.run(client.chat.completions.create(model="gpt-4o", messages=MESSAGES))
        assert result is response

        [record] = read_spool(tmp_path)
        captured = record["captured_output"]
        assert captured["output_text"] == "The grade is B+."
        assert captured["model"] == "gpt-4o-2024-08-06"
        assert captured["usage"]["input_tokens"] == 12
        assert captured["latency_ms"] > 0

    def test_sync_decorated_async_stream_does_not_crash_await(self, tmp_path):
        """Pre-fix, the sync wrapper wrapped the coroutine in a stream proxy and
        the host's ``await`` raised TypeError. The stream must survive an
        async round trip and the capture must accumulate normally."""
        chunks = [text_chunk("Bon"), text_chunk("jour", finish="stop"), usage_chunk()]
        completions = DecoratedAsyncCompletions(FakeStream(chunks))
        client = FakeClient(completions)
        wrap_openai(client, spool_dir=tmp_path)

        async def consume():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini", messages=MESSAGES, stream=True
            )
            return [chunk for chunk in stream]

        received = asyncio.run(consume())
        assert len(received) == 3
        [record] = read_spool(tmp_path)
        assert record["captured_output"]["output_text"] == "Bonjour"
        assert record["captured_output"]["usage"]["output_tokens"] == 12


class TestToolLoopCapture:
    def test_second_turn_of_tool_loop_spools_uploadable_messages(self, tmp_path):
        """The follow-up call of a tool loop carries the API's own assistant
        message (content=None + tool_calls) and possibly a developer prompt.
        The spooled record must hold the normalized, uploadable shape."""
        response = make_response(content="It is 21C in Paris.")
        client = FakeClient(FakeCompletions(result=response))
        wrap_openai(client, spool_dir=tmp_path)

        tool_loop_messages = [
            {"role": "developer", "content": "Answer with the tool result only."},
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_w1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_w1", "content": '{"temp_c": 21}'},
        ]
        client.chat.completions.create(model="gpt-4o", messages=tool_loop_messages)

        [record] = read_spool(tmp_path)
        roles = [m["role"] for m in record["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]
        assert record["messages"][2]["content"] == ""
        assert record["messages"][2]["tool_calls"][0]["id"] == "call_w1"
        # The provider saw the ORIGINAL, unmodified messages.
        sent = client.chat.completions.calls[0]["messages"]
        assert sent[0]["role"] == "developer"
        assert sent[2]["content"] is None
