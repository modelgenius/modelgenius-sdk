"""Tests for modelgenius_sdk.capture.anthropic — the Anthropic Messages wrapper.

All provider objects are hand-rolled fakes (SimpleNamespace / plain classes);
the ``anthropic`` package is never imported. Debug mode is enabled by default
so internal capture errors surface as test failures; the fail-open test
disables it explicitly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from modelgenius_sdk.capture import core
from modelgenius_sdk.capture.anthropic import wrap_anthropic
from modelgenius_sdk.capture.core import FORMAT_TAG, CaptureConfig


@pytest.fixture(autouse=True)
def _reset_core_state(monkeypatch):
    """Isolate module-global core state; surface capture errors via debug mode."""
    monkeypatch.setattr(core, "_writers", {})
    monkeypatch.setattr(core, "_warned_keys", set())
    for env_var in (core.ENV_DIR, core.ENV_SAMPLE_RATE, core.ENV_DISABLE, core.ENV_DEBUG):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv(core.ENV_DEBUG, "1")


def read_spool_records(spool_dir: Path) -> list:
    records = []
    for path in sorted(spool_dir.glob("snapshots-*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeMessage:
    """Anthropic Message lookalike with a pydantic-style model_dump()."""

    def __init__(self, content, model="claude-sonnet-4-20250514", stop_reason="end_turn",
                 usage=None, message_id="msg_01ABC"):
        self.id = message_id
        self.content = content
        self.model = model
        self.stop_reason = stop_reason
        self.usage = usage

    def model_dump(self, mode=None):
        return {
            "id": self.id,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "content": [vars(block) for block in self.content],
            "usage": vars(self.usage) if self.usage is not None else None,
        }


class FakeMessages:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)


class FakeAsyncMessages:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeAsyncClient:
    def __init__(self, result):
        self.messages = FakeAsyncMessages(result)


class FakeSyncStream:
    def __init__(self, events):
        self._events = list(events)
        self._pos = 0
        self.closed = False
        self.inner_attr = "inner-value"

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= len(self._events):
            raise StopIteration
        event = self._events[self._pos]
        self._pos += 1
        return event

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        self.closed = True


class FakeAsyncStream:
    def __init__(self, events):
        self._events = list(events)
        self._pos = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pos >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._pos]
        self._pos += 1
        return event

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        return False


def request_kwargs(**extra):
    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": "You are a concise travel assistant.",
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
    }
    kwargs.update(extra)
    return kwargs


def text_stream_events():
    return [
        NS(type="message_start",
           message=NS(id="msg_stream1", model="claude-3-5-haiku-20241022",
                      usage=NS(input_tokens=9, output_tokens=1))),
        NS(type="content_block_start", index=0, content_block=NS(type="text", text="")),
        NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="Bonjour ")),
        NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="Paris")),
        NS(type="content_block_stop", index=0),
        NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=NS(output_tokens=4)),
        NS(type="message_stop"),
    ]


def tool_stream_events():
    return [
        NS(type="message_start",
           message=NS(id="msg_stream2", model="claude-sonnet-4-20250514",
                      usage=NS(input_tokens=42, output_tokens=2))),
        NS(type="content_block_start", index=0, content_block=NS(type="text", text="")),
        NS(type="content_block_delta", index=0,
           delta=NS(type="text_delta", text="Checking the weather.")),
        NS(type="content_block_stop", index=0),
        NS(type="content_block_start", index=1,
           content_block=NS(type="tool_use", id="toolu_stream", name="get_weather", input={})),
        NS(type="content_block_delta", index=1,
           delta=NS(type="input_json_delta", partial_json='{"city": "Par')),
        NS(type="content_block_delta", index=1,
           delta=NS(type="input_json_delta", partial_json='is", "unit": "celsius"}')),
        NS(type="content_block_stop", index=1),
        NS(type="message_delta", delta=NS(stop_reason="tool_use"), usage=NS(output_tokens=31)),
        NS(type="message_stop"),
    ]


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


class TestNonStream:
    def test_text_only_response(self, tmp_path):
        response = FakeMessage(
            content=[NS(type="text", text="Hello"), NS(type="text", text="World")],
            usage=NS(input_tokens=12, output_tokens=5),
        )
        client = FakeClient(response)
        wrap_anthropic(client, spool_dir=tmp_path, workflow_name="travel_bot",
                       call_name="answer")

        kwargs = request_kwargs(temperature=0.5)
        result = client.messages.create(**kwargs)

        # Call behavior unchanged: identity return, exact kwargs through.
        assert result is response
        assert client.messages.calls == [kwargs]

        records = read_spool_records(tmp_path)
        assert len(records) == 1
        record = records[0]

        # Request side is provider-native, envelope on top.
        assert record["format"] == FORMAT_TAG
        assert record["snapshot_id"]
        assert record["timestamp"]
        assert record["system"] == kwargs["system"]
        assert record["messages"] == kwargs["messages"]
        assert record["max_tokens"] == 1024
        assert record["temperature"] == 0.5
        assert "api_surface" not in record  # top-level omitted for anthropic shape
        assert record["metadata"]["workflow_name"] == "travel_bot"
        assert record["metadata"]["call_name"] == "answer"
        assert record["metadata"]["capture_sdk"]["name"] == "modelgenius-sdk"

        co = record["captured_output"]
        assert co["output_text"] == "Hello\nWorld"
        assert co["tool_calls"] is None
        assert co["model"] == "claude-sonnet-4-20250514"
        assert co["usage"] == {
            "input_tokens": 12,
            "output_tokens": 5,
            "thinking_tokens": None,
            "total_tokens": 17,
        }
        assert co["cost_usd"] is None
        assert isinstance(co["latency_ms"], float)
        assert co["api_surface"] == "anthropic"
        assert co["provider_metadata"]["stop_reason"] == "end_turn"
        assert co["provider_metadata"]["request_id"] == "msg_01ABC"
        assert co["request_timestamp"]
        assert co["response_timestamp"]
        # raw_response captured via the fake's model_dump.
        assert co["raw_response"]["id"] == "msg_01ABC"
        assert co["raw_response"]["stop_reason"] == "end_turn"

    def test_tool_use_response_openai_shape(self, tmp_path):
        tool_input = {"city": "Paris", "unit": "celsius"}
        response = FakeMessage(
            content=[
                NS(type="text", text="Let me check."),
                NS(type="tool_use", id="toolu_01", name="get_weather", input=tool_input),
            ],
            stop_reason="tool_use",
            usage=NS(input_tokens=30, output_tokens=18),
        )
        client = FakeClient(response)
        wrap_anthropic(client, spool_dir=tmp_path)

        client.messages.create(**request_kwargs())

        [record] = read_spool_records(tmp_path)
        co = record["captured_output"]
        assert co["output_text"] == "Let me check."
        assert isinstance(co["tool_calls"], list) and len(co["tool_calls"]) == 1
        call = co["tool_calls"][0]
        assert call["id"] == "toolu_01"
        assert call["type"] == "function"
        assert call["function"]["name"] == "get_weather"
        # arguments must be a JSON *string* round-tripping to the input dict.
        assert isinstance(call["function"]["arguments"], str)
        assert json.loads(call["function"]["arguments"]) == tool_input
        assert co["provider_metadata"]["stop_reason"] == "tool_use"

    def test_thinking_blocks_dropped(self, tmp_path):
        response = FakeMessage(
            content=[
                NS(type="thinking", thinking="secret chain of thought", signature="sig1"),
                NS(type="text", text="Final answer."),
            ],
            usage=None,
        )
        client = FakeClient(response)
        wrap_anthropic(client, spool_dir=tmp_path)

        client.messages.create(**request_kwargs())

        [record] = read_spool_records(tmp_path)
        co = record["captured_output"]
        assert co["output_text"] == "Final answer."
        assert "secret chain of thought" not in co["output_text"]
        assert co["tool_calls"] is None
        assert co["usage"] is None  # no usage on the response

    def test_exception_propagates_no_record(self, tmp_path):
        client = FakeClient(ValueError("bad request"))
        wrap_anthropic(client, spool_dir=tmp_path)

        with pytest.raises(ValueError, match="bad request"):
            client.messages.create(**request_kwargs())

        assert read_spool_records(tmp_path) == []

    def test_double_wrap_noop(self, tmp_path):
        response = FakeMessage(content=[NS(type="text", text="hi")], usage=None)
        client = FakeClient(response)
        wrap_anthropic(client, spool_dir=tmp_path)
        first_create = client.messages.create

        result = wrap_anthropic(client, spool_dir=tmp_path)

        assert result is client
        assert client.messages.create is first_create  # not re-wrapped

        client.messages.create(**request_kwargs())
        assert len(read_spool_records(tmp_path)) == 1  # captured once, not twice

    def test_capture_failure_is_fail_open(self, tmp_path, monkeypatch):
        monkeypatch.delenv(core.ENV_DEBUG, raising=False)

        class ExplodingResponse:
            model = "claude-sonnet-4-20250514"

            @property
            def content(self):
                raise RuntimeError("boom during capture")

        response = ExplodingResponse()
        client = FakeClient(response)
        wrap_anthropic(client, spool_dir=tmp_path)

        result = client.messages.create(**request_kwargs())  # must not raise

        assert result is response
        assert read_spool_records(tmp_path) == []

    def test_sample_rate_zero_skips_capture_and_stream_proxy(self, tmp_path):
        stream = FakeSyncStream(text_stream_events())
        client = FakeClient(stream)
        wrap_anthropic(client, config=CaptureConfig(spool_dir=tmp_path, sample_rate=0.0))

        result = client.messages.create(stream=True, **request_kwargs())

        # Decision made before dispatch: raw stream returned, nothing captured.
        assert result is stream
        list(result)
        assert read_spool_records(tmp_path) == []

    def test_lazy_reexport_points_here(self):
        import modelgenius_sdk.capture as capture_pkg

        assert capture_pkg.wrap_anthropic is wrap_anthropic


# ---------------------------------------------------------------------------
# Sync streaming
# ---------------------------------------------------------------------------


class TestSyncStreaming:
    def test_stream_text_accumulation_emits_once(self, tmp_path):
        events = text_stream_events()
        client = FakeClient(FakeSyncStream(events))
        wrap_anthropic(client, spool_dir=tmp_path)

        stream = client.messages.create(stream=True, **request_kwargs())
        with stream as s:
            seen = list(s)  # full consumption inside the CM

        assert seen == events  # transparent delegation
        records = read_spool_records(tmp_path)
        assert len(records) == 1  # emitted exactly once (StopIteration, not also __exit__)
        co = records[0]["captured_output"]
        assert co["output_text"] == "Bonjour Paris"
        assert co["model"] == "claude-3-5-haiku-20241022"
        assert co["tool_calls"] is None
        assert co["usage"] == {
            "input_tokens": 9,
            "output_tokens": 4,
            "thinking_tokens": None,
            "total_tokens": 13,
        }
        assert co["provider_metadata"]["stop_reason"] == "end_turn"
        assert co["provider_metadata"]["request_id"] == "msg_stream1"
        assert "truncated_stream" not in co["provider_metadata"]
        assert co["raw_response"]["stream"] is True
        assert co["raw_response"]["chunk_count"] == len(events)
        assert co["raw_response"]["reconstructed"]["content"] == [
            {"type": "text", "text": "Bonjour Paris"}
        ]
        assert co["request_timestamp"] and co["response_timestamp"]

    def test_stream_tool_use_input_json_delta_concatenation(self, tmp_path):
        client = FakeClient(FakeSyncStream(tool_stream_events()))
        wrap_anthropic(client, spool_dir=tmp_path)

        for _ in client.messages.create(stream=True, **request_kwargs()):
            pass  # plain iteration, no context manager

        [record] = read_spool_records(tmp_path)
        co = record["captured_output"]
        assert co["output_text"] == "Checking the weather."
        [call] = co["tool_calls"]
        assert call["id"] == "toolu_stream"
        assert call["type"] == "function"
        assert call["function"]["name"] == "get_weather"
        assert call["function"]["arguments"] == '{"city": "Paris", "unit": "celsius"}'
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "unit": "celsius"}
        assert co["usage"]["input_tokens"] == 42
        assert co["usage"]["output_tokens"] == 31
        assert co["provider_metadata"]["stop_reason"] == "tool_use"

    def test_stream_abandoned_midway_marks_truncated(self, tmp_path):
        inner = FakeSyncStream(text_stream_events())
        client = FakeClient(inner)
        wrap_anthropic(client, spool_dir=tmp_path)

        stream = client.messages.create(stream=True, **request_kwargs())
        with stream as s:
            iterator = iter(s)
            next(iterator)  # message_start
            next(iterator)  # content_block_start
            next(iterator)  # first text_delta
            # abandon here; CM exit must still emit what accumulated

        assert inner.closed  # inner CM exit delegated
        [record] = read_spool_records(tmp_path)
        co = record["captured_output"]
        assert co["provider_metadata"]["truncated_stream"] is True
        assert co["output_text"] == "Bonjour "
        assert co["usage"] == {
            "input_tokens": 9,
            "output_tokens": None,
            "thinking_tokens": None,
            "total_tokens": None,
        }

    def test_stream_close_emits_truncated(self, tmp_path):
        inner = FakeSyncStream(text_stream_events())
        client = FakeClient(inner)
        wrap_anthropic(client, spool_dir=tmp_path)

        stream = client.messages.create(stream=True, **request_kwargs())
        next(iter(stream))  # message_start only
        stream.close()

        assert inner.closed
        [record] = read_spool_records(tmp_path)
        assert record["captured_output"]["provider_metadata"]["truncated_stream"] is True

    def test_stream_proxy_delegates_attributes(self, tmp_path):
        client = FakeClient(FakeSyncStream(text_stream_events()))
        wrap_anthropic(client, spool_dir=tmp_path)

        stream = client.messages.create(stream=True, **request_kwargs())
        assert stream.inner_attr == "inner-value"


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class TestAsync:
    def test_async_text_response(self, tmp_path):
        response = FakeMessage(
            content=[NS(type="text", text="Async hello")],
            usage=NS(input_tokens=7, output_tokens=3),
        )
        client = FakeAsyncClient(response)
        wrap_anthropic(client, spool_dir=tmp_path)

        import inspect as _inspect

        assert _inspect.iscoroutinefunction(client.messages.create)

        async def call():
            return await client.messages.create(**request_kwargs())

        result = asyncio.run(call())

        assert result is response
        [record] = read_spool_records(tmp_path)
        co = record["captured_output"]
        assert co["output_text"] == "Async hello"
        assert co["api_surface"] == "anthropic"
        assert co["usage"]["total_tokens"] == 10
        assert "api_surface" not in record

    def test_async_exception_propagates_no_record(self, tmp_path):
        client = FakeAsyncClient(RuntimeError("provider down"))
        wrap_anthropic(client, spool_dir=tmp_path)

        async def call():
            await client.messages.create(**request_kwargs())

        with pytest.raises(RuntimeError, match="provider down"):
            asyncio.run(call())

        assert read_spool_records(tmp_path) == []

    def test_async_stream_accumulation(self, tmp_path):
        events = tool_stream_events()
        client = FakeAsyncClient(FakeAsyncStream(events))
        wrap_anthropic(client, spool_dir=tmp_path)

        async def consume():
            stream = await client.messages.create(stream=True, **request_kwargs())
            seen = []
            async with stream as s:
                async for event in s:
                    seen.append(event)
            return seen

        seen = asyncio.run(consume())

        assert seen == events
        [record] = read_spool_records(tmp_path)
        co = record["captured_output"]
        assert co["output_text"] == "Checking the weather."
        [call] = co["tool_calls"]
        assert call["function"]["arguments"] == '{"city": "Paris", "unit": "celsius"}'
        assert "truncated_stream" not in co["provider_metadata"]
        assert co["raw_response"]["chunk_count"] == len(events)

    def test_async_stream_abandoned_marks_truncated(self, tmp_path):
        inner = FakeAsyncStream(text_stream_events())
        client = FakeAsyncClient(inner)
        wrap_anthropic(client, spool_dir=tmp_path)

        async def consume_partial():
            stream = await client.messages.create(stream=True, **request_kwargs())
            async with stream as s:
                iterator = s.__aiter__()
                await iterator.__anext__()  # message_start
                await iterator.__anext__()  # content_block_start

        asyncio.run(consume_partial())

        assert inner.closed
        [record] = read_spool_records(tmp_path)
        assert record["captured_output"]["provider_metadata"]["truncated_stream"] is True


# ---------------------------------------------------------------------------
# Regressions: detection pin, real-SDK async detection, async close()
# ---------------------------------------------------------------------------


class TestSystemDetectionPin:
    def test_missing_system_pinned_to_empty_string(self, tmp_path):
        """Without a system prompt, an Anthropic-native record would be
        misdetected as OpenAI-shaped upstream (stop_sequences silently dropped,
        native tools never converted). The wrapper must pin `"system": ""`."""
        response = FakeMessage(
            content=[NS(type="text", text="Sure.")],
            usage=NS(input_tokens=4, output_tokens=2),
        )
        client = FakeClient(response)
        wrap_anthropic(client, spool_dir=tmp_path)

        kwargs = request_kwargs()
        kwargs.pop("system")
        client.messages.create(**kwargs)

        [record] = read_spool_records(tmp_path)
        assert record["system"] == ""
        # The provider call itself was NOT given a system kwarg.
        assert "system" not in client.messages.calls[0]

    def test_explicit_system_never_overwritten(self, tmp_path):
        response = FakeMessage(
            content=[NS(type="text", text="Oui.")],
            usage=NS(input_tokens=4, output_tokens=2),
        )
        client = FakeClient(response)
        wrap_anthropic(client, spool_dir=tmp_path)
        client.messages.create(**request_kwargs())
        [record] = read_spool_records(tmp_path)
        assert record["system"] == "You are a concise travel assistant."


class DecoratedAsyncMessages:
    """Mimics the real AsyncAnthropic: ``create`` is a SYNC functools.wraps
    wrapper around an async def, defeating plain iscoroutinefunction."""

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
        response = FakeMessage(
            content=[NS(type="text", text="Bonjour Paris")],
            usage=NS(input_tokens=9, output_tokens=3),
        )
        client = NS(messages=DecoratedAsyncMessages(response))
        wrap_anthropic(client, spool_dir=tmp_path)

        result = asyncio.run(client.messages.create(**request_kwargs()))
        assert result is response

        [record] = read_spool_records(tmp_path)
        captured = record["captured_output"]
        assert captured["output_text"] == "Bonjour Paris"
        assert captured["usage"]["input_tokens"] == 9
        assert captured["latency_ms"] > 0


class AsyncCloseStream(FakeAsyncStream):
    """Real anthropic AsyncStream exposes ``async def close()`` (no aclose)."""

    async def close(self):
        self.closed = True


class TestAsyncStreamClose:
    def test_await_close_emits_truncated_record(self, tmp_path):
        """Aborting via ``await stream.close()`` (no context manager) must
        still emit the truncated capture — pre-fix the async proxy only
        intercepted ``aclose`` and dropped the record entirely."""
        inner = AsyncCloseStream(text_stream_events())
        client = FakeAsyncClient(inner)
        wrap_anthropic(client, spool_dir=tmp_path)

        async def consume_then_close():
            stream = await client.messages.create(stream=True, **request_kwargs())
            iterator = stream.__aiter__()
            await iterator.__anext__()  # message_start
            await iterator.__anext__()  # content_block_start
            await iterator.__anext__()  # first text delta
            await stream.close()

        asyncio.run(consume_then_close())

        assert inner.closed
        [record] = read_spool_records(tmp_path)
        captured = record["captured_output"]
        assert captured["provider_metadata"]["truncated_stream"] is True
        assert captured["output_text"] == "Bonjour "
