"""Anthropic Messages API capture wrapper.

:func:`wrap_anthropic` patches ``client.messages.create`` **on the client
instance** (sync and async clients both supported; the class is never
touched) so every production call is captured to the local JSONL spool while
the call itself is completely unaffected: the original method is invoked with
the exact args/kwargs and its return value — or exception — propagates
unchanged.

Record shape
------------
The **request** side of the record is written provider-native: ``system``,
``messages`` (content blocks included), ``max_tokens``, ``stop_sequences``,
``top_k``, ``thinking`` … land in the record exactly as the application
passed them, with no conversion, and the top-level ``api_surface`` key is
omitted (the ModelGenius server detects Anthropic-shaped records and sets
it). Detection is structural, so calls that carry no Anthropic marker of
their own (no ``system`` prompt, plain string-content messages) get
``"system": ""`` pinned into the record — the empty string guarantees
Anthropic-side handling server-side without adding a system message. The
**captured_output** is always the canonical, OpenAI-flavored shape:

- ``output_text`` — ``text`` blocks joined with ``"\\n"``.
- ``tool_calls`` — ``tool_use`` blocks converted to OpenAI function-call
  dicts (``arguments`` is ``json.dumps(block.input)``).
- ``thinking`` blocks are dropped.
- ``usage`` — mapped from ``input_tokens`` / ``output_tokens`` (``total``
  is their sum when both are present).
- ``stop_reason`` — recorded in ``provider_metadata``.
- ``api_surface`` — ``"anthropic"``.
- ``request_timestamp`` / ``response_timestamp`` — always present.

Streaming
---------
``messages.create(stream=True)`` is supported for sync and async clients:
the returned stream is wrapped in a transparent proxy that accumulates
``message_start`` / ``content_block_start`` / ``content_block_delta``
(``text_delta`` and ``input_json_delta``) / ``message_delta`` events and
emits a single record when iteration completes or the context manager exits.
A stream abandoned mid-iteration is emitted with
``provider_metadata.truncated_stream = true``. Accumulation failures never
break the host's iteration — capture stops, events keep flowing.

Not captured in v0
------------------
- The ``client.messages.stream()`` helper (the event/context-manager
  convenience API) is **not** captured in v0 — only
  ``messages.create(stream=True)``. Calls made through ``messages.stream()``
  behave exactly as without the wrapper.
- Failed provider calls: if ``create`` raises, the exception is re-raised
  immediately and nothing is captured.

Everything is duck-typed — this module never imports the ``anthropic``
package. All capture work is fail-open: internal errors are logged once per
error class and swallowed, unless ``MODELGENIUS_CAPTURE_DEBUG`` is set, in
which case they re-raise (for tests/debugging).
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import os
import time
from typing import Any, Optional

from modelgenius_sdk.capture import core

__all__ = ["wrap_anthropic"]

#: Attribute set on a patched ``create`` so double-wrapping is a no-op.
_WRAP_MARKER = "_modelgenius_capture_wrapped"

_DEBUG_VALUES = ("1", "true", "yes")


def _debug_enabled() -> bool:
    return os.environ.get(core.ENV_DEBUG, "").strip().lower() in _DEBUG_VALUES


def _swallow(exc: Exception, where: str) -> None:
    """Fail-open handler: warn once per (site, error class), re-raise on debug."""
    if _debug_enabled():
        raise exc
    core.warn_once(
        f"anthropic_{where}:" + type(exc).__name__,
        f"modelgenius-sdk capture (anthropic {where}): {type(exc).__name__}: {exc}; "
        "capture skipped, further errors of this type suppressed",
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Duck-typed field access: dict key or attribute, else default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_async_callable(original: Any) -> bool:
    """True when ``original`` is (or wraps) a coroutine function.

    The real anthropic SDK decorates ``messages.create`` with a *synchronous*
    ``functools.wraps`` wrapper (same pattern as openai's ``@required_args``),
    so a bare ``inspect.iscoroutinefunction`` returns False for
    ``AsyncAnthropic`` clients. Follow ``__wrapped__`` via ``inspect.unwrap``
    to see the underlying ``async def``.
    """
    if inspect.iscoroutinefunction(original):
        return True
    try:
        return inspect.iscoroutinefunction(inspect.unwrap(original))
    except Exception:
        return False


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _usage_dict(input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[dict]:
    if input_tokens is None and output_tokens is None:
        return None
    total = None
    if input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": None,
        "total_tokens": total,
    }


def _tool_call_dict(tool_id: Any, name: Any, arguments: str) -> dict:
    """OpenAI function-call shape for an Anthropic ``tool_use`` block."""
    return {
        "id": tool_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _captured_output_from_response(
    response: Any, *, request_ts: str, response_ts: str, latency_ms: float
) -> dict:
    """Map a non-streaming Anthropic Message to the canonical captured_output."""
    texts = []
    tool_calls = []
    for block in _get(response, "content") or []:
        block_type = _get(block, "type")
        if block_type == "text":
            text = _get(block, "text")
            texts.append(text if isinstance(text, str) else "")
        elif block_type == "tool_use":
            tool_input = _get(block, "input")
            arguments = json.dumps(core.to_jsonable(tool_input if tool_input is not None else {}))
            tool_calls.append(_tool_call_dict(_get(block, "id"), _get(block, "name"), arguments))
        # thinking / redacted_thinking / anything unknown: dropped.

    usage_obj = _get(response, "usage")
    raw = core.to_jsonable(response)
    return {
        "output_text": "\n".join(texts),
        "tool_calls": tool_calls or None,
        "raw_response": raw if isinstance(raw, dict) else {},
        "model": _get(response, "model"),
        "usage": _usage_dict(
            _int_or_none(_get(usage_obj, "input_tokens")),
            _int_or_none(_get(usage_obj, "output_tokens")),
        ),
        "cost_usd": None,
        "latency_ms": latency_ms,
        "api_surface": "anthropic",
        "provider_metadata": {
            "stop_reason": _get(response, "stop_reason"),
            "request_id": _get(response, "id"),
        },
        "request_timestamp": request_ts,
        "response_timestamp": response_ts,
    }


def _capture_non_stream(
    config: core.CaptureConfig, request_kwargs: dict, request_ts: str, t0: float, response: Any
) -> None:
    """Build and emit a record for a completed non-streaming call. Fail-open."""
    try:
        response_ts = core.utcnow_iso()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        captured = _captured_output_from_response(
            response, request_ts=request_ts, response_ts=response_ts, latency_ms=latency_ms
        )
        record = core.build_envelope(
            config,
            request_kwargs=request_kwargs,
            api_surface=None,  # Anthropic-shaped: top-level api_surface omitted.
            request_ts=request_ts,
            captured_output=captured,
        )
        core.emit(config, record)
    except Exception as exc:
        _swallow(exc, "capture")


class _StreamAccumulator:
    """Accumulates Anthropic streaming events into one canonical record.

    Emits exactly once (``finalize`` is idempotent). ``feed`` failures mark
    the accumulator broken: capture stops, iteration is never disturbed, and
    no record is written.
    """

    def __init__(
        self, config: core.CaptureConfig, request_kwargs: dict, request_ts: str, t0: float
    ):
        self.config = config
        self.request_kwargs = request_kwargs
        self.request_ts = request_ts
        self.t0 = t0
        self.model: Optional[str] = None
        self.request_id: Any = None
        self.stop_reason: Any = None
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.chunk_count = 0
        # insertion-ordered: block index -> accumulated block state
        self.blocks: dict = {}
        self.completed = False
        self.emitted = False
        self.broken = False

    def feed(self, event: Any) -> None:
        if self.broken:
            return
        try:
            self.chunk_count += 1
            event_type = _get(event, "type")
            if event_type == "message_start":
                message = _get(event, "message")
                model = _get(message, "model")
                if model is not None:
                    self.model = model
                request_id = _get(message, "id")
                if request_id is not None:
                    self.request_id = request_id
                input_tokens = _int_or_none(_get(_get(message, "usage"), "input_tokens"))
                if input_tokens is not None:
                    self.input_tokens = input_tokens
            elif event_type == "content_block_start":
                index = _get(event, "index")
                block = _get(event, "content_block")
                block_type = _get(block, "type")
                if block_type == "tool_use":
                    self.blocks[index] = {
                        "type": "tool_use",
                        "id": _get(block, "id"),
                        "name": _get(block, "name"),
                        "arguments": "",
                    }
                elif block_type == "text":
                    text = _get(block, "text")
                    self.blocks[index] = {
                        "type": "text",
                        "text": text if isinstance(text, str) else "",
                    }
                else:
                    # thinking / unknown blocks: tracked so deltas have a home, never rendered.
                    self.blocks[index] = {"type": block_type}
            elif event_type == "content_block_delta":
                index = _get(event, "index")
                delta = _get(event, "delta")
                delta_type = _get(delta, "type")
                if delta_type == "text_delta":
                    entry = self.blocks.setdefault(index, {"type": "text", "text": ""})
                    if entry.get("type") == "text":
                        entry["text"] = entry.get("text", "") + (_get(delta, "text") or "")
                elif delta_type == "input_json_delta":
                    entry = self.blocks.setdefault(
                        index, {"type": "tool_use", "id": None, "name": None, "arguments": ""}
                    )
                    if entry.get("type") == "tool_use":
                        entry["arguments"] = entry.get("arguments", "") + (
                            _get(delta, "partial_json") or ""
                        )
                # thinking_delta / signature_delta: dropped.
            elif event_type == "message_delta":
                stop_reason = _get(_get(event, "delta"), "stop_reason")
                if stop_reason is not None:
                    self.stop_reason = stop_reason
                output_tokens = _int_or_none(_get(_get(event, "usage"), "output_tokens"))
                if output_tokens is not None:
                    self.output_tokens = output_tokens
            # content_block_stop / message_stop / ping: nothing to accumulate.
        except Exception as exc:
            self.broken = True
            _swallow(exc, "stream_accumulate")

    def finalize(self, truncated: bool) -> None:
        """Emit the accumulated record exactly once. Fail-open."""
        if self.emitted or self.broken:
            return
        self.emitted = True
        try:
            response_ts = core.utcnow_iso()
            latency_ms = (time.perf_counter() - self.t0) * 1000.0

            texts = []
            tool_calls = []
            content = []
            for entry in self.blocks.values():
                if entry.get("type") == "text":
                    text = entry.get("text", "")
                    texts.append(text)
                    content.append({"type": "text", "text": text})
                elif entry.get("type") == "tool_use":
                    arguments = entry.get("arguments") or "{}"
                    tool_calls.append(
                        _tool_call_dict(entry.get("id"), entry.get("name"), arguments)
                    )
                    try:
                        parsed_input: Any = json.loads(arguments)
                    except ValueError:
                        parsed_input = arguments
                    content.append(
                        {
                            "type": "tool_use",
                            "id": entry.get("id"),
                            "name": entry.get("name"),
                            "input": parsed_input,
                        }
                    )

            provider_metadata: dict = {
                "stop_reason": self.stop_reason,
                "request_id": self.request_id,
            }
            if truncated:
                provider_metadata["truncated_stream"] = True

            captured = {
                "output_text": "\n".join(texts),
                "tool_calls": tool_calls or None,
                "raw_response": {
                    "stream": True,
                    "chunk_count": self.chunk_count,
                    "reconstructed": {
                        "id": self.request_id,
                        "model": self.model,
                        "stop_reason": self.stop_reason,
                        "content": content,
                        "usage": {
                            "input_tokens": self.input_tokens,
                            "output_tokens": self.output_tokens,
                        },
                    },
                },
                "model": self.model,
                "usage": _usage_dict(self.input_tokens, self.output_tokens),
                "cost_usd": None,
                "latency_ms": latency_ms,
                "api_surface": "anthropic",
                "provider_metadata": provider_metadata,
                "request_timestamp": self.request_ts,
                "response_timestamp": response_ts,
            }
            record = core.build_envelope(
                self.config,
                request_kwargs=self.request_kwargs,
                api_surface=None,
                request_ts=self.request_ts,
                captured_output=captured,
            )
            core.emit(self.config, record)
        except Exception as exc:
            _swallow(exc, "stream_emit")


class _CapturedSyncStream:
    """Transparent proxy around a sync Anthropic event stream."""

    def __init__(self, inner: Any, accumulator: _StreamAccumulator):
        self._mg_inner = inner
        self._mg_acc = accumulator
        self._mg_iter: Any = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._mg_iter is None:
            self._mg_iter = iter(self._mg_inner)
        try:
            event = next(self._mg_iter)
        except StopIteration:
            self._mg_acc.completed = True
            self._mg_acc.finalize(truncated=False)
            raise
        self._mg_acc.feed(event)
        return event

    def __enter__(self):
        enter = getattr(self._mg_inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            exit_ = getattr(self._mg_inner, "__exit__", None)
            result = exit_(exc_type, exc, tb) if exit_ is not None else False
        finally:
            self._mg_acc.finalize(truncated=not self._mg_acc.completed)
        return result

    def close(self):
        try:
            close = getattr(self._mg_inner, "close", None)
            if close is not None:
                close()
        finally:
            self._mg_acc.finalize(truncated=not self._mg_acc.completed)

    def __getattr__(self, name):
        return getattr(self._mg_inner, name)


class _CapturedAsyncStream:
    """Transparent proxy around an async Anthropic event stream."""

    def __init__(self, inner: Any, accumulator: _StreamAccumulator):
        self._mg_inner = inner
        self._mg_acc = accumulator
        self._mg_iter: Any = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._mg_iter is None:
            self._mg_iter = self._mg_inner.__aiter__()
        try:
            event = await self._mg_iter.__anext__()
        except StopAsyncIteration:
            self._mg_acc.completed = True
            self._mg_acc.finalize(truncated=False)
            raise
        self._mg_acc.feed(event)
        return event

    async def __aenter__(self):
        aenter = getattr(self._mg_inner, "__aenter__", None)
        if aenter is not None:
            await aenter()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            aexit = getattr(self._mg_inner, "__aexit__", None)
            result = await aexit(exc_type, exc, tb) if aexit is not None else False
        finally:
            self._mg_acc.finalize(truncated=not self._mg_acc.completed)
        return result

    def close(self):
        """Intercept ``close`` — the real anthropic AsyncStream exposes
        ``async def close()`` (there is no ``aclose``), so an early abort via
        ``await stream.close()`` must finalize here or the partial record is
        silently dropped. Returns the inner ``close()`` result (a coroutine
        for real async streams — the caller awaits it)."""
        try:
            close = getattr(self._mg_inner, "close", None)
            return close() if close is not None else None
        finally:
            self._mg_acc.finalize(truncated=not self._mg_acc.completed)

    async def aclose(self):
        try:
            aclose = getattr(self._mg_inner, "aclose", None)
            if aclose is not None:
                await aclose()
        finally:
            self._mg_acc.finalize(truncated=not self._mg_acc.completed)

    def __getattr__(self, name):
        return getattr(self._mg_inner, name)


def _wrap_stream(
    config: core.CaptureConfig,
    request_kwargs: dict,
    request_ts: str,
    t0: float,
    response: Any,
    is_async: bool,
) -> Any:
    """Wrap a stream in the capturing proxy; on failure return the raw stream."""
    try:
        accumulator = _StreamAccumulator(config, request_kwargs, request_ts, t0)
        if is_async:
            return _CapturedAsyncStream(response, accumulator)
        return _CapturedSyncStream(response, accumulator)
    except Exception as exc:
        _swallow(exc, "stream_wrap")
        return response


def _record_request_kwargs(kwargs: dict) -> dict:
    """Pin the Anthropic shape onto the request kwargs written to the record.

    The ModelGenius server detects Anthropic-shaped records structurally, and
    a plain call with no ``system`` prompt and string-content messages carries
    no Anthropic marker at all — the server would misread it as OpenAI-shaped,
    silently dropping ``stop_sequences`` and leaving native ``tools`` /
    ``tool_choice`` unconverted. Emitting ``"system": ""`` pins detection:
    the empty string triggers Anthropic handling server-side without adding a
    system message. The caller's kwargs are never mutated (the copy is only
    what lands in the record).
    """
    if isinstance(kwargs.get("system"), (str, list)):
        return kwargs
    pinned = dict(kwargs)
    pinned["system"] = ""
    return pinned


async def _capture_awaited(
    awaitable: Any,
    config: core.CaptureConfig,
    kwargs: dict,
    record_kwargs: dict,
    request_ts: str,
    t0: float,
) -> Any:
    """Await a response that reached the sync wrapper, then capture it.

    Runtime backstop for async clients that slip past async detection: chain
    onto the coroutine so the host's ``await`` receives the real response
    (never a proxy around a coroutine) and capture observes the real body.
    """
    response = await awaitable
    if kwargs.get("stream"):
        return _wrap_stream(config, record_kwargs, request_ts, t0, response, is_async=True)
    _capture_non_stream(config, record_kwargs, request_ts, t0, response)
    return response


def _make_sync_create(original: Any, config: core.CaptureConfig) -> Any:
    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        # Capture decision is made once, BEFORE dispatch; streams inherit it.
        if not core.should_capture(config, kwargs):
            return original(*args, **kwargs)
        request_ts = core.utcnow_iso()
        t0 = time.perf_counter()
        # Provider exceptions propagate unchanged; v0 captures nothing for failed calls.
        response = original(*args, **kwargs)
        record_kwargs = _record_request_kwargs(kwargs)
        if inspect.isawaitable(response):
            # Async client mis-detected as sync: never duck-type the coroutine.
            return _capture_awaited(response, config, kwargs, record_kwargs, request_ts, t0)
        if kwargs.get("stream"):
            return _wrap_stream(config, record_kwargs, request_ts, t0, response, is_async=False)
        _capture_non_stream(config, record_kwargs, request_ts, t0, response)
        return response

    return create


def _make_async_create(original: Any, config: core.CaptureConfig) -> Any:
    @functools.wraps(original)
    async def create(*args: Any, **kwargs: Any) -> Any:
        if not core.should_capture(config, kwargs):
            return await original(*args, **kwargs)
        request_ts = core.utcnow_iso()
        t0 = time.perf_counter()
        response = await original(*args, **kwargs)
        record_kwargs = _record_request_kwargs(kwargs)
        if kwargs.get("stream"):
            return _wrap_stream(config, record_kwargs, request_ts, t0, response, is_async=True)
        _capture_non_stream(config, record_kwargs, request_ts, t0, response)
        return response

    return create


def wrap_anthropic(
    client: Any, config: Optional[core.CaptureConfig] = None, **config_overrides: Any
) -> Any:
    """Patch ``client.messages.create`` on the instance to capture snapshots.

    Works with sync and async Anthropic clients — async ``create`` is
    detected via ``inspect.iscoroutinefunction`` after ``inspect.unwrap``
    (the real SDK decorates ``create`` with a synchronous ``functools.wraps``
    wrapper, so the bare check would miss async clients). Returns the same
    client object.
    Wrapping an already-wrapped client is a no-op. ``config_overrides`` are
    :class:`~modelgenius_sdk.capture.core.CaptureConfig` field kwargs applied
    on top of ``config`` with ``dataclasses.replace`` semantics (an unknown
    field name raises ``TypeError`` at wrap time — a developer error surfaced
    deliberately; per-call capture work is always fail-open).
    """
    if config is None:
        config = core.CaptureConfig()
    if config_overrides:
        config = dataclasses.replace(config, **config_overrides)

    messages = client.messages
    original = messages.create
    if getattr(original, _WRAP_MARKER, False):
        return client

    if _is_async_callable(original):
        wrapped = _make_async_create(original, config)
    else:
        wrapped = _make_sync_create(original, config)
    setattr(wrapped, _WRAP_MARKER, True)
    messages.create = wrapped
    return client
