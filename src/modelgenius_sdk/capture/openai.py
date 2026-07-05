"""OpenAI client wrapper — capture chat-completions calls to the local spool.

:func:`wrap_openai` patches ``client.chat.completions.create`` **on the client
instance** (never on a class) so every call is transparently captured as a
ModelGenius snapshot record while the call itself is left completely
untouched: the original method receives the exact ``args``/``kwargs`` and its
return value (or exception) propagates unchanged.

Scope and limitations (v0):

- Only the chat-completions surface is wrapped. ``client.responses`` (the
  OpenAI Responses API), embeddings, and every other surface are NOT wrapped.
- Failed calls capture nothing: a provider exception re-raises immediately
  and no record is written.
- ``stream=True`` calls are captured by wrapping the returned stream in a
  transparent proxy that accumulates chunks and emits a single record when
  iteration completes or the stream's context manager exits / the stream is
  closed. A stream abandoned mid-iteration is emitted with
  ``provider_metadata.truncated_stream = true`` on context-manager exit.
- ``cost_usd`` is always ``null`` (the OpenAI response carries no cost).

The module is fully duck-typed — it never imports the ``openai`` package —
and fail-open: capture-internal errors are swallowed (logged once per error
class) unless ``MODELGENIUS_CAPTURE_DEBUG`` is set, in which case they
re-raise (for tests/debugging). No network calls are ever made.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import os
import time
from typing import Any, Optional

from modelgenius_sdk.capture.core import (
    ENV_DEBUG,
    CaptureConfig,
    build_envelope,
    emit,
    should_capture,
    to_jsonable,
    utcnow_iso,
    warn_once,
)

_API_SURFACE = "chat"
_WRAPPED_MARKER = "_modelgenius_sdk_wrapped"
_DEBUG_VALUES = frozenset({"1", "true", "yes"})


def _debug_enabled() -> bool:
    """True when MODELGENIUS_CAPTURE_DEBUG asks internal errors to re-raise."""
    return os.environ.get(ENV_DEBUG, "").strip().lower() in _DEBUG_VALUES


def _capture_error(key: str, exc: Exception) -> None:
    """Fail-open handler for capture-internal errors (re-raise under debug)."""
    if _debug_enabled():
        raise exc
    warn_once(
        key + ":" + type(exc).__name__,
        "modelgenius-sdk capture (openai): capture skipped "
        f"({type(exc).__name__}: {exc}); further errors of this type suppressed",
    )


def _pick(obj: Any, name: str) -> Any:
    """Duck-typed field access: attribute first, dict key as fallback."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_int(value: Any) -> Optional[int]:
    """Coerce a token count to int, or None when it isn't numeric."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _canonical_usage(usage: Any) -> Optional[dict]:
    """Map an OpenAI usage object/dict to the canonical TokenUsage shape.

    ``thinking_tokens`` is populated from
    ``usage.completion_tokens_details.reasoning_tokens`` when present.
    """
    if usage is None:
        return None
    details = _pick(usage, "completion_tokens_details")
    thinking = _as_int(_pick(details, "reasoning_tokens")) if details is not None else None
    return {
        "input_tokens": _as_int(_pick(usage, "prompt_tokens")),
        "output_tokens": _as_int(_pick(usage, "completion_tokens")),
        "thinking_tokens": thinking,
        "total_tokens": _as_int(_pick(usage, "total_tokens")),
    }


def _captured_output_from_response(
    response: Any,
    *,
    request_ts: str,
    response_ts: str,
    latency_ms: float,
) -> dict:
    """Build the canonical ``captured_output`` dict from a non-stream response."""
    choices = _pick(response, "choices") or []
    choice = choices[0] if choices else None
    message = _pick(choice, "message")

    content = _pick(message, "content")
    output_text = content or ""
    if not isinstance(output_text, str):
        output_text = str(output_text)

    tool_calls_raw = _pick(message, "tool_calls")
    tool_calls = to_jsonable(tool_calls_raw) if tool_calls_raw else None

    raw_response = to_jsonable(response)
    if not isinstance(raw_response, dict):
        # Plain-attrs objects without model_dump()/to_dict() stringify; the
        # canonical shape requires a dict, so fall back to empty.
        raw_response = {}

    model = _pick(response, "model")
    return {
        "output_text": output_text,
        "tool_calls": tool_calls,
        "raw_response": raw_response,
        "model": model if isinstance(model, str) else (str(model) if model is not None else None),
        "usage": _canonical_usage(_pick(response, "usage")),
        "cost_usd": None,
        "latency_ms": latency_ms,
        "api_surface": _API_SURFACE,
        "provider_metadata": {
            "finish_reason": _pick(choice, "finish_reason"),
            "request_id": _pick(response, "id"),
        },
        "request_timestamp": request_ts,
        "response_timestamp": response_ts,
    }


def _capture_non_stream(
    config: CaptureConfig,
    request_kwargs: dict,
    response: Any,
    request_ts: str,
    start_perf: float,
) -> None:
    """Assemble and emit one record for a completed non-stream call. Fail-open."""
    try:
        latency_ms = (time.perf_counter() - start_perf) * 1000.0
        captured = _captured_output_from_response(
            response,
            request_ts=request_ts,
            response_ts=utcnow_iso(),
            latency_ms=latency_ms,
        )
        record = build_envelope(
            config,
            request_kwargs=request_kwargs,
            api_surface=_API_SURFACE,
            request_ts=request_ts,
            captured_output=captured,
        )
        emit(config, record)
    except Exception as exc:
        _capture_error("openai_capture", exc)


class _ChatStreamAccumulator:
    """Merges chat-completion stream chunks into one canonical output.

    - ``delta.content`` strings are concatenated.
    - ``delta.tool_calls`` are merged by their ``index``: ``id`` / ``type`` /
      ``function.name`` are set once, ``function.arguments`` fragments are
      concatenated.
    - ``model`` / ``id`` come from any chunk bearing them; ``finish_reason``
      is the last non-null one; ``usage`` comes from a usage-bearing chunk
      (present when the caller sets ``stream_options={"include_usage": True}``
      — that final chunk may have empty ``choices``).

    Only the first choice (``index`` 0 or missing) is accumulated in v0.
    """

    def __init__(self) -> None:
        self.text_parts: list = []
        self.tool_calls_by_index: dict = {}
        self.model: Optional[str] = None
        self.finish_reason: Optional[str] = None
        self.usage: Any = None
        self.request_id: Optional[str] = None
        self.chunk_count = 0

    def observe(self, chunk: Any) -> None:
        self.chunk_count += 1
        model = _pick(chunk, "model")
        if model:
            self.model = model
        if self.request_id is None:
            request_id = _pick(chunk, "id")
            if request_id:
                self.request_id = request_id
        usage = _pick(chunk, "usage")
        if usage is not None:
            self.usage = usage
        for choice in _pick(chunk, "choices") or []:
            index = _pick(choice, "index")
            if index not in (None, 0):
                continue
            finish_reason = _pick(choice, "finish_reason")
            if finish_reason:
                self.finish_reason = finish_reason
            delta = _pick(choice, "delta")
            if delta is None:
                continue
            content = _pick(delta, "content")
            if isinstance(content, str):
                self.text_parts.append(content)
            for tool_call in _pick(delta, "tool_calls") or []:
                self._merge_tool_call(tool_call)

    def _merge_tool_call(self, tool_call: Any) -> None:
        index = _pick(tool_call, "index")
        if index is None:
            index = 0
        entry = self.tool_calls_by_index.get(index)
        if entry is None:
            entry = {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
            self.tool_calls_by_index[index] = entry
        call_id = _pick(tool_call, "id")
        if call_id and entry["id"] is None:
            entry["id"] = call_id
        call_type = _pick(tool_call, "type")
        if call_type:
            entry["type"] = call_type
        function = _pick(tool_call, "function")
        if function is not None:
            name = _pick(function, "name")
            if name and entry["function"]["name"] is None:
                entry["function"]["name"] = name
            arguments = _pick(function, "arguments")
            if isinstance(arguments, str):
                entry["function"]["arguments"] += arguments

    def merged_tool_calls(self) -> Optional[list]:
        if not self.tool_calls_by_index:
            return None
        return [self.tool_calls_by_index[i] for i in sorted(self.tool_calls_by_index)]

    def to_captured_output(
        self,
        *,
        request_ts: str,
        response_ts: str,
        latency_ms: float,
        truncated: bool,
    ) -> dict:
        output_text = "".join(self.text_parts)
        tool_calls = self.merged_tool_calls()
        provider_metadata: dict = {
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
        }
        if truncated:
            provider_metadata["truncated_stream"] = True
        reconstructed = {
            "id": self.request_id,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": to_jsonable(self.usage),
        }
        return {
            "output_text": output_text,
            "tool_calls": tool_calls,
            "raw_response": {
                "stream": True,
                "chunk_count": self.chunk_count,
                "reconstructed": reconstructed,
            },
            "model": self.model,
            "usage": _canonical_usage(self.usage),
            "cost_usd": None,
            "latency_ms": latency_ms,
            "api_surface": _API_SURFACE,
            "provider_metadata": provider_metadata,
            "request_timestamp": request_ts,
            "response_timestamp": response_ts,
        }


class _CaptureStreamProxy:
    """Transparent proxy around an OpenAI (sync or async) stream object.

    Delegates iteration, the (a)context-manager protocol, ``close`` /
    ``aclose``, and every other attribute to the wrapped stream while
    accumulating chunks. The snapshot record is emitted exactly once: when
    iteration completes naturally (``StopIteration`` /
    ``StopAsyncIteration``), or when the context manager exits / the stream
    is closed — whichever happens first. If iteration was abandoned mid-way,
    the record carries ``provider_metadata.truncated_stream = true``.

    Accumulation failures never break the host's iteration: capture stops
    (nothing is emitted) and chunks keep flowing.
    """

    def __init__(
        self,
        inner: Any,
        config: CaptureConfig,
        request_kwargs: dict,
        request_ts: str,
        start_perf: float,
    ) -> None:
        self._mg_inner = inner
        self._mg_config = config
        self._mg_request_kwargs = request_kwargs
        self._mg_request_ts = request_ts
        self._mg_start_perf = start_perf
        self._mg_acc = _ChatStreamAccumulator()
        self._mg_completed = False
        self._mg_emitted = False
        self._mg_broken = False
        self._mg_sync_iter: Any = None
        self._mg_async_iter: Any = None

    # -- capture internals -------------------------------------------------

    def _mg_observe(self, chunk: Any) -> None:
        if self._mg_broken:
            return
        try:
            self._mg_acc.observe(chunk)
        except Exception as exc:
            self._mg_broken = True
            _capture_error("openai_stream_accumulate", exc)

    def _mg_finish(self) -> None:
        if self._mg_emitted:
            return
        self._mg_emitted = True
        if self._mg_broken:
            return
        try:
            latency_ms = (time.perf_counter() - self._mg_start_perf) * 1000.0
            captured = self._mg_acc.to_captured_output(
                request_ts=self._mg_request_ts,
                response_ts=utcnow_iso(),
                latency_ms=latency_ms,
                truncated=not self._mg_completed,
            )
            record = build_envelope(
                self._mg_config,
                request_kwargs=self._mg_request_kwargs,
                api_surface=_API_SURFACE,
                request_ts=self._mg_request_ts,
                captured_output=captured,
            )
            emit(self._mg_config, record)
        except Exception as exc:
            _capture_error("openai_stream_emit", exc)

    # -- sync iteration ----------------------------------------------------

    def __iter__(self) -> _CaptureStreamProxy:
        return self

    def __next__(self) -> Any:
        if self._mg_sync_iter is None:
            self._mg_sync_iter = iter(self._mg_inner)
        try:
            chunk = next(self._mg_sync_iter)
        except StopIteration:
            self._mg_completed = True
            self._mg_finish()
            raise
        self._mg_observe(chunk)
        return chunk

    # -- async iteration ---------------------------------------------------

    def __aiter__(self) -> _CaptureStreamProxy:
        return self

    async def __anext__(self) -> Any:
        if self._mg_async_iter is None:
            self._mg_async_iter = self._mg_inner.__aiter__()
        try:
            chunk = await self._mg_async_iter.__anext__()
        except StopAsyncIteration:
            self._mg_completed = True
            self._mg_finish()
            raise
        self._mg_observe(chunk)
        return chunk

    # -- context managers --------------------------------------------------

    def __enter__(self) -> _CaptureStreamProxy:
        enter = getattr(self._mg_inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            exit_ = getattr(self._mg_inner, "__exit__", None)
            return exit_(exc_type, exc, tb) if exit_ is not None else False
        finally:
            self._mg_finish()

    async def __aenter__(self) -> _CaptureStreamProxy:
        aenter = getattr(self._mg_inner, "__aenter__", None)
        if aenter is not None:
            await aenter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            aexit = getattr(self._mg_inner, "__aexit__", None)
            if aexit is not None:
                return await aexit(exc_type, exc, tb)
            return False
        finally:
            self._mg_finish()

    # -- explicit close ----------------------------------------------------

    def close(self) -> Any:
        try:
            return self._mg_inner.close()
        finally:
            self._mg_finish()

    async def aclose(self) -> Any:
        try:
            return await self._mg_inner.aclose()
        finally:
            self._mg_finish()

    # -- everything else ---------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_mg_"):
            raise AttributeError(name)
        return getattr(self._mg_inner, name)


def _wrap_stream(
    response: Any,
    config: CaptureConfig,
    request_kwargs: dict,
    request_ts: str,
    start_perf: float,
) -> Any:
    """Wrap a stream in the capture proxy; on failure return it unwrapped."""
    try:
        return _CaptureStreamProxy(response, config, request_kwargs, request_ts, start_perf)
    except Exception as exc:
        _capture_error("openai_stream_wrap", exc)
        return response


def _is_async_callable(original: Any) -> bool:
    """True when ``original`` is (or wraps) a coroutine function.

    The real openai/anthropic v1 SDKs decorate ``create`` with a *synchronous*
    ``functools.wraps`` wrapper (e.g. openai's ``@required_args``), so a bare
    ``inspect.iscoroutinefunction`` returns False for ``AsyncOpenAI`` /
    ``AsyncAnthropic`` clients. Follow ``__wrapped__`` via ``inspect.unwrap``
    to see the underlying ``async def``.
    """
    if inspect.iscoroutinefunction(original):
        return True
    try:
        return inspect.iscoroutinefunction(inspect.unwrap(original))
    except Exception:
        return False


async def _capture_awaited(
    awaitable: Any,
    config: CaptureConfig,
    kwargs: dict,
    request_ts: str,
    start_perf: float,
) -> Any:
    """Await a response that reached the sync wrapper, then capture it.

    Runtime backstop for async clients that slip past
    :func:`_is_async_callable` (e.g. a future SDK decorator change): instead
    of duck-typing a coroutine into a corrupt record — or crashing the host's
    ``await`` on the stream path — chain onto the coroutine so the host's
    ``await`` receives the real response and capture stays correct.
    """
    response = await awaitable
    if kwargs.get("stream"):
        return _wrap_stream(response, config, kwargs, request_ts, start_perf)
    _capture_non_stream(config, kwargs, response, request_ts, start_perf)
    return response


def _make_sync_wrapper(original: Any, config: CaptureConfig) -> Any:
    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        capture = should_capture(config, kwargs)
        if not capture:
            return original(*args, **kwargs)
        request_ts = utcnow_iso()
        start_perf = time.perf_counter()
        response = original(*args, **kwargs)
        if inspect.isawaitable(response):
            # Async client mis-detected as sync: never capture the coroutine
            # object itself — chain, so the host's `await` still works.
            return _capture_awaited(response, config, kwargs, request_ts, start_perf)
        if kwargs.get("stream"):
            return _wrap_stream(response, config, kwargs, request_ts, start_perf)
        _capture_non_stream(config, kwargs, response, request_ts, start_perf)
        return response

    return create


def _make_async_wrapper(original: Any, config: CaptureConfig) -> Any:
    @functools.wraps(original)
    async def create(*args: Any, **kwargs: Any) -> Any:
        capture = should_capture(config, kwargs)
        if not capture:
            return await original(*args, **kwargs)
        request_ts = utcnow_iso()
        start_perf = time.perf_counter()
        response = await original(*args, **kwargs)
        if kwargs.get("stream"):
            return _wrap_stream(response, config, kwargs, request_ts, start_perf)
        _capture_non_stream(config, kwargs, response, request_ts, start_perf)
        return response

    return create


def wrap_openai(client: Any, config: Optional[CaptureConfig] = None, **config_overrides: Any) -> Any:
    """Patch ``client.chat.completions.create`` to capture snapshot records.

    Works with both sync (``OpenAI``) and async (``AsyncOpenAI``) v1 clients —
    async originals are detected via ``inspect.iscoroutinefunction`` after
    ``inspect.unwrap`` (the real SDK decorates ``create`` with a synchronous
    ``functools.wraps`` wrapper, so the bare check would miss async clients).
    The patch is applied to the **instance**, never the class, and wrapping an
    already-wrapped client is a no-op. Returns the same ``client``.

    ``config_overrides`` are :class:`CaptureConfig` field kwargs (e.g.
    ``workflow_name=...``, ``spool_dir=...``) applied on top of ``config``
    via ``dataclasses.replace`` semantics.

    The wrapper never changes call behavior: the original method receives the
    exact args/kwargs and its return value or exception propagates unchanged.
    Failed calls capture nothing in v0. Only chat completions are wrapped;
    ``client.responses`` is NOT wrapped in v0.
    """
    if config is None:
        resolved_config = CaptureConfig(**config_overrides)
    elif config_overrides:
        resolved_config = dataclasses.replace(config, **config_overrides)
    else:
        resolved_config = config

    try:
        completions = client.chat.completions
        original = completions.create
    except AttributeError as exc:
        if _debug_enabled():
            raise
        warn_once(
            "wrap_openai_shape",
            "modelgenius-sdk capture (openai): client has no chat.completions.create; "
            f"not wrapping ({exc})",
        )
        return client

    if getattr(original, _WRAPPED_MARKER, False):
        return client

    if _is_async_callable(original):
        wrapper = _make_async_wrapper(original, resolved_config)
    else:
        wrapper = _make_sync_wrapper(original, resolved_config)
    setattr(wrapper, _WRAPPED_MARKER, True)
    completions.create = wrapper
    return client
