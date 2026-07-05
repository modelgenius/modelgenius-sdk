"""litellm success-callback integration for ModelGenius snapshot capture.

Usage::

    import litellm
    from modelgenius_sdk import make_litellm_callback

    litellm.success_callback.append(
        make_litellm_callback(workflow_name="support-bot", call_name="draft_reply")
    )

    # or, zero-config (environment-variable driven):
    from modelgenius_sdk.capture.litellm_callback import capture_success_callback
    litellm.success_callback.append(capture_success_callback)

The callback receives ``(kwargs, completion_response, start_time, end_time)``
from litellm after every successful completion and spools one snapshot record
(request params + captured output) as a JSONL line. Only chat completions are
captured: litellm invokes success callbacks for *all* call types (embedding,
image_generation, transcription, text_completion, ...), so the callback
early-returns unless ``kwargs["call_type"]`` is a chat completion (or absent)
and a non-empty ``messages`` list is present. Everything is duck-typed —
this module never imports ``litellm`` (or any provider SDK) and makes zero
network calls.

The request part of the record is the intersection of litellm's ``kwargs``
with the canonical OpenAI chat parameter names (``messages``, ``model``,
``temperature``, …). Missing canonical params are additionally looked up in
``kwargs["optional_params"]`` / ``kwargs["litellm_params"]`` when those dicts
are present, because litellm moves most optional params off the top level.
``cost_usd`` comes from ``kwargs["response_cost"]`` with a fallback to
``completion_response._hidden_params["response_cost"]``.

v0 is **sync-only**: litellm accepts synchronous callbacks in
``litellm.success_callback`` for both its sync and async completion paths
(async paths invoke sync callbacks too), so a plain function is sufficient.
An async-native callback may come later.

Fail-open guarantee: the callback body never raises into the host application
— any internal error is swallowed and logged once per error class (set
``MODELGENIUS_CAPTURE_DEBUG=1`` to re-raise, for tests/debugging).
"""

from __future__ import annotations

import dataclasses
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from modelgenius_sdk.capture import core
from modelgenius_sdk.capture.core import CaptureConfig

#: Canonical OpenAI chat-completion param names copied from litellm kwargs
#: into the request part of the snapshot record.
_CANONICAL_PARAMS = (
    "messages",
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "tools",
    "tool_choice",
    "response_format",
    "frequency_penalty",
    "presence_penalty",
    "n",
    "reasoning_effort",
    "seed",
    "user",
)

#: Nested litellm kwarg dicts consulted (in order) for canonical params that
#: are absent at the top level. litellm relocates most optional params into
#: ``optional_params``; ``litellm_params`` occasionally carries a few more.
_NESTED_PARAM_SOURCES = ("optional_params", "litellm_params")

_DEBUG_VALUES = frozenset({"1", "true", "yes"})

_SENTINEL = object()


def _debug_enabled() -> bool:
    """True when MODELGENIUS_CAPTURE_DEBUG asks internal errors to re-raise."""
    return os.environ.get(core.ENV_DEBUG, "").strip().lower() in _DEBUG_VALUES


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Duck-typed field access: dict lookup, attribute, then ``.get()``.

    litellm's ``ModelResponse`` objects support both attribute-style and
    dict-style access; tests (and defensive callers) may hand us plain dicts,
    SimpleNamespace-alikes, or garbage. Never raises.
    """
    if obj is None:
        return default
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        value = getattr(obj, key, _SENTINEL)
        if value is not _SENTINEL:
            return value
        getter = getattr(obj, "get", None)
        if callable(getter):
            return getter(key, default)
    except Exception:
        pass
    return default


def _as_int(value: Any) -> Optional[int]:
    """Best-effort int coercion; None (or garbage) → None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    """Best-effort float coercion; None (or garbage) → None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_aware_utc(value: Any) -> Optional[datetime]:
    """Coerce litellm's start_time/end_time into a tz-aware UTC datetime.

    litellm passes **naive local-time** datetimes (``datetime.now()``), so a
    naive value is interpreted as local time and *converted* to UTC via
    ``astimezone`` (a no-op on UTC hosts; relabeling it as UTC would skew
    every captured timestamp by the host's UTC offset). Numbers are treated
    as unix epoch seconds (litellm has historically passed ``time.time()``
    floats on some paths). Anything else → None.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.astimezone(timezone.utc)
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _build_request_part(kwargs: dict) -> dict:
    """Copy the canonical OpenAI param intersection out of litellm kwargs.

    Top-level keys win; canonical params missing at the top level are filled
    from ``optional_params`` / ``litellm_params`` when present. ``None``
    values are skipped (litellm pads unset params with None).
    """
    request_part: dict = {}
    for name in _CANONICAL_PARAMS:
        value = kwargs.get(name)
        if value is not None:
            request_part[name] = value
    for source_key in _NESTED_PARAM_SOURCES:
        nested = kwargs.get(source_key)
        if not isinstance(nested, dict):
            continue
        for name in _CANONICAL_PARAMS:
            if name in request_part:
                continue
            value = nested.get(name)
            if value is not None:
                request_part[name] = value
    return request_part


def _extract_cost(kwargs: dict, completion_response: Any) -> Optional[float]:
    """cost_usd from kwargs['response_cost'], else the response's _hidden_params."""
    cost = kwargs.get("response_cost")
    if cost is None:
        hidden = _get(completion_response, "_hidden_params")
        if hidden is not None:
            cost = _get(hidden, "response_cost")
    return _as_float(cost)


def _map_usage(usage: Any) -> Optional[dict]:
    """OpenAI-shaped usage object → canonical TokenUsage dict (or None)."""
    if usage is None:
        return None
    thinking_tokens = None
    details = _get(usage, "completion_tokens_details")
    if details is not None:
        thinking_tokens = _as_int(_get(details, "reasoning_tokens"))
    return {
        "input_tokens": _as_int(_get(usage, "prompt_tokens")),
        "output_tokens": _as_int(_get(usage, "completion_tokens")),
        "thinking_tokens": thinking_tokens,
        "total_tokens": _as_int(_get(usage, "total_tokens")),
    }


def _build_captured_output(
    kwargs: dict,
    completion_response: Any,
    request_ts: str,
    response_ts: str,
    latency_ms: Optional[float],
) -> dict:
    """Map an OpenAI-shaped litellm ModelResponse to canonical captured_output."""
    output_text = ""
    tool_calls = None
    finish_reason = None
    choices = _get(completion_response, "choices") or []
    if isinstance(choices, (list, tuple)) and choices:
        first = choices[0]
        message = _get(first, "message")
        content = _get(message, "content")
        if isinstance(content, str):
            output_text = content
        raw_tool_calls = _get(message, "tool_calls")
        if raw_tool_calls:
            tool_calls = core.to_jsonable(raw_tool_calls)
        finish_reason = _get(first, "finish_reason")

    try:
        raw_response = core.to_jsonable(completion_response)
    except Exception:
        raw_response = None  # hostile object (raising attributes) — keep the record
    if not isinstance(raw_response, dict):
        raw_response = {}

    if not isinstance(finish_reason, str):
        finish_reason = core.to_jsonable(finish_reason)
    model = _get(completion_response, "model")
    return {
        "output_text": output_text,
        "tool_calls": tool_calls,
        "raw_response": raw_response,
        "model": model if isinstance(model, str) else None,
        "usage": _map_usage(_get(completion_response, "usage")),
        "cost_usd": _extract_cost(kwargs, completion_response),
        "latency_ms": latency_ms,
        "api_surface": "chat",
        "provider_metadata": {
            "finish_reason": finish_reason,
            "request_id": core.to_jsonable(_get(completion_response, "id")),
        },
        "request_timestamp": request_ts,
        "response_timestamp": response_ts,
    }


def make_litellm_callback(
    config: Optional[CaptureConfig] = None, **config_overrides: Any
) -> Callable[[Any, Any, Any, Any], None]:
    """Build a litellm success callback that spools ModelGenius snapshots.

    Returns ``fn(kwargs, completion_response, start_time, end_time)`` suitable
    for ``litellm.success_callback.append(...)``. ``config_overrides`` are
    :class:`CaptureConfig` field kwargs (e.g. ``workflow_name=...``,
    ``spool_dir=...``) applied on top of ``config`` (or a fresh env-backed
    config when ``config`` is None).

    The callback is sync-only in v0 (litellm invokes sync success callbacks
    on both its sync and async completion paths) and fully fail-open: it
    never raises into the host application unless ``MODELGENIUS_CAPTURE_DEBUG``
    is set.
    """
    if config is None:
        cfg = CaptureConfig(**config_overrides)
    elif config_overrides:
        cfg = dataclasses.replace(config, **config_overrides)
    else:
        cfg = config

    def _callback(kwargs: Any, completion_response: Any, start_time: Any, end_time: Any) -> None:
        try:
            request_kwargs = kwargs if isinstance(kwargs, dict) else {}
            # litellm fires success callbacks for EVERY call type (embedding,
            # image_generation, transcription, text_completion, ...) and even
            # synthesizes chat-shaped `messages` for embeddings — only real
            # chat completions may become snapshots, or the spool fills with
            # bogus "chat" workloads.
            if request_kwargs.get("call_type") not in (None, "completion", "acompletion"):
                return
            messages = request_kwargs.get("messages")
            if not isinstance(messages, list) or not messages:
                return
            if not core.should_capture(cfg, request_kwargs):
                return

            start_dt = _to_aware_utc(start_time)
            end_dt = _to_aware_utc(end_time)
            request_ts = start_dt.isoformat() if start_dt is not None else core.utcnow_iso()
            response_ts = end_dt.isoformat() if end_dt is not None else core.utcnow_iso()
            latency_ms: Optional[float] = None
            if start_dt is not None and end_dt is not None:
                latency_ms = (end_dt - start_dt).total_seconds() * 1000.0

            captured_output = _build_captured_output(
                request_kwargs, completion_response, request_ts, response_ts, latency_ms
            )
            record = core.build_envelope(
                cfg,
                request_kwargs=_build_request_part(request_kwargs),
                api_surface="chat",
                request_ts=request_ts,
                captured_output=captured_output,
            )
            core.emit(cfg, record)
        except Exception as exc:
            if _debug_enabled():
                raise
            core.warn_once(
                "litellm_callback:" + type(exc).__name__,
                "modelgenius-sdk capture: litellm success callback failed; snapshot dropped "
                f"({type(exc).__name__}: {exc}); further errors of this type suppressed",
            )

    return _callback


_default_callback: Optional[Callable[[Any, Any, Any, Any], None]] = None
_default_callback_lock = threading.Lock()


def capture_success_callback(kwargs: Any, completion_response: Any, start_time: Any, end_time: Any) -> None:
    """Zero-config litellm success callback (environment-variable driven).

    Append directly: ``litellm.success_callback.append(capture_success_callback)``.
    The underlying callback is built lazily via :func:`make_litellm_callback`
    on the first invocation, so configuration comes from the
    ``MODELGENIUS_CAPTURE*`` environment variables at call time. Sync-only in
    v0 and fail-open, like every callback from :func:`make_litellm_callback`.
    """
    global _default_callback
    callback = _default_callback
    if callback is None:
        with _default_callback_lock:
            callback = _default_callback
            if callback is None:
                callback = make_litellm_callback()
                _default_callback = callback
    callback(kwargs, completion_response, start_time, end_time)
