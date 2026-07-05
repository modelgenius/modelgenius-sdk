"""Core capture plumbing shared by all ModelGenius capture integrations.

This module implements the frozen core API used by the provider wrappers
(``wrap_openai`` / ``wrap_anthropic``) and the litellm success callback:

- :class:`CaptureConfig` — capture behaviour knobs with env-var fallbacks.
- :class:`SpoolWriter` / :func:`get_writer` — thread-safe, fail-open JSONL
  spool writer with size-based rotation.
- :func:`should_capture` — per-call sampling / kill-switch / filter decision.
- :func:`build_envelope` / :func:`emit` — snapshot record assembly + writing.
- :func:`to_jsonable` / :func:`sanitize_request_kwargs` — best-effort
  JSON-safe conversion of duck-typed provider objects.

Trust posture: everything here is local-only. This module makes **zero
network calls**, has no telemetry, and never imports provider SDKs. Capture
is fail-open — internal errors are swallowed (logged once per error class via
:func:`warn_once`) so they can never break the host application. Setting the
``MODELGENIUS_CAPTURE_DEBUG=1`` environment variable re-raises internal
errors instead (intended for tests and debugging).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TextIO, Union

from modelgenius_sdk import __version__ as _SDK_VERSION

FORMAT_TAG = "modelgenius.snapshot/v1"

ENV_DIR = "MODELGENIUS_CAPTURE_DIR"
ENV_SAMPLE_RATE = "MODELGENIUS_CAPTURE_SAMPLE_RATE"
ENV_DISABLE = "MODELGENIUS_CAPTURE"
ENV_DEBUG = "MODELGENIUS_CAPTURE_DEBUG"

_DEFAULT_SPOOL_DIR = "./modelgenius_snapshots"
_DISABLE_VALUES = frozenset({"0", "false", "no"})
_DEBUG_VALUES = frozenset({"1", "true", "yes"})

#: Transport/auth kwargs that must never land in a snapshot record.
_DROPPED_REQUEST_KEYS = frozenset({"extra_headers", "extra_query", "timeout", "stream_cls"})

_logger = logging.getLogger("modelgenius_sdk.capture")

_warned_keys: set[str] = set()
_warned_lock = threading.Lock()

_writers: dict[str, SpoolWriter] = {}
_writers_lock = threading.Lock()


def _debug_enabled() -> bool:
    """True when MODELGENIUS_CAPTURE_DEBUG asks internal errors to re-raise."""
    return os.environ.get(ENV_DEBUG, "").strip().lower() in _DEBUG_VALUES


def warn_once(key: str, msg: str) -> None:
    """Log ``msg`` on the ``modelgenius_sdk.capture`` logger once per ``key``.

    Subsequent calls with the same key are no-ops for the process lifetime,
    so a hot loop hitting the same failure cannot flood the host app's logs.
    """
    with _warned_lock:
        if key in _warned_keys:
            return
        _warned_keys.add(key)
    _logger.warning(msg)


@dataclass
class CaptureConfig:
    """Configuration for snapshot capture.

    Every field is optional; unset fields fall back to environment variables
    and then to defaults via the ``resolved_*`` accessors:

    - ``spool_dir``: ``MODELGENIUS_CAPTURE_DIR`` → ``./modelgenius_snapshots``
    - ``sample_rate``: ``MODELGENIUS_CAPTURE_SAMPLE_RATE`` → ``1.0``
    - ``enabled``: ``MODELGENIUS_CAPTURE`` (``"0"``/``"false"``/``"no"``,
      case-insensitive, disables) → ``True``

    ``redact`` receives the full assembled record and may return a modified
    record or ``None`` to drop it entirely. ``request_filter`` receives the
    raw request kwargs before any capture work happens and returns whether
    the call should be captured. ``metadata`` is merged into the record's
    ``metadata`` object alongside ``workflow_name`` / ``call_name``.
    """

    spool_dir: Union[str, Path, None] = None
    workflow_name: Optional[str] = None
    call_name: Optional[str] = None
    workflow_id: Optional[str] = None
    sample_rate: Optional[float] = None
    redact: Optional[Callable[[dict], Optional[dict]]] = None
    request_filter: Optional[Callable[[dict], bool]] = None
    enabled: Optional[bool] = None
    max_file_bytes: int = 64 * 1024 * 1024
    metadata: Optional[dict] = None

    def resolved_spool_dir(self) -> Path:
        """Spool directory: explicit value, else env, else the default."""
        if self.spool_dir is not None:
            return Path(self.spool_dir)
        env_value = os.environ.get(ENV_DIR)
        if env_value:
            return Path(env_value)
        return Path(_DEFAULT_SPOOL_DIR)

    def resolved_sample_rate(self) -> float:
        """Sample rate: explicit value, else env, else 1.0 (capture all)."""
        if self.sample_rate is not None:
            return float(self.sample_rate)
        env_value = os.environ.get(ENV_SAMPLE_RATE)
        if env_value:
            try:
                return float(env_value)
            except ValueError:
                warn_once(
                    "sample_rate_env",
                    f"modelgenius-sdk capture: invalid {ENV_SAMPLE_RATE}={env_value!r}; "
                    "falling back to 1.0",
                )
        return 1.0

    def resolved_enabled(self) -> bool:
        """Enabled flag: explicit value, else env kill switch, else True."""
        if self.enabled is not None:
            return bool(self.enabled)
        env_value = os.environ.get(ENV_DISABLE)
        if env_value is not None and env_value.strip().lower() in _DISABLE_VALUES:
            return False
        return True


class SpoolWriter:
    """Thread-safe, fail-open JSONL spool writer with size-based rotation.

    Files are named ``<spool_dir>/snapshots-<YYYYMMDD>-<pid>-<seq:03d>.jsonl``,
    opened in append mode, one ``json.dumps(..., ensure_ascii=False,
    default=str)`` line per record, flushed after every write. When the
    current file's size reaches/exceeds ``config.max_file_bytes`` the writer
    rotates to ``seq + 1`` before the next record. The spool directory is
    created lazily on the first write. All exceptions are swallowed (logged
    once per error class via :func:`warn_once`) unless
    ``MODELGENIUS_CAPTURE_DEBUG`` is set, in which case they re-raise.
    """

    def __init__(self, config: CaptureConfig):
        self._config = config
        self._lock = threading.Lock()
        self._fh: Optional[TextIO] = None
        self._path: Optional[Path] = None
        self._seq = 0
        self._bytes = 0

    def write(self, record: dict) -> None:
        """Append one record as a JSONL line. Never raises (unless debug)."""
        try:
            data = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            encoded_len = len(data.encode("utf-8"))
            with self._lock:
                fh = self._ensure_open_locked()
                fh.write(data)
                fh.flush()
                self._bytes += encoded_len
        except Exception as exc:
            if _debug_enabled():
                raise
            warn_once(
                "spool_write:" + type(exc).__name__,
                "modelgenius-sdk capture: dropping snapshot record "
                f"({type(exc).__name__}: {exc}); further errors of this type suppressed",
            )

    def close(self) -> None:
        """Close the current spool file (if open). Never raises (unless debug)."""
        try:
            with self._lock:
                self._close_locked()
        except Exception as exc:
            if _debug_enabled():
                raise
            warn_once(
                "spool_close:" + type(exc).__name__,
                f"modelgenius-sdk capture: error closing spool file ({type(exc).__name__}: {exc})",
            )

    def _ensure_open_locked(self) -> TextIO:
        if self._fh is not None and self._bytes >= self._config.max_file_bytes:
            self._close_locked()
            self._seq += 1
        if self._fh is None:
            spool_dir = self._config.resolved_spool_dir()
            spool_dir.mkdir(parents=True, exist_ok=True)
            day = datetime.now(timezone.utc).strftime("%Y%m%d")
            path = spool_dir / f"snapshots-{day}-{os.getpid()}-{self._seq:03d}.jsonl"
            fh = open(path, "a", encoding="utf-8")
            try:
                size = os.fstat(fh.fileno()).st_size
            except OSError:
                size = 0
            self._fh = fh
            self._path = path
            self._bytes = size
        return self._fh

    def _close_locked(self) -> None:
        fh = self._fh
        self._fh = None
        self._path = None
        if fh is not None:
            fh.close()


def get_writer(config: CaptureConfig) -> SpoolWriter:
    """Return the process-wide :class:`SpoolWriter` for the config's spool dir.

    Writers are cached per resolved (absolute) spool directory, so all
    configs pointing at the same directory share one writer (and one
    rotation sequence). The first config to reach a directory decides that
    writer's ``max_file_bytes``.
    """
    key = os.path.abspath(os.path.expanduser(os.fspath(config.resolved_spool_dir())))
    with _writers_lock:
        writer = _writers.get(key)
        if writer is None:
            writer = SpoolWriter(config)
            _writers[key] = writer
        return writer


def should_capture(config: CaptureConfig, request_kwargs: dict) -> bool:
    """Decide whether this call should be captured.

    Returns False when capture is disabled, when ``random.random() >=
    sample_rate``, or when ``config.request_filter`` declines the request.
    Wrappers must make this decision once per call, BEFORE dispatching to the
    provider — streamed responses inherit the decision. Fail-open: internal
    errors (e.g. a raising ``request_filter``) result in ``False`` unless
    ``MODELGENIUS_CAPTURE_DEBUG`` is set.
    """
    try:
        if not config.resolved_enabled():
            return False
        if random.random() >= config.resolved_sample_rate():
            return False
        if config.request_filter is not None and not config.request_filter(request_kwargs):
            return False
        return True
    except Exception as exc:
        if _debug_enabled():
            raise
        warn_once(
            "should_capture:" + type(exc).__name__,
            "modelgenius-sdk capture: should_capture failed; skipping capture "
            f"({type(exc).__name__}: {exc})",
        )
        return False


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string with timezone offset."""
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of an arbitrary (duck-typed) object to JSON-safe data.

    Order: passthrough for None/str/int/float/bool; ``obj.model_dump(mode="json")``
    (pydantic v2 style, falling back to no-arg ``model_dump()``); ``obj.to_dict()``;
    dataclass instances field-by-field; dict / list / tuple / set recursion;
    final fallback ``str(obj)``.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return to_jsonable(model_dump(mode="json"))
        except Exception:
            try:
                return to_jsonable(model_dump())
            except Exception:
                pass
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_jsonable(to_dict())
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in obj]
    return str(obj)


def _normalize_message_entry(message: Any) -> Any:
    """Normalize one (already JSON-safe) chat message for the snapshot record.

    Two shapes that are routine in production OpenAI traffic are rejected by
    the ModelGenius upload path, so they are normalized at capture time:

    - ``content: null`` (or a missing ``content`` key): the standard
      tool-calling loop appends the API's own assistant message — which has
      ``content: null`` plus ``tool_calls`` — back into ``messages`` for the
      follow-up call. Coerced to ``""`` (lossless: the tool_calls carry the
      payload; this matches the canonical shape the server itself produces
      for Anthropic assistant tool_use turns).
    - ``role: "developer"`` (the o-series/gpt-5 convention): mapped to
      ``"system"``, its replay-equivalent canonical role.
    """
    if not isinstance(message, dict):
        return message
    normalized = dict(message)
    if normalized.get("content") is None:
        normalized["content"] = ""
    if normalized.get("role") == "developer":
        normalized["role"] = "system"
    return normalized


def sanitize_request_kwargs(kwargs: dict) -> dict:
    """Return a JSON-safe copy of request kwargs with transport/auth junk removed.

    Drops ``extra_headers`` / ``extra_query`` / ``timeout`` / ``stream_cls``
    and passes every remaining value through :func:`to_jsonable`. A top-level
    ``messages`` list is additionally normalized so every entry is acceptable
    to the ModelGenius upload path (see :func:`_normalize_message_entry`:
    ``content: null`` → ``""``, ``role: "developer"`` → ``"system"``). The
    input dict is never mutated.
    """
    sanitized = {
        key: to_jsonable(value)
        for key, value in kwargs.items()
        if key not in _DROPPED_REQUEST_KEYS
    }
    messages = sanitized.get("messages")
    if isinstance(messages, list):
        sanitized["messages"] = [_normalize_message_entry(m) for m in messages]
    return sanitized


def build_envelope(
    config: CaptureConfig,
    *,
    request_kwargs: dict,
    api_surface: Optional[str],
    request_ts: str,
    captured_output: Optional[dict],
) -> dict:
    """Assemble a full snapshot record from request kwargs + capture context.

    The record is the sanitized request kwargs overlaid with the envelope
    fields — envelope keys always win over colliding request kwarg keys:
    ``format`` (:data:`FORMAT_TAG`), a fresh ``snapshot_id``
    (``uuid4().hex``), ``timestamp`` (= ``request_ts``), ``api_surface``
    (omitted when falsy — Anthropic-shaped records leave it to the server),
    ``workflow_id`` (only when set on the config), a merged ``metadata``
    object (request-kwarg metadata, then ``config.metadata``, then
    ``workflow_name`` / ``call_name``, then ``capture_sdk`` provenance), and
    ``captured_output`` when not None.
    """
    record = sanitize_request_kwargs(request_kwargs)

    metadata: dict = {}
    existing_metadata = record.get("metadata")
    if isinstance(existing_metadata, dict):
        metadata.update(existing_metadata)
    if config.metadata:
        metadata.update(to_jsonable(config.metadata))
    if config.workflow_name is not None:
        metadata["workflow_name"] = config.workflow_name
    if config.call_name is not None:
        metadata["call_name"] = config.call_name
    metadata["capture_sdk"] = {"name": "modelgenius-sdk", "version": _SDK_VERSION}

    record["format"] = FORMAT_TAG
    record["snapshot_id"] = uuid.uuid4().hex
    record["timestamp"] = request_ts
    if api_surface:
        record["api_surface"] = api_surface
    if config.workflow_id is not None:
        record["workflow_id"] = config.workflow_id
    record["metadata"] = metadata
    if captured_output is not None:
        record["captured_output"] = captured_output
    return record


def emit(config: CaptureConfig, record: dict) -> None:
    """Apply ``config.redact`` and write the record to the config's spool.

    ``redact`` returning ``None`` drops the record silently. Fail-open: any
    internal error (including a raising ``redact``) is swallowed and logged
    once per error class, unless ``MODELGENIUS_CAPTURE_DEBUG`` is set.
    """
    try:
        final_record: Optional[dict] = record
        if config.redact is not None:
            final_record = config.redact(record)
            if final_record is None:
                return
        get_writer(config).write(final_record)
    except Exception as exc:
        if _debug_enabled():
            raise
        warn_once(
            "emit:" + type(exc).__name__,
            "modelgenius-sdk capture: dropping snapshot record "
            f"({type(exc).__name__}: {exc}); further errors of this type suppressed",
        )
