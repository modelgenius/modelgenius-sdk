"""modelgenius-sdk — capture and validate ModelGenius LLM call snapshots, locally.

Public entry points (all lazily re-exported so importing ``modelgenius_sdk``
never eagerly imports provider wrappers or ``jsonschema``):

- :class:`CaptureConfig` — capture configuration (``modelgenius_sdk.capture.core``)
- :func:`wrap_openai` — instrument an OpenAI client (``modelgenius_sdk.capture.openai``)
- :func:`wrap_anthropic` — instrument an Anthropic client (``modelgenius_sdk.capture.anthropic``)
- :func:`make_litellm_callback` — litellm success callback factory
  (``modelgenius_sdk.capture.litellm_callback``)
- :func:`validate_path` — validate snapshot files (``modelgenius_sdk.validate.validator``)

Capture is local-only: no network calls, no telemetry. Uploading snapshots to
ModelGenius is a separate, deliberate step done elsewhere.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.3.0"

__all__ = [
    "CaptureConfig",
    "__version__",
    "make_litellm_callback",
    "validate_path",
    "wrap_anthropic",
    "wrap_openai",
]

_LAZY_ATTRS = {
    "CaptureConfig": ("modelgenius_sdk.capture.core", "CaptureConfig"),
    "wrap_openai": ("modelgenius_sdk.capture.openai", "wrap_openai"),
    "wrap_anthropic": ("modelgenius_sdk.capture.anthropic", "wrap_anthropic"),
    "make_litellm_callback": ("modelgenius_sdk.capture.litellm_callback", "make_litellm_callback"),
    "validate_path": ("modelgenius_sdk.validate.validator", "validate_path"),
}


def __getattr__(name: str) -> Any:
    # PEP 562 lazy re-exports: resolve on first attribute access, then cache.
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
