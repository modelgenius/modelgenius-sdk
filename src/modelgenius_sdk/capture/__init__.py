"""Capture subpackage — spool production LLM calls to local JSONL snapshot files.

Core plumbing (:class:`CaptureConfig`, :class:`SpoolWriter`, :func:`get_writer`)
is re-exported eagerly; the provider integrations (``wrap_openai``,
``wrap_anthropic``, ``make_litellm_callback``) are re-exported lazily via PEP
562 module ``__getattr__`` so importing this package never pulls in the
wrapper modules (or anything they might need) until first use.
"""

from __future__ import annotations

import importlib
from typing import Any

from modelgenius_sdk.capture.core import CaptureConfig, SpoolWriter, get_writer

__all__ = [
    "CaptureConfig",
    "SpoolWriter",
    "get_writer",
    "make_litellm_callback",
    "wrap_anthropic",
    "wrap_openai",
]

_LAZY_ATTRS = {
    "wrap_openai": ("modelgenius_sdk.capture.openai", "wrap_openai"),
    "wrap_anthropic": ("modelgenius_sdk.capture.anthropic", "wrap_anthropic"),
    "make_litellm_callback": ("modelgenius_sdk.capture.litellm_callback", "make_litellm_callback"),
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
