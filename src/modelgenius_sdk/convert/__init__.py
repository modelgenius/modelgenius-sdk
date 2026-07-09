"""Converters from foreign trace formats to ModelGenius snapshot records.

Currently supported: OTLP/JSON trace exports following the OpenTelemetry
GenAI semantic conventions (``modelgenius_sdk.convert.otel``). Everything
runs locally; this package makes no network calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from modelgenius_sdk.convert.otel import (
    OtlpConversionResult,
    convert_otlp_payload,
    is_otlp_traces,
)

__all__ = [
    "ConvertFileResult",
    "OtlpConversionResult",
    "convert_otlp_file",
    "convert_otlp_payload",
    "is_otlp_traces",
]


@dataclass
class ConvertFileResult:
    """Outcome of converting one OTLP export file to snapshot records."""

    records: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    payloads: int = 0
    skipped_non_genai: int = 0
    skipped_non_inference: int = 0

    @property
    def converted(self) -> int:
        return len(self.records)


def convert_otlp_file(path: Union[str, Path]) -> ConvertFileResult:
    """Convert an OTLP/JSON trace export file to snapshot records.

    Accepts a ``.json`` file holding one OTLP payload (an object with
    ``resourceSpans``) or an array of them, or a ``.jsonl`` file with one
    OTLP payload per line (the OpenTelemetry collector file exporter's
    format). Raises ``OSError`` on unreadable input and ``ValueError`` on
    non-JSON or non-OTLP content; per-span conversion problems are collected
    in ``errors``, never raised.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    payloads: list[dict] = []
    if path.suffix == ".jsonl":
        for line_num, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num}: invalid JSON: {e}") from None
            if is_otlp_traces(obj):
                payloads.append(obj)
            else:
                raise ValueError(
                    f"{path}:{line_num}: not an OTLP trace payload (no resourceSpans)."
                )
    else:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON: {e}") from None
        candidates = obj if isinstance(obj, list) else [obj]
        for i, candidate in enumerate(candidates):
            if is_otlp_traces(candidate):
                payloads.append(candidate)
            else:
                raise ValueError(
                    f"{path}[{i}]: not an OTLP trace payload (no resourceSpans)."
                )

    result = ConvertFileResult(payloads=len(payloads))
    for payload in payloads:
        converted: OtlpConversionResult = convert_otlp_payload(payload)
        result.records.extend(converted.records)
        result.errors.extend(converted.errors)
        result.skipped_non_genai += converted.skipped_non_genai
        result.skipped_non_inference += converted.skipped_non_inference
    return result
