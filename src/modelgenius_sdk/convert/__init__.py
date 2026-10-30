"""Converters from foreign trace formats to ModelGenius snapshot records.

Currently supported: OTLP/JSON trace exports following the OpenTelemetry
GenAI semantic conventions (``modelgenius_sdk.convert.otel``). Everything
runs locally; this package makes no network calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from modelgenius_sdk.convert.otel import (
    OtlpConversionResult,
    convert_otlp_payload,
    convert_otlp_payloads,
    is_otlp_traces,
)

__all__ = [
    "ConvertFileResult",
    "OtlpConversionResult",
    "convert_otlp_file",
    "convert_otlp_payload",
    "convert_otlp_payloads",
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
    skipped_duplicate: int = 0
    no_content: int = 0
    unsupported_openinference: int = 0

    @property
    def converted(self) -> int:
        return len(self.records)


def _json_lines(text: str) -> Optional[list]:
    """Every non-empty line parsed as JSON, or None if any line isn't."""
    parsed = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    return parsed or None


def convert_otlp_file(path: Union[str, Path]) -> ConvertFileResult:
    """Convert an OTLP/JSON trace export file to snapshot records.

    Accepts a ``.json`` file holding one OTLP payload (an object with
    ``resourceSpans``) or an array of them, or one OTLP payload per line (the
    OpenTelemetry collector file exporter's format, whether the file is named
    ``.jsonl`` or ``.json``). All payloads are converted together, so two
    copies of one call on different lines become one record. Raises
    ``OSError`` on unreadable input and ``ValueError`` on non-JSON or
    non-OTLP content (including binary protobuf exports, which need the
    collector's ``format: json``); per-span conversion problems are collected
    in ``errors``, never raised.
    """
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"{path}: not UTF-8 text. Binary (protobuf) OTLP exports are not "
            "supported here; export JSON instead (collector file exporter "
            "format: json)."
        ) from None

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
            lines = _json_lines(text)
            if lines is None or len(lines) < 2:
                raise ValueError(f"{path}: invalid JSON: {e}") from None
            obj = lines  # one payload per line, as the collector writes
        candidates = obj if isinstance(obj, list) else [obj]
        for i, candidate in enumerate(candidates):
            if is_otlp_traces(candidate):
                payloads.append(candidate)
            else:
                raise ValueError(
                    f"{path}[{i}]: not an OTLP trace payload (no resourceSpans)."
                )

    converted: OtlpConversionResult = convert_otlp_payloads(payloads)
    return ConvertFileResult(
        records=converted.records,
        errors=converted.errors,
        payloads=len(payloads),
        skipped_non_genai=converted.skipped_non_genai,
        skipped_non_inference=converted.skipped_non_inference,
        skipped_duplicate=converted.skipped_duplicate,
        no_content=converted.no_content,
        unsupported_openinference=converted.unsupported_openinference,
    )
