"""Local snapshot validation with the ModelGenius upload path's accept/reject semantics.

Validates snapshot records (JSON / JSONL files or directories of them) against
the packaged ``modelgenius.snapshot/v1`` JSON Schema, mirroring the server-side
loader semantics:

- OpenAI-shaped records are validated against the full canonical schema.
- Anthropic-shaped records (Messages-API native) get structural checks, and
  their ``captured_output`` — which is ALWAYS canonical (OpenAI-flavored) shape —
  is validated against the schema's ``$defs/CapturedOutput`` subschema.
- A timestamp must be resolvable for every record: top-level ``timestamp``,
  else ``metadata.timestamp``, else ``metadata.request_timestamp``, else
  ``captured_output.request_timestamp``.
- The server-side loader parses records with Pydantic v2 in lax mode, which
  coerces epoch numbers to datetimes and numeric strings to numbers. Records
  are validated through a coerced copy that mirrors those conversions, so the
  local verdict matches the upload path (see ``_coerce_record``).

Everything runs locally; this module makes no network calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Union

from jsonschema import Draft202012Validator

__all__ = ["Issue", "ValidationReport", "validate_path", "validate_record"]

FORMAT_TAG = "modelgenius.snapshot/v1"

_SCHEMA_PACKAGE = "modelgenius_sdk.schema"
_SCHEMA_FILENAME = "modelgenius.snapshot.v1.schema.json"

# Mirrors the private loader's error text VERBATIM (modelgenius/ingestion/loader.py).
_TIMESTAMP_REQUIRED_MESSAGE = (
    "timestamp is required. Provide top-level timestamp "
    "or compatibility metadata.timestamp, metadata.request_timestamp, "
    "or captured_output.request_timestamp."
)

# "What you unlock" nudges. Aggregated per file: at most one Issue of each code
# per file, with the record count interpolated.
_WARNING_TEMPLATES = {
    "no_captured_output": (
        "{n} record(s) have no captured_output — ModelGenius will replay your "
        "baseline model to reconstruct outputs (slower, costs replay tokens, "
        "loses real production latency/cost)."
    ),
    "no_grouping_keys": (
        "{n} record(s) have neither workflow_id nor metadata.workflow_name/"
        "call_name — grouping will be inferred from content."
    ),
    "no_usage": (
        "{n} record(s) have captured_output without usage — include provider "
        "token usage to unlock an honest savings baseline from real production "
        "traffic."
    ),
    "no_cost": (
        "{n} record(s) have captured_output without cost_usd — include cost to "
        "unlock real-dollar savings comparisons against your baseline spend."
    ),
    "no_model": (
        "{n} record(s) have no top-level model — include it to unlock automatic "
        "baseline model identification."
    ),
    "empty_messages": (
        "{n} record(s) have empty messages — there is nothing to replay or "
        "evaluate for these records."
    ),
    "needs_response_format_schema": (
        "{n} record(s) were imported without their response_format JSON schema "
        "(OTel traces cannot represent it) — supply the schema at upload to "
        "restore schema/contract checks and faithful structured-output replay."
    ),
    "needs_tool_definitions": (
        "{n} record(s) contain tool calls but no tool definitions — these "
        "cannot be replayed until the definitions are supplied."
    ),
}


@dataclass
class Issue:
    """A single validation finding for one file/record."""

    file: str
    line: Optional[int]  # 1-based JSONL line; None for .json array (use index)
    index: Optional[int]  # record index within file
    severity: str  # "error" | "warning"
    code: str
    message: str


@dataclass
class ValidationReport:
    """Aggregated result of validating a path (file or directory)."""

    files: int = 0
    records: int = 0
    valid_records: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def ok(self, strict: bool = False) -> bool:
        if self.errors:
            return False
        if strict and self.warnings:
            return False
        return True


# ---------- schema loading ----------


@lru_cache(maxsize=1)
def _load_schema() -> dict:
    try:
        from importlib.resources import files

        text = (files(_SCHEMA_PACKAGE) / _SCHEMA_FILENAME).read_text(encoding="utf-8")
    except Exception:
        # Namespace-package fallback (importlib.resources.files() does not
        # support namespace packages before Python 3.12).
        schema_path = Path(__file__).resolve().parent.parent / "schema" / _SCHEMA_FILENAME
        text = schema_path.read_text(encoding="utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def _snapshot_validator() -> Draft202012Validator:
    return Draft202012Validator(_load_schema())


@lru_cache(maxsize=1)
def _captured_output_validator() -> Draft202012Validator:
    schema = _load_schema()
    # Wrapper schema so "#/$defs/..." refs inside CapturedOutput still resolve.
    wrapper = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "#/$defs/CapturedOutput",
        "$defs": schema["$defs"],
    }
    return Draft202012Validator(wrapper)


# ---------- format detection (mirrors modelgenius/ingestion/normalizer.py) ----------


def _detect_format(raw: Any) -> str:
    """Decide which provider format ``raw`` is in, by structural shape only.

    Mirrors the private ``_detect_format`` exactly: Anthropic iff top-level
    ``system`` is a str/list, OR any message has list content containing a block
    with ``type`` in {"tool_use", "tool_result"} or ``type == "image"`` with a
    dict ``source``. Otherwise OpenAI.
    """
    if not isinstance(raw, dict):
        return "openai"
    if "system" in raw and isinstance(raw["system"], (str, list)):
        return "anthropic"
    for msg in raw.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t in ("tool_use", "tool_result"):
                return "anthropic"
            if t == "image" and isinstance(block.get("source"), dict):
                return "anthropic"
    return "openai"


# ---------- lax coercion (mirrors the server-side Pydantic v2 lax parsing) ----------

# The upload path parses records with Pydantic v2 in lax mode, which accepts
# epoch numbers for datetime fields and numeric strings for number/integer
# fields. The JSON Schema, read strictly, would reject those — so before
# schema validation we build a coerced copy of the record applying the same
# conversions at the same locations. Unparseable values are left untouched so
# the schema still rejects them (e.g. ``temperature: "warm"``).

_NUMBER_FIELDS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")
_INTEGER_FIELDS = ("max_tokens", "n")
_CAPTURED_NUMBER_FIELDS = ("cost_usd", "latency_ms")
_USAGE_INTEGER_FIELDS = ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")

# Pydantic interprets bare numbers as epoch *seconds* up to this magnitude and
# as epoch *milliseconds* beyond it.
_EPOCH_MS_WATERSHED = 2e10


def _epoch_to_iso(value: Any) -> Optional[str]:
    """Convert an int/float epoch to an ISO-8601 UTC string, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = value if abs(value) <= _EPOCH_MS_WATERSHED else value / 1000.0
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _coerce_timestamp(value: Any) -> Any:
    iso = _epoch_to_iso(value)
    return iso if iso is not None else value


def _coerce_number(value: Any) -> Any:
    """Numeric string → float (Pydantic lax float parsing); else unchanged."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _coerce_integer(value: Any) -> Any:
    """Integer-valued string → int (Pydantic lax int parsing accepts "512"
    and "512.0", but not "512.5"); else unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return int(value.strip())
    except ValueError:
        pass
    try:
        as_float = float(value)
    except ValueError:
        return value
    if as_float.is_integer():
        return int(as_float)
    return value


def _coerce_record(record: dict) -> dict:
    """Return a copy of ``record`` with the loader's lax coercions applied.

    Only the known datetime locations (top-level ``timestamp``, the
    ``metadata.timestamp`` / ``metadata.request_timestamp`` fallbacks, and
    ``captured_output.request_timestamp`` / ``response_timestamp``) and the
    known numeric/integer fields are touched. The input is never mutated.
    """
    out = dict(record)
    if "timestamp" in out:
        out["timestamp"] = _coerce_timestamp(out["timestamp"])
    for key in _NUMBER_FIELDS:
        if key in out:
            out[key] = _coerce_number(out[key])
    for key in _INTEGER_FIELDS:
        if key in out:
            out[key] = _coerce_integer(out[key])

    metadata = out.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        for key in ("timestamp", "request_timestamp"):
            if key in metadata:
                metadata[key] = _coerce_timestamp(metadata[key])
        out["metadata"] = metadata

    captured = out.get("captured_output")
    if isinstance(captured, dict):
        captured = dict(captured)
        for key in ("request_timestamp", "response_timestamp"):
            if key in captured:
                captured[key] = _coerce_timestamp(captured[key])
        for key in _CAPTURED_NUMBER_FIELDS:
            if key in captured:
                captured[key] = _coerce_number(captured[key])
        usage = captured.get("usage")
        if isinstance(usage, dict):
            usage = dict(usage)
            for key in _USAGE_INTEGER_FIELDS:
                if key in usage:
                    usage[key] = _coerce_integer(usage[key])
            captured["usage"] = usage
        out["captured_output"] = captured

    return out


# ---------- per-record validation ----------


def validate_record(
    record: dict,
    *,
    file: str = "<memory>",
    line: Optional[int] = None,
    index: Optional[int] = None,
) -> list[Issue]:
    """Validate one snapshot record; returns all issues (errors + warnings)."""

    def make(severity: str, code: str, message: str) -> Issue:
        return Issue(file=file, line=line, index=index, severity=severity, code=code, message=message)

    if not isinstance(record, dict):
        return [
            make(
                "error",
                "invalid_record",
                f"record must be a JSON object, got {type(record).__name__}",
            )
        ]

    issues: list[Issue] = []

    if "format" in record and record["format"] != FORMAT_TAG:
        issues.append(
            make(
                "error",
                "format_tag",
                f'format must be "{FORMAT_TAG}" when present, got {record["format"]!r}',
            )
        )

    # Mirror the loader's Pydantic-v2 lax coercions (epoch numbers → datetimes,
    # numeric strings → numbers) before applying the strict JSON Schema.
    coerced = _coerce_record(record)

    if _detect_format(record) == "anthropic":
        issues.extend(_validate_anthropic(coerced, make))
    else:
        issues.extend(_validate_openai(coerced, make))

    issues.extend(_check_timestamps(coerced, make))
    issues.extend(_record_warnings(record, make))
    return issues


def _schema_message(path_elems: list[Any], message: str) -> str:
    pointer = "/" + "/".join(str(p) for p in path_elems) if path_elems else "(root)"
    return f"{pointer}: {message}"


def _sorted_errors(validator: Draft202012Validator, instance: Any) -> list:
    return sorted(
        validator.iter_errors(instance),
        key=lambda e: ("/".join(str(p) for p in e.absolute_path), e.message),
    )


def _flatten_schema_error(err: Any, prefix: list[Any]) -> list[tuple[list[Any], str]]:
    """Expand anyOf/oneOf context errors into leaf (path, message) pairs.

    The canonical schema spells optional fields as ``anyOf: [X, null]``, so the
    raw top-level message ("is not valid under any of the given schemas") hides
    the real cause. Descend into ``err.context`` and drop the vacuous
    "is not of type 'null'" branch whenever a more specific branch exists.
    Context errors are linked to their parent, so their ``absolute_path``
    already includes every ancestor segment within the validated instance —
    recursion therefore keeps the ORIGINAL instance ``prefix`` rather than
    re-adding the accumulated path (which would duplicate segments).
    """
    path = prefix + list(err.absolute_path)
    if not err.context:
        return [(path, err.message)]
    informative = [
        sub
        for sub in err.context
        if not (sub.validator == "type" and sub.validator_value == "null")
    ]
    pairs: list[tuple[list[Any], str]] = []
    for sub in informative or list(err.context):
        pairs.extend(_flatten_schema_error(sub, prefix))
    return pairs


def _schema_issues(
    validator: Draft202012Validator,
    instance: Any,
    prefix: list[Any],
    make: Callable[[str, str, str], Issue],
    skip_paths: tuple[tuple[str, ...], ...] = (),
) -> list[Issue]:
    out: list[Issue] = []
    for err in _sorted_errors(validator, instance):
        if tuple(str(p) for p in err.absolute_path) in skip_paths:
            continue
        for path_elems, message in _flatten_schema_error(err, list(prefix)):
            out.append(make("error", "schema", _schema_message(path_elems, message)))
    return out


def _validate_openai(record: dict, make: Callable[[str, str, str], Issue]) -> list[Issue]:
    # /format violations are already reported with the dedicated format_tag code.
    return _schema_issues(_snapshot_validator(), record, [], make, skip_paths=(("format",),))


def _validate_anthropic(record: dict, make: Callable[[str, str, str], Issue]) -> list[Issue]:
    out: list[Issue] = []

    if "system" in record and not isinstance(record["system"], (str, list)):
        out.append(
            make("error", "anthropic_system", "/system: must be a string or a list of content blocks")
        )

    # The server normalizer forwards non-dict metadata into the canonical
    # Snapshot, which rejects it per-record (siblings in the file still load).
    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        out.append(make("error", "anthropic_metadata", "/metadata: must be an object"))

    messages = record.get("messages")
    if "messages" in record and messages is not None:
        if not isinstance(messages, list):
            out.append(make("error", "anthropic_messages", "/messages: must be a list"))
        else:
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    out.append(make("error", "anthropic_messages", f"/messages/{i}: must be an object"))
                    continue
                role = msg.get("role")
                if role not in ("user", "assistant"):
                    out.append(
                        make(
                            "error",
                            "anthropic_role",
                            f"/messages/{i}/role: must be 'user' or 'assistant', got {role!r}",
                        )
                    )
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for j, block in enumerate(content):
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and "text" in block
                        and not isinstance(block["text"], str)
                    ):
                        out.append(
                            make(
                                "error",
                                "anthropic_text_block",
                                f"/messages/{i}/content/{j}/text: must be a string",
                            )
                        )

    # A non-list tools value is silently dropped by the server normalizer, so
    # only list entries are checked here.
    tools = record.get("tools")
    if isinstance(tools, list):
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                out.append(make("error", "anthropic_tools", f"/tools/{i}: must be an object"))

    out.extend(_check_anthropic_scalar_params(record, make))

    # captured_output is ALWAYS canonical shape, even on Anthropic-shaped
    # records — validate it against the CapturedOutput subschema.
    captured = record.get("captured_output")
    if captured is not None:
        out.extend(_schema_issues(_captured_output_validator(), captured, ["captured_output"], make))
    return out


def _check_anthropic_scalar_params(record: dict, make: Callable[[str, str, str], Issue]) -> list[Issue]:
    """Check the scalar request params that flow into the canonical Snapshot.

    Only ``temperature``, ``top_p``, ``max_tokens``, and ``stop_sequences``
    are forwarded by the server normalizer on the Anthropic-shaped path;
    OpenAI-only params (``frequency_penalty``, ``presence_penalty``, ``n``)
    are ignored by the loader and therefore not checked here. The record has
    already been lax-coerced, so numeric strings arrive as numbers and any
    remaining string is unparseable.
    """
    out: list[Issue] = []

    for key in ("temperature", "top_p"):
        value = record.get(key)
        if value is not None and not isinstance(value, (int, float)):
            out.append(
                make("error", "anthropic_param", f"/{key}: must be a number, got {value!r}")
            )

    max_tokens = record.get("max_tokens")
    if max_tokens is not None and not (
        isinstance(max_tokens, int)
        or (isinstance(max_tokens, float) and max_tokens.is_integer())
    ):
        out.append(
            make("error", "anthropic_param", f"/max_tokens: must be an integer, got {max_tokens!r}")
        )

    stop_sequences = record.get("stop_sequences")
    if stop_sequences is not None and not (
        isinstance(stop_sequences, str)
        or (isinstance(stop_sequences, list) and all(isinstance(s, str) for s in stop_sequences))
    ):
        out.append(
            make(
                "error",
                "anthropic_stop_sequences",
                "/stop_sequences: must be a string or a list of strings",
            )
        )

    return out


def _parseable_iso8601(value: str) -> bool:
    try:
        # Python 3.9 fromisoformat does not accept a trailing "Z".
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _check_timestamps(record: dict, make: Callable[[str, str, str], Issue]) -> list[Issue]:
    out: list[Issue] = []
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    captured = record.get("captured_output") if isinstance(record.get("captured_output"), dict) else {}

    # Resolution order mirrors Snapshot._normalize_timestamp.
    candidates = (
        ("timestamp", record.get("timestamp")),
        ("metadata.timestamp", metadata.get("timestamp")),
        ("metadata.request_timestamp", metadata.get("request_timestamp")),
        ("captured_output.request_timestamp", captured.get("request_timestamp")),
    )
    if all(value is None for _, value in candidates):
        out.append(make("error", "timestamp_missing", _TIMESTAMP_REQUIRED_MESSAGE))

    checked = candidates + (
        ("captured_output.response_timestamp", captured.get("response_timestamp")),
    )
    for label, value in checked:
        if value is None:
            continue
        if not isinstance(value, str) or not _parseable_iso8601(value):
            out.append(
                make(
                    "error",
                    "timestamp_invalid",
                    f"{label}: {value!r} is not a valid ISO-8601 timestamp",
                )
            )
    return out


def _record_warnings(record: dict, make: Callable[[str, str, str], Issue]) -> list[Issue]:
    out: list[Issue] = []

    def warn(code: str) -> None:
        out.append(make("warning", code, _WARNING_TEMPLATES[code].format(n=1)))

    captured = record.get("captured_output")
    if captured is None:
        warn("no_captured_output")
    elif isinstance(captured, dict):
        if captured.get("usage") is None:
            warn("no_usage")
        if captured.get("cost_usd") is None:
            warn("no_cost")

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if not (record.get("workflow_id") or metadata.get("workflow_name") or metadata.get("call_name")):
        warn("no_grouping_keys")

    if record.get("model") is None:
        warn("no_model")

    if record.get("messages") == []:
        warn("empty_messages")

    enrichment = metadata.get("enrichment_needed")
    if isinstance(enrichment, list):
        if "response_format_schema" in enrichment:
            warn("needs_response_format_schema")
        if "tool_definitions" in enrichment:
            warn("needs_tool_definitions")

    return out


# ---------- file / path validation ----------


def validate_path(path: Union[str, Path], *, strict: bool = False) -> ValidationReport:
    """Validate a snapshot file or a directory of snapshot files.

    Supports (mirroring the private loader):
    - a single ``.json`` file containing one record or an array of records
    - a ``.jsonl`` file with one record per line (blank lines skipped)
    - a directory containing top-level ``.json`` and/or ``.jsonl`` files
      (non-recursive; ``.json`` files sorted first, then ``.jsonl``)

    ``strict`` does not change the report content; it is accepted for symmetry
    so callers can thread the flag through — use ``report.ok(strict=...)`` to
    apply it.
    """
    del strict  # content is identical either way; see docstring
    path = Path(path)
    report = ValidationReport()

    if not path.exists():
        report.issues.append(
            Issue(
                file=str(path),
                line=None,
                index=None,
                severity="error",
                code="path_not_found",
                message=f"Path does not exist: {path}",
            )
        )
        return report

    if path.is_dir():
        files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
        if not files:
            report.issues.append(
                Issue(
                    file=str(path),
                    line=None,
                    index=None,
                    severity="error",
                    code="no_files_found",
                    message=f"No .json or .jsonl files found in {path}",
                )
            )
            return report
        for file_path in files:
            _validate_file(file_path, report)
        return report

    if path.suffix in (".json", ".jsonl"):
        _validate_file(path, report)
        return report

    report.issues.append(
        Issue(
            file=str(path),
            line=None,
            index=None,
            severity="error",
            code="unsupported_file_type",
            message=f"Unsupported file type: {path.suffix}",
        )
    )
    return report


def _validate_file(path: Path, report: ValidationReport) -> None:
    entries, issues = _read_records(path)
    report.files += 1
    report.records += len(entries)

    for record, line, index in entries:
        record_issues = validate_record(record, file=str(path), line=line, index=index)
        if not any(i.severity == "error" for i in record_issues):
            report.valid_records += 1
        issues.extend(record_issues)

    report.issues.extend(_aggregate_file_warnings(str(path), issues))


def _read_records(path: Path) -> tuple[list[tuple[Any, Optional[int], int]], list[Issue]]:
    """Parse a .json/.jsonl file into (record, line, index) entries + parse issues.

    An unreadable or non-UTF-8 file yields a single ``io_error`` issue for
    that file instead of raising, so a directory scan continues past it.
    """
    entries: list[tuple[Any, Optional[int], int]] = []
    issues: list[Issue] = []

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        issues.append(
            Issue(
                file=str(path),
                line=None,
                index=None,
                severity="error",
                code="io_error",
                message=f"cannot read file: {e}",
            )
        )
        return entries, issues

    if path.suffix == ".jsonl":
        index = 0
        for line_num, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(
                    Issue(
                        file=str(path),
                        line=line_num,
                        index=None,
                        severity="error",
                        code="invalid_json",
                        message=f"invalid JSON: {e}",
                    )
                )
                continue
            entries.append((record, line_num, index))
            index += 1
        return entries, issues

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        issues.append(
            Issue(
                file=str(path),
                line=None,
                index=None,
                severity="error",
                code="invalid_json",
                message=f"invalid JSON: {e}",
            )
        )
        return entries, issues

    records = raw if isinstance(raw, list) else [raw]
    for i, record in enumerate(records):
        entries.append((record, None, i))
    return entries, issues


def _aggregate_file_warnings(file: str, issues: list[Issue]) -> list[Issue]:
    """Collapse per-record warnings into at most one Issue per code per file."""
    errors = [i for i in issues if i.severity == "error"]
    counts: dict = {}
    extras: list[Issue] = []
    for issue in issues:
        if issue.severity != "warning":
            continue
        if issue.code in _WARNING_TEMPLATES:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        else:
            extras.append(issue)  # unknown warning codes pass through untouched

    aggregated = [
        Issue(
            file=file,
            line=None,
            index=None,
            severity="warning",
            code=code,
            message=_WARNING_TEMPLATES[code].format(n=counts[code]),
        )
        for code in _WARNING_TEMPLATES
        if code in counts
    ]
    return errors + aggregated + extras
