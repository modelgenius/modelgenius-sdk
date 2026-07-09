"""OpenTelemetry GenAI (OTLP/JSON) trace exports → canonical snapshot records.

Converts traces that follow the OpenTelemetry GenAI semantic conventions
(https://github.com/open-telemetry/semantic-conventions-genai — Development
status) into canonical ModelGenius snapshot records (OpenAI chat shape).

This module is intentionally pure stdlib and SOURCE-IDENTICAL between the
private repo (``modelgenius/ingestion/otel.py``) and the public SDK
(``modelgenius_sdk/convert/otel.py``). Do not add package-local imports; keep
the two copies byte-for-byte in sync — the private repo's contract test
(``tests/test_public_sdk_contract.py``) compares both the module sources and
their conversion outputs over a shared fixture.

What maps where (OTel → snapshot record):
    gen_ai.input.messages / gen_ai.system_instructions → messages
    gen_ai.output.messages (first choice)     → captured_output.output_text/.tool_calls
    gen_ai.request.{temperature,top_p,...}    → first-class sampling params
    gen_ai.request.{seed,top_k}               → provider_params (captured, not auto-replayed)
    gen_ai.usage.*                            → captured_output.usage
    span start/end times                      → timestamp, captured_output timestamps, latency_ms
    gen_ai.tool.definitions                   → tools (OpenAI function wrappers)
    gen_ai.prompt.name                        → workflow_id
    trace/span/conversation ids, provider, …  → metadata["otel"]

Message content is read, in order of preference, from: span attributes; the
``gen_ai.client.inference.operation.details`` span event; legacy flattened
OpenLLMetry-style attributes (``gen_ai.prompt.{i}.role/.content``,
``gen_ai.completion.0.*``).

Known gaps in the conventions are detected and surfaced, never fabricated:

* ``response_format`` JSON schemas are not representable in OTel
  (``gen_ai.output.type`` is a modality hint only). When ``output.type`` is
  ``json`` the record gets ``metadata["enrichment_needed"]`` containing
  ``"response_format_schema"`` so downstream UIs can ask for the real schema.
* ``tool_choice`` does not exist in the conventions at all.
* Tool *calls* without tool *definitions* (definitions are opt-in and often
  omitted) yield ``"tool_definitions"`` in ``metadata["enrichment_needed"]``.
* Message content capture is opt-in in OTel instrumentations; a GenAI span
  with no content converts to a per-span error explaining how to enable it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

FORMAT_TAG = "modelgenius.snapshot/v1"

# gen_ai.operation.name values that describe a model inference call we can
# snapshot. Other well-known operations (embeddings, execute_tool,
# invoke_agent, create_agent, retrieve) are structural/agentic spans — counted
# and skipped, never errors.
INFERENCE_OPERATIONS = frozenset({"chat", "text_completion", "generate_content"})

# Roles seen in the wild that must fold onto the canonical four.
_ROLE_ALIASES = {"model": "assistant", "ai": "assistant", "human": "user"}

# Span attributes consumed into first-class record fields; everything else
# with a GenAI-ish prefix lands in metadata["otel"]["attributes"].
_CONSUMED_ATTRS = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.system",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
        "gen_ai.request.model",
        "gen_ai.request.temperature",
        "gen_ai.request.top_p",
        "gen_ai.request.frequency_penalty",
        "gen_ai.request.presence_penalty",
        "gen_ai.request.max_tokens",
        "gen_ai.request.stop_sequences",
        "gen_ai.request.choice.count",
        "gen_ai.request.seed",
        "gen_ai.request.top_k",
        "gen_ai.request.reasoning.level",
        "gen_ai.request.stream",
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.response.finish_reasons",
        "gen_ai.response.time_to_first_chunk",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.reasoning.output_tokens",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.cache_creation.input_tokens",
        "gen_ai.conversation.id",
        "gen_ai.prompt.name",
        "gen_ai.prompt.version",
        "gen_ai.output.type",
        "llm.request.type",
    }
)

_GENAI_ATTR_PREFIXES = ("gen_ai.", "llm.", "openai.", "anthropic.", "aws.bedrock.")

_CONTENT_CAPTURE_HINT = (
    "Enable message-content capture in your OTel instrumentation (for many "
    "SDKs: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true, or your "
    "vendor's equivalent content-capture option) and re-export the traces, "
    "or capture with the modelgenius-sdk client wrappers, which record full "
    "requests by default."
)


@dataclass
class OtlpConversionResult:
    """Outcome of converting one OTLP/JSON payload."""

    records: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped_non_genai: int = 0
    skipped_non_inference: int = 0

    @property
    def converted(self) -> int:
        return len(self.records)


def is_otlp_traces(raw: Any) -> bool:
    """True when ``raw`` looks like an OTLP/JSON trace export payload."""
    if not isinstance(raw, dict):
        return False
    spans = raw.get("resourceSpans", raw.get("resource_spans"))
    return isinstance(spans, list)


def convert_otlp_payload(raw: dict) -> OtlpConversionResult:
    """Convert one OTLP/JSON payload (an ExportTraceServiceRequest) to records."""
    result = OtlpConversionResult()
    resource_spans = raw.get("resourceSpans", raw.get("resource_spans")) or []
    for rs in resource_spans:
        if not isinstance(rs, dict):
            continue
        resource_attrs = _decode_attrs((rs.get("resource") or {}).get("attributes"))
        scope_spans = (
            rs.get("scopeSpans")
            or rs.get("scope_spans")
            or rs.get("instrumentationLibrarySpans")
            or []
        )
        for ss in scope_spans:
            if not isinstance(ss, dict):
                continue
            scope_name = ((ss.get("scope") or {}).get("name")) or None
            for span in ss.get("spans") or []:
                if not isinstance(span, dict):
                    continue
                _convert_span(span, resource_attrs, scope_name, result)
    return result


# ---------- OTLP plumbing ----------


def _decode_any_value(value: Any) -> Any:
    """Decode an OTLP/JSON AnyValue; pass through already-plain values."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])  # proto3 JSON encodes int64 as string
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [
            _decode_any_value(v) for v in (value["arrayValue"] or {}).get("values", [])
        ]
    if "kvlistValue" in value:
        return {
            kv.get("key"): _decode_any_value(kv.get("value"))
            for kv in (value["kvlistValue"] or {}).get("values", [])
            if isinstance(kv, dict)
        }
    if "bytesValue" in value:
        return value["bytesValue"]
    return value


def _decode_attrs(attr_list: Any) -> dict[str, Any]:
    """OTLP attribute list [{key, value}, ...] → plain dict."""
    out: dict[str, Any] = {}
    for entry in attr_list or []:
        if isinstance(entry, dict) and "key" in entry:
            out[entry["key"]] = _decode_any_value(entry.get("value"))
    return out


def _maybe_json(value: Any) -> Any:
    """Complex attributes may arrive as JSON strings on spans; decode if so."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _span_time_iso(span: dict, *keys: str) -> tuple[str | None, int | None]:
    """(ISO-8601 UTC string, raw nanos) for the first present time key."""
    for key in keys:
        raw = span.get(key)
        if raw is None:
            continue
        try:
            nanos = int(raw)
        except (TypeError, ValueError):
            continue
        dt = datetime.fromtimestamp(nanos / 1e9, tz=timezone.utc)
        return dt.isoformat(), nanos
    return None, None


# ---------- message mapping ----------


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _tool_call_entry(part: dict) -> dict:
    args = part.get("arguments")
    arguments = args if isinstance(args, str) else json.dumps(args if args is not None else {})
    return {
        "id": part.get("id") or f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": part.get("name") or "tool", "arguments": arguments},
    }


def _parts_messages_to_openai(messages: list) -> list[dict]:
    """OTel role/parts messages → OpenAI chat messages.

    Part types per the conventions: ``text``, ``tool_call``,
    ``tool_call_response``. Reasoning parts are response artifacts and are
    dropped; unknown part types are JSON-serialized into text so nothing is
    silently lost.
    """
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        role = _ROLE_ALIASES.get(role, role)
        parts = msg.get("parts")
        if parts is None and "content" in msg:
            out.append({"role": role, "content": _stringify(msg.get("content"))})
            continue

        texts: list[str] = []
        tool_calls: list[dict] = []
        tool_responses: list[tuple[str, str]] = []
        for part in parts or []:
            if not isinstance(part, dict):
                texts.append(_stringify(part))
                continue
            ptype = part.get("type")
            if ptype == "text":
                texts.append(_stringify(part.get("content")))
            elif ptype == "tool_call":
                tool_calls.append(_tool_call_entry(part))
            elif ptype == "tool_call_response":
                tool_responses.append(
                    (part.get("id") or "", _stringify(part.get("response")))
                )
            elif ptype in ("reasoning", "thinking"):
                continue
            else:
                texts.append(json.dumps(part, sort_keys=True))

        if role == "assistant":
            entry: dict = {"role": "assistant", "content": "\n".join(texts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            for call_id, response in tool_responses:
                out.append(
                    {"role": "tool", "tool_call_id": call_id, "content": response}
                )
        elif role == "tool":
            if tool_responses:
                for call_id, response in tool_responses:
                    out.append(
                        {"role": "tool", "tool_call_id": call_id, "content": response}
                    )
            elif texts:
                out.append({"role": "tool", "content": "\n".join(texts)})
        else:
            if texts:
                out.append({"role": role, "content": "\n".join(texts)})
            for call_id, response in tool_responses:
                out.append(
                    {"role": "tool", "tool_call_id": call_id, "content": response}
                )
            if tool_calls:
                # tool_call parts on a non-assistant role: preserve on an
                # assistant turn rather than dropping them.
                out.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    return out


def _system_instructions_text(value: Any) -> str:
    """gen_ai.system_instructions: string, or a list of {type, content} parts."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for part in value:
            if isinstance(part, dict):
                chunks.append(_stringify(part.get("content")))
            elif isinstance(part, str):
                chunks.append(part)
        return "\n\n".join(c for c in chunks if c)
    return ""


def _tool_definitions_to_openai(defs: Any) -> list[dict] | None:
    """gen_ai.tool.definitions entries → OpenAI function-tool wrappers."""
    if not isinstance(defs, list) or not defs:
        return None
    tools: list[dict] = []
    for entry in defs:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("function"), dict):
            tools.append(entry)  # already OpenAI-wrapped
            continue
        function: dict = {"name": entry.get("name") or "tool"}
        if entry.get("description"):
            function["description"] = entry["description"]
        function["parameters"] = entry.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        tools.append({"type": "function", "function": function})
    return tools or None


def _legacy_flat_messages(attrs: dict) -> tuple[list[dict], str | None, str | None]:
    """OpenLLMetry-era flattened attributes → (messages, output_text, finish).

    ``gen_ai.prompt.{i}.role/.content`` for input; ``gen_ai.completion.0.*``
    for the first output choice. Tool structures in the legacy flattening are
    not reconstructed.
    """
    indices: set[int] = set()
    for key in attrs:
        if key.startswith("gen_ai.prompt."):
            head = key[len("gen_ai.prompt.") :].split(".", 1)[0]
            if head.isdigit():
                indices.add(int(head))
    messages: list[dict] = []
    for i in sorted(indices):
        role = attrs.get(f"gen_ai.prompt.{i}.role") or "user"
        role = _ROLE_ALIASES.get(role, role)
        content = attrs.get(f"gen_ai.prompt.{i}.content")
        if content is None:
            continue
        messages.append({"role": role, "content": _stringify(content)})
    output_text = attrs.get("gen_ai.completion.0.content")
    finish = attrs.get("gen_ai.completion.0.finish_reason")
    return messages, (
        _stringify(output_text) if output_text is not None else None
    ), finish


# ---------- span conversion ----------


def _convert_span(
    span: dict,
    resource_attrs: dict,
    scope_name: str | None,
    result: OtlpConversionResult,
) -> None:
    attrs = _decode_attrs(span.get("attributes"))

    details_attrs: dict[str, Any] = {}
    for event in span.get("events") or []:
        if isinstance(event, dict) and event.get("name") == "gen_ai.client.inference.operation.details":
            details_attrs = _decode_attrs(event.get("attributes"))
            break
    merged = {**details_attrs, **attrs}  # span attributes win over the event

    operation = merged.get("gen_ai.operation.name")
    has_legacy_prompt = any(
        k.startswith("gen_ai.prompt.") and k[len("gen_ai.prompt.") :].split(".", 1)[0].isdigit()
        for k in merged
    )
    is_genai = (
        operation is not None
        or has_legacy_prompt
        or "gen_ai.provider.name" in merged
        or "gen_ai.system" in merged
        or "llm.request.type" in merged
    )
    if not is_genai:
        result.skipped_non_genai += 1
        return
    if operation is not None and operation not in INFERENCE_OPERATIONS:
        result.skipped_non_inference += 1
        return

    span_name = span.get("name") or "?"
    trace_id = span.get("traceId") or span.get("trace_id")
    span_id = span.get("spanId") or span.get("span_id")
    label = f"otel span {span_name!r} (trace_id={trace_id or '?'}, span_id={span_id or '?'})"

    start_iso, start_ns = _span_time_iso(span, "startTimeUnixNano", "start_time_unix_nano")
    end_iso, end_ns = _span_time_iso(span, "endTimeUnixNano", "end_time_unix_nano")
    if start_iso is None:
        result.errors.append(f"{label}: span has no start time; cannot derive the snapshot timestamp.")
        return

    # --- input messages ---
    messages: list[dict] = []
    content_source: str | None = None
    input_raw = _maybe_json(merged.get("gen_ai.input.messages"))
    if isinstance(input_raw, list) and input_raw:
        messages = _parts_messages_to_openai(input_raw)
        content_source = (
            "span_attributes" if "gen_ai.input.messages" in attrs else "operation_details_event"
        )

    legacy_output_text: str | None = None
    legacy_finish: str | None = None
    if not messages and has_legacy_prompt:
        messages, legacy_output_text, legacy_finish = _legacy_flat_messages(merged)
        if messages:
            content_source = "legacy_flat_attributes"

    system_text = _system_instructions_text(_maybe_json(merged.get("gen_ai.system_instructions")))
    if system_text and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_text})

    if not any(m.get("role") != "system" for m in messages):
        result.errors.append(
            f"{label}: GenAI span carries no message content (metadata-only telemetry). "
            + _CONTENT_CAPTURE_HINT
        )
        return

    # --- output / captured response ---
    output_text: str | None = legacy_output_text
    output_tool_calls: list[dict] | None = None
    finish_reason: str | None = legacy_finish
    output_choice_count: int | None = None
    output_raw = _maybe_json(merged.get("gen_ai.output.messages"))
    if isinstance(output_raw, list) and output_raw:
        output_choice_count = len(output_raw)
        first_choice = output_raw[0]
        converted_choice = _parts_messages_to_openai([first_choice])
        if converted_choice:
            output_text = converted_choice[0].get("content") or ""
            output_tool_calls = converted_choice[0].get("tool_calls")
        if isinstance(first_choice, dict) and first_choice.get("finish_reason"):
            finish_reason = first_choice["finish_reason"]

    usage: dict[str, int] = {}
    input_tokens = merged.get("gen_ai.usage.input_tokens", merged.get("gen_ai.usage.prompt_tokens"))
    output_tokens = merged.get(
        "gen_ai.usage.output_tokens", merged.get("gen_ai.usage.completion_tokens")
    )
    reasoning_tokens = merged.get("gen_ai.usage.reasoning.output_tokens")
    if isinstance(input_tokens, int):
        usage["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        usage["output_tokens"] = output_tokens
    if isinstance(reasoning_tokens, int):
        usage["thinking_tokens"] = reasoning_tokens
    if "input_tokens" in usage and "output_tokens" in usage:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

    provider = merged.get("gen_ai.provider.name") or merged.get("gen_ai.system")
    api_surface = "anthropic" if provider == "anthropic" else "chat"
    request_model = merged.get("gen_ai.request.model")
    response_model = merged.get("gen_ai.response.model")

    captured_output: dict | None = None
    if output_text is not None or output_tool_calls:
        provider_metadata: dict[str, Any] = {}
        for source_key, meta_key in (
            ("gen_ai.response.id", "response_id"),
            ("gen_ai.response.finish_reasons", "finish_reasons"),
            ("gen_ai.response.time_to_first_chunk", "time_to_first_chunk"),
            ("gen_ai.usage.cache_read.input_tokens", "cache_read_input_tokens"),
            ("gen_ai.usage.cache_creation.input_tokens", "cache_creation_input_tokens"),
        ):
            if merged.get(source_key) is not None:
                provider_metadata[meta_key] = merged[source_key]
        if finish_reason is not None:
            provider_metadata["finish_reason"] = finish_reason
        captured_output = {
            "output_text": output_text or "",
            "model": response_model or request_model,
            "api_surface": api_surface,
            "request_timestamp": start_iso,
            "response_timestamp": end_iso or start_iso,
        }
        if output_tool_calls:
            captured_output["tool_calls"] = output_tool_calls
        if usage:
            captured_output["usage"] = usage
        if start_ns is not None and end_ns is not None and end_ns >= start_ns:
            captured_output["latency_ms"] = round((end_ns - start_ns) / 1e6, 3)
        if provider_metadata:
            captured_output["provider_metadata"] = provider_metadata

    # --- request parameters ---
    record: dict[str, Any] = {
        "format": FORMAT_TAG,
        "snapshot_id": (
            f"otel-{trace_id}-{span_id}" if trace_id and span_id else f"otel-{uuid.uuid4().hex}"
        ),
        "timestamp": start_iso,
        "messages": messages,
        "model": request_model or response_model,
        "api_surface": api_surface,
    }
    for attr_key, record_key in (
        ("gen_ai.request.temperature", "temperature"),
        ("gen_ai.request.top_p", "top_p"),
        ("gen_ai.request.frequency_penalty", "frequency_penalty"),
        ("gen_ai.request.presence_penalty", "presence_penalty"),
        ("gen_ai.request.max_tokens", "max_tokens"),
        ("gen_ai.request.stop_sequences", "stop"),
        ("gen_ai.request.reasoning.level", "reasoning_effort"),
    ):
        if merged.get(attr_key) is not None:
            record[record_key] = merged[attr_key]
    choice_count = merged.get("gen_ai.request.choice.count")
    if isinstance(choice_count, int) and choice_count != 1:
        record["n"] = choice_count

    provider_params: dict[str, Any] = {}
    if merged.get("gen_ai.request.seed") is not None:
        provider_params["seed"] = merged["gen_ai.request.seed"]
    if merged.get("gen_ai.request.top_k") is not None:
        provider_params["top_k"] = merged["gen_ai.request.top_k"]
    if provider_params:
        record["provider_params"] = provider_params

    tools = _tool_definitions_to_openai(_maybe_json(merged.get("gen_ai.tool.definitions")))
    if tools:
        record["tools"] = tools

    # --- enrichment gaps: detect, surface, never fabricate ---
    enrichment_needed: list[str] = []
    output_type = merged.get("gen_ai.output.type")
    if output_type == "json":
        # The conventions cannot carry the response_format JSON schema; the
        # customer supplies it once per workflow to restore contract checks.
        enrichment_needed.append("response_format_schema")
    saw_tool_calls = bool(output_tool_calls) or any(m.get("tool_calls") for m in messages)
    if saw_tool_calls and not tools:
        enrichment_needed.append("tool_definitions")

    # --- metadata ---
    otel_meta: dict[str, Any] = {"span_name": span_name}
    for value, key in (
        (trace_id, "trace_id"),
        (span_id, "span_id"),
        (provider, "provider"),
        (operation, "operation"),
        (scope_name, "scope"),
        (resource_attrs.get("service.name"), "service_name"),
        (merged.get("gen_ai.conversation.id"), "conversation_id"),
        (merged.get("gen_ai.prompt.name"), "prompt_name"),
        (merged.get("gen_ai.prompt.version"), "prompt_version"),
        (output_type, "output_type"),
        (merged.get("gen_ai.request.stream"), "stream"),
        (content_source, "content_source"),
    ):
        if value is not None:
            otel_meta[key] = value
    if output_choice_count is not None and output_choice_count > 1:
        otel_meta["output_choice_count"] = output_choice_count

    prompt_variables = {
        key[len("gen_ai.prompt.variable.") :]: value
        for key, value in merged.items()
        if key.startswith("gen_ai.prompt.variable.")
    }
    if prompt_variables:
        otel_meta["prompt_variables"] = prompt_variables

    extra_attrs = {
        key: value
        for key, value in merged.items()
        if key.startswith(_GENAI_ATTR_PREFIXES)
        and key not in _CONSUMED_ATTRS
        and not key.startswith("gen_ai.prompt.variable.")
        and not (
            key.startswith(("gen_ai.prompt.", "gen_ai.completion."))
            and key.split(".", 2)[2].split(".", 1)[0].isdigit()
        )
    }
    if extra_attrs:
        otel_meta["attributes"] = extra_attrs

    metadata: dict[str, Any] = {"source_format": "otel_genai", "otel": otel_meta}
    if enrichment_needed:
        metadata["enrichment_needed"] = enrichment_needed
    record["metadata"] = metadata

    workflow_id = merged.get("gen_ai.prompt.name")
    if workflow_id:
        record["workflow_id"] = workflow_id
    if captured_output is not None:
        record["captured_output"] = captured_output

    result.records.append(record)
