# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `convert otlp`: Traceloop's `gen_ai.request.structured_output_schema` now
  becomes the record's `response_format` (it was ignored, and not flagged).
- `convert otlp`: Anthropic-style tool definitions (`input_schema`, as
  Traceloop's Anthropic instrumentation records them) keep their parameter
  schema instead of converting with empty parameters.
- `convert otlp`: one call traced twice (the Claude SDK's client span and the
  Claude API's server span, Traceloop plus the SDK's own span, or the same
  span delivered twice) converts to one record, the most complete copy; the
  others are listed in `metadata.otel.duplicate_span_ids`.
- `convert otlp`: spans that are not inference calls but carry a provider
  (file uploads, the Claude API's `anthropic.model_request`) no longer
  produce "no message content" errors.
- `convert otlp`: the content-capture hint names settings that work today
  (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_only`; current
  official instrumentations ignore `=true`) and the Claude Console's
  "Full content" mode.
- `convert otlp`: a collector file export named `.json` (one payload per
  line) converts; a binary file gets a clear error.
- `convert otlp`: Claude spans keep their stop reason (read from
  `anthropic.message.stop_reason` when the output message and
  `gen_ai.response.finish_reasons` carry none), and the Claude Console's
  `gen_ai.usage.cache_write.input_tokens` is kept as
  `cache_creation_input_tokens`.

### Added

- Snapshots can carry an optional `reference`: a known-correct answer for the
  request — the full answer as `output`, the values a correct answer must
  contain as `values`, or both — with optional `notes`, `importance`
  (`"ranking"` or `"must_pass"`) and `source`. ModelGenius judges every
  model's output against it; see
  [docs/snapshot_format.md](docs/snapshot_format.md#reference-a-known-correct-answer).
  The schema and the local validator check it on both record shapes: at least
  one of `output` or `values` is required.
- `examples/reference.jsonl`.
- Records use Traceloop's `traceloop.workflow.name` as `workflow_id` (when
  there is no `gen_ai.prompt.name`) and `traceloop.entity.path` as
  `metadata.call_name`, so calls group by workflow and task.
- `metadata.otel` records `response_id`, `span_kind` and `parent_span_id`.
- OpenInference (Arize Phoenix) LLM spans are reported in one error instead of
  being skipped silently.
- `convert_otlp_payloads()` converts several payloads as one set of traces;
  `ConvertFileResult` / `OtlpConversionResult` count `skipped_duplicate`,
  `no_content` and `unsupported_openinference`.

### Changed

- `reference` is now a known top-level field. A record that used a top-level
  `reference` key for something else (a ticket ID string, say) now fails
  validation, as it does on upload: rename that key or move it under
  `metadata`.

## [0.2.0] - 2026-07-09

### Added

- `modelgenius-sdk convert otlp PATH [-o OUTPUT.jsonl]` — convert OTLP/JSON
  trace exports following the OpenTelemetry GenAI semantic conventions into
  snapshot JSONL, then validate the result. Reads message content from span
  attributes, the `gen_ai.client.inference.operation.details` event, or
  legacy flattened OpenLLMetry attributes; metadata-only spans (content
  capture disabled) produce actionable per-span errors.
- `modelgenius_sdk.convert` — `convert_otlp_file()` / `convert_otlp_payload()`
  / `is_otlp_traces()` public API (pure stdlib, local-only).
- Validator warnings `needs_response_format_schema` and
  `needs_tool_definitions` for imported records whose trace format could not
  carry the response-format JSON schema or tool definitions
  (`metadata.enrichment_needed`).

## [0.1.0] - 2026-07-05

Initial public release.

### Added

- `wrap_openai(client, ...)` — instance-level capture wrapper for OpenAI
  `chat.completions.create` (sync and async clients, streaming included).
- `wrap_anthropic(client, ...)` — instance-level capture wrapper for Anthropic
  `messages.create` (sync and async clients, `stream=True` included).
- `make_litellm_callback(...)` — success-callback factory for LiteLLM, with
  `cost_usd` filled from litellm's `response_cost`.
- `CaptureConfig` — spool directory, workflow/call naming, sampling,
  redaction hook, request filter, kill switch, rotation size; all overridable
  via `MODELGENIUS_CAPTURE*` environment variables.
- Thread-safe, fail-open JSONL spool writer with size-based rotation
  (`snapshots-<YYYYMMDD>-<pid>-<seq>.jsonl`).
- `modelgenius-sdk validate` CLI and `validate_path()` API implementing the
  same accept/reject semantics as the ModelGenius upload path, plus
  capability-unlock warnings (`--json`, `--strict`, `--quiet`).
- Bundled JSON Schema for the canonical record shape
  (`modelgenius.snapshot/v1`) and the format specification
  (`docs/snapshot_format.md`).
- Validated example snapshot files under `examples/`.

[0.1.0]: https://github.com/modelgenius/modelgenius-sdk/releases/tag/v0.1.0
