# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
