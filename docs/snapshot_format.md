# ModelGenius Snapshot Format (v1)

A **snapshot** is one captured production LLM request — optionally together
with the response your baseline model actually produced. Snapshot files are
what you upload to ModelGenius, and what `modelgenius-sdk` captures and
validates locally.

- File formats: **JSONL** (one JSON object per line, blank lines ignored) or
  **`.json`** (a single object or an array of objects). A directory of such
  files also works.
- Format tag: records *may* carry `"format": "modelgenius.snapshot/v1"`. It is
  optional, but when present it must be exactly that string. The capture SDK
  always writes it.
- Encoding: UTF-8. All timestamps are ISO-8601 strings with a timezone
  (trailing `Z` or an explicit offset).

## Why request parameters matter

ModelGenius answers "is it safe to switch models, and is it worth it?" by
replaying your captured requests against candidate models and comparing. That
comparison is only honest if the replay recreates the conditions your baseline
actually ran under. If production ran with `reasoning_effort=high` and a
`max_tokens=512` cap, but the replay runs default-effort and uncapped, you are
not measuring the same thing — both the quality verdict and the cost
projection drift. So capture the request parameters alongside the prompt; the
SDK wrappers do this for you automatically.

## A minimal valid snapshot

Only two things are strictly required: `messages` and a resolvable
`timestamp`.

```json
{"messages": [{"role": "user", "content": "Classify this ticket: ..."}], "timestamp": "2026-06-11T14:32:05Z"}
```

Everything else is optional — but each optional field you include makes the
analysis more faithful and unlocks capability (see
[What each optional field unlocks](#what-each-optional-field-unlocks)).

A well-populated snapshot looks like this (one JSONL line, shown expanded):

```json
{
  "format": "modelgenius.snapshot/v1",
  "snapshot_id": "tkt-84512-triage",
  "timestamp": "2026-06-11T14:47:19Z",
  "model": "gpt-4o-mini",
  "temperature": 0,
  "messages": [
    {"role": "system", "content": "You triage inbound support tickets..."},
    {"role": "user", "content": "The export-to-CSV button does nothing..."}
  ],
  "metadata": {"workflow_name": "support_ticket_triage", "call_name": "classify_ticket"},
  "captured_output": {
    "output_text": "bug",
    "model": "gpt-4o-mini-2024-07-18",
    "usage": {"input_tokens": 174, "output_tokens": 2, "thinking_tokens": null, "total_tokens": 176},
    "cost_usd": 0.0000274,
    "latency_ms": 412.8,
    "request_timestamp": "2026-06-11T14:47:19Z",
    "response_timestamp": "2026-06-11T14:47:20Z"
  }
}
```

The [`examples/`](../examples/) directory contains complete, validated files
for each major variation.

## Two accepted request shapes

A record's *request* portion can be in either of two shapes, auto-detected
per line — a single file may freely mix them:

1. **OpenAI-shaped (canonical)** — top-level `messages` in OpenAI chat format,
   plus the optional parameters listed below.
2. **Anthropic-shaped (Messages API native)** — top-level `system` (string or
   block list), content-block messages (`tool_use` / `tool_result` / `image`
   blocks), `max_tokens`, `stop_sequences`, `top_k`, `thinking`, and so on,
   exactly as your app sent them to the Anthropic API.

**Detection rule** (structural only — the model name and file name are never
consulted): a record is treated as Anthropic-shaped iff its top-level `system`
is a string or list, **or** any message has list-shaped content containing a
block whose `type` is `tool_use` or `tool_result`, or an `image` block with a
dict `source`. Otherwise it is OpenAI-shaped.

Anthropic-shaped records are converted to the canonical shape at upload time:

| Anthropic field | Canonical result |
| --- | --- |
| `system` (string or text-block list) | First message with `role: "system"` |
| `tool_use` content blocks | Assistant `tool_calls` (OpenAI function-call shape) |
| `tool_result` content blocks | `role: "tool"` messages with `tool_call_id` |
| `tools[*].input_schema` | `{"type": "function", "function": {..., "parameters": ...}}` |
| `tool_choice` `{auto\|any\|tool}` | `"auto"` / `"required"` / `{"type": "function", ...}` |
| `thinking.budget_tokens` | `reasoning_effort` tier (`low`/`medium`/`high`) |
| `stop_sequences` | `stop` |
| `top_k`, raw `thinking` block | `provider_params` (captured, not auto-replayed) |

Write the request the way your app actually made it; do not convert it
yourself.

## Envelope fields

| Field | Required? | Meaning |
| --- | --- | --- |
| `format` | optional | Self-describing tag; must be `"modelgenius.snapshot/v1"` when present. |
| `snapshot_id` | optional | Stable ID for the record. A UUID is generated if absent. |
| `timestamp` | **required (resolvable)** | When the production request happened. See [Timestamps](#timestamps). |
| `messages` | **required** | The conversation, in the request's native shape. |
| `workflow_id` | optional | Explicit grouping key, used when `metadata.workflow_name` is absent — see [Grouping precedence](#grouping-precedence). |
| `metadata` | optional | Free-form dict. Recognized keys: `workflow_name`, `call_name`, `timestamp`, `request_timestamp`. The capture SDK adds `capture_sdk: {"name", "version"}`. |
| `captured_output` | optional | The real production response. See [captured_output](#captured_output-the-real-production-response). |
| `api_surface` | optional | Which request surface produced the snapshot (`"chat"` for OpenAI chat completions; set server-side for Anthropic records). Provenance for replay-side parameter renaming. |

Unknown top-level fields are permitted (preserved or ignored, never a
rejection) — that is part of the v1 compatibility promise.

## Grouping precedence

ModelGenius groups snapshots into workflow/call groups (grouping quality
directly drives the quality of the extracted evaluation criteria). The group
key is resolved per snapshot, in order:

1. `metadata.workflow_name` (paired with `metadata.call_name`, defaulting to
   `default_call`) — wins whenever present.
2. `workflow_id` — used when `workflow_name` is absent.
3. Content-derived clustering — the fallback when neither is set. It works,
   but is fuzzier than telling us explicitly.

The SDK wrappers set `metadata.workflow_name` / `metadata.call_name` from
`CaptureConfig(workflow_name=..., call_name=...)`.

## Timestamps

All timestamp fields are ISO-8601 strings with timezone information, e.g.
`"2026-06-11T14:32:05Z"` or `"2026-06-11T14:32:05+00:00"`.

Every snapshot must have a **resolvable** timestamp. If the top-level
`timestamp` is absent, it is backfilled from, in order:

1. `metadata.timestamp`
2. `metadata.request_timestamp`
3. `captured_output.request_timestamp`

If none of these exist, the record is rejected with:

> timestamp is required. Provide top-level timestamp or compatibility
> metadata.timestamp, metadata.request_timestamp, or
> captured_output.request_timestamp.

Three distinct timestamps can appear on a record:

- `timestamp` — where the request belongs in the workflow timeline.
- `captured_output.request_timestamp` — when the provider request started.
- `captured_output.response_timestamp` — when the provider response completed.

`request_timestamp` and `response_timestamp` are **both required whenever
`captured_output` is present** (real latency is derived from the pair).

## Request parameter fields

| Field | What it is | Replay behavior |
| --- | --- | --- |
| `model` | Baseline model that served production | Identifies the baseline to compare against |
| `temperature` | Sampling temperature | Reproduced |
| `top_p` | Nucleus sampling cutoff | Reproduced |
| `max_tokens` | Output-token cap (canonical name) | Reproduced |
| `stop` | Stop sequence(s) | Reproduced |
| `reasoning_effort` | Reasoning depth (`minimal`/`low`/`medium`/`high`/…) | Reproduced, mapped per target model (see below) |
| `frequency_penalty`, `presence_penalty`, `n` | Standard sampling knobs | Reproduced |
| `tools`, `tool_choice`, `response_format` | Tool / structured-output contract | Reproduced; also drives deterministic output checks |
| `api_surface` | Request surface provenance | Drives replay-side parameter renaming |
| `provider_params` | Capture-only bag | **Not** auto-replayed (see below) |

Surface-specific spellings of the token cap are canonicalized for you:
`max_completion_tokens` and `max_output_tokens` both map to `max_tokens`.

### Replay policy: faithful baseline, mapped candidate

Output-shaping parameters (`temperature`, `top_p`, `max_tokens`, `stop`,
`reasoning_effort`) are applied to **every** replayed model. For
`reasoning_effort` the captured value is run through a per-model capability
registry: it is an **identity** for the baseline's own model (faithful
reproduction) and is **coerced** onto a different candidate's supported set
(for example, an OpenAI `minimal`/`low`/`medium`/`high` tier is translated to
an Anthropic thinking budget or a Gemini thinking level). When a captured
effort cannot be faithfully mapped to a candidate (say, a non-reasoning
model), the parameter is **dropped** and the candidate runs at its natural
default — a graceful, logged degrade rather than a hard failure.

### `provider_params`: captured but not auto-replayed

Some parameters are recorded for provenance but deliberately **not** applied
on replay: `seed`, `logit_bias`, `service_tier`, `parallel_tool_calls`,
`verbosity`, `user`, `top_k`, and the raw Anthropic `thinking` block. You can
send these at the top level of a record (the SDK captures them wherever your
app put them); they are swept into `provider_params` automatically at upload.
Two reasons they are off by default:

- **`seed` is the load-bearing case.** ModelGenius measures behavior *across*
  stochastic variation — pass rates with confidence intervals only mean
  something if the dice are actually rolling. Pinning a captured seed would
  freeze one point of the output distribution and make the measured variance
  artificially tight. Seed is recorded for a faithful record of the request
  but is not applied on replay.
- **`logit_bias` is tokenizer-specific.** Token IDs are meaningful only for
  the baseline's tokenizer; transferring them to a different candidate is at
  best meaningless and at worst harmful.

## `captured_output`: the real production response

`captured_output` carries the response your baseline model actually produced.
It is optional — but strongly recommended. When present, ModelGenius scores
the baseline's real output directly instead of replaying the baseline model,
which saves replay tokens and preserves the true production latency, cost,
and token usage.

> **The canonical-shape rule:** `captured_output` is **always** written in the
> canonical (OpenAI-flavored) shape — *even when the request part of the
> record is Anthropic-shaped*. The upload pipeline converts Anthropic-shaped
> *requests* but passes `captured_output` through verbatim, so a
> provider-native response body here would be misread. The SDK wrappers do
> this conversion for you; if you build records by hand from Anthropic
> responses, apply the same mapping:
>
> - `output_text` = the `text` blocks concatenated with `"\n"` (thinking
>   blocks dropped).
> - Each `tool_use` block becomes an OpenAI-style tool call:
>   `{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<JSON string of input>"}}`.

| Field | Type | Meaning |
| --- | --- | --- |
| `output_text` | string (default `""`) | The assistant's text output. |
| `tool_calls` | list \| null | OpenAI function-call shape (see above). |
| `raw_response` | object (default `{}`) | Full provider response, JSON-safe. For streams: `{"stream": true, "chunk_count": N, "reconstructed": {...}}`. |
| `model` | string \| null | Model name from the response (often more precise than the requested alias). |
| `usage` | object \| null | `{input_tokens, output_tokens, thinking_tokens, total_tokens}`, each int or null. |
| `cost_usd` | number \| null | Cost of the production call in USD. |
| `latency_ms` | number \| null | Observed production latency. |
| `api_surface` | string \| null | `"chat"` for OpenAI chat completions, `"anthropic"` for Anthropic messages. |
| `provider_metadata` | object (default `{}`) | e.g. `{"request_id": ..., "stop_reason": ..., "finish_reason": ...}`. |
| `request_timestamp` | **required** | ISO-8601, when the provider request started. |
| `response_timestamp` | **required** | ISO-8601, when the provider response completed. |

## What each optional field unlocks

| You provide | ModelGenius can then |
| --- | --- |
| `captured_output` | Skip the baseline replay entirely — score your real production outputs, keep real latency/cost/usage, save replay tokens. |
| `captured_output.usage` + `cost_usd` | Compute an honest savings baseline from what production actually cost, instead of estimating from token counts and price sheets. |
| `metadata.workflow_name` + `metadata.call_name`, or `workflow_id` | Group snapshots into workflows precisely. **Precedence:** `workflow_name` (+ `call_name`) wins; else `workflow_id`; else grouping is inferred by content clustering — see [Grouping precedence](#grouping-precedence). |
| `tools` / `response_format` | Run deterministic contract checks (JSON parseability, schema conformance, required fields, type validity) against every candidate's outputs. |
| `model` | Identify the baseline automatically and anchor the cost/quality comparison. |
| `temperature`, `max_tokens`, `reasoning_effort`, … | Replay under the same conditions production ran under — honest quality verdicts and cost projections. |

## Versioning promise

`modelgenius.snapshot/v1` evolves **additively only**:

- New optional fields may be added; existing fields will not be removed or
  change meaning within v1.
- Unknown top-level fields are accepted today and will remain accepted.
- A file that validates cleanly against v1 today will continue to validate
  against v1.

The machine-readable contract is the JSON Schema shipped inside the package
at `modelgenius_sdk/schema/modelgenius.snapshot.v1.schema.json` (it describes
the canonical shape; Anthropic-shaped requests are additionally accepted per
the detection rule above).

## Validating locally

```bash
modelgenius-sdk validate ./modelgenius_snapshots
```

The validator applies the same accept/reject semantics as the ModelGenius
upload path, and emits warnings for missing optional fields that would unlock
capability (no `captured_output`, no grouping keys, no usage/cost, and so on).
Use `--strict` to treat warnings as failures and `--json` for
machine-readable output.
