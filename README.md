# modelgenius-sdk

[![CI](https://github.com/modelgenius/modelgenius-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/modelgenius/modelgenius-sdk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/modelgenius-sdk.svg)](https://pypi.org/project/modelgenius-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/modelgenius-sdk.svg)](https://pypi.org/project/modelgenius-sdk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Capture your production LLM calls as local **snapshot** files, and validate
them against the ModelGenius snapshot format — entirely on your machine, with
**zero network calls and zero telemetry**.

[ModelGenius](https://github.com/modelgenius) answers "is it safe to switch
LLM models, and is it worth it?" by replaying your real production requests
against candidate models and comparing quality, cost, and latency. The input
to that analysis is a set of snapshots: your requests, the parameters they
ran with, and (ideally) the responses your current model actually produced.
This SDK is the capture side. It has one runtime dependency (`jsonschema`)
and never imports `openai`, `anthropic`, or `litellm` — it duck-types your
existing client objects.

Three ways to adopt it:

1. **Wrap your client** — one line at startup; every call is spooled to local
   JSONL as it happens, with real outputs, token usage, and latency attached.
2. **Bring your own logs** — if you already log LLM traffic, write it in the
   [snapshot format](docs/snapshot_format.md) and use `modelgenius-sdk
   validate` to check it before uploading.
3. **Bring your OpenTelemetry traces** — if your app is already instrumented
   with the [OTel GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
   (tested with Traceloop/OpenLLMetry, the official
   `opentelemetry-instrumentation-*` packages, and Claude SDK and Claude API
   tracing), convert an OTLP/JSON export with `modelgenius-sdk convert otlp`
   — no app changes needed. Requires message content capture to be enabled
   in your instrumentation (it is off by default in OTel; metadata-only spans
   convert to an actionable error). Legacy flattened OpenLLMetry attributes
   (`gen_ai.prompt.N.*`) are also understood; OpenInference (Arize Phoenix)
   spans are not yet, and are reported as such. Two gaps are inherent to the
   conventions and flagged rather than guessed: `response_format` JSON
   schemas (Traceloop records them, and they are used) and often tool
   definitions are not carried in traces — converted records note
   `enrichment_needed` in metadata and the validator reminds you to supply
   them at upload.

## Install

```bash
pip install modelgenius-sdk
```

## Quickstart

**OpenAI** (sync or async client; streaming supported):

```python
import modelgenius_sdk
from openai import OpenAI

client = modelgenius_sdk.wrap_openai(OpenAI(), workflow_name="support_ticket_triage")
# use `client` exactly as before — every chat.completions call is snapshotted locally
```

**Anthropic** (sync or async client; `messages.create` streaming supported):

```python
import modelgenius_sdk
from anthropic import Anthropic

client = modelgenius_sdk.wrap_anthropic(Anthropic(), workflow_name="math_tutoring")
```

**LiteLLM** (success callback):

```python
import litellm
import modelgenius_sdk

litellm.success_callback.append(modelgenius_sdk.make_litellm_callback(workflow_name="ap_automation"))
```

**Validate** what you captured (or logs you wrote yourself):

```bash
modelgenius-sdk validate ./modelgenius_snapshots
```

The wrapper returns the same client object and never changes call behavior:
your arguments pass through untouched, return values and exceptions propagate
unchanged, and any internal capture error is swallowed (logged once), never
raised into your app.

## What gets captured

One JSON object per call, appended to a local JSONL spool file: the request
(messages, model, temperature, tools, response_format, and the rest of the
request parameters), plus a `captured_output` block with the real response —
output text, tool calls, token usage, latency, and, via the LiteLLM callback,
cost. See [docs/snapshot_format.md](docs/snapshot_format.md) for the full
format, and [examples/](examples/) for validated sample files.

The more you capture, the more the analysis can do:

| You provide | You unlock |
| --- | --- |
| `captured_output` | Baseline scored from real production outputs — no baseline replay, real latency/cost preserved |
| `usage` + `cost_usd` | Honest savings baseline from actual spend |
| `workflow_name` + `call_name` (or `workflow_id`) | Precise workflow grouping instead of content-based inference |
| `tools` / `response_format` | Deterministic output-contract checks on every candidate |
| `reference` | Every model judged against your known-correct answers (see [docs](docs/snapshot_format.md#reference-a-known-correct-answer)) |
| `model` | Automatic baseline identification |

Full table with grouping precedence: [docs/snapshot_format.md](docs/snapshot_format.md#what-each-optional-field-unlocks).

## Configuration

Everything can be set in code via `CaptureConfig`, or via environment
variables:

```python
from modelgenius_sdk import CaptureConfig, wrap_openai

config = CaptureConfig(
    spool_dir="/var/log/myapp/snapshots",
    workflow_name="support_ticket_triage",
    call_name="classify_ticket",
    sample_rate=0.25,          # capture 25% of calls
)
client = wrap_openai(OpenAI(), config)
```

| Environment variable | Effect | Default |
| --- | --- | --- |
| `MODELGENIUS_CAPTURE` | Set to `0`, `false`, or `no` (case-insensitive) to disable capture entirely — the kill switch | enabled |
| `MODELGENIUS_CAPTURE_DIR` | Spool directory for snapshot files | `./modelgenius_snapshots` |
| `MODELGENIUS_CAPTURE_SAMPLE_RATE` | Fraction of calls to capture, `0.0`–`1.0` | `1.0` |
| `MODELGENIUS_CAPTURE_DEBUG` | Set to `1` to make internal capture errors raise instead of being swallowed (for tests/debugging) | off |

Explicit `CaptureConfig` values win over environment variables. The sampling
decision is made once per call, before dispatch, so a streamed response is
either fully captured or not at all.

### Redaction and filtering

You stay in control of what lands on disk:

```python
def redact(record: dict):
    # Called with the full record just before it is written.
    for message in record.get("messages", []):
        if isinstance(message.get("content"), str):
            message["content"] = scrub_pii(message["content"])
    if record.get("metadata", {}).get("tenant_tier") == "enterprise":
        return None  # returning None drops the record entirely
    return record

def only_prod_traffic(request_kwargs: dict) -> bool:
    # Called with the request kwargs before capture is attempted.
    return request_kwargs.get("user") != "internal-smoke-test"

config = CaptureConfig(
    workflow_name="support_ticket_triage",
    redact=redact,
    request_filter=only_prod_traffic,
    sample_rate=0.1,
)
```

### Spool files and rotation

Records are appended to
`<spool_dir>/snapshots-<YYYYMMDD>-<pid>-<seq>.jsonl` — one JSON object per
line, flushed after every write, safe across processes (each PID writes its
own file). When the current file exceeds `max_file_bytes` (default 64 MiB)
the writer rotates to the next sequence number. The spool directory is
created lazily on first write. Point `modelgenius-sdk validate` (or the
ModelGenius upload step) at the directory when you are ready.

## Trust posture

- **Local-only.** The library makes no network calls of its own — the only
  network traffic is your unmodified provider call. Uploading snapshots to
  ModelGenius is a separate, deliberate step you take elsewhere.
- **No telemetry.** Nothing is phoned home. Ever.
- **Fail-open.** Capture errors never raise into your application and never
  alter the provider call, its return value, or its exceptions.
- **You control the data.** Redaction hook, request filter, sampling, and a
  one-variable kill switch (`MODELGENIUS_CAPTURE=0`).
- **Small and auditable.** A handful of modules, one dependency, no provider
  SDK imports — read the whole thing in one sitting.

## CLI

```bash
modelgenius-sdk validate PATH [--json] [--strict] [--quiet]
modelgenius-sdk convert otlp PATH [-o OUTPUT.jsonl] [--json] [--quiet]
modelgenius-sdk version
```

`validate` accepts a `.jsonl` file, a `.json` file (object or array), or a
directory of both. Exit code 0 means every record is accepted by the same
semantics as the ModelGenius upload path; warnings point out optional fields
that would unlock capability. `--strict` turns warnings into failures (useful
in CI — this repo validates its own `examples/` on every push).

`convert otlp` converts an OTLP/JSON trace export (a `.json` payload, or a
collector file export with one payload per line, named `.jsonl` or `.json`)
into snapshot JSONL and validates the result in one step. When one call was
traced twice (say the Claude SDK's span and the Claude API's, or the same span
delivered twice), the copies become one record. Exit code 0 means every GenAI
span converted and validated; per-span problems (e.g. content capture
disabled in your instrumentation) are reported individually and exit 1.
Binary (protobuf) exports are not read; use the collector's `format: json`.

## Limitations (v0)

- Failed provider calls are not captured (only successful responses).
- OpenAI: `chat.completions` only; the `responses` API surface is not wrapped yet.
- Anthropic: `messages.create` (including `stream=True`) is captured; the
  `messages.stream()` helper is not yet.
- LiteLLM: the callback is sync-only (litellm accepts sync callbacks for
  async paths).

## Roadmap

- Direct upload client (`modelgenius-sdk push`) as an explicit, opt-in step
- Continuous monitoring mode (rolling capture windows + scheduled re-evaluation)
- TypeScript SDK
- OpenAI `responses` surface and Anthropic `messages.stream()` capture

## License

[Apache-2.0](LICENSE).
