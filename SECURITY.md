# Security Policy

## Data-handling posture

`modelgenius-sdk` is designed so that you can verify its behavior by reading
the source:

- **Local-only.** The library makes **no network calls**. Capture writes
  JSONL files to a local spool directory; validation reads local files. The
  only network traffic in your process is your own, unmodified provider call.
  Uploading snapshots to ModelGenius is a separate, deliberate step performed
  outside this library.
- **No telemetry.** The library collects and transmits nothing about you,
  your usage, or your data.
- **No credential access.** Provider clients are duck-typed and passed
  through; the SDK never reads or stores API keys. Transport/auth kwargs
  (`extra_headers`, `extra_query`, `timeout`) are stripped from records
  before they are written.
- **Fail-open by design.** Capture errors are swallowed (logged once per
  error class) and never alter your application's calls, return values, or
  exceptions.

### Treat spool files as sensitive

Snapshot files contain your prompts and (when captured) model outputs — which
may include end-user content or PII. Recommendations:

- Point `MODELGENIUS_CAPTURE_DIR` at a directory with appropriate filesystem
  permissions, and apply the same retention and access policy you use for
  application logs containing user content.
- Use the `redact` hook on `CaptureConfig` to scrub or drop sensitive records
  before they are written, and `request_filter` / `sample_rate` to limit what
  is captured at all.
- `MODELGENIUS_CAPTURE=0` disables capture entirely.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | yes |

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/ClassGenius/modelgenius-sdk/security/advisories/new)
("Report a vulnerability" on the repository's Security tab). Do **not** open
a public issue for security reports.

Include what you can: affected version, reproduction steps, and impact
assessment. You will receive an acknowledgment within a few business days,
and we will coordinate a fix and disclosure timeline with you.
