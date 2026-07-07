# Contributing to modelgenius-sdk

Thanks for helping out. This is a deliberately small, auditable library —
please keep it that way.

## Development setup

```bash
git clone https://github.com/modelgenius/modelgenius-sdk
cd modelgenius-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the checks that CI runs:

```bash
ruff check src tests
pytest -q
modelgenius-sdk validate examples/
```

## Ground rules

- **Python >= 3.9.** No `match` statements, no `datetime.UTC` (use
  `timezone.utc`), and every module starts with
  `from __future__ import annotations`.
- **One runtime dependency.** `jsonschema` only. Do not add runtime
  dependencies.
- **Never import provider SDKs.** `openai`, `anthropic`, and `litellm` must
  not be imported anywhere in `src/` or `tests/` — provider objects are
  duck-typed, and tests use hand-rolled fakes.
- **Capture is fail-open.** Nothing in the capture path may raise into the
  host application or make a network call. Tests must not touch the network.
- **Schema files are generated upstream.** Do **not** hand-edit
  `src/modelgenius_sdk/schema/*.json` in a PR — the schema is exported from
  the ModelGenius canonical models and any drift is caught by an upstream
  contract test. If you believe the schema is wrong, open an issue instead.
- **Examples must validate.** If you touch `examples/`, every row must still
  pass `modelgenius-sdk validate examples/` (CI enforces this), keep a
  resolvable `timestamp`, and stay realistic.
- **Format changes are additive-only** within `modelgenius.snapshot/v1`. See
  the versioning promise in [docs/snapshot_format.md](docs/snapshot_format.md).

## Pull requests

- Keep PRs focused; include tests for behavior changes (happy path *and*
  fail-open paths).
- Update `CHANGELOG.md` under an `Unreleased` heading.
- Security reports do not belong in PRs or issues — see
  [SECURITY.md](SECURITY.md).
