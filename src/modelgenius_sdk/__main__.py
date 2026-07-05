"""Allow `python -m modelgenius_sdk` to run the CLI."""

from __future__ import annotations

from modelgenius_sdk.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
