"""Command-line interface for modelgenius-sdk.

Subcommands:
    modelgenius-sdk validate PATH [--json] [--strict] [--quiet]
    modelgenius-sdk version

Exit codes: 0 = ok; 1 = validation errors (or warnings with --strict);
2 = bad invocation / IO problem (missing path, unsupported file type, empty
directory, unreadable or non-UTF-8 file, argparse errors).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Optional

from modelgenius_sdk.validate.validator import Issue, validate_path

# Issue codes for io-level problems — these exit 2 rather than 1. A directory
# scan continues past an unreadable file (its siblings still validate), but the
# run still exits 2 so automation notices the file that was never read.
_IO_ERROR_CODES = frozenset({"path_not_found", "unsupported_file_type", "no_files_found", "io_error"})


def _resolve_version() -> str:
    try:
        from modelgenius_sdk import __version__  # type: ignore[attr-defined]

        return str(__version__)
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("modelgenius-sdk")
    except Exception:
        return "0.1.0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelgenius-sdk",
        description="Capture and validate ModelGenius snapshot files — locally, no network calls.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate",
        help="Validate snapshot files against the ModelGenius snapshot format.",
    )
    validate.add_argument("path", help="A .json/.jsonl file or a directory of snapshot files.")
    validate.add_argument("--json", action="store_true", help="Emit the validation report as JSON.")
    validate.add_argument(
        "--strict", action="store_true", help="Treat warnings as failures (exit code 1)."
    )
    validate.add_argument(
        "--quiet", action="store_true", help="Suppress text output; rely on the exit code."
    )

    sub.add_parser("version", help="Print the SDK version.")
    return parser


def _issue_location(issue: Issue) -> str:
    if issue.line is not None:
        return f"{issue.file}:{issue.line}"
    if issue.index is not None:
        return f"{issue.file}[{issue.index}]"
    return issue.file


def _cmd_validate(args: argparse.Namespace) -> int:
    report = validate_path(args.path, strict=args.strict)
    ok = report.ok(strict=args.strict)

    if args.json:
        payload = {
            "files": report.files,
            "records": report.records,
            "valid_records": report.valid_records,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "ok": ok,
            "issues": [asdict(issue) for issue in report.issues],
        }
        print(json.dumps(payload, indent=2))
    elif not args.quiet:
        print(f"Validating snapshots from: {args.path}")
        for issue in report.issues:
            if issue.severity == "error":
                print(f"  ERROR {_issue_location(issue)}: {issue.message}")
            else:
                print(f"  WARN  {issue.file}: {issue.message}")
        print(
            f"Loaded: {report.valid_records} snapshot(s), "
            f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
        )
        if not report.errors:
            print("All snapshots valid.")

    if any(issue.code in _IO_ERROR_CODES for issue in report.issues):
        return 2
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(_resolve_version())
        return 0
    return _cmd_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
