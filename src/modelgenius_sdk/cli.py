"""Command-line interface for modelgenius-sdk.

Subcommands:
    modelgenius-sdk validate PATH [--json] [--strict] [--quiet]
    modelgenius-sdk convert otlp PATH [-o OUTPUT.jsonl] [--json] [--quiet]
    modelgenius-sdk version

Exit codes: 0 = ok; 1 = validation/conversion errors (or warnings with
--strict); 2 = bad invocation / IO problem (missing path, unsupported file
type, empty directory, unreadable or non-UTF-8 file, argparse errors).
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

    convert = sub.add_parser(
        "convert",
        help="Convert foreign trace exports to snapshot JSONL (local, no network).",
    )
    convert_sub = convert.add_subparsers(dest="convert_format", required=True)
    otlp = convert_sub.add_parser(
        "otlp",
        help=(
            "Convert an OTLP/JSON trace export (OpenTelemetry GenAI semantic "
            "conventions) to snapshot JSONL, then validate the result."
        ),
    )
    otlp.add_argument(
        "path",
        help="A .json OTLP export (one payload or an array) or a collector .jsonl file.",
    )
    otlp.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output snapshot JSONL path (default: <input>.snapshots.jsonl).",
    )
    otlp.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    otlp.add_argument(
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


def _cmd_convert_otlp(args: argparse.Namespace) -> int:
    from pathlib import Path

    from modelgenius_sdk.convert import convert_otlp_file

    input_path = Path(args.path)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_suffix(input_path.suffix + ".snapshots.jsonl")
    )

    try:
        result = convert_otlp_file(input_path)
    except OSError as e:
        if not args.quiet and not args.json:
            print(f"ERROR: cannot read {input_path}: {e}")
        elif args.json:
            print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        return 2
    except ValueError as e:
        if not args.quiet and not args.json:
            print(f"ERROR: {e}")
        elif args.json:
            print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        return 2

    with open(output_path, "w", encoding="utf-8") as f:
        for record in result.records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = validate_path(str(output_path)) if result.records else None
    validation_ok = report.ok() if report is not None else True
    ok = validation_ok and not result.errors and bool(result.records)

    if args.json:
        payload = {
            "input": str(input_path),
            "output": str(output_path),
            "payloads": result.payloads,
            "converted": result.converted,
            "skipped_non_genai": result.skipped_non_genai,
            "skipped_non_inference": result.skipped_non_inference,
            "conversion_errors": result.errors,
            "validation": (
                {
                    "records": report.records,
                    "valid_records": report.valid_records,
                    "errors": len(report.errors),
                    "warnings": len(report.warnings),
                    "issues": [asdict(issue) for issue in report.issues],
                }
                if report is not None
                else None
            ),
            "ok": ok,
        }
        print(json.dumps(payload, indent=2))
    elif not args.quiet:
        print(f"Converting OTLP traces from: {input_path}")
        for message in result.errors:
            print(f"  ERROR {message}")
        skipped_bits = []
        if result.skipped_non_genai:
            skipped_bits.append(f"{result.skipped_non_genai} non-GenAI span(s)")
        if result.skipped_non_inference:
            skipped_bits.append(f"{result.skipped_non_inference} non-inference GenAI span(s)")
        skipped_note = f" (skipped {', '.join(skipped_bits)})" if skipped_bits else ""
        print(
            f"Converted: {result.converted} snapshot(s), "
            f"{len(result.errors)} error(s){skipped_note}"
        )
        print(f"Wrote: {output_path}")
        if report is not None:
            for issue in report.issues:
                if issue.severity == "error":
                    print(f"  ERROR {_issue_location(issue)}: {issue.message}")
                else:
                    print(f"  WARN  {issue.message}")
            print(
                f"Validated: {report.valid_records}/{report.records} snapshot(s) valid, "
                f"{len(report.warnings)} warning(s)"
            )
        if not result.records:
            print("No snapshots produced — see errors above.")
        elif ok:
            print("All converted snapshots valid.")

    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(_resolve_version())
        return 0
    if args.command == "convert":
        return _cmd_convert_otlp(args)
    return _cmd_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
