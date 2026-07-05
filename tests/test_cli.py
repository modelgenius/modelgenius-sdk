"""Tests for the modelgenius-sdk CLI (validate / version subcommands)."""

from __future__ import annotations

import json
import os
import re

import pytest

from modelgenius_sdk.cli import main
from modelgenius_sdk.validate.validator import validate_path


def minimal_openai(**overrides) -> dict:
    record = {
        "messages": [{"role": "user", "content": "Draft a renewal reminder email."}],
        "timestamp": "2026-07-01T12:00:00+00:00",
    }
    record.update(overrides)
    return record


@pytest.fixture
def valid_file(tmp_path):
    p = tmp_path / "spool.jsonl"
    p.write_text(json.dumps(minimal_openai()) + "\n")
    return p


@pytest.fixture
def error_file(tmp_path):
    record = minimal_openai()
    del record["timestamp"]
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(record) + "\n")
    return p


def test_validate_valid_file_exit0_and_text_output(valid_file, capsys):
    rc = main(["validate", str(valid_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"Validating snapshots from: {valid_file}" in out
    assert "Loaded: 1 snapshot(s), 0 error(s)" in out
    assert "All snapshots valid." in out


def test_validate_warn_lines_present(valid_file, capsys):
    main(["validate", str(valid_file)])
    out = capsys.readouterr().out
    assert f"  WARN  {valid_file}: " in out
    assert "no captured_output" in out


def test_validate_error_file_exit1_and_error_line(error_file, capsys):
    rc = main(["validate", str(error_file)])
    out = capsys.readouterr().out
    assert rc == 1
    assert f"  ERROR {error_file}:1: timestamp is required." in out
    assert "All snapshots valid." not in out
    assert "Loaded: 0 snapshot(s), 1 error(s)" in out


def test_strict_flips_warnings_to_failure(valid_file):
    assert main(["validate", str(valid_file)]) == 0
    assert main(["validate", str(valid_file), "--strict"]) == 1


def test_strict_on_warning_free_file_passes(tmp_path):
    record = minimal_openai(
        model="gpt-4o",
        workflow_id="wf-billing",
        captured_output={
            "output_text": "Done.",
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            "cost_usd": 0.0001,
            "request_timestamp": "2026-07-01T12:00:00Z",
            "response_timestamp": "2026-07-01T12:00:01Z",
        },
    )
    p = tmp_path / "full.jsonl"
    p.write_text(json.dumps(record) + "\n")
    assert main(["validate", str(p), "--strict"]) == 0


def test_json_output_matches_report(valid_file, capsys):
    rc = main(["validate", str(valid_file), "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    report = validate_path(valid_file)
    assert rc == 0
    assert payload["files"] == report.files == 1
    assert payload["records"] == report.records == 1
    assert payload["valid_records"] == report.valid_records == 1
    assert payload["errors"] == len(report.errors) == 0
    assert payload["warnings"] == len(report.warnings)
    assert payload["ok"] is True
    assert len(payload["issues"]) == len(report.issues)
    assert {i["code"] for i in payload["issues"]} == {i.code for i in report.issues}
    issue_keys = set(payload["issues"][0].keys())
    assert issue_keys == {"file", "line", "index", "severity", "code", "message"}


def test_json_output_error_file(error_file, capsys):
    rc = main(["validate", str(error_file), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert payload["errors"] == 1
    assert any(i["code"] == "timestamp_missing" for i in payload["issues"])


def test_json_strict_reflects_in_ok(valid_file, capsys):
    rc = main(["validate", str(valid_file), "--json", "--strict"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False


def test_quiet_suppresses_all_text_output(valid_file, error_file, capsys):
    rc = main(["validate", str(valid_file), "--quiet"])
    assert rc == 0
    assert capsys.readouterr().out == ""
    rc = main(["validate", str(error_file), "--quiet"])
    assert rc == 1
    assert capsys.readouterr().out == ""


def test_missing_path_exit2(tmp_path, capsys):
    rc = main(["validate", str(tmp_path / "nope.jsonl")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "Path does not exist" in out


def test_unsupported_suffix_exit2(tmp_path):
    p = tmp_path / "spool.txt"
    p.write_text("hello")
    assert main(["validate", str(p)]) == 2


def test_empty_directory_exit2(tmp_path):
    assert main(["validate", str(tmp_path)]) == 2


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_file_exit2_no_traceback(tmp_path, capsys):
    p = tmp_path / "locked.jsonl"
    p.write_text(json.dumps(minimal_openai()) + "\n")
    p.chmod(0)
    rc = main(["validate", str(p)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "cannot read file" in out


def test_non_utf8_file_exit2_no_traceback(tmp_path, capsys):
    p = tmp_path / "binary.jsonl"
    p.write_bytes(b"\xff\xfe{} not utf-8")
    rc = main(["validate", str(p)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "cannot read file" in out


def test_directory_with_unreadable_file_continues_and_exit2(tmp_path, capsys):
    (tmp_path / "a.jsonl").write_bytes(b"\xff\xfe")
    (tmp_path / "b.jsonl").write_text(json.dumps(minimal_openai()) + "\n")
    rc = main(["validate", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["files"] == 2
    assert payload["valid_records"] == 1  # the readable file still validates
    assert any(i["code"] == "io_error" for i in payload["issues"])


def test_version_prints_version(capsys):
    rc = main(["version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert re.match(r"^\d+\.\d+\.\d+", out)


def test_no_subcommand_is_bad_invocation_exit2():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_unknown_subcommand_exit2():
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate"])
    assert excinfo.value.code == 2
