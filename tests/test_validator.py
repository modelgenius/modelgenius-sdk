"""Tests for modelgenius_sdk.validate.validator."""

from __future__ import annotations

import json
import os

import pytest

from modelgenius_sdk.validate.validator import (
    ValidationReport,
    validate_path,
    validate_record,
)

TIMESTAMP_REQUIRED_MESSAGE = (
    "timestamp is required. Provide top-level timestamp "
    "or compatibility metadata.timestamp, metadata.request_timestamp, "
    "or captured_output.request_timestamp."
)


# ---------- record builders ----------


def minimal_openai(**overrides) -> dict:
    record = {
        "messages": [{"role": "user", "content": "Summarize this support ticket."}],
        "timestamp": "2026-07-01T12:00:00+00:00",
    }
    record.update(overrides)
    return record


def captured_output(**overrides) -> dict:
    out = {
        "output_text": "The customer reports a billing discrepancy.",
        "request_timestamp": "2026-07-01T12:00:00Z",
        "response_timestamp": "2026-07-01T12:00:01Z",
    }
    out.update(overrides)
    return out


def anthropic_record(**overrides) -> dict:
    record = {
        "system": "You are a claims-processing assistant.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "File a claim for $120."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "file_claim",
                        "input": {"amount": 120},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}],
            },
        ],
        "max_tokens": 512,
        "timestamp": "2026-07-01T09:30:00Z",
    }
    record.update(overrides)
    return record


def errors_of(issues):
    return [i for i in issues if i.severity == "error"]


def warnings_of(issues):
    return [i for i in issues if i.severity == "warning"]


def codes(issues):
    return [i.code for i in issues]


# ---------- validate_record: shapes ----------


def test_valid_minimal_openai_record():
    issues = validate_record(minimal_openai())
    assert errors_of(issues) == []
    warn_codes = codes(warnings_of(issues))
    assert "no_captured_output" in warn_codes
    assert "no_model" in warn_codes
    assert "no_grouping_keys" in warn_codes


def test_valid_anthropic_record_with_tool_use():
    issues = validate_record(anthropic_record())
    assert errors_of(issues) == []


def test_anthropic_detection_via_tool_use_block_without_system():
    # No top-level system: only the tool_use block can trigger Anthropic
    # detection. Role "system" is fine for OpenAI shape but rejected by the
    # Anthropic structural checks — so an anthropic_role error proves the
    # detection fired on the block type.
    record = {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            }
        ],
        "timestamp": "2026-07-01T12:00:00Z",
    }
    issues = validate_record(record)
    assert "anthropic_role" in codes(errors_of(issues))


def test_anthropic_bad_role_rejected():
    record = anthropic_record()
    record["messages"] = [{"role": "tool", "content": [{"type": "text", "text": "hi"}]}]
    issues = validate_record(record)
    assert "anthropic_role" in codes(errors_of(issues))


def test_openai_schema_error_bad_role():
    record = minimal_openai(messages=[{"role": "banana", "content": "hi"}])
    issues = validate_record(record)
    errs = errors_of(issues)
    assert codes(errs) == ["schema"]
    assert "/messages/0/role" in errs[0].message


# ---------- timestamps ----------


def test_missing_timestamp_error_message_verbatim():
    record = minimal_openai()
    del record["timestamp"]
    issues = validate_record(record)
    errs = errors_of(issues)
    assert codes(errs) == ["timestamp_missing"]
    assert errs[0].message == TIMESTAMP_REQUIRED_MESSAGE


def test_timestamp_only_in_metadata_request_timestamp_is_valid():
    record = minimal_openai()
    del record["timestamp"]
    record["metadata"] = {"request_timestamp": "2026-07-01T12:00:00+00:00"}
    assert errors_of(validate_record(record)) == []


def test_timestamp_only_in_metadata_timestamp_is_valid():
    record = minimal_openai()
    del record["timestamp"]
    record["metadata"] = {"timestamp": "2026-07-01T12:00:00Z"}
    assert errors_of(validate_record(record)) == []


def test_timestamp_from_captured_output_request_timestamp_is_valid():
    record = minimal_openai(captured_output=captured_output())
    del record["timestamp"]
    assert errors_of(validate_record(record)) == []


def test_bad_iso_timestamp_is_timestamp_invalid():
    record = minimal_openai(timestamp="last tuesday-ish")
    issues = validate_record(record)
    errs = errors_of(issues)
    assert "timestamp_invalid" in codes(errs)
    # present-but-invalid still counts as resolved: no timestamp_missing
    assert "timestamp_missing" not in codes(errs)


def test_z_suffix_timestamp_accepted():
    record = minimal_openai(timestamp="2026-07-01T12:00:00Z")
    assert errors_of(validate_record(record)) == []


def test_bad_response_timestamp_in_captured_output_flagged():
    record = minimal_openai(
        captured_output=captured_output(response_timestamp="soon")
    )
    issues = validate_record(record)
    errs = errors_of(issues)
    assert "timestamp_invalid" in codes(errs)
    assert any("captured_output.response_timestamp" in e.message for e in errs)


# ---------- lax coercion (loader parity) ----------


def test_epoch_int_timestamp_accepted():
    assert errors_of(validate_record(minimal_openai(timestamp=1719859200))) == []


def test_epoch_float_timestamp_accepted():
    assert errors_of(validate_record(minimal_openai(timestamp=1719859200.5))) == []


def test_epoch_timestamp_in_metadata_resolves():
    record = minimal_openai()
    del record["timestamp"]
    record["metadata"] = {"request_timestamp": 1719859200}
    assert errors_of(validate_record(record)) == []


def test_epoch_timestamps_in_captured_output_accepted():
    record = minimal_openai(
        captured_output={
            "output_text": "ok",
            "request_timestamp": 1719859200,
            "response_timestamp": 1719859201.25,
        }
    )
    del record["timestamp"]  # resolvable from captured_output.request_timestamp
    assert errors_of(validate_record(record)) == []


def test_bool_timestamp_rejected():
    # Bools are not epochs: the loader rejects them for datetime fields.
    errs = errors_of(validate_record(minimal_openai(timestamp=True)))
    assert "timestamp_invalid" in codes(errs)


def test_numeric_string_params_accepted():
    record = minimal_openai(
        temperature="0.7",
        top_p="0.9",
        frequency_penalty="0.1",
        presence_penalty="0.2",
        max_tokens="512",
        n="3",
    )
    assert errors_of(validate_record(record)) == []


def test_integer_field_float_string_accepted():
    # Pydantic lax int parsing accepts "512.0" (but not "512.5") — mirror it.
    assert errors_of(validate_record(minimal_openai(max_tokens="512.0"))) == []
    errs = errors_of(validate_record(minimal_openai(max_tokens="512.5")))
    assert codes(errs) == ["schema"]


def test_unparseable_numeric_string_still_rejected():
    errs = errors_of(validate_record(minimal_openai(temperature="warm")))
    assert codes(errs) == ["schema"]


def test_numeric_string_captured_output_fields_accepted():
    record = minimal_openai(
        captured_output=captured_output(
            usage={"input_tokens": "21", "output_tokens": "9", "total_tokens": "30"},
            cost_usd="0.003",
            latency_ms="12.5",
        )
    )
    assert errors_of(validate_record(record)) == []


def test_unparseable_usage_string_still_rejected():
    record = minimal_openai(captured_output=captured_output(usage={"input_tokens": "lots"}))
    errs = errors_of(validate_record(record))
    assert codes(errs) == ["schema"]


def test_coercion_never_mutates_the_input_record():
    record = minimal_openai(
        timestamp=1719859200,
        temperature="0.7",
        metadata={"request_timestamp": 1719859200},
        captured_output=captured_output(usage={"input_tokens": "21"}, cost_usd="0.003"),
    )
    validate_record(record)
    assert record["timestamp"] == 1719859200
    assert record["temperature"] == "0.7"
    assert record["metadata"]["request_timestamp"] == 1719859200
    assert record["captured_output"]["usage"]["input_tokens"] == "21"
    assert record["captured_output"]["cost_usd"] == "0.003"


# ---------- schema error pointer rendering ----------


def test_schema_pointer_bad_temperature_rendered_once():
    errs = errors_of(validate_record(minimal_openai(temperature="warm")))
    assert len(errs) == 1
    assert errs[0].message == "/temperature: 'warm' is not of type 'number'"


def test_schema_pointer_bad_usage_input_tokens_rendered_once():
    record = minimal_openai(captured_output=captured_output(usage={"input_tokens": "lots"}))
    errs = errors_of(validate_record(record))
    assert len(errs) == 1
    assert errs[0].message == "/captured_output/usage/input_tokens: 'lots' is not of type 'integer'"


def test_schema_pointer_anthropic_captured_output_usage_rendered_once():
    record = anthropic_record(captured_output=captured_output(usage={"input_tokens": "lots"}))
    errs = errors_of(validate_record(record))
    assert len(errs) == 1
    assert errs[0].message == "/captured_output/usage/input_tokens: 'lots' is not of type 'integer'"


# ---------- anthropic structural checks ----------


def test_anthropic_non_dict_metadata_rejected():
    issues = validate_record(anthropic_record(metadata="xy"))
    assert "anthropic_metadata" in codes(errors_of(issues))


def test_anthropic_dict_metadata_accepted():
    issues = validate_record(anthropic_record(metadata={"workflow_name": "claims"}))
    assert errors_of(issues) == []


def test_anthropic_tools_non_object_entry_rejected():
    issues = validate_record(anthropic_record(tools=[123]))
    errs = errors_of(issues)
    assert "anthropic_tools" in codes(errs)
    assert any("/tools/0" in e.message for e in errs)


def test_anthropic_tools_list_of_objects_accepted():
    tools = [{"name": "file_claim", "input_schema": {"type": "object", "properties": {}}}]
    assert errors_of(validate_record(anthropic_record(tools=tools))) == []


def test_anthropic_non_list_tools_ignored():
    # The server normalizer silently drops a non-list tools value; mirror it.
    assert errors_of(validate_record(anthropic_record(tools="file_claim"))) == []


def test_anthropic_text_block_non_string_text_rejected():
    record = anthropic_record()
    record["messages"] = [{"role": "user", "content": [{"type": "text", "text": 123}]}]
    errs = errors_of(validate_record(record))
    assert "anthropic_text_block" in codes(errs)
    assert any("/messages/0/content/0/text" in e.message for e in errs)


def test_anthropic_temperature_unparseable_rejected():
    issues = validate_record(anthropic_record(temperature="warm"))
    assert "anthropic_param" in codes(errors_of(issues))


def test_anthropic_numeric_string_params_accepted():
    record = anthropic_record(temperature="0.7", top_p="0.9", max_tokens="512")
    assert errors_of(validate_record(record)) == []


def test_anthropic_max_tokens_non_integer_rejected():
    issues = validate_record(anthropic_record(max_tokens=512.5))
    assert "anthropic_param" in codes(errors_of(issues))


def test_anthropic_openai_only_params_ignored():
    # frequency_penalty / presence_penalty / n never flow into the canonical
    # Snapshot on the anthropic path — the loader ignores them, so must we.
    record = anthropic_record(frequency_penalty="warm", presence_penalty="cold", n="banana")
    assert errors_of(validate_record(record)) == []


def test_anthropic_stop_sequences_shapes():
    assert errors_of(validate_record(anthropic_record(stop_sequences="END"))) == []
    assert errors_of(validate_record(anthropic_record(stop_sequences=["END", "STOP"]))) == []
    issues = validate_record(anthropic_record(stop_sequences=[1]))
    assert "anthropic_stop_sequences" in codes(errors_of(issues))
    issues = validate_record(anthropic_record(stop_sequences=5))
    assert "anthropic_stop_sequences" in codes(errors_of(issues))


def test_anthropic_epoch_timestamp_accepted():
    assert errors_of(validate_record(anthropic_record(timestamp=1719859200))) == []


# ---------- captured_output ----------


def test_captured_output_missing_response_timestamp_openai_is_schema_error():
    co = captured_output()
    del co["response_timestamp"]
    record = minimal_openai(captured_output=co)
    issues = validate_record(record)
    errs = errors_of(issues)
    assert codes(errs) == ["schema"]
    assert "response_timestamp" in errs[0].message


def test_captured_output_missing_response_timestamp_anthropic_is_schema_error():
    co = captured_output()
    del co["response_timestamp"]
    record = anthropic_record(captured_output=co)
    issues = validate_record(record)
    errs = errors_of(issues)
    assert codes(errs) == ["schema"]
    assert "/captured_output" in errs[0].message
    assert "response_timestamp" in errs[0].message


def test_captured_output_with_usage_and_cost_no_warnings():
    record = minimal_openai(
        model="gpt-4o",
        workflow_id="wf-1",
        captured_output=captured_output(
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            cost_usd=0.0021,
        ),
    )
    issues = validate_record(record)
    assert errors_of(issues) == []
    warn_codes = codes(warnings_of(issues))
    for code in ("no_captured_output", "no_usage", "no_cost", "no_model", "no_grouping_keys"):
        assert code not in warn_codes


def test_captured_output_without_usage_and_cost_warns():
    record = minimal_openai(captured_output=captured_output())
    warn_codes = codes(warnings_of(validate_record(record)))
    assert "no_usage" in warn_codes
    assert "no_cost" in warn_codes
    assert "no_captured_output" not in warn_codes


# ---------- envelope ----------


def test_unknown_extra_top_level_key_accepted():
    record = minimal_openai(my_custom_field={"anything": [1, 2, 3]}, seed=42)
    assert errors_of(validate_record(record)) == []


def test_wrong_format_tag_is_single_format_tag_error():
    record = minimal_openai(format="modelgenius.snapshot/v2")
    errs = errors_of(validate_record(record))
    assert codes(errs) == ["format_tag"]  # no duplicate schema error at /format


def test_correct_format_tag_accepted():
    record = minimal_openai(format="modelgenius.snapshot/v1")
    assert errors_of(validate_record(record)) == []


def test_empty_messages_warns():
    record = minimal_openai(messages=[])
    issues = validate_record(record)
    assert errors_of(issues) == []
    assert "empty_messages" in codes(warnings_of(issues))


def test_non_dict_record_rejected():
    issues = validate_record([1, 2])  # type: ignore[arg-type]
    assert codes(issues) == ["invalid_record"]
    assert issues[0].severity == "error"


def test_issue_carries_file_line_index():
    record = minimal_openai()
    del record["timestamp"]
    issues = validate_record(record, file="spool.jsonl", line=7, index=3)
    err = errors_of(issues)[0]
    assert (err.file, err.line, err.index) == ("spool.jsonl", 7, 3)


# ---------- validate_path: files & directories ----------


def test_json_single_object(tmp_path):
    p = tmp_path / "one.json"
    p.write_text(json.dumps(minimal_openai()))
    report = validate_path(p)
    assert (report.files, report.records, report.valid_records) == (1, 1, 1)
    assert report.errors == []
    assert report.ok()


def test_json_array_with_error_index(tmp_path):
    bad = minimal_openai()
    del bad["timestamp"]
    p = tmp_path / "batch.json"
    p.write_text(json.dumps([minimal_openai(), bad]))
    report = validate_path(p)
    assert (report.files, report.records, report.valid_records) == (1, 2, 1)
    errs = report.errors
    assert len(errs) == 1
    assert errs[0].code == "timestamp_missing"
    assert errs[0].line is None
    assert errs[0].index == 1


def test_jsonl_file_skips_blank_lines(tmp_path):
    p = tmp_path / "spool.jsonl"
    p.write_text(
        json.dumps(minimal_openai()) + "\n\n" + json.dumps(anthropic_record()) + "\n"
    )
    report = validate_path(p)
    assert (report.files, report.records, report.valid_records) == (1, 2, 2)
    assert report.errors == []


def test_jsonl_invalid_json_line(tmp_path):
    p = tmp_path / "spool.jsonl"
    p.write_text(json.dumps(minimal_openai()) + "\n{not json}\n" + json.dumps(minimal_openai()) + "\n")
    report = validate_path(p)
    assert (report.records, report.valid_records) == (2, 2)
    errs = report.errors
    assert len(errs) == 1
    assert errs[0].code == "invalid_json"
    assert errs[0].line == 2
    assert "invalid JSON" in errs[0].message


def test_invalid_json_whole_file(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{oops")
    report = validate_path(p)
    assert report.records == 0
    assert [i.code for i in report.errors] == ["invalid_json"]


def test_directory_json_then_jsonl(tmp_path):
    (tmp_path / "b.jsonl").write_text(json.dumps(minimal_openai()) + "\n")
    (tmp_path / "a.json").write_text(json.dumps(minimal_openai()))
    (tmp_path / "notes.txt").write_text("ignore me")
    report = validate_path(tmp_path)
    assert (report.files, report.records, report.valid_records) == (2, 2, 2)
    assert report.errors == []
    # .json files validate before .jsonl (mirrors the private loader ordering):
    # the aggregated warnings come out grouped per file in that order.
    warn_files = [i.file for i in report.warnings]
    assert warn_files == sorted(warn_files, key=lambda f: (f.endswith(".jsonl"), f))


def test_directory_empty(tmp_path):
    report = validate_path(tmp_path)
    assert [i.code for i in report.errors] == ["no_files_found"]
    assert f"No .json or .jsonl files found in {tmp_path}" in report.errors[0].message


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "snapshots.txt"
    p.write_text("hello")
    report = validate_path(p)
    assert [i.code for i in report.errors] == ["unsupported_file_type"]
    assert report.errors[0].message == "Unsupported file type: .txt"


def test_missing_path(tmp_path):
    p = tmp_path / "nope.jsonl"
    report = validate_path(p)
    assert [i.code for i in report.errors] == ["path_not_found"]
    assert report.errors[0].message == f"Path does not exist: {p}"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_file_is_io_error_not_crash(tmp_path):
    p = tmp_path / "locked.jsonl"
    p.write_text(json.dumps(minimal_openai()) + "\n")
    p.chmod(0)
    report = validate_path(p)
    assert [i.code for i in report.errors] == ["io_error"]
    assert report.errors[0].file == str(p)
    assert "cannot read file" in report.errors[0].message
    assert report.records == 0


def test_non_utf8_file_is_io_error_not_crash(tmp_path):
    p = tmp_path / "binary.jsonl"
    p.write_bytes(b"\xff\xfe\x00 not utf-8")
    report = validate_path(p)
    assert [i.code for i in report.errors] == ["io_error"]
    assert "cannot read file" in report.errors[0].message


def test_directory_continues_past_unreadable_file(tmp_path):
    bad = tmp_path / "a.jsonl"
    bad.write_bytes(b"\xff\xfe")
    good = tmp_path / "b.jsonl"
    good.write_text(json.dumps(minimal_openai()) + "\n")
    report = validate_path(tmp_path)
    assert report.files == 2
    assert report.valid_records == 1
    assert [i.code for i in report.errors] == ["io_error"]
    assert report.errors[0].file == str(bad)


# ---------- warning aggregation ----------


def test_warnings_aggregated_one_per_code_per_file_with_counts(tmp_path):
    p = tmp_path / "spool.jsonl"
    p.write_text("".join(json.dumps(minimal_openai()) + "\n" for _ in range(3)))
    report = validate_path(p)
    assert report.errors == []
    no_co = [i for i in report.warnings if i.code == "no_captured_output"]
    assert len(no_co) == 1
    assert no_co[0].message.startswith("3 record(s)")
    assert no_co[0].file == str(p)
    no_model = [i for i in report.warnings if i.code == "no_model"]
    assert len(no_model) == 1
    assert "3 record(s)" in no_model[0].message
    # aggregated warnings are file-level: no line/index
    assert all(i.line is None and i.index is None for i in report.warnings)


def test_warnings_aggregated_separately_per_file(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps(minimal_openai()) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(minimal_openai()) + "\n")
    report = validate_path(tmp_path)
    no_co = [i for i in report.warnings if i.code == "no_captured_output"]
    assert len(no_co) == 2
    assert {i.file for i in no_co} == {str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")}


# ---------- report semantics ----------


def test_report_ok_strict_flips_on_warnings(tmp_path):
    p = tmp_path / "one.jsonl"
    p.write_text(json.dumps(minimal_openai()) + "\n")
    report = validate_path(p)
    assert report.warnings  # minimal record triggers unlock nudges
    assert report.ok() is True
    assert report.ok(strict=True) is False


def test_report_ok_false_on_errors():
    report = ValidationReport()
    record = minimal_openai()
    del record["timestamp"]
    report.issues.extend(validate_record(record))
    assert report.ok() is False


@pytest.mark.parametrize("strict", [False, True])
def test_validate_path_strict_kwarg_accepted(tmp_path, strict):
    p = tmp_path / "one.json"
    p.write_text(json.dumps(minimal_openai()))
    report = validate_path(p, strict=strict)
    assert report.records == 1


# ---------- reference answers ----------


@pytest.mark.parametrize(
    "reference",
    [
        {"output": "billing"},
        {"output": {"category": "billing"}},
        {"values": {"category": "billing", "priority": "high"}, "importance": "must_pass"},
        {"output": "billing", "notes": "case-insensitive", "source": "human labels", "importance": "ranking"},
    ],
)
def test_a_reference_answer_is_accepted_on_either_shape(reference):
    assert errors_of(validate_record(minimal_openai(reference=reference))) == []
    record = anthropic_record()
    record["reference"] = reference
    assert errors_of(validate_record(record)) == []


@pytest.mark.parametrize(
    "reference",
    [
        {},
        {"notes": "no answer given"},
        {"output": ""},
        {"values": {}},
        {"output": "billing", "importance": "critical"},
        "billing",
    ],
)
def test_a_reference_without_an_answer_or_with_a_bad_importance_is_an_error(reference):
    assert errors_of(validate_record(minimal_openai(reference=reference)))
    record = anthropic_record()
    record["reference"] = reference
    assert errors_of(validate_record(record))


def test_the_reference_example_file_validates_cleanly():
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "examples" / "reference.jsonl"
    report = validate_path(path)
    assert report.records == 2
    assert report.errors == [], [str(i) for i in report.errors]
