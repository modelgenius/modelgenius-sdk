"""The converter on real instrumentation-library output.

Fixtures in tests/fixtures/libraries/ come from the private repo's
real-library set (current Traceloop/OpenLLMetry, OpenInference, and Claude SDK
+ Claude API spans built from Anthropic's span reference).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from modelgenius_sdk.convert import convert_otlp_file

LIB = Path(__file__).parent / "fixtures" / "libraries"


def _by_response_id(result):
    return {r["metadata"]["otel"].get("response_id"): r for r in result.records}


def test_traceloop_calls_convert_cleanly():
    result = convert_otlp_file(LIB / "traceloop__default__spans.jsonl")
    assert result.converted == 7
    assert result.errors == []
    assert result.skipped_duplicate == 3  # the Claude SDK's own copies
    for record in result.records:
        assert record["workflow_id"] == "grading"
        assert record["metadata"]["call_name"] == "grade_essay"


def test_traceloop_structured_output_and_anthropic_tools():
    records = _by_response_id(convert_otlp_file(LIB / "traceloop__default__spans.jsonl"))
    structured = records["chatcmpl-CV9hQ3xYz8KpL2mN4oR6sT05"]
    assert structured["response_format"]["type"] == "json_schema"
    assert structured["response_format"]["json_schema"]["schema"]["required"] == [
        "score",
        "feedback",
    ]
    assert "enrichment_needed" not in structured["metadata"]
    tool = records["msg_01XFDUDYJgAACzvnptvVoY12"]["tools"][0]["function"]
    assert tool["parameters"]["properties"]["criterion"]["enum"] == [
        "thesis",
        "evidence",
        "style",
    ]


def test_claude_sdk_and_api_copies_merge():
    result = convert_otlp_file(LIB / "anthropic_sdk_plus_api.jsonl")
    assert result.errors == []
    assert result.converted == 1
    assert result.skipped_duplicate == 1
    assert result.records[0]["metadata"]["otel"]["span_kind"] == "client"


def test_collector_file_named_json(tmp_path):
    target = tmp_path / "export.json"
    shutil.copy(LIB / "traceloop__default__spans.jsonl", target)
    result = convert_otlp_file(target)
    assert result.converted == 7
    assert result.payloads == 7


def test_binary_export_gets_clear_error(tmp_path):
    target = tmp_path / "export.json"
    shutil.copy(LIB / "traceloop__default__spans.pb", target)
    with pytest.raises(ValueError, match="not UTF-8 text"):
        convert_otlp_file(target)


def test_openinference_reported_not_silent():
    result = convert_otlp_file(LIB / "openinference_openai__default__spans.jsonl")
    assert result.converted == 0
    assert result.unsupported_openinference == 4
    assert len(result.errors) == 1
    assert "OpenInference" in result.errors[0]
