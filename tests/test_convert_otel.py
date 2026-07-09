"""Tests for the OTLP/JSON → snapshot converter and the `convert otlp` CLI.

The fixture (tests/fixtures/otlp_traces.json) is shared verbatim with the
private repo, whose contract tests pin both converter copies to identical
output. Eight spans: content on span attributes, content on the
operation.details event (structured form), structured-output gap, tool calls
with and without definitions, metadata-only span (error), non-GenAI span
(skipped), legacy OpenLLMetry flattened attributes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelgenius_sdk.cli import main
from modelgenius_sdk.convert import convert_otlp_file, convert_otlp_payload, is_otlp_traces

FIXTURE = Path(__file__).parent / "fixtures" / "otlp_traces.json"

SPAN_CHAT = "otel-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331"
SPAN_JSON = "otel-2cf7651916cd43dd8448eb211c80319e-d9ad6b7169203333"
SPAN_TOOLS_MISSING = "otel-4ef7651916cd43dd8448eb211c8031a0-f2ad6b7169203335"


def _fixture_records():
    result = convert_otlp_file(FIXTURE)
    return result, {r["snapshot_id"]: r for r in result.records}


class TestConverter:
    def test_detection(self):
        assert is_otlp_traces({"resourceSpans": []})
        assert not is_otlp_traces({"messages": []})

    def test_fixture_conversion_counts(self):
        result, _ = _fixture_records()
        assert result.payloads == 1
        assert result.converted == 6
        assert result.skipped_non_genai == 1
        assert len(result.errors) == 1
        assert "no message content" in result.errors[0]
        assert "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT" in result.errors[0]

    def test_span_attribute_content_record(self):
        _, by_id = _fixture_records()
        record = by_id[SPAN_CHAT]
        assert record["format"] == "modelgenius.snapshot/v1"
        assert record["model"] == "gpt-4o"
        assert record["temperature"] == 0.2
        assert record["max_tokens"] == 512
        assert record["provider_params"] == {"seed": 42}
        assert record["workflow_id"] == "support_ticket_triage"
        assert [m["role"] for m in record["messages"]] == ["system", "user"]
        co = record["captured_output"]
        assert co["output_text"] == "Category: account-access. Priority: high."
        assert co["usage"]["total_tokens"] == 914
        assert co["latency_ms"] == 1234.0
        assert record["metadata"]["source_format"] == "otel_genai"
        assert record["metadata"]["otel"]["prompt_variables"] == {"customer_tier": "gold"}

    def test_enrichment_flags(self):
        _, by_id = _fixture_records()
        assert by_id[SPAN_JSON]["metadata"]["enrichment_needed"] == ["response_format_schema"]
        assert "response_format" not in by_id[SPAN_JSON]
        assert by_id[SPAN_TOOLS_MISSING]["metadata"]["enrichment_needed"] == ["tool_definitions"]

    def test_jsonl_input(self, tmp_path):
        payload = json.loads(FIXTURE.read_text())
        path = tmp_path / "collector.jsonl"
        path.write_text(json.dumps(payload) + "\n" + json.dumps(payload) + "\n")
        result = convert_otlp_file(path)
        assert result.payloads == 2
        assert result.converted == 12

    def test_non_otlp_input_raises(self, tmp_path):
        path = tmp_path / "native.json"
        path.write_text(json.dumps({"messages": []}))
        with pytest.raises(ValueError, match="not an OTLP trace payload"):
            convert_otlp_file(path)

    def test_non_inference_spans_skipped(self):
        payload = {
            "resourceSpans": [{"scopeSpans": [{"spans": [{
                "name": "embeddings text-embedding-3-small",
                "startTimeUnixNano": "1751500000000000000",
                "attributes": [
                    {"key": "gen_ai.operation.name", "value": {"stringValue": "embeddings"}},
                ],
            }]}]}]
        }
        result = convert_otlp_payload(payload)
        assert result.skipped_non_inference == 1
        assert result.converted == 0 and result.errors == []


class TestConvertCli:
    def test_convert_and_validate_roundtrip(self, tmp_path, capsys):
        out = tmp_path / "snapshots.jsonl"
        code = main(["convert", "otlp", str(FIXTURE), "-o", str(out)])
        captured = capsys.readouterr().out
        # Exit 1: the metadata-only span is a conversion error worth noticing.
        assert code == 1
        assert "Converted: 6 snapshot(s), 1 error(s)" in captured
        assert "skipped 1 non-GenAI span(s)" in captured
        assert "no message content" in captured
        # The converted records themselves validate cleanly...
        assert "Validated: 6/6 snapshot(s) valid" in captured
        # ...and the enrichment gaps surface as warnings.
        assert "response_format JSON schema" in captured
        assert "tool calls but no tool definitions" in captured

        lines = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(lines) == 6
        assert all(line["format"] == "modelgenius.snapshot/v1" for line in lines)

    def test_convert_clean_file_exits_zero(self, tmp_path, capsys):
        payload = json.loads(FIXTURE.read_text())
        # Keep only the first (fully clean) span.
        payload["resourceSpans"] = [payload["resourceSpans"][0]]
        payload["resourceSpans"][0]["scopeSpans"][0]["spans"] = (
            payload["resourceSpans"][0]["scopeSpans"][0]["spans"][:1]
        )
        src = tmp_path / "clean.json"
        src.write_text(json.dumps(payload))
        code = main(["convert", "otlp", str(src)])
        captured = capsys.readouterr().out
        assert code == 0
        assert "All converted snapshots valid." in captured
        default_out = tmp_path / "clean.json.snapshots.jsonl"
        assert default_out.exists()

    def test_convert_json_report(self, tmp_path, capsys):
        out = tmp_path / "snapshots.jsonl"
        code = main(["convert", "otlp", str(FIXTURE), "-o", str(out), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 1
        assert payload["converted"] == 6
        assert payload["skipped_non_genai"] == 1
        assert len(payload["conversion_errors"]) == 1
        assert payload["validation"]["valid_records"] == 6
        assert payload["ok"] is False

    def test_convert_missing_file_exits_two(self, tmp_path, capsys):
        code = main(["convert", "otlp", str(tmp_path / "nope.json")])
        assert code == 2

    def test_convert_non_otlp_exits_two(self, tmp_path, capsys):
        src = tmp_path / "native.json"
        src.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
        code = main(["convert", "otlp", str(src)])
        captured = capsys.readouterr().out
        assert code == 2
        assert "not an OTLP trace payload" in captured
