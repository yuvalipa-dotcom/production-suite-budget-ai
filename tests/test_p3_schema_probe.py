"""The P3 schema probe, offline: the Anthropic API is mocked at the httpx transport. ALL DATA IS SYNTHETIC.

The tool definitions a Crew 1 agent sends (captured from a real, mocked crew kickoff) and the ones the
probe sends are identical, and equal the committed reference the probe checks at run time.
"""

from __future__ import annotations

import builtins
import io
import json
import tempfile
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

import crew1.tools as T
import scripts.p3_schema_probe as P
from crew1.tools import eda as E
from crew1.tools.workspace import REPO
from tests.crew1_mock_anthropic import (DUMMY_KEY, MockAnthropic, api_error, correct_plans,  # noqa: F401
                                        no_external_network, run_mocked_crew, tool_use)

pytestmark = pytest.mark.usefixtures("no_external_network")
EXPECTED = json.loads(P.EXPECTED_TOOLS.read_text(encoding="utf-8"))
SECTIONS_OK = {s: f"Synthetic placeholder sentence for the {s} section." for s in E.SECTIONS}
DESCRIPTIONS_OK = {"column_descriptions": [{"column": c, "description": f"Synthetic description of {c}."}
                                           for c in P.DESCRIBED]}


def responder(report=None, contract=None, first_status=None):
    def respond(body, n):
        if first_status and n == 1:
            return api_error(first_status)
        name = body["tools"][0]["name"]
        if name == "assemble_eda_report":
            return tool_use([(name, SECTIONS_OK if report is None else report)], usage=(900, 300))
        return tool_use([(name, DESCRIPTIONS_OK if contract is None else contract)], usage=(700, 120))
    return respond


def run_probe(monkeypatch, tmp_path, respond, approve=True, key=True):
    if approve:
        monkeypatch.setenv(P.APPROVAL_ENV, P.APPROVAL)
    else:
        monkeypatch.delenv(P.APPROVAL_ENV, raising=False)
    if key:
        monkeypatch.setenv(P.KEY_ENV, DUMMY_KEY)
    else:
        monkeypatch.delenv(P.KEY_ENV, raising=False)
    mock = MockAnthropic(respond).install(monkeypatch)
    code, out = P.run_probe(out_root=tmp_path / "probe_output")
    result = json.loads((out / "probe_result.json").read_text(encoding="utf-8")) if out else None
    return code, result, mock, out


def test_the_probe_refuses_without_approval_or_key_and_sends_nothing(monkeypatch, tmp_path):
    code, result, mock, _ = run_probe(monkeypatch, tmp_path, responder(), approve=False)
    assert (code, result, mock.requests) == (2, None, [])
    monkeypatch.setenv(P.APPROVAL_ENV, "approve p3 schema probe")                  # not the exact phrase
    assert P.run_probe(out_root=tmp_path / "x") == (2, None) and mock.requests == []
    code, result, mock, _ = run_probe(monkeypatch, tmp_path, responder(), key=False)
    assert (code, result, mock.requests) == (2, None, [])
    assert not (tmp_path / "probe_output").exists()


def test_success_path_passes_with_exactly_two_forced_requests(monkeypatch, tmp_path):
    code, result, mock, _ = run_probe(monkeypatch, tmp_path, responder())
    assert code == 0 and result["result"] == "PASS", result["criteria"]
    assert all(c["status"] == "PASS" for c in result["criteria"].values())
    assert len(mock.requests) == 2
    assert [b["tool_choice"] for b in mock.requests] == [{"type": "tool", "name": n} for n in P.PROBED]
    assert [len(b["tools"]) for b in mock.requests] == [1, 1]
    for b in mock.requests:
        assert (b["model"], b["max_tokens"], b["temperature"], b["stream"]) == (
            "claude-haiku-4-5-20251001", 1500, 0.0, False)
    assert result["usage"] == {"input_tokens": 1600, "output_tokens": 420, "successful_requests": 2}
    assert result["cost_estimate"]["estimated_usd"] == round((1600 * 1.0 + 420 * 5.0) / 1e6, 6)
    assert "docs.claude.com" in result["cost_estimate"]["source"]


def test_agent_and_probe_send_identical_tool_definitions(monkeypatch, tmp_path):
    crew, script, agent_mock, _, ws, error = run_mocked_crew(monkeypatch, correct_plans())
    ws.cleanup()
    assert error is None and script.forced_final == {k: 0 for k in script.forced_final}
    _, result, probe_mock, _ = run_probe(monkeypatch, tmp_path, responder())
    for name in P.PROBED:
        agent_defs, probe_defs = agent_mock.tool_definitions(name), probe_mock.tool_definitions(name)
        assert agent_defs and len(probe_defs) == 1
        assert all(d == agent_defs[0] for d in agent_defs)            # the same in every request of the agent
        assert probe_defs[0] == agent_defs[0] == EXPECTED[name], name
        assert result["sent_tool_definitions"][name] == [EXPECTED[name]]
    report = EXPECTED["assemble_eda_report"]
    assert report["strict"] is True and report["input_schema"]["additionalProperties"] is False
    assert sorted(report["input_schema"]["required"]) == sorted(E.SECTIONS)
    assert all(p["type"] == "string" for p in report["input_schema"]["properties"].values())
    contract = EXPECTED["build_dataset_contract"]["input_schema"]["properties"]["column_descriptions"]
    array = next(x for x in contract["anyOf"] if x["type"] == "array")
    assert sorted(array["items"]["required"]) == ["column", "description"]
    assert array["items"]["additionalProperties"] is False


def test_the_old_empty_object_failure_is_reported_as_fail(monkeypatch, tmp_path):
    code, result, mock, _ = run_probe(monkeypatch, tmp_path, responder(report={"sections": {}}))
    c = result["criteria"]
    assert code == 1 and result["result"] == "FAIL" and len(mock.requests) == 2
    assert c["report_nine_sections_nonempty"]["status"] == "FAIL"
    assert c["report_tool_accepts_and_writes"]["status"] == "FAIL"
    assert c["contract_three_descriptions_nonempty"]["status"] == "PASS"
    assert "validation failed" in result["calls"][0]["tool_error"]


def test_a_tool_definition_unlike_the_agents_is_reported_as_fail(monkeypatch, tmp_path):
    class OldReportArgs(BaseModel):                                          # the pre-p3-live-003 argument
        sections: dict[str, str]

    monkeypatch.setitem(T.SPECS, "assemble_eda_report", (OldReportArgs, T.SPECS["assemble_eda_report"][1]))
    code, result, _, _ = run_probe(monkeypatch, tmp_path, responder(report={"sections": {}}))
    c = result["criteria"]
    assert code == 1 and c["tool_definitions_match_crew1_agents"]["status"] == "FAIL"
    assert c["tool_definitions_match_crew1_agents"]["detail"] == {"assemble_eda_report": "DIFFERENT",
                                                                 "build_dataset_contract": "identical"}
    assert c["report_nine_sections_nonempty"]["status"] == "FAIL"


def test_retries_are_off_at_every_layer(monkeypatch, tmp_path):
    code, result, mock, _ = run_probe(monkeypatch, tmp_path, responder(first_status=500))
    assert code == 1 and len(mock.requests) == 2                      # the failed request is not retried
    assert [b["tools"][0]["name"] for b in mock.requests] == list(P.PROBED)
    s = result["settings"]
    assert (s["max_retries"], s["sdk_max_retries"], s["litellm_installed"]) == (0, 0, False)
    assert "500" in result["calls"][0]["error"] and result["criteria"]["report_response_is_tool_call"][
        "status"] == "FAIL"
    assert result["criteria"]["retries_off_and_limits_set"]["status"] == "PASS"


def test_a_third_request_is_refused_before_it_is_sent():
    guard = P._guard_class()()
    req = lambda: httpx.Request("POST", "https://api.anthropic.com/v1/messages", json={"tools": [], "model": "m"})
    guard.on_outbound(req())
    guard.on_outbound(req())
    with pytest.raises(P.ProbeAbort):
        guard.on_outbound(req())
    assert (len(guard.sent), guard.refused) == (2, 1)


def test_nothing_outside_the_temp_folder_is_written_and_no_frozen_data_is_read(monkeypatch, tmp_path):
    opened: list[tuple[Path, str]] = []
    real = builtins.open

    def tracing_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)):
            opened.append((Path(file).resolve(), mode))
        return real(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracing_open)
    monkeypatch.setattr(io, "open", tracing_open)
    code, _, _, out = run_probe(monkeypatch, tmp_path, responder())
    monkeypatch.setattr(builtins, "open", real)
    monkeypatch.setattr(io, "open", real)
    assert code == 0
    temp_root = Path(tempfile.gettempdir()).resolve()
    writes = [p for p, m in opened if any(c in m for c in "wax+")]
    probe_dirs = {p.relative_to(temp_root).parts[0] for p in writes if p.is_relative_to(temp_root)
                  and p.relative_to(temp_root).parts[0].startswith("p3-schema-probe-")}
    assert len(probe_dirs) == 1                                         # one fresh temporary folder
    for p in writes:
        assert (p.is_relative_to(temp_root / next(iter(probe_dirs))) or p.is_relative_to(out.resolve())), p
    assert not (temp_root / next(iter(probe_dirs))).exists()            # removed afterwards
    for p, _ in opened:
        for forbidden in (REPO / "data", REPO / "ground_truth", REPO / "artifacts", REPO / "out",
                          REPO / "contracts" / "frozen_split.json"):
            assert not p.is_relative_to(forbidden.resolve()), p
    repo_reads = {p.relative_to(REPO.resolve()).as_posix() for p, _ in opened if p.is_relative_to(REPO.resolve())
                  and not p.is_relative_to(out.resolve()) and "site-packages" not in p.parts}
    assert {"contracts/column_roles.json", "contracts/dataset_contract.schema.json"} <= repo_reads
    assert not {r for r in repo_reads if r.startswith("contracts/")} - {
        "contracts/column_roles.json", "contracts/dataset_contract.schema.json"}


def test_saved_output_has_no_request_body_and_no_key(monkeypatch, tmp_path):
    _, _, _, out = run_probe(monkeypatch, tmp_path, responder())
    text = (out / "probe_result.json").read_text(encoding="utf-8")
    assert DUMMY_KEY not in text and P.SYSTEM not in text and P.REPORT_PROMPT not in text
    assert '"messages"' not in text and [p.name for p in out.iterdir()] == ["probe_result.json"]


def test_a_key_pattern_in_the_output_withholds_the_result(monkeypatch, tmp_path):
    leaked = "sk-" + "ant-" + "x" * 30                                       # assembled: no key-shaped literal
    code, result, _, _ = run_probe(monkeypatch, tmp_path, responder(report={**SECTIONS_OK, "overview": leaked}))
    assert code == 1 and result == {"result": "FAIL", "reason": "a key pattern was found; the result was withheld"}
