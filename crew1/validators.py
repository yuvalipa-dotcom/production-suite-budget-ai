"""Post-run deterministic validators for a Crew-1 run (P3 instruction 2.7). ALL DATA IS SYNTHETIC.

A run is PASS only if every validator passes. Quantitative claims in the narrative text are
checked against ``stats.json`` and the contracts with their units (docs/crew1.md, "Validators").

Claim rules, in order:
1. Identifiers and metadata are masked first and never validated: production and record ids
   (``SYN-0001``, ``SYN-0001-C01``), dates and years in date context, version strings, seeds,
   ordinals and percentile labels (``95th``), category bucket labels (``2+`` revisions), the confidence level of an interval (``95% Wilson
   interval``), labelled references (``Chart 2``, ``Table 1``, ``Section 3``, ``Crew 2``, ``stage 2``),
   list numbering at the start of a line. Letter-attached numbers (``S0``-``S3``, ``H1``, ``P2``,
   ``v2``) are never numbers.
2. Every other number is a QUANTITATIVE CLAIM. Its unit comes from the text next to it:
   ``%``/percent -> percent; pp/percentage points/points -> pp; ILS/NIS/shekel -> ILS; days -> days;
   ``x``/``×`` -> ratio; no unit -> a plain integer or decimal. An unlabelled number in an interval
   or range takes the unit of its partner (``[3.2, 26.3] pp``, ``40 to 50%``).
3. A claim matches a source value of a compatible unit when it equals that value rounded to the
   precision shown (half-up or half-even). Percent claims match fractions x 100; pp claims match only
   pp values; ILS claims only ILS values; days only days; plain integers match counts and plain
   numbers; plain decimals match ratios, fractions and plain numbers. An unsigned claim matches the
   magnitude; a signed claim must match the sign. Only after "about", "around", "approximately",
   "roughly", "nearly" or "~" may an integer's trailing zeros mark rounding to that place, and only
   with at least two significant digits.
4. A claim without a matching source value of a compatible unit fails the run.
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path

from .tools import cleaning as C
from .tools import contract as K
from .tools import eda as E
from .tools.workspace import Sources

REQUIRED_ARTIFACTS = ("clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json")
INSIGHT_SECTIONS = ("Observations (descriptive, from eligible synthetic snapshots)",
                    "Hypotheses for Crew 2 (not predictions)",
                    "Frozen split note",
                    "Data quality and Policy C exclusions",
                    "Limitations")
MANIFEST_FIELDS = ("run_id", "status", "mode", "model", "llm_settings", "limits", "package_versions",
                   "input_sha256", "policy_c_counts", "token_usage", "timings", "artifacts_sha256")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "successful_requests")
TIMING_FIELDS = ("started_at", "finished_at", "duration_seconds", "tasks")

# ---------------------------------------------------------------- claim extraction

MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
MASKS = [
    ("record_id", re.compile(r"\bSYN-\d{4}(?:-[CMP]\d{2})?\b")),
    ("date", re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b")),
    ("date", re.compile(rf"\b\d{{1,2}}[-\s]{MONTH}[-\s,]+\d{{4}}\b", re.I)),
    ("date", re.compile(rf"\b{MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I)),
    ("date", re.compile(rf"\b{MONTH}\.?\s+\d{{4}}\b", re.I)),
    ("year", re.compile(r"\b(?:in|year|years|since|during|until|before|after|fy)\s+(?:19|20)\d{2}\b", re.I)),
    ("version", re.compile(r"\b\d+\.\d+\.\d+(?:\.\d+)*\b")),
    ("seed", re.compile(r"\bseeds?\s*(?:=|:|of)?\s*\d+\b", re.I)),
    ("confidence_level", re.compile(r"\b(?:80|90|95|99)\s*%\s*(?:(?:wilson|bootstrap|percentile|confidence|"
                                    r"production[- ]level|two[- ]sided|score)\s+)*(?:intervals?|ci|cis|level)\b",
                                    re.I)),
    ("ordinal", re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I)),
    ("bucket_label", re.compile(r"(?<![\w.])\d+\+(?![\d.])")),
    ("reference", re.compile(r"\b(?:chart|figure|fig\.?|table|section|step|appendix|hypothesis|observation|"
                             r"limitation|item|part|phase|crew|stage|rule|question|task|agent)\s*(?:#|no\.?\s*)?"
                             r"\d+[a-z]?\b", re.I)),
    ("list_number", re.compile(r"(?m)^[ \t]*(?:[-*+][ \t]+)?(?:#+[ \t]*)?\(?\d+[.)](?=\s)")),
]
NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[+\-−–]?)(?P<pre>(?:ILS|NIS|₪)\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?!\d)(?P<mult>[kKM](?![A-Za-z]))?"
    r"(?P<post>\s*(?:%|percent\b|per\s+cent\b|percentage[\s-]+points?\b|pp\b|p\.p\.|points?\b|ILS\b|NIS\b|"
    r"₪|shekels?\b|days?\b|×|x\b))?(?![A-Za-z0-9_])")
SIGN_CONTEXT = set(" \t\n([{=:;,") | {" "}
APPROX = re.compile(r"(?:about|around|approximately|approx\.?|roughly|nearly|almost|some|~|≈)\s*$", re.I)
PAIR_GAP = re.compile(r"^\s*(?:,|–|-|to|and|vs\.?|versus)\s*$", re.I)
TRAILING_UNIT = re.compile(r"^\s*[\])]\s*(?P<post>%|percent\b|percentage[\s-]+points?\b|pp\b|points?\b|ILS\b|"
                           r"NIS\b|days?\b)", re.I)


@dataclass
class Claim:
    text: str
    value: Decimal
    places: int            # decimal places shown (negative after a k/M multiplier)
    unit: str
    signed: bool
    approx: bool
    trailing_zeros: int


def _unit(post: str | None, pre: str | None) -> str | None:
    p = (post or "").strip().lower()
    if pre or p.startswith(("ils", "nis", "₪", "shekel")):
        return "ILS"
    if p.startswith(("%", "percent", "per cent")) and "point" not in p:
        return "percent"
    if p.startswith(("pp", "p.p.", "percentage", "point")):
        return "pp"
    if p.startswith("day"):
        return "days"
    if p in ("x", "×"):
        return "ratio"
    return None


def mask_metadata(text: str) -> str:
    for _, rx in MASKS:
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    return text


def extract_claims(text: str) -> list[Claim]:
    masked = mask_metadata(text)
    found = []
    for m in NUMBER.finditer(masked):
        digits = m["num"].replace(",", "")
        places = len(digits.split(".")[1]) if "." in digits else 0
        value = Decimal(digits)
        mult = {"k": 3, "K": 3, "M": 6}.get(m["mult"] or "", 0)
        value = value.scaleb(mult)
        sign = m["sign"]
        before = masked[m.start() - 1] if m.start() else " "
        if sign in ("-", "–") and before not in SIGN_CONTEXT:
            sign = ""                              # a range dash ("48.4%-69.5%"), not a minus sign
        if sign in ("-", "−", "–"):
            value = -value
        trailing = 0
        if not places and not mult:
            trailing = len(digits) - len(digits.rstrip("0")) if digits.strip("0") else 0
        found.append([m, Claim(m.group(0).strip(), value, places - mult, _unit(m["post"], m["pre"]) or "",
                               bool(sign), bool(APPROX.search(masked[max(0, m.start() - 16):m.start()])),
                               trailing)])
    for i in range(len(found) - 1, -1, -1):        # interval and range partners share a unit
        m, c = found[i]
        if c.unit:
            continue
        after = masked[m.end():m.end() + 30]
        t = TRAILING_UNIT.match(after)
        if t:
            c.unit = _unit(t["post"], None) or ""
        elif i + 1 < len(found) and found[i + 1][1].unit and PAIR_GAP.match(masked[m.end():found[i + 1][0].start()]):
            c.unit = found[i + 1][1].unit
    claims = []
    for m, c in found:
        if not c.unit:
            c.unit = "decimal" if c.places > 0 else "integer"
        claims.append(c)
    return claims


# ---------------------------------------------------------------- sources and matching

COMPATIBLE = {"percent": ("fraction",), "pp": ("pp",), "ILS": ("ILS",), "days": ("days",),
              "ratio": ("ratio", "fraction", "number"), "integer": ("count", "number"),
              "decimal": ("ratio", "fraction", "number")}


def contract_sources(contract: dict | None, frozen_counts: dict | None) -> list[tuple[float, str]]:
    out = []
    if contract:
        n, excl = contract.get("row_count"), contract.get("exclusion_counts") or {}
        cols = contract.get("columns") or []
        out += [(float(n), "count")] if isinstance(n, int) else []
        out += [(float(v), "count") for v in excl.values() if isinstance(v, int)]
        if isinstance(n, int):
            total = sum(v for v in excl.values() if isinstance(v, int))
            out += [(float(total), "count"), (float(n + total), "count")]
        roles = [c.get("role") for c in cols if isinstance(c, dict)]
        out += [(float(len(cols)), "count")] + [(float(roles.count(r)), "count") for r in set(roles)]
        out += [(float(len(contract.get("outcome_only_exclusions") or [])), "count")]
        for c in cols:
            for k in ("min", "max"):
                if isinstance(c, dict) and isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool):
                    out.append((float(c[k]), "number"))
    if frozen_counts:
        for v in frozen_counts.values():
            if isinstance(v, dict):
                out += [(float(x), "count") for x in v.values()]
            elif isinstance(v, int):
                out.append((float(v), "count"))
    return out


def claim_sources(stats: dict | None, contract: dict | None, frozen_counts: dict | None) -> dict[str, list[float]]:
    by_unit: dict[str, list[float]] = {}
    leaves = list(E.numeric_leaves(stats)) if stats else []
    for v, u in leaves + contract_sources(contract, frozen_counts):
        by_unit.setdefault(u, []).append(v)
    return by_unit


def _rounds_to(source: float, claim: Claim, places: int) -> bool:
    try:
        s = Decimal(repr(source))
        q = Decimal(1).scaleb(-places)
        candidates = {s.quantize(q, rounding=ROUND_HALF_UP), s.quantize(q, rounding=ROUND_HALF_EVEN)}
    except InvalidOperation:
        return False
    target = claim.value if claim.signed else abs(claim.value)
    return any((c if claim.signed else abs(c)) == target for c in candidates)


def claim_matches(claim: Claim, sources: dict[str, list[float]]) -> bool:
    values = [v * (100 if claim.unit == "percent" else 1)
              for u in COMPATIBLE[claim.unit] for v in sources.get(u, [])]
    places = [claim.places]
    if claim.approx and claim.trailing_zeros and len(str(abs(int(claim.value))).rstrip("0")) >= 2:
        places.append(-claim.trailing_zeros)
    return any(_rounds_to(v, claim, p) for v in values for p in places)


def unmatched_claims(text: str, sources: dict[str, list[float]]) -> list[str]:
    return [f"{c.text} [{c.unit}]" for c in extract_claims(text) if not claim_matches(c, sources)]


# ---------------------------------------------------------------- forbidden claims

NEGATION = re.compile(r"\b(?:no|not|never|without|nor|cannot|neither|none|nothing)\b|n't\b", re.I)
FORBIDDEN = {
    "real_world_accuracy": re.compile(r"real[-\s]world\s+(?:accuracy|performance|predictive|validity|validation|"
                                      r"reliability|effectiveness|results?)|(?:validated|proven|demonstrated)\s+"
                                      r"(?:on|with|in|against)\s+real", re.I),
    "will_predict": re.compile(r"\bwill\s+(?:\w+\s+)?predict", re.I),
    "resplit_or_rebalance": re.compile(r"\bre-?split|\bre-?draw|\bredraw|\bre-?shuffl|\brebalanc|\bre-?balanc|"
                                       r"\bre-?stratif|\bover-?sampl|\bunder-?sampl|\bSMOTE\b|"
                                       r"\bstratif\w*\s+the\s+split|\bre-?assign\w*\s+(?:the\s+)?(?:split|groups?|folds?)",
                                       re.I),
}


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def forbidden_claims(text: str) -> list[str]:
    hits = []
    for s in sentences(text):
        for rule, rx in FORBIDDEN.items():
            if rx.search(s) and not NEGATION.search(s):
                hits.append(f"{rule}: {s.strip()[:160]}")
    return hits


# ---------------------------------------------------------------- documents

class _Narrative(HTMLParser):
    """Collect the text of <div class="narrative" data-section="..."> blocks and the embedded images."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections: dict[str, list[str]] = {}
        self.images: list[str] = []
        self._stack: list[str | None] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img":
            self.images.append(a.get("src") or "")
            return
        if tag == "div":
            sid = a.get("data-section") if "narrative" in (a.get("class") or "").split() else None
            self._stack.append(sid)
            if sid is not None:
                self.sections.setdefault(sid, [])

    def handle_endtag(self, tag):
        if tag == "div" and self._stack:
            self._stack.pop()

    def handle_data(self, data):
        current = next((s for s in reversed(self._stack) if s is not None), None)
        if current is not None:
            self.sections[current].append(data)


def eda_narrative(html_text: str) -> tuple[dict[str, str], list[str]]:
    p = _Narrative()
    p.feed(html_text)
    return {k: "\n\n".join(v).strip() for k, v in p.sections.items()}, p.images


def _norm_heading(s: str) -> str:
    s = re.sub(r"^\s*\d+[.)]\s*", "", s)
    s = re.sub(r"[*_`]", "", s).strip().rstrip(":").strip()
    return re.sub(r"\s+", " ", s).lower()


def markdown_sections(md: str) -> dict[str, str]:
    out, current, buf = {}, None, []
    for line in md.splitlines():
        m = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current, buf = _norm_heading(m.group(1)), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def _png_ok(src: str) -> bool:
    prefix = "data:image/png;base64,"
    if not src.startswith(prefix):
        return False
    try:
        return base64.b64decode(src[len(prefix):], validate=True).startswith(b"\x89PNG\r\n\x1a\n")
    except ValueError:
        return False


# ---------------------------------------------------------------- the validators

def _res(vid: str, name: str, problems: list) -> dict:
    return {"id": vid, "name": name, "status": "FAIL" if problems else "PASS",
            "details": [str(p) for p in problems[:25]] + ([f"... {len(problems) - 25} more"] if len(problems) > 25 else [])}


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fresh_handoff(sources: Sources) -> tuple[bytes, dict, dict]:
    """A fresh deterministic recomputation from the repository raw export: (csv bytes, log, contract)."""
    from gate.checks import load_raw, raw_manifest_hash
    raw = load_raw(sources.raw_dir)
    frozen = json.loads((sources.contracts_dir / "frozen_split.json").read_text(encoding="utf-8"))
    sha = raw_manifest_hash(sources.raw_dir)
    text, log = C.build_clean_data(raw, frozen, sha)
    data = text.encode("utf-8")
    contract = K.build_contract(data, sources.contracts_dir, sha, log["exclusion_counts"])
    return data, log, contract


def v1_artifacts(out: Path) -> dict:
    bad = [f"missing or empty: {n}" for n in REQUIRED_ARTIFACTS
           if not (out / n).is_file() or (out / n).stat().st_size == 0]
    return _res("V1", "required artifacts exist and are non-empty", bad)


def v2_clean_data(out: Path, expected: bytes) -> dict:
    p = out / "clean_data.csv"
    ok = p.is_file() and p.read_bytes() == expected
    return _res("V2", "clean_data.csv is byte-identical to a fresh Policy C recomputation",
                [] if ok else ["clean_data.csv differs from the fresh recomputation"])


def v3_contract(out: Path, expected_contract: dict) -> dict:
    delivered = _load_json(out / "dataset_contract.json")
    if not isinstance(delivered, dict):
        return _res("V3", "contract structure equals a fresh build", ["dataset_contract.json missing or invalid"])
    want, got = K.structural(expected_contract), K.structural(delivered)
    bad = [f"structural field differs: {k}" for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)]
    try:
        K.validate_descriptions({c["name"]: c["description"] for c in delivered.get("columns", [])
                                 if isinstance(c, dict) and "description" in c},
                                [c.get("name") for c in delivered.get("columns", []) if isinstance(c, dict)])
    except K.ContractError as e:
        bad.append(f"invalid description: {e}")
    return _res("V3", "contract structure equals a fresh build (descriptions excepted)", bad)


def v4_gate(out: Path, sources: Sources) -> tuple[dict, dict | None]:
    from gate.run import main as gate_main
    with tempfile.TemporaryDirectory() as tmp:
        code = gate_main(["--stage", "handoff", "--raw", str(sources.raw_dir), "--out", tmp,
                          "--data", str(out / "clean_data.csv"), "--contract", str(out / "dataset_contract.json")])
        result = _load_json(Path(tmp) / "gate_result.json")
    bad = [] if code == 0 and result and result.get("status") == "PASS" else \
        [f"{c['id']}: {c['status']}" for c in (result or {}).get("checks", [])
         if c["severity"] == "HARD" and c["status"] != "PASS"] or ["gate did not pass"]
    delivered = _load_json(out / "gate_result.json")
    if not delivered or delivered.get("status") != "PASS":
        bad.append("the crew's own gate_result.json is missing or not PASS")
    return _res("V4", "P2 handoff gate PASS (re-run independently)", bad), result


def v5_claims(out: Path, sources: Sources) -> dict:
    stats = _load_json(out / "stats.json")
    contract = _load_json(out / "dataset_contract.json")
    frozen = _load_json(sources.contracts_dir / "frozen_split.json")
    if not isinstance(stats, dict):
        return _res("V5", "quantitative claims match stats.json or the contracts", ["stats.json missing or invalid"])
    try:
        src = claim_sources(stats, contract if isinstance(contract, dict) else None,
                            (frozen or {}).get("counts"))
    except ValueError as e:
        return _res("V5", "quantitative claims match stats.json or the contracts", [f"stats.json: {e}"])
    bad = []
    html_path, md_path = out / "eda_report.html", out / "insights.md"
    if html_path.is_file():
        narrative, _ = eda_narrative(html_path.read_text(encoding="utf-8"))
        for sid, text in narrative.items():
            bad += [f"eda_report.html [{sid}]: {c}" for c in unmatched_claims(text, src)]
    if md_path.is_file():
        bad += [f"insights.md: {c}" for c in unmatched_claims(md_path.read_text(encoding="utf-8"), src)]
    return _res("V5", "quantitative claims match stats.json or the contracts (value, unit, rounding)", bad)


def v6_eda(out: Path) -> dict:
    p = out / "eda_report.html"
    if not p.is_file():
        return _res("V6", "EDA report sections and charts", ["eda_report.html missing"])
    narrative, images = eda_narrative(p.read_text(encoding="utf-8"))
    bad = [f"missing or empty section: {sid}" for sid in E.SECTIONS if len(narrative.get(sid, "")) < 40]
    charts = sum(1 for s in images if _png_ok(s))
    if charts < E.MIN_CHARTS:
        bad.append(f"{charts} embedded PNG charts, at least {E.MIN_CHARTS} required")
    bad += [f"forbidden claim in eda_report.html: {h}" for t in narrative.values() for h in forbidden_claims(t)]
    return _res("V6", "EDA report has every required section and at least 4 embedded charts", bad)


def frozen_note_problems(section: str, stats: dict | None) -> list[str]:
    bad = []
    a = ((stats or {}).get("analyses") or {}).get("frozen_split_prevalence_difference")
    if not a:
        return ["stats.json has no frozen_split_prevalence_difference analysis"]
    claims = [c for c in extract_claims(section) if c.unit == "pp"]
    for key in ("test_minus_train_pp", "test_minus_train_lo_pp", "test_minus_train_hi_pp"):
        if not any(claim_matches(c, {"pp": [a[key]]}) for c in claims):
            bad.append(f"frozen split note does not state {key} from stats.json (in pp)")
    for word, rx in (("frozen", r"\bfrozen\b"), ("not tuned", r"\bnot\s+(?:been\s+)?tuned\b|\buntuned\b"),
                     ("sampling outcome", r"\bsampling\b"), ("independent of the outcome mechanism", r"\bindependent")):
        if not re.search(rx, section, re.I):
            bad.append(f"frozen split note does not state: {word}")
    return bad


def v7_insights(out: Path) -> dict:
    p = out / "insights.md"
    if not p.is_file():
        return _res("V7", "insights.md sections and forbidden claims", ["insights.md missing"])
    md = p.read_text(encoding="utf-8")
    secs = markdown_sections(md)
    bad = [f"missing or empty section: {t}" for t in INSIGHT_SECTIONS if not secs.get(_norm_heading(t))]
    bad += frozen_note_problems(secs.get(_norm_heading("Frozen split note"), ""), _load_json(out / "stats.json"))
    lim = secs.get(_norm_heading("Limitations"), "")
    if not re.search(r"\bsynthetic\b", lim, re.I) or not re.search(r"real[-\s]world", lim, re.I):
        bad.append("Limitations must state that the data are synthetic and there is no real-world validation")
    if not re.search(r"policy\s*c", secs.get(_norm_heading("Data quality and Policy C exclusions"), ""), re.I):
        bad.append("Data quality section does not discuss Policy C")
    bad += [f"forbidden claim: {h}" for h in forbidden_claims(md)]
    return _res("V7", "insights.md has every required section and no forbidden claim", bad)


def v8_manifest(manifest: dict | None) -> dict:
    if not isinstance(manifest, dict):
        return _res("V8", "run_manifest.json complete", ["run_manifest.json missing"])
    bad = [f"missing field: {f}" for f in MANIFEST_FIELDS if manifest.get(f) in (None, "", {}, [])]
    usage = manifest.get("token_usage") or {}
    bad += [f"missing token usage field: {f}" for f in TOKEN_FIELDS if not isinstance(usage.get(f), int)]
    timing = manifest.get("timings") or {}
    bad += [f"missing timing field: {f}" for f in TIMING_FIELDS if timing.get(f) in (None, "", [])]
    if not (manifest.get("model") or {}).get("id"):
        bad.append("missing model id")
    for pkg in ("python", "crewai", "anthropic", "matplotlib", "pandas"):
        if not (manifest.get("package_versions") or {}).get(pkg):
            bad.append(f"missing package version: {pkg}")
    return _res("V8", "run_manifest.json is complete", bad)


SECRET_PATTERNS = (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), re.compile(r"\bsk-[A-Za-z0-9]{20,}"))
PATH_PATTERNS = (re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]"), re.compile(r"\\\\[A-Za-z0-9.?]"),
                 re.compile(r"(?:^|[\s\"'(=:])/(?:home|Users|tmp|var|etc|root|mnt|opt|private)/"))


def trace_problems(text: str, secret: str | None) -> list[str]:
    bad = []
    if secret and secret in text:
        bad.append("the API key value appears in the trace")
    bad += [f"secret-shaped value ({rx.pattern[:12]}...)" for rx in SECRET_PATTERNS if rx.search(text)]
    bad += [f"absolute path ({rx.pattern[:18]}...)" for rx in PATH_PATTERNS if rx.search(text)]
    return bad


def v9_trace(out: Path, secret: str | None) -> dict:
    p = out / "trace.jsonl"
    if not p.is_file():
        return _res("V9", "trace.jsonl redacted", ["trace.jsonl missing"])
    text = p.read_text(encoding="utf-8")
    bad = trace_problems(text, secret)
    for n, line in enumerate(text.splitlines(), 1):
        try:
            json.loads(line)
        except ValueError:
            bad.append(f"line {n} is not JSON")
            break
    return _res("V9", "trace.jsonl is redacted (no key, no absolute path)", bad)


def run_validators(out: Path, sources: Sources, manifest: dict | None, secret: str | None) -> list[dict]:
    expected_csv, _, expected_contract = fresh_handoff(sources)
    gate_res, _ = v4_gate(out, sources)
    return [v1_artifacts(out), v2_clean_data(out, expected_csv), v3_contract(out, expected_contract), gate_res,
            v5_claims(out, sources), v6_eda(out), v7_insights(out), v8_manifest(manifest), v9_trace(out, secret)]


def lint_narrative(text: str, stats: dict | None, contract: dict | None, frozen_counts: dict | None) -> dict:
    """Feedback for the authoring tools (non-blocking; the post-run validators decide)."""
    try:
        src = claim_sources(stats, contract, frozen_counts)
    except ValueError:
        src = {}
    return {"unverified_quantitative_claims": unmatched_claims(text, src), "forbidden_claims": forbidden_claims(text)}
