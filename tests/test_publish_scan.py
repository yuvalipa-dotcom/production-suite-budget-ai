"""The pre-publication scan must catch what it should, and pass healthy content.

Planted values are assembled at runtime so that this file itself contains no
secret-shaped literal.
"""

import base64
import hashlib
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_scan", REPO / "tools" / "publish_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

FAKE_GH = "gh" + "p_" + "Ab1" * 12


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


FAKE_JWT = ".".join([_b64(b'{"alg":"HS256","typ":"JWT"}'), _b64(b'{"sub":"planted-test-value"}'),
                     _b64(hashlib.sha256(b"planted").digest())])
FAKE_DB = "postgres" + "ql://" + "user:pw@host:5432/db"
DIGEST = hashlib.sha256(b"planted").hexdigest()


def rules(findings):
    return {f["rule"] for f in findings}


def test_content_scan_catches_planted_values():
    text = "\n".join([f"token = {FAKE_GH}", f"auth: {FAKE_JWT}", f"url = {FAKE_DB}",
                      "path C:" + "\\Users\\someone\\x", "import x from '" + "@/lib/x'"])
    found = rules(scan._content_findings("docs/note.md", text, []))
    assert {"github_token", "jwt", "postgres_url", "local_windows_path", "product_code_import"} <= found


def test_content_scan_catches_env_files_and_private_denylist():
    assert "env_file_name" in rules(scan._content_findings("config/." + "env", "", []))
    assert "env_file_name" in rules(scan._content_findings("." + "env.local", "", []))
    assert "private_denylist[0]" in rules(scan._content_findings("a.md", "see Example-Corp here", ["example-corp"]))


def test_content_scan_passes_healthy_content():
    healthy = "production_id,stage,budget_pre_vat\nSYN-0001,S0,254237.29\n"
    assert scan._content_findings("data/synthetic/v1/snapshots.csv", healthy, ["example-corp"]) == []
    assert scan._content_findings(".gitignore", "." + "env*\n", []) == []


def test_secret_scan_flags_a_token_and_spares_only_manifest_digests(tmp_path):
    secret_file = tmp_path / "notes.txt"
    secret_file.write_text(f'github_token = "{FAKE_GH}"\n', encoding="utf-8")
    assert scan._secret_findings(secret_file, "notes.txt")

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{\n  "files_sha256": {\n    "snapshots.csv": "' + DIGEST + '"\n  }\n}\n', encoding="utf-8")
    assert scan._secret_findings(manifest, "data/synthetic/v1/manifest.json") == []

    elsewhere = tmp_path / "other.json"
    elsewhere.write_text('{\n  "api_secret": "' + DIGEST + '"\n}\n', encoding="utf-8")
    assert scan._secret_findings(elsewhere, "other.json"), "the same digest outside the manifest must be flagged"


# ---------- P2: the two approved digest structures, and nothing else ----------

BASELINE = "tests/fixtures/policy_c_baseline.json"
FROZEN = "contracts/frozen_split.json"
V1, V2 = "data/synthetic/v1/productions.csv", "data/synthetic/v2/productions.csv"
DIGEST2 = hashlib.sha256(b"planted-2").hexdigest()


def _baseline(digest=DIGEST, extra=""):
    return '{\n  "algorithm": "sha256",\n  "digest": "' + digest + '",\n' + extra + '  "ineligible": []\n}\n'


def _frozen(v1=DIGEST, v2=DIGEST2, extra=""):
    return ('{\n  "provenance": {\n    "seed": 42,\n    "source_files_sha256": {\n'
            f'      "{V1}": "{v1}",\n      "{V2}": "{v2}"\n    }}\n  }},\n{extra}  "version": "1.0.0"\n}}\n')


def _flagged(tmp_path, logical, text):
    disk = tmp_path / "f.json"
    disk.write_text(text, encoding="utf-8")
    return scan._secret_findings(disk, logical)


def _lines(findings):
    return sorted(f["line"] for f in findings)


def test_approved_digest_structures_are_exempted(tmp_path):
    assert _flagged(tmp_path, "x.json", _baseline()), "the detector must flag the digest without the exemption"
    assert _flagged(tmp_path, BASELINE, _baseline()) == []
    assert _flagged(tmp_path, FROZEN, _frozen()) == []
    assert _flagged(tmp_path, FROZEN, _frozen(v1=DIGEST, v2=DIGEST)) == []      # v1 and v2 files may be identical


def test_committed_files_are_exempted():
    for logical in (BASELINE, FROZEN):
        assert scan._secret_findings(REPO / logical, logical) == [], logical


def test_same_digest_in_another_field_or_file_is_detected(tmp_path):
    # another field in the approved file. detect-secrets reports a repeated value once, at its first line;
    # that line is the approved field, but the value is no longer unique to it, so it is not spared.
    assert _lines(_flagged(tmp_path, BASELINE, _baseline(extra=f'  "other": "{DIGEST}",\n'))) == [3]
    before = '{\n  "other": "' + DIGEST + '",\n' + _baseline()[2:]
    assert _lines(_flagged(tmp_path, BASELINE, before)) == [2]
    assert _flagged(tmp_path, FROZEN, _frozen(extra=f'  "checksum": "{DIGEST}",\n'))
    # the approved key at an unapproved JSON location
    nested = '  "nested": {\n    "digest": "' + DIGEST2 + '"\n  },\n'
    assert _lines(_flagged(tmp_path, BASELINE, _baseline(extra=nested))) == [5]
    moved = f'  "{V1}": "{DIGEST2}",\n'
    assert _lines(_flagged(tmp_path, FROZEN, _frozen(v1=DIGEST, v2=DIGEST, extra=moved))) == [9]
    # the approved structure in another file
    assert _flagged(tmp_path, "tests/fixtures/other.json", _baseline())
    assert _flagged(tmp_path, "contracts/other.json", _frozen())
    assert _flagged(tmp_path, "docs/" + BASELINE, _baseline())


def test_values_not_lowercase_hex64_are_not_exempted(tmp_path):
    for bad in (DIGEST.upper(), DIGEST[:63], DIGEST + "a"):
        assert _flagged(tmp_path, BASELINE, _baseline(digest=bad)), bad
        assert _flagged(tmp_path, FROZEN, _frozen(v1=bad)), bad
    for bad in (DIGEST.upper(), DIGEST[:63], DIGEST + "0", "g" + DIGEST[1:]):
        line = f'  "digest": "{bad}",'
        assert not scan._approved_digest(BASELINE, line, _baseline(digest=bad), None), bad


def test_unrelated_secret_rules_remain_effective_in_approved_files(tmp_path):
    token = f'  "token": "{FAKE_GH}",\n'
    assert _flagged(tmp_path, BASELINE, _baseline(extra=token))
    assert _flagged(tmp_path, FROZEN, _frozen(extra=token))
    # another secret on the same line as an approved digest is never spared
    same_line = '{\n  "digest": "' + DIGEST + '", "token": "' + FAKE_GH + '"\n}\n'
    assert _flagged(tmp_path, BASELINE, same_line)
    # a high-entropy hex value that is not the line's approved value is never spared
    assert not scan._approved_digest(BASELINE, f'  "digest": "{DIGEST}",', _baseline(), DIGEST2)


# ---------- P2: the public W-3 baseline commit id in exactly two lines ----------

CHECKER, CHECKER_TEST = "tools/publication_check.py", "tests/test_publication_check.py"
BASE = scan.APPROVED_GIT_BASELINE
OTHER_SHA = hashlib.sha1(b"planted-commit").hexdigest()


def _checker_src(value=BASE, extra=""):
    return f'import x\n\nPUBLISHED_BASELINE = "{value}"\n{extra}BASELINE_DESCENDANTS = ("p2-quality-gate",)\n'


def _checker_test_src(value=BASE, extra=""):
    return f'def test_x():\n    assert check.PUBLISHED_BASELINE == "{value}"\n{extra}'


def test_git_baselines_are_public_origin_main_commits():
    """The W-3 baseline was origin/main before PR #3; the P3 baseline is the PR #3 merge, whose first parent
    is the W-3 baseline. Both stay ancestors of origin/main (equality would break at every later merge)."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)

    pc_spec = importlib.util.spec_from_file_location("publication_check", REPO / "tools" / "publication_check.py")
    pc = importlib.util.module_from_spec(pc_spec)
    pc_spec.loader.exec_module(pc)
    for value in (BASE, P3):
        assert len(value) == 40 and set(value) <= set("0123456789abcdef")
    assert BASE == pc.PUBLISHED_BASELINE and P3 == pc.P3_BASELINE and BASE != P3
    origin = git("rev-parse", "--verify", "-q", "refs/remotes/origin/main").stdout.strip()
    if origin:                                    # absent in a remote-less dry-run clone
        assert git("rev-parse", "--verify", "-q", f"{P3}^1").stdout.strip() == BASE
        for value in (BASE, P3):
            assert git("merge-base", "--is-ancestor", value, origin).returncode == 0, value


def test_git_baseline_lines_are_exempted(tmp_path):
    assert _flagged(tmp_path, "tools/other.py", _checker_src()), "the detector must flag it without the exemption"
    assert _flagged(tmp_path, CHECKER, _checker_src()) == []
    assert _flagged(tmp_path, CHECKER_TEST, _checker_test_src()) == []
    for logical in (CHECKER, CHECKER_TEST, "tools/publish_scan.py"):
        assert scan._secret_findings(REPO / logical, logical) == [], logical


def test_git_baseline_modified_values_are_detected(tmp_path):
    for bad in (OTHER_SHA, BASE.upper(), BASE[:39], BASE + "0", BASE[:-1] + ("0" if BASE[-1] != "0" else "1")):
        assert _flagged(tmp_path, CHECKER, _checker_src(value=bad)), bad
        assert _flagged(tmp_path, CHECKER_TEST, _checker_test_src(value=bad)), bad


def test_git_baseline_unrelated_paths_and_lines_are_detected(tmp_path):
    for logical in ("tools/publication_check_v2.py", "tests/test_other.py", "docs/" + CHECKER):
        assert _flagged(tmp_path, logical, _checker_src()), logical
        assert _flagged(tmp_path, logical, _checker_test_src()), logical
    variants = [f'PUBLISHED_BASELINE = "{BASE}"  # note', f'  PUBLISHED_BASELINE = "{BASE}"',
                f"PUBLISHED_BASELINE = '{BASE}'", f'OTHER_BASELINE = "{BASE}"']
    for line in variants:
        assert _flagged(tmp_path, CHECKER, f"import x\n{line}\n"), line
    # the approved line plus the same value anywhere else in the file: not spared
    assert _flagged(tmp_path, CHECKER, _checker_src(extra=f'MIRROR = "{BASE}"\n'))
    assert _flagged(tmp_path, CHECKER_TEST, _checker_test_src(extra=f'    other = "{BASE}"\n'))
    assert not scan._approved_git_baseline(CHECKER, f'PUBLISHED_BASELINE = "{BASE}"', _checker_src(), OTHER_SHA)


def test_git_baseline_files_still_detect_additional_secrets(tmp_path):
    token = f'TOKEN = "{FAKE_GH}"\n'
    assert _flagged(tmp_path, CHECKER, _checker_src(extra=token))
    assert _flagged(tmp_path, CHECKER_TEST, _checker_test_src(extra="    " + token))
    assert _flagged(tmp_path, CHECKER, _checker_src(extra=f'OTHER = "{OTHER_SHA}"\n'))
    assert _flagged(tmp_path, CHECKER, f'PUBLISHED_BASELINE = "{BASE}"; T = "{FAKE_GH}"\n')
    assert "github_token" in rules(scan._content_findings(CHECKER, _checker_src(extra=token), []))


# ---------- P3: the public P2 merge commit (P3 starting baseline) in exactly two lines ----------

P3 = scan.APPROVED_P3_BASELINE


def _p3_checker_src(value=P3, extra=""):
    return f'{_checker_src()}P3_BASELINE = "{value}"\n{extra}P3_BASELINE_DESCENDANTS = ("p3-crew1",)\n'


def _p3_checker_test_src(value=P3, extra=""):
    return f'{_checker_test_src()}    assert check.P3_BASELINE == "{value}"\n{extra}'


def test_p3_baseline_exemption_is_exact():
    assert scan.APPROVED_COMMIT_IDS == ((BASE, scan.GIT_BASELINE_LINES), (P3, scan.P3_BASELINE_LINES))
    assert set(scan.P3_BASELINE_LINES) == {CHECKER, CHECKER_TEST}
    assert scan.GIT_BASELINE_LINES == {CHECKER: 'PUBLISHED_BASELINE = "{}"',
                                       CHECKER_TEST: '    assert check.PUBLISHED_BASELINE == "{}"'}


def test_p3_baseline_lines_are_exempted(tmp_path):
    assert _flagged(tmp_path, "tools/other.py", _p3_checker_src()), "the detector must flag it without the exemption"
    assert _flagged(tmp_path, CHECKER, _p3_checker_src()) == []
    assert _flagged(tmp_path, CHECKER_TEST, _p3_checker_test_src()) == []


def test_p3_baseline_modified_values_are_detected(tmp_path):
    for bad in (OTHER_SHA, P3.upper(), P3[:39], P3 + "0", P3[:-1] + ("0" if P3[-1] != "0" else "1"), BASE):
        assert _flagged(tmp_path, CHECKER, _p3_checker_src(value=bad)), bad
        assert _flagged(tmp_path, CHECKER_TEST, _p3_checker_test_src(value=bad)), bad


def test_p3_baseline_unrelated_paths_lines_and_swaps_are_detected(tmp_path):
    for logical in ("tools/publication_check_v2.py", "tests/test_other.py", "docs/" + CHECKER, "crew1/run.py"):
        assert _flagged(tmp_path, logical, _p3_checker_src()), logical
        assert _flagged(tmp_path, logical, _p3_checker_test_src()), logical
    variants = [f'P3_BASELINE = "{P3}"  # note', f'  P3_BASELINE = "{P3}"', f"P3_BASELINE = '{P3}'",
                f'OTHER_BASELINE = "{P3}"', f'PUBLISHED_BASELINE = "{P3}"']
    for line in variants:
        assert _flagged(tmp_path, CHECKER, f"import x\n{line}\n"), line
    # each approved value is spared only in its own line: swapping the two values is detected
    assert _flagged(tmp_path, CHECKER, f'import x\nP3_BASELINE = "{BASE}"\n')
    assert _flagged(tmp_path, CHECKER_TEST, f'    assert check.P3_BASELINE == "{BASE}"\n')
    assert _flagged(tmp_path, CHECKER_TEST, f'    assert check.PUBLISHED_BASELINE == "{P3}"\n')
    # the approved line plus the same value anywhere else in the file: not spared
    assert _flagged(tmp_path, CHECKER, _p3_checker_src(extra=f'MIRROR = "{P3}"\n'))
    assert _flagged(tmp_path, CHECKER_TEST, _p3_checker_test_src(extra=f'    other = "{P3}"\n'))
    assert not scan._approved_git_baseline(CHECKER, f'P3_BASELINE = "{P3}"', _p3_checker_src(), OTHER_SHA)
    assert not scan._approved_git_baseline(CHECKER, f'P3_BASELINE = "{P3}"', _p3_checker_src(), BASE)


def test_p3_baseline_files_still_detect_additional_secrets(tmp_path):
    token = f'TOKEN = "{FAKE_GH}"\n'
    assert _flagged(tmp_path, CHECKER, _p3_checker_src(extra=token))
    assert _flagged(tmp_path, CHECKER_TEST, _p3_checker_test_src(extra="    " + token))
    assert _flagged(tmp_path, CHECKER, _p3_checker_src(extra=f'OTHER = "{OTHER_SHA}"\n'))
    assert _flagged(tmp_path, CHECKER, f'P3_BASELINE = "{P3}"; T = "{FAKE_GH}"\n')
    assert "github_token" in rules(scan._content_findings(CHECKER, _p3_checker_src(extra=token), []))


def test_manifest_exemption_is_unchanged():
    assert scan.MANIFEST_DIGEST_LINE.pattern == r'^\s*"[a-z_]+\.(csv|json)": "[0-9a-f]{64}",?\s*$'


def test_private_patterns_with_allow_list():
    deny = scan.Denylist([r"re:\bzork\b", r"re:zork[-_]", r"allow:unzork|zorkish", "plainword"])
    assert len(deny) == 3
    assert "private_pattern[0]" in rules(scan._content_findings("a.md", "met Zork today", deny))
    assert "private_pattern[1]" in rules(scan._content_findings("a.md", "zork-team", deny))
    assert scan._content_findings("a.md", "unzork_x is ok", deny) == []      # allow-listed line
    assert scan._content_findings("a.md", "zorkless word", deny) == []          # word boundary respected
    # allow never suppresses a literal entry
    assert "private_denylist[0]" in rules(scan._content_findings("a.md", "unzork plainword", deny))
