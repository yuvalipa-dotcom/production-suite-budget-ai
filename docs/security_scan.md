# Pre-publication security scan: summary

**Date:** 2026-10-18 (re-run after the Policy C changes)
**Outcome:** **0 findings.** Secret scan 0 · content scan 0 · commit-message scan 0 · commercial identity-term check: **passed**.
No excluded paths, submodules, large-file pointers, files over 5 MB or binary files.

## Tools

| Tool | Version | Purpose |
|---|---|---|
| `detect-secrets` | 1.5.0 | secret detection with all default plugins (high-entropy strings, keyword, cloud and VCS token formats, private keys) |
| `tools/publish_scan.py` | in this repository | content scan: environment files, key and token shapes, database URLs, local absolute paths, code imported from a product codebase, plus a private term list kept outside the repository |
| `tools/publication_check.py` | in this repository | publication dry run: refs, full-history, working-tree and commit-message scans, excluded paths, object and ignore-rule checks |
| git | 2.55.0 | history and ref enumeration |

Runtime: Python 3.13.15.

## Scope

The check ran in a fresh, disposable clone with its remote removed.

- **Publication refs:** exactly the three branches `main`, `p1-synthetic-generator` and `p1-remediation`. No tags, stashes, notes or other refs.
- **History:** every file of every commit reachable from those branches. That is 28 commits, including the commit that updates this summary.
- **Commit messages:** every message, in full, including trailers. It was scanned with the same secret rules as files (each detect-secrets plugin's per-line rule) and with the content and private-term rules, except for the one check listed under *Coverage gap*.
- **Working trees:** each of the three branches checked out and scanned file by file.
- **Commercial identity terms:** a check against the identity terms of the commercial product was run over the same scope, and **passed**.

## Documented exclusion

The dataset manifests list the SHA-256 digests of the data files. These are high-entropy hexadecimal strings by design, not secrets. Only lines with exactly that shape inside `manifest.json` are excluded. A test proves that the same digest anywhere else is still flagged.

**P2 addition.** Two further legitimate SHA-256 digest structures are exempted, after an owner-approved review of five false-positive findings:

- `digest` in `tests/fixtures/policy_c_baseline.json`;
- the `data/synthetic/v1/productions.csv` and `data/synthetic/v2/productions.csv` keys under `provenance.source_files_sha256` in `contracts/frozen_split.json`.

A flagged line is spared only when all of the following hold:

- the file path is exact;
- the whole line is that one key with a lowercase hex-64 value;
- the parsed JSON holds the value at exactly that location;
- the value occurs nowhere else in the file;
- the flagged secret is that value.

Tests in `tests/test_publish_scan.py` prove that the same digest in another field or file is still flagged. They also prove that a value that is not lowercase hex-64 is not exempted, and that other secrets in those files, including on the same line, are still detected.

**P2 publication candidate.** The dry-run checker (`tools/publication_check.py`) expects exactly four branches: the three W-3 branches plus `p2-quality-gate`. It applies the same history, working-tree, commit-message, path and object checks to each, and it still rejects any other ref. It also requires `p2-quality-gate` to descend from the published W-3 baseline, `origin/main` = `5951725`. Tests are in `tests/test_publication_check.py`.

**P2 baseline commit id.** The checker records that baseline's full 40-character commit id, a public Git object name, not a secret. With owner approval, the id is exempted in exactly two complete lines:

- the `PUBLISHED_BASELINE` assignment in `tools/publication_check.py`;
- its assertion in `tests/test_publication_check.py`.

A flagged line is spared only when all of the following hold:

- the file path is exact;
- the line equals its template filled with the one approved value;
- the flagged secret is that value;
- the value occurs once in the file.

The scanner holds the approved value as an integer literal, so its own source contains no quoted hex string. The tests build every commit id at runtime. Tests prove that the approved value is the checker's baseline and the repository's `origin/main`. They also prove that modified values, other paths, other lines, a repeated value and additional secrets are all still detected.

**P3 publication candidate (owner-approved compatibility fix).** After PR #3 merged, `origin/main` moved to the P2 merge commit `7140231`, whose first parent is the W-3 baseline `5951725` and whose tree equals the P2 tip. The checker now expects exactly five branches, the four above plus `p3-crew1`, and still rejects any other ref. The historical check is kept unchanged: `p2-quality-gate` must descend from `5951725`. A separate check requires `p3-crew1` to descend from the P3 baseline `7140231`, and requires that commit's first parent to be `5951725`. The scanner test no longer requires `origin/main` to *equal* a fixed id, which broke at the merge. It now requires both baselines to be ancestors of `origin/main` and the parent link above to hold.

**P3 baseline commit id.** The P3 baseline's full commit id is exempted under the same four rules as the W-3 id. It applies in exactly two complete lines: the `P3_BASELINE` assignment in `tools/publication_check.py` and its assertion in `tests/test_publication_check.py`. Each approved id is spared only in its own line template, so swapping the two values is detected. The scanner again holds the value as an integer literal. Tests prove that modified values, the other baseline's value in the P3 line, other paths and lines, a repeated value, an unrelated commit id and additional secrets are all still detected.

**P3 run digests in the p3-live-005 artifacts (owner-approved).** Four JSON artifacts of the published live run record SHA-256 provenance digests: `run_manifest.json` (`artifacts_sha256`, `input_sha256`), `gate_result.json` (`input_sha256s`), and `dataset_contract.json` and `cleaning_log.json` (`raw_manifest_sha256`). The scanner flags 25 of them as `Hex High Entropy String`: 17, 6, 1 and 1 respectively. The other five `artifacts_sha256` lines already match the dataset-manifest rule above, which is unchanged.

All 30 digests in these files were verified independently, and none is a secret. Each equals the SHA-256 of the file it names: either an artifact of the same run, or a committed input at the run's code commit `5cca25f` (`raw/*` is `data/synthetic/v2/raw/*`). `tests/test_publish_scan.py` repeats this check for the 25 approved values.

A finding is spared only when all of the following hold:

- the path is exactly one of the four approved paths under `artifacts/crew1/p3-live-005/`;
- the whole file is byte-identical to its approved blob (git blob id), so any change to the run record, inside or outside the digests, removes the exemption;
- the finding type is exactly `Hex High Entropy String`;
- the whole flagged line is one key with a lowercase hex-64 value, and the key names exactly one approved JSON location in that file, which holds that value;
- the value occurs once in the file.

There is no value or pattern allow-list. The blob ids are integer literals, so the scanner holds no quoted hex string. The exemption is applied where the history and working-tree scans collect findings (`_scan_tree`).

Tests prove the following are all still detected:

- a different digest in an approved slot;
- any other edit to the file;
- the same bytes at another path, including the same file under `p3-live-004`;
- a planted token or an extra hex value in an approved file;
- another finding type, or another line;
- a synthetic file with the approved structure at an approved path.

## Proof that the checks work

- `tests/test_publish_scan.py` checks both directions: planted token shapes, database URLs, local paths, environment-file names and private terms (including pattern rules with allow-lists) **are** detected, and healthy synthetic content passes. Planted values are built at runtime, so the test file contains no secret-shaped literal.
- The dry-run checker was also shown to flag an unexpected ref and excluded paths.

## Coverage gap (stated, not bypassed)

One commit-message pattern check, for references to environment-file names, was **not executed**. A safety hook in the development environment blocked the command that contained the pattern, and the block was **not bypassed** with another tool or wording.

Commit messages were still scanned for secrets, private terms, absolute local paths, internal file names and e-mail addresses. Separately, environment *files* are excluded and checked by path in every commit tree and working tree.

Raw scanner output is kept locally only and is not part of this repository.

## P4 exemption: the pinned clean-data digest in one historical file version (owner-approved)

**Finding.** The scan flagged a `Hex High Entropy String` at `crew2/ml/data.py` line 36 in the local P4 commit `b2146c5`. Both the P3 read-only scan and the P4 scan found it. That line is `EXPECTED_CLEAN_SHA256 = "<64 hex characters>"`.

**Why it is not a secret.**
- The value is the SHA-256 of the `clean_data.csv` text that the deterministic Policy C cleaning function (`crew1.tools.cleaning.build_clean_data`) rebuilds from the committed synthetic raw export.
- It is a public fingerprint of public synthetic data. It grants no access and is not a credential, token or key.
- The same digest is reproducible by anyone who runs the cleaning on the committed raw data.

**How it was verified independently.**
- The clean data was rebuilt in-process and hashed. The result equals the flagged value.
- `tests/test_crew2_data.py` ties that output to the independent reference handoff, built from the committed clean snapshots and the ground truth rather than from the cleaning code.
- The value occurs once in the file.
- Exactly one historical file version contains it as a quoted literal. That version is git blob `e0c7b3a…`, the file as it is in `b2146c5`.
- From `44a81d5` onwards the file holds the digest as an integer literal. No tracked file at `HEAD` contains it as a quoted string.

**Exact scope** (`tools/publish_scan.py`, `_approved_p4_clean_digest`). A flagged secret is spared only when **all** of these hold:
1. the path is exactly `crew2/ml/data.py`;
2. the finding type is exactly `Hex High Entropy String`;
3. the whole line equals `EXPECTED_CLEAN_SHA256 = "<value>"` filled with the one approved value, and the flagged secret is that value;
4. the value occurs once in the file;
5. the whole file is byte-identical to the approved historical blob. Its git blob id must equal the approved id, so any other content, version or path is not spared.

Both approved values are held in the scanner as integer literals, so the scanner contains no quoted hex string. No other rule, file or value is affected: there is no hex allow-listing and no scanner disabling. The earlier exemptions are unchanged.

**Security implications.**
- The only thing spared is one known, public digest in one immutable historical blob.
- A real secret anywhere is still detected, including in that same file. Any edit to the file changes its blob id and voids the exemption.
- Git history was **not** rewritten. Commit `b2146c5` and the pre-registration commits `dc830a8` and `aa16b98` are preserved.

**Proof.** `tests/test_publish_scan_p4.py`:
- The historical blob is spared at its path, and the detector does flag it without the exemption.
- The same file at any other path is detected.
- A different digest in the approved line is detected.
- A one-byte edit, an appended line or a repeated value is detected.
- **A deliberately planted, synthetic token and a different hex digest added to the approved file are both still detected** (owner step 6). The planted values are built at runtime; no real key is used.
- The finding type, value, line and path are each required exactly.
- The earlier exemptions are unchanged.

**Scan result** (P4 closure, all local branches, every reachable commit including `b2146c5`, plus the tracked working tree): 52 commits, 233 distinct blob/path pairs, **0 secret findings, 0 content findings**.
- The private denylist was not applied (0 entries). It is kept outside this worktree, and the P4 instruction limits work to this worktree.
- That denylist scan must still be run before publication.
