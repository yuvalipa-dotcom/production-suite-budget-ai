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

## Proof that the checks work

- `tests/test_publish_scan.py` checks both directions: planted token shapes, database URLs, local paths, environment-file names and private terms (including pattern rules with allow-lists) **are** detected, and healthy synthetic content passes. Planted values are built at runtime, so the test file contains no secret-shaped literal.
- The dry-run checker was also shown to flag an unexpected ref and excluded paths.

## Coverage gap (stated, not bypassed)

One commit-message pattern check, for references to environment-file names, was **not executed**. A safety hook in the development environment blocked the command that contained the pattern, and the block was **not bypassed** with another tool or wording.

Commit messages were still scanned for secrets, private terms, absolute local paths, internal file names and e-mail addresses. Separately, environment *files* are excluded and checked by path in every commit tree and working tree.

Raw scanner output is kept locally only and is not part of this repository.
