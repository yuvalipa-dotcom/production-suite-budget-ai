# Pre-publication security scan: summary

**Date:** 2026-10-18
**Outcome:** **0 findings.** Secret scan 0 · content scan 0 · commercial identity-term check: **passed**.
No excluded paths, submodules, large-file pointers, files over 5 MB or binary files.

## Tools

| Tool | Version | Purpose |
|---|---|---|
| `detect-secrets` | 1.5.0 | secret detection with all default plugins (high-entropy strings, keyword, cloud and VCS token formats, private keys) |
| `tools/publish_scan.py` | in this repository | content scan: environment files, key and token shapes, database URLs, local absolute paths, code imported from a product codebase, plus a private term list kept outside the repository |
| `tools/publication_check.py` | in this repository | publication dry run: refs, full-history and working-tree scans, excluded paths, object and ignore-rule checks |
| git | 2.55.0 | history and ref enumeration |

Runtime: Python 3.13.15.

## Scope

The check ran in a fresh, disposable clone with its remote removed.

- **Publication refs:** exactly the three branches `main`, `p1-synthetic-generator` and `p1-remediation`. No tags, stashes, notes or other refs.
- **History:** every file of every commit reachable from those branches. That is 21 commits, including the commit that adds this summary.
- **Working trees:** each of the three branches checked out and scanned file by file.
- **Commercial identity terms:** a check against the identity terms of the commercial product was run over the same scope, and **passed**.

## Documented exclusion

The dataset manifests list the SHA-256 digests of the data files. These are high-entropy hexadecimal strings by design, not secrets. Only lines with exactly that shape inside `manifest.json` are excluded. A test proves that the same digest anywhere else is still flagged.

## Proof that the checks work

- `tests/test_publish_scan.py` checks both directions: planted token shapes, database URLs, local paths, environment-file names and private terms (including pattern rules with allow-lists) **are** detected, and healthy synthetic content passes. Planted values are built at runtime, so the test file contains no secret-shaped literal.
- The dry-run checker was also shown to flag an unexpected ref and excluded paths.

Raw scanner output is kept locally only and is not part of this repository.
