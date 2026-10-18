# Pre-publication security scan: summary

**Date:** 2026-10-18
**Outcome:** **0 findings** (secret scan: 0 · content scan: 0)

## Tools

| Tool | Version | Purpose |
|---|---|---|
| `detect-secrets` | 1.5.0 | secret detection with all default plugins (high-entropy strings, keyword, cloud and VCS token formats, private keys) |
| `tools/publish_scan.py` | in this repository | content scan: environment files, key and token shapes, database URLs, local absolute paths, code imported from a product codebase, plus a private denylist that is kept outside the repository |

Runtime: Python 3.13.15.

## Scope

- **Every file of every commit** reachable from every local branch (`main` and the P1 branch), including the commit that adds this document.
- The current working tree (all tracked files).

## Documented exclusion

The dataset manifest lists the SHA-256 digests of the data files. These are high-entropy hexadecimal strings by design, not secrets. Only lines with exactly that shape inside `manifest.json` are excluded. A test proves that the same digest anywhere else is still flagged.

## Proof that the scan works

`tests/test_publish_scan.py` checks both directions:
- planted token shapes, database URLs, local paths, environment-file names and denylisted terms **are** detected;
- healthy synthetic content passes.

Planted values are built at runtime, so the test file itself contains no secret-shaped literal.

Raw scanner output is kept locally only and is not part of this repository.
