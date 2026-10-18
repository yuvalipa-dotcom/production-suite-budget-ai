# Crew 1 isolation: required enforcement and future test

> **Design specification only.** Nothing here is implemented in P1-R: no CrewAI,
> no agents, no tools, no sandbox. **ALL DATA IS SYNTHETIC.**

The future data-analyst crew (Crew 1) must ingest, clean and validate the **raw export** using only `raw/` and `docs/raw_data_dictionary.md`. It must not be able to reach:
- the curated evaluation dataset (the v2 snapshot and ledger tables);
- the defect catalogue;
- the defect log;
- any other ground-truth file.

An instruction in a prompt is **not** a control. The isolation below is technical.

## Enforcement

1. **Fresh staged workspace outside the repository.**
   - Before every run, the orchestrator creates a new empty directory outside the repository tree (a new temporary directory per run).
   - It copies in only a copy of `raw/` and a copy of `docs/raw_data_dictionary.md`.
   - Crew 1's process starts with that workspace as its working directory.
   - The repository is never mounted, linked or reachable from it.
2. **No generic capabilities.**
   - Crew 1 gets no generic file-read, shell, code-execution or directory-listing tool.
   - Every data tool takes a **logical dataset name** (for example `raw.commitments`), never a path.
   - The tool resolves the name through a fixed allow-list to a file **inside the staged workspace**.
3. **Path hardening in every tool.**
   - The resolved path is canonicalised with a real-path call.
   - It is rejected unless it lies inside the workspace root. The following are rejected:
     - absolute paths;
     - `..` traversal;
     - symbolic links;
     - Windows junctions and reparse points;
     - drive-relative, UNC and device paths (`\\?\`, `\\.\`);
     - 8.3 short names;
     - alternate data streams (`file:stream`).
   - Links are checked **before** opening, and the opened handle's final path is checked **after** opening.
4. **No leaking configuration.**
   - No environment variable, config value, prompt, tool description or tool argument given to Crew 1 contains:
     - the repository path;
     - the evaluation dataset path;
     - the ground-truth path.
   - Crew 1 runs with a minimal environment (an allow-list of variables), not the parent environment.
5. **Write only to its own output folder.**
   - The single writable location is `<workspace>/out/`.
   - Writes elsewhere are refused by the output tool.
   - The orchestrator copies the expected artefacts out of `out/` after the run.

## Future isolation test (to implement in a later approved phase)

1. **Direct access attempts must all fail.** For every Crew 1 tool, try to read the evaluation dataset and each ground-truth file:
   - by logical name;
   - by relative traversal (`../../…`);
   - by absolute path;
   - through a planted symbolic link;
   - through a planted junction (Windows);
   - through UNC, device-path and 8.3 forms.
2. **Canary tokens.**
   - Plant unique canary strings in the evaluation dataset and in every ground-truth file.
   - Run Crew 1 end to end.
   - Assert that **no canary** appears in any Crew 1 output, log, trace, tool result or model transcript.
3. **Environment check.** Assert that Crew 1's process environment and configuration contain none of the forbidden paths.
4. **Two-way proof.** A deliberately misconfigured run, with the repository mounted, must make the canary test **fail**. That shows the test can detect a leak.
