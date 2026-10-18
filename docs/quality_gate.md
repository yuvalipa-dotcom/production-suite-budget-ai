# Deterministic quality gate

P2 validates source raw files and Crew-1 handoffs offline. The gate derives evidence from raw records; tests use committed synthetic snapshots and ground truth only as an independent oracle. It emits byte-stable JSON and Markdown reports.

Source checks: `G-SRC-MANIFEST`, `G-SRC-ROLES`, `G-SRC-OUTCOME`. Handoff checks: `G-IN-FILES`, `G-CONTRACT-SCHEMA`, `G-CONTRACT-PROVENANCE`, `G-SCHEMA-COLUMNS`, `G-SCHEMA-DTYPES`, `G-ROLES`, `G-LEAKAGE-OUTCOME`, `G-KEY-UNIQUE`, `G-ELIGIBILITY-SET`, `G-ELIGIBILITY-COUNTS`, `G-TARGET-GAP`, `G-TARGET-LABEL`, `G-TARGET-CONSISTENCY`, `G-FIN-RECON`, `G-FIN-ASOF-BUDGET`, `G-NONFIN-RULES`, `G-SPLIT-GROUP`, `G-SPLIT-FOLD`, `G-NULLS`, `G-RANGES`; `W-PREVALENCE-IMBALANCE` and `I-SYNTHETIC` never fail.

P5 Flow runs source before handoff and handoff before model consumption. Corruption demonstrations use copied handoffs and internally consistent copied contracts; independent raw reconciliation must still fail.
