"""The P5 Flow: the file-based handoff from Crew 1 to Crew 2. ALL DATA IS SYNTHETIC.

Strict crew boundary: nothing in this package imports Crew 1 (directly or through a package
``__init__``), reads the raw export under ``data/synthetic/``, or calls Crew 1's cleaning code.
Crew 2's Flow path receives only two explicit files written by Crew 1, ``clean_data.csv`` and
``dataset_contract.json``, validates them against the versioned Flow contract (``flow-1``) and
a pinned reference profile, and redacts every test-split outcome label before any Crew 2
component sees the data.
"""
