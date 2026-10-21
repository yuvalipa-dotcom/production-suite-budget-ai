"""The deterministic modelling core of Crew 2 (P4). ALL DATA IS SYNTHETIC.

Pre-registered in docs/model_selection_rule.md (Addendum P4-1, erratum P4-1a). No module here
calls an LLM or the network; every number Crew 2 reports is computed here. Only
``data.build_handoff`` (the historical in-process rebuild) imports the Crew 1 cleaning function, and
it does so when called, so importing this package loads no Crew 1 module.
"""
