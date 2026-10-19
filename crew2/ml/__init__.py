"""The deterministic modelling core of Crew 2 (P4). ALL DATA IS SYNTHETIC.

Pre-registered in docs/model_selection_rule.md (Addendum P4-1, erratum P4-1a). No module here
calls an LLM or the network; every number Crew 2 reports is computed here. (Importing the Crew 1
cleaning function loads the ``crew1.tools`` package, which imports CrewAI's tool base class.)
"""
