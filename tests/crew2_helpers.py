"""Offline helpers for the Crew-2 tests. ALL DATA IS SYNTHETIC.

* ``small_grids``: replaces the pre-registered grids with one cheap setting per candidate, so the
  full protocol runs in seconds. It changes no rule, split or metric; the real grids are used by
  ``python -m crew2.run`` and are checked against the addendum in ``test_crew2_preregistration.py``.
* ``no_network``: Crew 1's socket blocker.
"""

from __future__ import annotations

import pytest

import crew2  # noqa: F401  (applies the safety environment first)
from crew2.ml import models as M
from tests.crew1_helpers import no_network  # noqa: F401

SMALL = {"B0": [{}], "LR": [{"C": 1.0}], "GB": [{"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}]}


@pytest.fixture(scope="module")
def small_grids():
    with pytest.MonkeyPatch.context() as mp:
        for k, v in SMALL.items():
            mp.setitem(M.GRIDS, k, v)
        yield SMALL


@pytest.fixture(scope="module")
def small_session(small_grids):
    """One full protocol run with the small grids (handoff, 9 CV runs, rule, final fit, test once, demo)."""
    from crew2.ml.protocol import Session
    s = Session()
    s.run_all()
    return s
