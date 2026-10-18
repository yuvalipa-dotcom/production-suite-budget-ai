"""VAT normalisation for amounts stored "as typed" with a per-amount VAT mode.

Semantics (shared with the product's financial model, re-implemented here):

* ``includes_vat`` -- the typed amount already contains VAT.
* ``excludes_vat`` -- the typed amount is net; VAT is added on payment.
* ``no_vat``       -- a VAT-exempt supplier; the typed amount is the full cost.

All analysis is done in pre-VAT shekels (ILS).
"""

from __future__ import annotations

VAT_MODES = ("includes_vat", "excludes_vat", "no_vat")


def pre_vat(amount: float, mode: str, rate: float) -> float:
    """Pre-VAT value of an amount typed in ``mode``."""
    if mode not in VAT_MODES:
        raise ValueError(f"unknown vat mode: {mode!r}")
    return amount / (1.0 + rate) if mode == "includes_vat" else amount


def gross_of(amount: float, mode: str, rate: float) -> float:
    """Amount actually paid (gross) for an amount typed in ``mode``."""
    if mode not in VAT_MODES:
        raise ValueError(f"unknown vat mode: {mode!r}")
    return amount * (1.0 + rate) if mode == "excludes_vat" else amount


def entered_from_pre_vat(pre: float, mode: str, rate: float) -> float:
    """The amount a user would type in ``mode`` for a given pre-VAT value."""
    if mode not in VAT_MODES:
        raise ValueError(f"unknown vat mode: {mode!r}")
    return pre * (1.0 + rate) if mode == "includes_vat" else pre


def fee_entered(budget_entered: float, fee_mode: str, fee_value: float) -> float:
    """Contractual production fee as typed: a percent of the typed budget, or a fixed amount."""
    if fee_mode == "percent":
        return budget_entered * (fee_value / 100.0)
    if fee_mode == "amount":
        return fee_value
    raise ValueError(f"unknown fee mode: {fee_mode!r}")
