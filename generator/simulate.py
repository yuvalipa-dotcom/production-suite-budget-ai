"""Simulate the full life cycle of synthetic productions.

ALL OUTPUT IS SYNTHETIC.

Each production is simulated as a process, not a formula:

1. A contract is signed (day 0) with a client-approved budget, a VAT mode and a
   contractual production fee (percent of budget or fixed amount; included in
   the budget or charged on top).
2. The planned vendor pool is split into category plans.
3. Supplier commitments are selected over the lead time (each with its own VAT
   mode and payment responsibility: paid by the company or by the client directly).
4. Some commitments are cancelled and substituted; a cancellation fee is recorded
   as a manual cost.
5. Unexpected expenses appear as manual costs, some before and many after the
   later planning stages.
6. Some clients revise the approved budget (and guest count) mid-way.
7. After the event every commitment receives a verified final cost (additions,
   reductions and credits relative to the original accepted quote), the company
   pays its share, and the production is financially closed.
8. A late fee concession may reduce the company's final profit; it never changes
   the overrun label.

Hidden latent factors (client volatility, planner discipline, venue complexity,
season pressure) drive the process. They are NEVER written to any output file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .vat import entered_from_pre_vat, fee_entered, pre_vat


def _r2(x: float) -> float:
    """Round to agorot (2 decimals) the way a stored money value would be."""
    return float(round(x + 0.0, 2))


@dataclass
class Commitment:
    commitment_id: str
    production_id: str
    category: str
    vat_mode: str
    payment_responsibility: str        # "company" | "client_direct"
    original_entered: float            # accepted quote, as typed
    original_pre_vat: float
    selected_day: int
    cancelled_day: int | None = None   # day the commitment was cancelled (substituted)
    final_pre_vat: float | None = None  # verified final supplier cost (pre-VAT), known only at close
    replaces: str | None = None


@dataclass
class ManualCost:
    manual_cost_id: str
    production_id: str
    category: str
    kind: str                          # "unexpected" | "cancellation_fee"
    vat_mode: str
    entered: float
    pre_vat: float
    created_day: int


@dataclass
class Payment:
    payment_id: str
    production_id: str
    commitment_id: str
    direction: str                     # always "out" (company -> supplier)
    amount_gross: float
    paid_day: int


@dataclass
class BudgetRevision:
    day: int
    budget_entered: float
    guest_count: int


@dataclass
class Production:
    production_id: str
    archetype: str
    contract_offset_days: int          # contract date = ANCHOR_DATE + offset
    lead_time_days: int                # event date = contract + lead
    close_after_event_days: int        # financial close = event + this
    event_month: int
    budget_vat_mode: str
    fee_mode: str
    fee_value: float
    fee_vat_mode: str
    fee_included: bool
    initial_budget_entered: float
    initial_guest_count: int
    revisions: list[BudgetRevision] = field(default_factory=list)
    category_plans_entered: dict[str, float] = field(default_factory=dict)
    category_plan_vat_mode: str = "includes_vat"
    commitments: list[Commitment] = field(default_factory=list)
    manual_costs: list[ManualCost] = field(default_factory=list)
    payments: list[Payment] = field(default_factory=list)
    fee_concession_pre_vat: float = 0.0

    # ---- state as of a given day (prediction-time knowledge only) ----
    def budget_entered_at(self, day: int) -> float:
        value = self.initial_budget_entered
        for rev in self.revisions:
            if rev.day <= day:
                value = rev.budget_entered
        return value

    def guest_count_at(self, day: int) -> int:
        value = self.initial_guest_count
        for rev in self.revisions:
            if rev.day <= day:
                value = rev.guest_count
        return value

    def revisions_to_date(self, day: int) -> int:
        return sum(1 for rev in self.revisions if rev.day <= day)

    def budget_pre_vat_at(self, day: int) -> float:
        return pre_vat(self.budget_entered_at(day), self.budget_vat_mode, C.VAT_RATE)

    def fee_pre_vat_at(self, day: int) -> float:
        """Contractual fee known at ``day`` (no later concession)."""
        entered = fee_entered(self.budget_entered_at(day), self.fee_mode, self.fee_value)
        return pre_vat(_r2(entered), self.fee_vat_mode, C.VAT_RATE)

    @property
    def close_day(self) -> int:
        return self.lead_time_days + self.close_after_event_days

    def final_cost_pre_vat(self) -> float:
        """Verified final supplier costs + all manual costs (pre-VAT). Known at close only."""
        active = sum(c.final_pre_vat for c in self.commitments if c.cancelled_day is None)
        manual = sum(m.pre_vat for m in self.manual_costs)
        return active + manual


def _pick_vat_mode(rng: np.random.Generator, includes_p: float) -> str:
    u = rng.random()
    if u < includes_p:
        return "includes_vat"
    if u < includes_p + (1 - includes_p) * 0.85:
        return "excludes_vat"
    return "no_vat"


def simulate_production(index: int, archetype: str, rng: np.random.Generator) -> Production:
    a = C.ARCHETYPES[archetype]
    pid = f"{C.ID_PREFIX}{index:04d}"

    # ---- hidden latent factors (never exported) ----
    volatility = rng.beta(2, 3)
    discipline = rng.beta(3, 2)
    complexity = float(np.clip(rng.beta(2, 2) + a["complexity_shift"], 0, 1))

    contract_offset = int(rng.integers(0, C.CONTRACT_WINDOW_DAYS))
    lo, hi = a["lead_days"]
    lead = int(rng.integers(lo, hi + 1))
    event_month = int((np.datetime64(C.ANCHOR_DATE) + np.timedelta64(contract_offset + lead, "D"))
                      .astype("datetime64[M]").astype(int) % 12 + 1)
    season = 1.0 if event_month in a["peak_months"] else 0.0

    budget_vat_mode = "includes_vat" if rng.random() < a["budget_vat_includes_p"] else "excludes_vat"
    budget_gross = a["budget_gross_median"] * float(np.exp(rng.normal(0, a["budget_sigma"])))
    budget_entered = float(round(budget_gross / 1000.0) * 1000.0)
    if budget_vat_mode == "excludes_vat":
        budget_entered = float(round(budget_entered / 1.18 / 1000.0) * 1000.0)
    budget_entered = max(budget_entered, 30_000.0)
    guests = int(max(30, round(a["guests_median"] * float(np.exp(rng.normal(0, a["guests_sigma"]))))))

    fee_included = bool(rng.random() < a["fee_included_p"])
    if rng.random() < 0.6:
        fee_mode, fee_value = "percent", float(round(rng.uniform(8, 15), 1))
    else:
        fee_mode = "amount"
        fee_value = float(round(budget_entered * rng.uniform(0.07, 0.14) / 500.0) * 500.0)
    fee_vat_mode = budget_vat_mode

    p = Production(
        production_id=pid, archetype=archetype, contract_offset_days=contract_offset,
        lead_time_days=lead, close_after_event_days=int(rng.integers(14, 61)),
        event_month=event_month, budget_vat_mode=budget_vat_mode, fee_mode=fee_mode,
        fee_value=fee_value, fee_vat_mode=fee_vat_mode, fee_included=fee_included,
        initial_budget_entered=budget_entered, initial_guest_count=guests,
    )

    # ---- budget revisions (scope changes) ----
    if rng.random() < 0.15 + 0.35 * volatility:
        n_rev = 1 if rng.random() < 0.7 else 2
        days = sorted(int(rng.integers(int(0.2 * lead), int(0.9 * lead))) for _ in range(n_rev))
        current_b, current_g = budget_entered, guests
        for d in days:
            factor = rng.uniform(1.03, 1.15) if rng.random() < 0.85 else rng.uniform(0.95, 0.99)
            current_b = float(round(current_b * factor / 1000.0) * 1000.0)
            current_g = int(round(current_g * (1 + (factor - 1) * rng.uniform(0.5, 1.5))))
            p.revisions.append(BudgetRevision(day=max(d, 1), budget_entered=current_b, guest_count=current_g))

    # ---- category plans (pre-VAT shares of the vendor pool) ----
    budget_pre = p.budget_pre_vat_at(0)
    fee_pre = p.fee_pre_vat_at(0)
    vendor_pool = budget_pre - (fee_pre if fee_included else 0.0)
    planning_ratio = 0.98 - 0.10 * discipline + rng.normal(0, 0.02)
    cats = list(a["categories"].keys())
    shares = np.array([a["categories"][c] for c in cats])
    keep = [c for c in cats if c in C.EARLY_CATEGORIES or rng.random() > 0.12]
    idx = [cats.index(c) for c in keep]
    weights = rng.dirichlet(shares[idx] * 60)
    p.category_plan_vat_mode = budget_vat_mode
    for c, w in zip(keep, weights):
        planned_pre = vendor_pool * planning_ratio * float(w)
        p.category_plans_entered[c] = float(round(entered_from_pre_vat(planned_pre, budget_vat_mode, C.VAT_RATE)))

    # ---- supplier commitments ----
    seq = 0

    def new_commitment(category: str, planned_pre: float, day: int, replaces: str | None = None,
                       markup: float = 1.0) -> Commitment:
        nonlocal seq
        seq += 1
        quote_ratio = float(np.exp(rng.normal(0.02 * season + 0.04 * (complexity - 0.5), 0.10))) * markup
        vat_mode = _pick_vat_mode(rng, 0.55 if budget_vat_mode == "includes_vat" else 0.20)
        orig_pre = planned_pre * quote_ratio
        orig_entered = float(round(entered_from_pre_vat(orig_pre, vat_mode, C.VAT_RATE) / 100.0) * 100.0)
        orig_entered = max(orig_entered, 500.0)
        responsibility = "client_direct" if rng.random() < a["client_direct_p"] else "company"
        return Commitment(
            commitment_id=f"{pid}-C{seq:02d}", production_id=pid, category=category, vat_mode=vat_mode,
            payment_responsibility=responsibility, original_entered=orig_entered,
            original_pre_vat=_r2(pre_vat(orig_entered, vat_mode, C.VAT_RATE)), selected_day=day,
            replaces=replaces,
        )

    for c in keep:
        planned_pre = pre_vat(p.category_plans_entered[c], budget_vat_mode, C.VAT_RATE)
        n_parts = 2 if (c not in C.EARLY_CATEGORIES and rng.random() < 0.2) else 1
        for part in range(n_parts):
            if c in C.EARLY_CATEGORIES:
                day = 0 if rng.random() < 0.5 else int(rng.integers(1, max(2, int(0.25 * lead))))
            else:
                start = 0.08 + 0.10 * (1 - discipline)
                frac = rng.uniform(start, min(0.95, start + 0.75))
                day = max(1, int(frac * lead))
            p.commitments.append(new_commitment(c, planned_pre / n_parts, day))

    # ---- cancellations and substitutions ----
    mseq = 0
    for cm in list(p.commitments):
        if rng.random() < 0.05 + 0.10 * volatility:
            earliest, latest = cm.selected_day + 5, lead - 7
            if earliest >= latest:
                continue
            cday = int(rng.integers(earliest, latest))
            cm.cancelled_day = cday
            planned_pre = cm.original_pre_vat
            p.commitments.append(new_commitment(cm.category, planned_pre, cday, replaces=cm.commitment_id,
                                                markup=float(rng.uniform(1.02, 1.25))))
            fee_frac = float(rng.uniform(0.0, 0.30))
            if fee_frac > 0.02:
                mseq += 1
                pre = cm.original_pre_vat * fee_frac
                entered = float(round(entered_from_pre_vat(pre, cm.vat_mode, C.VAT_RATE)))
                p.manual_costs.append(ManualCost(
                    manual_cost_id=f"{pid}-M{mseq:02d}", production_id=pid, category=cm.category,
                    kind="cancellation_fee", vat_mode=cm.vat_mode, entered=entered,
                    pre_vat=_r2(pre_vat(entered, cm.vat_mode, C.VAT_RATE)), created_day=cday,
                ))

    # ---- unexpected expenses ----
    n_unexpected = int(rng.poisson(0.6 + 2.0 * complexity + 1.5 * volatility))
    for _ in range(n_unexpected):
        if rng.random() < 0.4:
            day = int(rng.integers(max(1, int(0.2 * lead)), max(2, int(0.65 * lead))))
        else:
            day = int(rng.integers(int(0.65 * lead), lead + 11))
        pre = budget_pre * float(np.clip(np.exp(rng.normal(np.log(0.012), 0.6)), 0.002, 0.06))
        vat_mode = _pick_vat_mode(rng, 0.5)
        entered = float(round(entered_from_pre_vat(pre, vat_mode, C.VAT_RATE)))
        mseq += 1
        p.manual_costs.append(ManualCost(
            manual_cost_id=f"{pid}-M{mseq:02d}", production_id=pid,
            category=str(rng.choice(keep)), kind="unexpected", vat_mode=vat_mode, entered=entered,
            pre_vat=_r2(pre_vat(entered, vat_mode, C.VAT_RATE)), created_day=day,
        ))

    # ---- verified final costs (additions, reductions, credits) ----
    for cm in p.commitments:
        if cm.cancelled_day is not None:
            cm.final_pre_vat = 0.0
            continue
        mu = -0.015 + 0.08 * volatility + 0.05 * complexity - 0.04 * discipline
        sigma = 0.04 + 0.04 * volatility
        adj = float(np.clip(rng.normal(mu, sigma), -0.15, 0.45))
        cm.final_pre_vat = _r2(cm.original_pre_vat * (1 + adj))

    # ---- company payments (client-direct commitments are paid by the client, not recorded here) ----
    pseq = 0
    for cm in p.commitments:
        if cm.payment_responsibility != "company":
            continue
        final_gross = cm.final_pre_vat * (1 + C.VAT_RATE) if cm.vat_mode != "no_vat" else cm.final_pre_vat
        deposit = _r2(cm.original_entered * 0.3 * (1 + C.VAT_RATE if cm.vat_mode == "excludes_vat" else 1))
        if cm.cancelled_day is not None:
            continue  # the forfeited deposit is represented by the cancellation-fee manual cost
        pseq += 1
        p.payments.append(Payment(f"{pid}-P{pseq:02d}", pid, cm.commitment_id, "out",
                                  min(deposit, _r2(final_gross)),
                                  min(cm.selected_day + int(rng.integers(0, 8)), lead)))
        balance = _r2(final_gross - min(deposit, _r2(final_gross)))
        if balance > 0:
            pseq += 1
            p.payments.append(Payment(f"{pid}-P{pseq:02d}", pid, cm.commitment_id, "out", balance,
                                      lead + int(rng.integers(1, p.close_after_event_days + 1))))

    # ---- late fee concession: affects company profit only, never the overrun label ----
    if rng.random() < 0.15:
        p.fee_concession_pre_vat = _r2(p.fee_pre_vat_at(p.close_day) * float(rng.uniform(0.05, 0.30)))

    return p


def simulate_all(seed: int = C.SEED) -> list[Production]:
    """Simulate every production from one seed. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    productions: list[Production] = []
    index = 1
    for archetype, spec in C.ARCHETYPES.items():
        for _ in range(spec["n"]):
            child = np.random.default_rng(rng.integers(0, 2**63 - 1))
            productions.append(simulate_production(index, archetype, child))
            index += 1
    return productions
