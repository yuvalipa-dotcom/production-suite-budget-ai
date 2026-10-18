"""Fixed parameters of the synthetic generator (version 1)."""

from __future__ import annotations

SEED = 42
N_PRODUCTIONS = 400
VAT_RATE = 0.18
BATCH_ID = "syn-v1-seed42"
ID_PREFIX = "SYN-"
ANCHOR_DATE = "2024-01-01"          # contract dates fall in [anchor, anchor + CONTRACT_WINDOW_DAYS)
CONTRACT_WINDOW_DAYS = 730
OVERRUN_TOLERANCE_ILS = 0.5         # floating-point rounding tolerance only -- NOT a business threshold
WARNING_THRESHOLD = 0.02            # materiality threshold for warnings only -- never the overrun definition

STAGES = ("S0", "S1", "S2", "S3")
# S0 = contract signed, S1 = 35% of lead time, S2 = 65% of lead time, S3 = 14 days before the event.
S1_FRACTION = 0.35
S2_FRACTION = 0.65
S3_DAYS_BEFORE_EVENT = 14

SPLIT_RATIOS = {"train": 0.70, "test": 0.20, "demo": 0.10}
N_FOLDS = 5

# Category share templates per archetype (shares of the planned vendor pool).
ARCHETYPES = {
    "wedding": {
        "n": 100,
        "budget_gross_median": 320_000, "budget_sigma": 0.35,
        "guests_median": 320, "guests_sigma": 0.30,
        "lead_days": (150, 420),
        "budget_vat_includes_p": 0.85,
        "fee_included_p": 0.70,
        "client_direct_p": 0.35,
        "categories": {
            "venue_catering": 0.50, "music_entertainment": 0.09, "photo_video": 0.08,
            "decor_design": 0.10, "lighting_sound": 0.07, "attire_beauty": 0.06,
            "logistics_transport": 0.04, "gifts_hospitality": 0.03, "staffing_security": 0.03,
        },
        "peak_months": (5, 6, 9, 10),
        "complexity_shift": 0.0,
    },
    "bar_bat_mitzvah": {
        "n": 100,
        "budget_gross_median": 190_000, "budget_sigma": 0.35,
        "guests_median": 220, "guests_sigma": 0.30,
        "lead_days": (120, 330),
        "budget_vat_includes_p": 0.85,
        "fee_included_p": 0.65,
        "client_direct_p": 0.30,
        "categories": {
            "venue_catering": 0.45, "music_entertainment": 0.12, "photo_video": 0.09,
            "decor_design": 0.10, "lighting_sound": 0.08, "gifts_hospitality": 0.06,
            "logistics_transport": 0.05, "staffing_security": 0.05,
        },
        "peak_months": (3, 4, 11, 12),
        "complexity_shift": -0.05,
    },
    "corporate": {
        "n": 100,
        "budget_gross_median": 210_000, "budget_sigma": 0.45,
        "guests_median": 250, "guests_sigma": 0.45,
        "lead_days": (45, 180),
        "budget_vat_includes_p": 0.25,
        "fee_included_p": 0.40,
        "client_direct_p": 0.15,
        "categories": {
            "venue_catering": 0.35, "staging_av": 0.18, "lighting_sound": 0.08,
            "music_entertainment": 0.07, "printing_branding": 0.10, "logistics_transport": 0.08,
            "gifts_hospitality": 0.07, "staffing_security": 0.07,
        },
        "peak_months": (6, 11, 12),
        "complexity_shift": 0.05,
    },
    "conference": {
        "n": 100,
        "budget_gross_median": 380_000, "budget_sigma": 0.45,
        "guests_median": 500, "guests_sigma": 0.45,
        "lead_days": (90, 270),
        "budget_vat_includes_p": 0.20,
        "fee_included_p": 0.35,
        "client_direct_p": 0.10,
        "categories": {
            "venue_catering": 0.30, "staging_av": 0.22, "printing_branding": 0.10,
            "logistics_transport": 0.12, "staffing_security": 0.10, "gifts_hospitality": 0.06,
            "photo_video": 0.05, "lighting_sound": 0.05,
        },
        "peak_months": (3, 5, 10, 11),
        "complexity_shift": 0.10,
    },
}

# Categories that are typically booked at or right after contract signing.
EARLY_CATEGORIES = ("venue_catering",)
