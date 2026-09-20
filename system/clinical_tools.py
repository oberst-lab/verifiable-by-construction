"""Vetted external clinical calculators the agent can surface."""

from __future__ import annotations

from .models import ClinicalCalculator

# AHA PREVENT — covers 10/30-year CVD, ASCVD, and heart-failure risk.
PREVENT = ClinicalCalculator(
    name="AHA PREVENT Risk Calculator",
    url="https://professional.heart.org/en/guidelines-and-statements/prevent-calculator",
    description=(
        "American Heart Association PREVENT™ calculator for 10-year and 30-year "
        "risk of cardiovascular disease (CVD), atherosclerotic CVD (ASCVD), and "
        "heart failure."
    ),
)
