"""Latent per-company demand: the single source of the planted signal.

Transactions are sampled from `demand`; stock returns respond to lagged
`shock`. Both consumers read this one panel, so the correlation the model is
asked to find is exactly the correlation that was planted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def build_demand_panel(seed: int = config.SEED) -> pd.DataFrame:
    days = config.trading_days(config.START_DATE, config.END_DATE)
    n = len(days)
    rows = []

    for idx, company in enumerate(config.COMPANIES):
        rng = np.random.default_rng(seed * 1000 + idx)

        # Innovations drive both demand level and (lagged) price response.
        innovations = rng.standard_normal(n)

        # Persistent component: demand trends rather than jumping day to day.
        ar = np.zeros(n)
        phi = 0.85
        for t in range(1, n):
            ar[t] = phi * ar[t - 1] + innovations[t]
        ar_std = ar.std()
        level = ar / ar_std if ar_std > 0 else ar

        t_index = np.arange(n)
        annual = 0.12 * np.sin(2 * np.pi * t_index / 252.0 + idx)
        trend = 0.15 * t_index / n
        dow = np.array([_dow_factor(d.weekday(), idx) for d in days])

        base = 1.0 + trend + annual + 0.20 * level
        demand_values = np.clip(base, 0.15, None) * dow

        shock = (innovations - innovations.mean()) / innovations.std()

        rows.append(pd.DataFrame({
            "ticker": company.ticker,
            "day": days,
            "demand": demand_values,
            "shock": shock,
        }))

    return pd.concat(rows, ignore_index=True)


def _dow_factor(weekday: int, company_idx: int) -> float:
    """Weekday seasonality, phase-shifted per company."""
    patterns = (
        (1.05, 1.00, 1.00, 1.02, 1.18, 1.00, 1.00),  # Friday-heavy
        (0.95, 0.98, 1.02, 1.05, 1.20, 1.00, 1.00),  # ramps into weekend
        (1.00, 0.96, 0.96, 1.00, 1.15, 1.00, 1.00),
    )
    return patterns[company_idx % len(patterns)][weekday]
