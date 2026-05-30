"""
Synthetic demand generator: three ATM archetypes modulated by day-of-week factors.
Produces forecast (mean/safety) and actual demand with stochastic noise layers.
"""

import random
from typing import TypedDict

import numpy as np

from .spatial import SpatialData


# Archetype demand-scale buckets. Selected by ATM-ID hash modulo 3, so that
# determinism is preserved across runs without relying on configuration order.
# Ranges chosen to span the empirical real-data scale (low <16k, mid 18-55k,
# high >55k TL daily mean). Will be discarded once real demand is available.
_ATM_ARCHETYPES = {
    "low":  {"mean_range": (2_500, 16_000),  "cv_range": (0.80, 1.30)},
    "mid":  {"mean_range": (18_000, 55_000), "cv_range": (0.70, 1.10)},
    "high": {"mean_range": (55_000, 98_000), "cv_range": (0.60, 0.95)},
}

# Day-of-week multipliers indexed by (day - 1) % 7. Pattern derived from the
# Ekinci et al. (2015) Istanbul ATM study and matches the observed weekly
# seasonality in the real-data file (Mon-Sun, with Sun lowest).
_DOW_FACTORS = [1.29, 1.09, 1.04, 1.05, 1.22, 0.84, 0.47]


class DemandData(TypedDict):
    """Container for forecast inputs and ground-truth realisations (keys: (atm_id, day))."""
    d_mean: dict[tuple[str, int], float]            # forecast point estimate
    d_safety: dict[tuple[str, int], float]          # forecast safety-level estimate
    actual_demand: dict[tuple[str, int], float]     # realised demand (ground truth)
    initial_inventory: dict[str, float]             # starting cash level per ATM


class DemandConfig(TypedDict):
    """CONFIG keys consumed by this module. Other keys are ignored."""
    SIMULATION_DAYS: int
    PLANNING_HORIZON: int
    ATM_CAPACITY: float
    INITIAL_INV_LOW: float
    INITIAL_INV_HIGH: float
    SEED: int


def _assign_archetype(atm_id: str) -> str:
    """
    Deterministic archetype assignment via ATM-ID character-sum hash.
    Stable across Python sessions because it does not rely on hash().
    """
    h = sum(ord(c) for c in atm_id)
    if h % 3 == 0:
        return "low"
    elif h % 3 == 1:
        return "mid"
    return "high"


def generate_master_timeseries(
    spatial: SpatialData,
    cfg: DemandConfig,
    capacity_per_atm: dict[str, float] | None = None,
) -> DemandData:
    """
    Generate full demand series (simulation_days + planning_horizon days).
    capacity_per_atm: per-ATM capacity (hetero) or None for uniform mode.
    Seed both RNGs for reproducibility. Caps: d_mean ≤ 60K, d_safety ≤ 130K, actual ≤ capacity.
    """
    atms = sorted(spatial["atm_location"].keys())
    total_days = cfg["SIMULATION_DAYS"] + cfg["PLANNING_HORIZON"]
    use_hetero = capacity_per_atm is not None
    rng_np = np.random.default_rng(cfg["SEED"])
    rng_py = random.Random(cfg["SEED"])

    d_mean: dict[tuple[str, int], float] = {}
    d_safety: dict[tuple[str, int], float] = {}
    actual_demand: dict[tuple[str, int], float] = {}
    initial_inventory: dict[str, float] = {}

    for a in atms:
        archetype = _assign_archetype(a)
        base_lo, base_hi = _ATM_ARCHETYPES[archetype]["mean_range"]
        cv_lo, cv_hi = _ATM_ARCHETYPES[archetype]["cv_range"]

        base_mean = rng_py.uniform(base_lo, base_hi)
        rng_py.uniform(cv_lo, cv_hi)  # Discarded draw for consistent seed sequence
        safety_mult = rng_py.uniform(1.80, 4.00)

        C_a = capacity_per_atm[a] if use_hetero else cfg["ATM_CAPACITY"]

        for t in range(1, total_days + 1):
            dow = (t - 1) % 7
            mu_t = base_mean * _DOW_FACTORS[dow]

            forecast_noise = rng_np.normal(loc=1.0, scale=0.15)
            mean_val = max(0.0, mu_t * forecast_noise)
            jitter = rng_np.uniform(0.95, 1.08)
            safety_val = max(mean_val, mean_val * safety_mult * jitter)

            mean_val = min(mean_val, 60_000)
            safety_val = min(safety_val, 130_000)
            safety_val = max(safety_val, mean_val)

            actual_noise = rng_np.normal(loc=1.0, scale=0.20)
            actual_val = min(max(0.0, mu_t * actual_noise), C_a)

            d_mean[(a, t)] = mean_val
            d_safety[(a, t)] = safety_val
            actual_demand[(a, t)] = actual_val

        # Single uniform draw for consistent seed sequence; mode-agnostic band.
        initial_inventory[a] = rng_py.uniform(
            cfg["INITIAL_INV_LOW"], cfg["INITIAL_INV_HIGH"]
        ) * C_a

    return {
        "d_mean": d_mean,
        "d_safety": d_safety,
        "actual_demand": actual_demand,
        "initial_inventory": initial_inventory,
    }