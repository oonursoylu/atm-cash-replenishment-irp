"""Real-data demand loader: prediction CSV -> DemandData contract.

Missing (atm, day) cells (zero-filtered at forecast time) are filled with
operational floor values, preserving the inactive-day assumption while
giving IRP a small safety buffer.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .spatial import SpatialData
from .synthetic_demand import DemandData


_REQUIRED_COLS = ("DATE", "CASHP_ID", "WITHDRWLS", "d_mean", "d_safety")


def load_real_demand(
    spatial: SpatialData,
    cfg: dict,
    capacity_per_atm: dict[str, float],
) -> DemandData:
    """
    Required cfg keys: REAL_DEMAND_CSV_PATH, SIMULATION_DAYS, PLANNING_HORIZON,
    MISSING_DAY_EPS_MEAN, MISSING_DAY_EPS_SAFETY, SEED,
    INITIAL_INV_LOW, INITIAL_INV_HIGH.
    """
    csv_path = Path(cfg["REAL_DEMAND_CSV_PATH"])
    total_days = cfg["SIMULATION_DAYS"] + cfg["PLANNING_HORIZON"]
    eps_mean = cfg["MISSING_DAY_EPS_MEAN"]
    eps_safety = cfg["MISSING_DAY_EPS_SAFETY"]
    atms = sorted(spatial["atm_location"].keys())

    df = pd.read_csv(csv_path)
    _validate_csv_schema(df)
    df["DATE"] = pd.to_datetime(df["DATE"])

    date_to_t = _build_date_to_t_mapping(df, total_days)
    df = df[df["DATE"].isin(date_to_t)].copy()
    df = df[df["CASHP_ID"].isin(atms)].copy()
    df["t"] = df["DATE"].map(date_to_t)

    d_mean, d_safety, actual_demand = _build_dense_panels(
        df, atms, total_days, capacity_per_atm,
        eps_mean=eps_mean, eps_safety=eps_safety,
    )
    initial_inventory = _sample_initial_inventory(
        atms, capacity_per_atm,
        seed=cfg["SEED"],
        low_pct=cfg["INITIAL_INV_LOW"],
        high_pct=cfg["INITIAL_INV_HIGH"],
    )

    return {
        "d_mean": d_mean,
        "d_safety": d_safety,
        "actual_demand": actual_demand,
        "initial_inventory": initial_inventory,
    }


def _validate_csv_schema(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Prediction CSV missing required columns: {missing}. "
            f"Got: {list(df.columns)}"
        )


def _build_date_to_t_mapping(
    df: pd.DataFrame, total_days: int,
) -> dict[pd.Timestamp, int]:
    """First `total_days` distinct sorted dates -> t=1..total_days.
    Errors out if CSV has fewer dates than the simulation needs."""
    unique_dates = sorted(pd.to_datetime(df["DATE"].unique()))
    if len(unique_dates) < total_days:
        raise ValueError(
            f"CSV has {len(unique_dates)} unique dates; simulation needs "
            f"{total_days} (SIMULATION_DAYS + PLANNING_HORIZON). "
            f"Reduce SIMULATION_DAYS or extend the prediction window."
        )
    return {d: i + 1 for i, d in enumerate(unique_dates[:total_days])}


def _build_dense_panels(
    df: pd.DataFrame,
    atms: list[str],
    total_days: int,
    capacity_per_atm: dict[str, float],
    eps_mean: float,
    eps_safety: float,
) -> tuple[dict, dict, dict]:
    """Iterate the full (atm, t) grid; row present -> use it, missing -> eps."""
    indexed = df.set_index(["CASHP_ID", "t"]).sort_index()
    d_mean: dict[tuple[str, int], float] = {}
    d_safety: dict[tuple[str, int], float] = {}
    actual: dict[tuple[str, int], float] = {}

    for a in atms:
        cap_a = capacity_per_atm[a]
        for t in range(1, total_days + 1):
            try:
                row = indexed.loc[(a, t)]
                actual_val = min(max(0.0, float(row["WITHDRWLS"])), cap_a)
                m = float(row["d_mean"])
                s = max(float(row["d_safety"]), m)
            except KeyError:
                m = float(eps_mean)
                s = float(eps_safety)
                actual_val = 0.0
            d_mean[(a, t)] = m
            d_safety[(a, t)] = s
            actual[(a, t)] = actual_val

    return d_mean, d_safety, actual


def _sample_initial_inventory(
    atms: list[str],
    capacity_per_atm: dict[str, float],
    seed: int,
    low_pct: float,
    high_pct: float,
) -> dict[str, float]:
    rng = random.Random(seed)
    return {
        a: rng.uniform(low_pct, high_pct) * capacity_per_atm[a]
        for a in atms
    }
