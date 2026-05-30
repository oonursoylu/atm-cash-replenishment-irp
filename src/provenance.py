"""
Provenance stamping: a single helper that captures every input that determines a
simulation/optimisation result, so any result writer can embed a reproducibility
record alongside its KPIs.

Design: one pure-ish function, build_provenance(cfg, tt, backend), returning a
flat JSON-serialisable dict. It does no I/O beyond reading the forecast CSV to
hash it and reading library version strings. Writers attach the returned dict to
their output; run_simulation can also return it (see rolling_horizon).
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any


def _file_md5(path: str | Path) -> str | None:
    """md5 of a file's bytes, or None if it cannot be read."""
    try:
        p = Path(path)
        h = hashlib.md5()
        h.update(p.read_bytes())
        return h.hexdigest()
    except Exception:
        return None


def _matrix_hash(tt: dict[tuple[int, int], float] | None) -> str | None:
    """Canonical travel-matrix hash (matches travel.matrix_content_hash)."""
    if tt is None:
        return None
    items = sorted((int(i), int(j), round(float(v), 6)) for (i, j), v in tt.items())
    return hashlib.md5(repr(items).encode()).hexdigest()


def _alpha_safety_from_csv(csv_path: str | None) -> float | None:
    """
    Parse the forecast safety quantile from the prediction CSV filename, e.g.
    test_predictions_p0.55_s0.95.csv -> 0.95. Returns None if not encoded in the
    name (the project's only carrier of alpha_safety today; there is no cfg key).
    """
    if not csv_path:
        return None
    m = re.search(r"_s([0-9]*\.?[0-9]+)\.csv$", str(csv_path))
    return float(m.group(1)) if m else None


def _versions() -> dict[str, Any]:
    """Library/runtime versions relevant to reproducibility."""
    osmnx_v = cplex_v = docplex_v = None
    try:
        import osmnx
        osmnx_v = osmnx.__version__
    except Exception:
        pass
    try:
        import docplex
        docplex_v = docplex.__version__
    except Exception:
        pass
    try:
        import cplex
        cplex_v = cplex.__version__
    except Exception:
        pass
    return {
        "python_version": sys.version.split()[0],
        "osmnx_version": osmnx_v,
        "docplex_version": docplex_v,
        "cplex_version": cplex_v,
    }


def build_provenance(
    cfg: dict,
    tt: dict[tuple[int, int], float] | None = None,
    backend: str | None = None,
    *,
    deterministic: bool = False,
) -> dict[str, Any]:
    """
    Build a flat provenance record for one run.

    Args:
        cfg:     flattened CONFIG dict (see src/config.py).
        tt:      the travel matrix actually used (for its content hash). Optional;
                 pass it when available so the hash reflects the real matrix.
        backend: travel backend label ("OSM"/"HAVERSINE") if known.

    Returns a JSON-serialisable dict. Unknown fields are None rather than absent,
    so downstream schemas stay stable.
    """
    csv_path = cfg.get("REAL_DEMAND_CSV_PATH")
    rec: dict[str, Any] = {
        "travel_matrix_hash": _matrix_hash(tt),
        "backend": backend,
        "symmetrize": cfg.get("SYMMETRIZE_TRAVEL_MATRIX"),
        "forecast_csv_path": csv_path,
        "forecast_csv_hash": _file_md5(csv_path) if csv_path else None,
        "alpha_safety": _alpha_safety_from_csv(csv_path),
        "seed": cfg.get("SEED"),
        "mip_gap": cfg.get("MIP_GAP"),
        "days": cfg.get("SIMULATION_DAYS"),
        "horizon": cfg.get("PLANNING_HORIZON"),
        "num_vehicles": cfg.get("NUM_VEHICLES"),
        "use_heterogeneous_capacity": cfg.get("USE_HETEROGENEOUS_CAPACITY"),
        "stockout_penalty": cfg.get("STOCKOUT_PENALTY"),
        "safety_floor_pen": cfg.get("SAFETY_FLOOR_PEN"),
        "cplex_deterministic": deterministic,
        "cplex_parallel_mode": 1 if deterministic else 0,
        "cplex_threads": 0,
        "cpu_count": os.cpu_count(),
    }
    rec.update(_versions())
    return rec
