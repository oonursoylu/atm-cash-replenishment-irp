"""
Run baseline B2 -- point forecast + IRP MILP.

B2 reuses the proposed system's full simulation pipeline
(rolling_horizon.run_simulation and the CPLEX IRP) without any code change,
pointed at the point-forecast prediction CSV. Two configuration overrides make
the forecast purely point:

  * REAL_DEMAND_CSV_PATH -> predictions/test_predictions_point.csv, where
    d_safety equals d_mean on every present cell.
  * MISSING_DAY_EPS_SAFETY := MISSING_DAY_EPS_MEAN, so the (ATM, day) cells
    absent from the CSV -- filled with operational floor values by
    real_demand -- are also point-consistent. Without this override those
    cells would carry the default eps_safety floor, leaving a small residual
    safety buffer that the point-forecast variant should not have.

Everything else (capacities, fleet, travel matrix, SEED, MIP gap, cost
coefficients) is the validated thesis baseline. Compute is a full 73-day
CPLEX run at the optimize.yaml MIP gap.

Output: docs/point_irp_results.json, consumed by scripts/run_baselines.py.

Usage (from project root, after make_point_forecast_csv.py):
    python scripts/run_point_irp.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.sim.rolling_horizon import run_simulation

N_ATMS = 31


def main() -> None:
    cfg = load_config()

    point_csv = ROOT / "predictions" / "test_predictions_point.csv"
    if not point_csv.exists():
        raise FileNotFoundError(
            f"{point_csv} not found. Run scripts/make_point_forecast_csv.py first."
        )

    cfg["REAL_DEMAND_CSV_PATH"] = str(point_csv)
    cfg["MISSING_DAY_EPS_SAFETY"] = cfg["MISSING_DAY_EPS_MEAN"]

    print("=" * 82)
    print("BASELINE B2 -- point forecast + IRP MILP")
    print(f"Forecast CSV : {point_csv.name}  (d_safety = d_mean)")
    print(f"eps_safety   : {cfg['MISSING_DAY_EPS_SAFETY']} (= eps_mean, point-consistent)")
    print(f"MIP gap      : {cfg['MIP_GAP']} | days: {cfg['SIMULATION_DAYS']} | SEED: {cfg['SEED']}")
    print("=" * 82)

    t0 = time.time()
    kpis, provenance = run_simulation(cfg, return_provenance=True)
    elapsed = time.time() - t0

    op_cost = (
        kpis["travel_cost"]
        + kpis["dispatch_cost"]
        + kpis["drop_fees"]
        + kpis["holding_cost"]
    )
    n_atm_days = cfg["SIMULATION_DAYS"] * N_ATMS

    out = {
        "label": "B2 point-forecast + IRP",
        "stockouts": kpis["stockout_events"],
        "service_level": 1.0 - kpis["stockout_events"] / n_atm_days,
        "op_cost": op_cost,
        "reported_total": op_cost + kpis["stockout_cost"],
        "dispatches": kpis["total_dispatches"],
        "travel": kpis["travel_cost"],
        "dispatch_cost": kpis["dispatch_cost"],
        "drop_fees": kpis["drop_fees"],
        "holding": kpis["holding_cost"],
        "total_deliveries": kpis["total_deliveries"],
        "n_atm_days": n_atm_days,
        "elapsed_sec": elapsed,
        "provenance": provenance,
    }

    out_path = ROOT / "docs" / "point_irp_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(
        f"\nB2 result: {out['stockouts']} stockouts "
        f"({out['service_level'] * 100:.2f}% SL), op cost {op_cost:,.0f} TL"
    )
    print(f"Wrote {out_path}  (compute {elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()