"""
Official frozen-OSM Phase 4.J high-service appendix run.

Runs the Proposed system with the alpha_safety=0.99 forecast:
  High-service variant: quantile forecast + IRP MILP

This is a 73-day appendix/robustness policy variant, not a replacement for the
alpha_safety=0.95 headline. It keeps the frozen headline settings fixed and
changes only the safety-quantile forecast CSV.

Expected command from project root:
  python scripts/run_phase_4j_high_service_alpha099_frozen.py

Output:
  docs/results_frozen/phase_4j_high_service_alpha099_frozen_20260601.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results_frozen" / "phase_4j_high_service_alpha099_frozen_20260601.json"

N_ATMS = 31
TT_HASH = "76013f9295fe036d980740994878c3be"
FORECAST_CSV_REL = "predictions/test_predictions_p0.55_s0.99.csv"
FORECAST_MD5 = "e23806961b4c2cb7290d7dbf3905305f"
REFERENCE_HEADLINE_REL = "docs/results_frozen/phase_4j_proposed_headline_frozen_20260531.json"

PHASES = (
    ("cold_start", 1, 6),
    ("catch_up", 7, 13),
    ("steady_state", 14, 73),
)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_file(path: Path, expected_md5: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = md5_file(path)
    if actual != expected_md5:
        raise RuntimeError(f"{label} hash mismatch: {actual}, expected {expected_md5}")
    return actual


def apply_frozen_high_service_overrides(cfg: dict, forecast_csv: Path) -> dict:
    cfg["SIMULATION_DAYS"] = 73
    cfg["PLANNING_HORIZON"] = 7
    cfg["MIP_GAP"] = 0.02
    cfg["TIME_LIMIT_SEC"] = 600
    cfg["NUM_VEHICLES"] = 3
    cfg["USE_HETEROGENEOUS_CAPACITY"] = True
    cfg["STOCKOUT_PENALTY"] = 3000
    cfg["SAFETY_FLOOR_PEN"] = 0.1
    cfg["INITIAL_INV_LOW"] = 0.30
    cfg["INITIAL_INV_HIGH"] = 0.50
    cfg["SEED"] = 42
    cfg["USE_REAL_DEMAND"] = True
    cfg["REAL_DEMAND_CSV_PATH"] = str(forecast_csv)
    cfg["SYMMETRIZE_TRAVEL_MATRIX"] = True
    cfg["CPLEX_DETERMINISTIC"] = False
    return cfg


def phase_breakdown(daily_log: list[dict]) -> dict[str, int]:
    return {
        name: sum(int(r["stockouts"]) for r in daily_log if lo <= int(r["day"]) <= hi)
        for name, lo, hi in PHASES
    }


def attach_daily_logger(rh) -> list[dict]:
    """Patch rolling_horizon._execute_day locally to record day-level KPI deltas."""
    daily_log: list[dict] = []
    original_execute_day = rh._execute_day

    def recording_execute_day(*args, **kwargs):
        sim_day = int(kwargs["sim_day"])
        cfg = kwargs["cfg"]
        kpis = kwargs["kpis"]

        before = {
            "stockout_events": kpis["stockout_events"],
            "total_dispatches": kpis["total_dispatches"],
            "total_deliveries": kpis["total_deliveries"],
            "travel_cost": kpis["travel_cost"],
            "dispatch_cost": kpis["dispatch_cost"],
            "drop_fees": kpis["drop_fees"],
            "holding_cost": kpis["holding_cost"],
            "stockout_cost": kpis["stockout_cost"],
        }

        ok = original_execute_day(*args, **kwargs)
        if ok:
            drop_delta = kpis["drop_fees"] - before["drop_fees"]
            stops = int(round(drop_delta / cfg["DROP_FEE_PER_ATM"]))
            daily_log.append(
                {
                    "day": sim_day,
                    "stockouts": kpis["stockout_events"] - before["stockout_events"],
                    "dispatches": kpis["total_dispatches"] - before["total_dispatches"],
                    "stops": stops,
                    "delivered": round(kpis["total_deliveries"] - before["total_deliveries"], 2),
                    "travel": round(kpis["travel_cost"] - before["travel_cost"], 2),
                    "dispatch_cost": round(kpis["dispatch_cost"] - before["dispatch_cost"], 2),
                    "drop_fees": round(drop_delta, 2),
                    "holding": round(kpis["holding_cost"] - before["holding_cost"], 2),
                    "stockout_cost": round(kpis["stockout_cost"] - before["stockout_cost"], 2),
                }
            )
        return ok

    rh._execute_day = recording_execute_day
    return daily_log


def main() -> None:
    if OUT.exists():
        print(f"Output already exists: {OUT}")
        print("Remove it first if you intentionally want to rerun this appendix scenario.")
        return

    sys.path.insert(0, str(ROOT))

    from src.config import load_config
    import src.sim.rolling_horizon as rh

    started = time.time()
    forecast_csv = ROOT / FORECAST_CSV_REL
    forecast_md5 = verify_file(forecast_csv, FORECAST_MD5, "Forecast CSV")

    cfg = apply_frozen_high_service_overrides(load_config(), forecast_csv)
    daily_log = attach_daily_logger(rh)

    print("=" * 92)
    print("RUN HIGH-SERVICE VARIANT: alpha_safety=0.99 + IRP MILP, frozen 73-day settings")
    print("=" * 92)
    print(f"Output JSON   : {OUT}")
    print(f"Forecast CSV  : {FORECAST_CSV_REL}")
    print(f"Forecast hash : {forecast_md5}")
    print(f"Frozen TT hash: {TT_HASH}")
    print("Config        : days=73, mip_gap=0.02, seed=42, hetero, nv=3, stockout=3000, sf=0.1")
    print("Policy role   : appendix high-service variant; alpha=0.95 remains the headline")
    print("CPLEX mode    : legacy/default (non-deterministic)")
    print("=" * 92)

    kpis, provenance = rh.run_simulation(cfg, map_generator=None, return_provenance=True)
    compute_sec = round(time.time() - started, 1)

    if provenance["travel_matrix_hash"] != TT_HASH:
        raise RuntimeError(
            f"Travel matrix hash mismatch: {provenance['travel_matrix_hash']}, expected {TT_HASH}"
        )
    if provenance["forecast_csv_hash"] != FORECAST_MD5:
        raise RuntimeError(
            f"Forecast hash mismatch in provenance: {provenance['forecast_csv_hash']}, expected {FORECAST_MD5}"
        )
    if provenance["mip_gap"] != 0.02 or provenance["days"] != 73:
        raise RuntimeError(f"Unexpected headline provenance: {provenance}")
    if abs(float(provenance["alpha_safety"]) - 0.99) > 1e-12:
        raise RuntimeError(f"Unexpected alpha_safety in provenance: {provenance}")

    op_cost = (
        kpis["travel_cost"]
        + kpis["dispatch_cost"]
        + kpis["drop_fees"]
        + kpis["holding_cost"]
    )
    n_atm_days = cfg["SIMULATION_DAYS"] * N_ATMS

    payload = {
        "purpose": "Frozen-OSM alpha_safety=0.99 high-service 73-day appendix scenario.",
        "interpretation_note": (
            "This is not the thesis headline. It shows the service/cost implication of "
            "moving the safety quantile from alpha=0.95 to alpha=0.99 while keeping the "
            "frozen headline IRP settings fixed."
        ),
        "reference_headline": {
            "label": "Proposed alpha_safety=0.95 frozen headline",
            "json": REFERENCE_HEADLINE_REL,
        },
        "locked_config": {
            "travel_matrix": "frozen symmetric OSM",
            "travel_matrix_hash": TT_HASH,
            "forecast_csv": FORECAST_CSV_REL,
            "forecast_csv_hash": FORECAST_MD5,
            "alpha_safety": 0.99,
            "cplex_mode": "legacy/default",
            "mip_gap": 0.02,
            "seed": 42,
            "days": 73,
            "planning_horizon": 7,
            "num_vehicles": 3,
            "use_heterogeneous_capacity": True,
            "stockout_penalty": 3000,
            "safety_floor_pen": 0.1,
            "initial_inv_low": 0.30,
            "initial_inv_high": 0.50,
        },
        "result": {
            "system": "Proposed_alpha099_high_service",
            "label": "High-service alpha=0.99 (quantile + IRP)",
            "status": "complete",
            "compute_sec": compute_sec,
            "summary": {
                "stockouts": kpis["stockout_events"],
                "service_level": 1.0 - kpis["stockout_events"] / n_atm_days,
                "op_cost": round(op_cost, 2),
                "reported_total": round(op_cost + kpis["stockout_cost"], 2),
                "dispatches": kpis["total_dispatches"],
                "travel": round(kpis["travel_cost"], 2),
                "dispatch_cost": round(kpis["dispatch_cost"], 2),
                "drop_fees": round(kpis["drop_fees"], 2),
                "holding": round(kpis["holding_cost"], 2),
                "stockout_cost": round(kpis["stockout_cost"], 2),
                "total_deliveries": round(kpis["total_deliveries"], 2),
                "n_atm_days": n_atm_days,
            },
            "phase_breakdown": phase_breakdown(daily_log),
            "daily_log": daily_log,
            "provenance": provenance,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    s = payload["result"]["summary"]
    print("\n" + "=" * 92)
    print("HIGH-SERVICE FROZEN APPENDIX RUN COMPLETE")
    print("=" * 92)
    print(
        f"stockouts={s['stockouts']} | service={s['service_level'] * 100:.2f}% | "
        f"op_cost={s['op_cost']:,.2f} | reported={s['reported_total']:,.2f} | "
        f"dispatches={s['dispatches']}"
    )
    print(f"Wrote {OUT}")
    print(f"Total elapsed: {compute_sec / 60:.1f} min")


if __name__ == "__main__":
    main()
