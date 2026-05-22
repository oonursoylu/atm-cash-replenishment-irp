"""
Assemble the baseline comparison for the predict-then-optimize ablation study.

This script runs the two greedy baselines directly (B0 static s,S; B1 quantile
+ greedy) and assembles a four-system comparison against B2 (point + IRP) and
the proposed system. B2 is produced by a separate CPLEX run
(scripts/run_point_irp.py) and read here from docs/point_irp_results.json; the
proposed-system figures are the 73-day headline run values.

Each baseline removes exactly one component from the proposed system, so each
baseline-to-Proposed difference isolates a single effect.

Prerequisites (from project root):
    python scripts/dump_train_means.py         # B0 demand model
    python scripts/make_point_forecast_csv.py   # B2 input CSV
    python scripts/run_point_irp.py             # B2 CPLEX run -> JSON

Usage:
    python scripts/run_baselines.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.baselines.common import Instance, load_instance, summarise
from src.baselines.static_ss import run_b0
from src.baselines.forecast_threshold import run_b1


# Proposed system: 73-day headline run (docs/headline_73day_run_results.md).
PROPOSED = {
    "label": "Proposed (quantile + IRP)",
    "stockouts": 121,
    "op_cost": 156_295.0,
    "reported_total": 519_295.0,
    "dispatches": 87,
}

# Proposed-system three-phase day boundaries (headline 73-day run).
PHASES = (
    ("cold_start", 1, 6),
    ("catch_up", 7, 13),
    ("steady_state", 14, 73),
)

N_ATMS = 31


def load_train_means(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/dump_train_means.py first."
        )
    means: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            means[row["CASHP_ID"]] = float(row["train_mean_withdrawal"])
    return means


def load_b2(path: Path, n_atm_days: int) -> dict | None:
    """Read the B2 (point + IRP) result written by run_point_irp.py."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "label": "B2 (point + IRP)",
        "stockouts": d["stockouts"],
        "service_level": d.get("service_level", 1.0 - d["stockouts"] / n_atm_days),
        "op_cost": d["op_cost"],
        "reported_total": d["reported_total"],
        "dispatches": d["dispatches"],
    }


def phase_breakdown(daily_log: list[dict]) -> dict[str, int]:
    return {
        name: sum(r["stockouts"] for r in daily_log if lo <= r["day"] <= hi)
        for name, lo, hi in PHASES
    }


def _row(name: str, s: dict) -> str:
    return (
        f"  {name:<26}{s['stockouts']:>10}{s['service_level'] * 100:>10.2f}%"
        f"{s['op_cost']:>16,.0f}{s['reported_total']:>16,.0f}{s['dispatches']:>12}"
    )


def _delta(label: str, frm: dict, to: dict) -> str:
    return (
        f"  {label:<46}"
        f"stockouts {to['stockouts'] - frm['stockouts']:+d}, "
        f"op cost {to['op_cost'] - frm['op_cost']:+,.0f} TL"
    )


def main() -> None:
    cfg = load_config()
    n_atm_days = cfg["SIMULATION_DAYS"] * N_ATMS

    print("=" * 96)
    print("BASELINE COMPARISON  --  predict-then-optimize ablation study")
    print(
        f"Window: {cfg['SIMULATION_DAYS']} days | SEED={cfg['SEED']} | "
        f"heterogeneous capacity | nv={cfg['NUM_VEHICLES']} | {n_atm_days} ATM-days"
    )
    print("=" * 96)

    instance: Instance = load_instance(cfg)
    train_means = load_train_means(ROOT / "data" / "train_atm_means.csv")

    print("\nRunning B0 (static s,S) ...")
    b0 = run_b0(cfg, instance, train_means)
    print("Running B1 (quantile + greedy) ...")
    b1 = run_b1(cfg, instance)

    s0 = summarise(b0["kpis"], n_atm_days)
    s1 = summarise(b1["kpis"], n_atm_days)
    s0["label"], s1["label"] = "B0 (static s,S)", "B1 (quantile + greedy)"

    s2 = load_b2(ROOT / "docs" / "point_irp_results.json", n_atm_days)
    sp = {
        "label": PROPOSED["label"],
        "stockouts": PROPOSED["stockouts"],
        "service_level": 1.0 - PROPOSED["stockouts"] / n_atm_days,
        "op_cost": PROPOSED["op_cost"],
        "reported_total": PROPOSED["reported_total"],
        "dispatches": PROPOSED["dispatches"],
    }

    print("\n" + "-" * 96)
    print(
        f"  {'System':<26}{'Stockouts':>10}{'Service':>11}"
        f"{'Op cost (TL)':>16}{'Reported (TL)':>16}{'Dispatches':>12}"
    )
    print("-" * 96)
    print(_row(s0["label"], s0))
    print(_row(s1["label"], s1))
    if s2 is not None:
        print(_row(s2["label"], s2))
    else:
        print("  B2 (point + IRP)          [run scripts/run_point_irp.py to populate]")
    print(_row(sp["label"], sp))
    print("-" * 96)

    print("\nOne-component ablation deltas (operational cost and stockout count):")
    print(_delta("B1 -> Proposed   (IRP optimization layer):", s1, sp))
    if s2 is not None:
        print(_delta("B2 -> Proposed   (probabilistic forecasting):", s2, sp))
    else:
        print("  B2 -> Proposed   (probabilistic forecasting):  [B2 result pending]")
    print(_delta("B0 -> Proposed   (total predict-then-optimize):", s0, sp))

    print("\nThree-phase stockout breakdown (proposed-system day boundaries):")
    for label, run in (("B0", b0), ("B1", b1)):
        pb = phase_breakdown(run["daily_log"])
        print(
            f"  {label}: cold-start(1-6)={pb['cold_start']}  "
            f"catch-up(7-13)={pb['catch_up']}  "
            f"steady-state(14-73)={pb['steady_state']}"
        )
    print("  B2, Proposed: per-day profile in their CPLEX run logs "
          "(Proposed: docs/headline_73day_run_results.md).")

    out = {
        "config": {
            "simulation_days": cfg["SIMULATION_DAYS"],
            "seed": cfg["SEED"],
            "num_vehicles": cfg["NUM_VEHICLES"],
            "n_atm_days": n_atm_days,
        },
        "B0": {"summary": s0, "phase_breakdown": phase_breakdown(b0["daily_log"]),
               "daily_log": b0["daily_log"]},
        "B1": {"summary": s1, "phase_breakdown": phase_breakdown(b1["daily_log"]),
               "daily_log": b1["daily_log"]},
        "B2": s2,
        "proposed": sp,
    }
    out_path = ROOT / "docs" / "baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nRaw results written to {out_path}")
    if s2 is None:
        print("Note: B2 result not found; comparison is partial. "
              "Run scripts/run_point_irp.py, then re-run this script.")


if __name__ == "__main__":
    main()