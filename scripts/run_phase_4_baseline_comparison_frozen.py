"""
Official frozen-OSM baseline-system rerun for the thesis comparison.

Runs the frozen 73-day baseline systems:
  B0: static (s, S) + greedy routing
  B1: quantile forecast + greedy routing
  B2: point forecast + IRP MILP

The final proposed-system headline is intentionally not run here; it is the
separate frozen 73-day headline run. This file records the baseline side of
the comparison and marks Proposed as pending until that headline lands.

Does not mutate configs/optimize.yaml. Resume-safe: completed systems in the
output JSON are skipped on rerun.

Output:
  docs/results_frozen/phase_4_baseline_comparison_frozen_20260531.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results_frozen" / "phase_4_baseline_comparison_frozen_20260531.json"

N_ATMS = 31
TT_HASH = "76013f9295fe036d980740994878c3be"
FORECAST_CSV_REL = "predictions/test_predictions_p0.55_s0.95.csv"
FORECAST_MD5 = "b9432a2eba76b887b49597cc705f0d8e"
POINT_CSV_REL = "predictions/test_predictions_point.csv"
POINT_MD5 = "007705f27accba1da5e934b7f119547a"
TRAIN_MEANS_REL = "data/train_atm_means.csv"
TRAIN_MEANS_MD5 = "f7852e21df9d5c686c134cbc65d628c3"

PHASES = (
    ("cold_start", 1, 6),
    ("catch_up", 7, 13),
    ("steady_state", 14, 73),
)


B2_WORKER = r"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
point_csv_rel = sys.argv[2]
expected_point_md5 = sys.argv[3]
expected_tt_hash = sys.argv[4]

sys.path.insert(0, str(ROOT))

from src.config import load_config
import src.sim.rolling_horizon as rh


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


csv_path = ROOT / point_csv_rel
if not csv_path.exists():
    raise FileNotFoundError(f"Point forecast CSV not found: {csv_path}")
actual_md5 = md5_file(csv_path)
if actual_md5 != expected_point_md5:
    raise RuntimeError(
        f"Point forecast hash mismatch for {point_csv_rel}: {actual_md5}, expected {expected_point_md5}"
    )

cfg = load_config()
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
cfg["REAL_DEMAND_CSV_PATH"] = str(csv_path)
cfg["MISSING_DAY_EPS_SAFETY"] = cfg["MISSING_DAY_EPS_MEAN"]
cfg["SYMMETRIZE_TRAVEL_MATRIX"] = True
cfg["CPLEX_DETERMINISTIC"] = False

print("=" * 92)
print("RUN B2: point forecast + IRP MILP, frozen 73-day headline settings")
print("=" * 92)
print(f"Point forecast hash verified: {actual_md5}")

t0 = time.time()
kpis, provenance = rh.run_simulation(cfg, map_generator=None, return_provenance=True)
sec = round(time.time() - t0, 1)

if provenance["travel_matrix_hash"] != expected_tt_hash:
    raise RuntimeError(
        f"Travel matrix hash mismatch: {provenance['travel_matrix_hash']}, expected {expected_tt_hash}"
    )
if provenance["forecast_csv_hash"] != expected_point_md5:
    raise RuntimeError(
        f"Provenance point forecast hash mismatch: {provenance['forecast_csv_hash']}, expected {expected_point_md5}"
    )

op_cost = (
    kpis["travel_cost"]
    + kpis["dispatch_cost"]
    + kpis["drop_fees"]
    + kpis["holding_cost"]
)
n_atm_days = cfg["SIMULATION_DAYS"] * 31

row = {
    "system": "B2",
    "label": "B2 (point + IRP)",
    "status": "complete",
    "forecast_csv_rel": point_csv_rel,
    "forecast_csv_hash_checked": actual_md5,
    "compute_sec": sec,
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
    "provenance": provenance,
}

print("JSON_RESULT_START")
print(json.dumps(row, sort_keys=True))
print("JSON_RESULT_END")
"""


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


def load_existing() -> dict:
    if not OUT.exists():
        return {"results": {}}
    data = json.loads(OUT.read_text(encoding="utf-8"))
    data.setdefault("results", {})
    return data


def write_output(results: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "Official frozen-OSM baseline-system rerun for final thesis comparison.",
        "locked_config": {
            "travel_matrix": "frozen symmetric OSM",
            "travel_matrix_hash": TT_HASH,
            "forecast_csv": FORECAST_CSV_REL,
            "forecast_csv_hash": FORECAST_MD5,
            "point_forecast_csv": POINT_CSV_REL,
            "point_forecast_hash": POINT_MD5,
            "train_means_csv": TRAIN_MEANS_REL,
            "train_means_hash": TRAIN_MEANS_MD5,
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
        "results": results,
        "proposed": {
            "status": "pending_frozen_headline",
            "note": "Final Proposed (quantile + IRP) is produced by the separate frozen 73-day headline run.",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_train_means(path: Path) -> dict[str, float]:
    means: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            means[row["CASHP_ID"]] = float(row["train_mean_withdrawal"])
    return means


def apply_frozen_73day_overrides(cfg: dict, forecast_csv: Path) -> dict:
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


def run_greedy_baseline(system: str) -> dict:
    sys.path.insert(0, str(ROOT))

    from src.baselines.common import load_instance, summarise
    from src.baselines.forecast_threshold import run_b1
    from src.baselines.static_ss import run_b0
    from src.config import load_config
    from src.provenance import build_provenance

    forecast_csv = ROOT / FORECAST_CSV_REL
    train_means_csv = ROOT / TRAIN_MEANS_REL
    verify_file(forecast_csv, FORECAST_MD5, "Forecast CSV")
    verify_file(train_means_csv, TRAIN_MEANS_MD5, "Train means CSV")

    cfg = apply_frozen_73day_overrides(load_config(), forecast_csv)
    instance = load_instance(cfg)
    provenance = build_provenance(cfg, instance["travel_time"], "OSM", deterministic=False)
    if provenance["travel_matrix_hash"] != TT_HASH:
        raise RuntimeError(
            f"Travel matrix hash mismatch: {provenance['travel_matrix_hash']}, expected {TT_HASH}"
        )
    if provenance["forecast_csv_hash"] != FORECAST_MD5:
        raise RuntimeError(
            f"Forecast hash mismatch in provenance: {provenance['forecast_csv_hash']}, expected {FORECAST_MD5}"
        )

    n_atm_days = cfg["SIMULATION_DAYS"] * N_ATMS
    if system == "B0":
        print("\nRunning B0 (static s,S + greedy), frozen 73-day settings ...")
        run = run_b0(cfg, instance, load_train_means(train_means_csv))
        label = "B0 (static s,S + greedy)"
    elif system == "B1":
        print("\nRunning B1 (quantile forecast + greedy), frozen 73-day settings ...")
        run = run_b1(cfg, instance)
        label = "B1 (quantile + greedy)"
    else:
        raise ValueError(f"Unknown greedy baseline: {system}")

    summary = summarise(run["kpis"], n_atm_days)
    summary["n_atm_days"] = n_atm_days
    summary = {
        k: (round(v, 2) if isinstance(v, float) else v)
        for k, v in summary.items()
    }

    return {
        "system": system,
        "label": label,
        "status": "complete",
        "forecast_csv_rel": FORECAST_CSV_REL,
        "forecast_csv_hash_checked": FORECAST_MD5,
        "train_means_rel": TRAIN_MEANS_REL if system == "B0" else None,
        "train_means_hash_checked": TRAIN_MEANS_MD5 if system == "B0" else None,
        "summary": summary,
        "phase_breakdown": phase_breakdown(run["daily_log"]),
        "daily_log": run["daily_log"],
        "provenance": provenance,
    }


def run_b2() -> dict:
    print("\n" + "#" * 92)
    print("# START B2 (point + IRP)")
    print("#" * 92)

    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            B2_WORKER,
            str(ROOT),
            POINT_CSV_REL,
            POINT_MD5,
            TT_HASH,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines: list[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
        lines.append(line)

    rc = p.wait()
    text = "".join(lines)
    if rc != 0:
        raise SystemExit(rc)

    start = text.index("JSON_RESULT_START") + len("JSON_RESULT_START")
    end = text.index("JSON_RESULT_END")
    return json.loads(text[start:end].strip())


def print_summary(results: dict) -> None:
    print("\n" + "#" * 92)
    print("# SUMMARY")
    print("#" * 92)
    print(
        f"{'system':<4} {'SO':>5} {'SL %':>8} {'op':>12} {'reported':>12} "
        f"{'holding':>12} {'disp':>5} {'tt_hash':>34}"
    )
    for key in ("B0", "B1", "B2"):
        row = results.get(key)
        if not row or row.get("status") != "complete":
            print(f"{key:<4} {'PENDING':>5}")
            continue
        s = row["summary"]
        prov = row["provenance"]
        print(
            f"{key:<4} {s['stockouts']:>5} {s['service_level'] * 100:>8.2f} "
            f"{s['op_cost']:>12,.2f} {s['reported_total']:>12,.2f} "
            f"{s['holding']:>12,.2f} {s['dispatches']:>5} "
            f"{prov['travel_matrix_hash']:>34}"
        )


def main() -> None:
    started = time.time()
    verify_file(ROOT / FORECAST_CSV_REL, FORECAST_MD5, "Forecast CSV")
    verify_file(ROOT / POINT_CSV_REL, POINT_MD5, "Point forecast CSV")
    verify_file(ROOT / TRAIN_MEANS_REL, TRAIN_MEANS_MD5, "Train means CSV")

    data = load_existing()
    results = data.get("results", {})

    print(f"Loaded {len(results)} completed/pending systems from {OUT if OUT.exists() else '(none)'}")

    for key in ("B0", "B1"):
        if results.get(key, {}).get("status") == "complete":
            print(f"[SKIP] {key}")
            continue
        results[key] = run_greedy_baseline(key)
        write_output(results)

    if results.get("B2", {}).get("status") == "complete":
        print("[SKIP] B2")
    else:
        results["B2"] = run_b2()
        write_output(results)

    print_summary(results)
    print(f"\nWrote {OUT}")
    print("Proposed frozen headline is still separate/pending.")
    print(f"Total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
