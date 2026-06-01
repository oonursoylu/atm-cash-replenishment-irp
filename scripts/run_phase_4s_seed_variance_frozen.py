"""
Official frozen-OSM Phase 4.S multi-seed variance rerun.

Runs the Proposed calibration cell across seven seeds:
  quantile forecast + IRP MILP, frozen 30-day settings, mip_gap=0.05

This is a 30-day calibration/sensitivity variance run, not the 73-day final
headline evaluation. It does not mutate configs/optimize.yaml.

Expected command from project root:
  python scripts/run_phase_4s_seed_variance_frozen.py

Output:
  docs/results_frozen/phase_4s_seed_variance_frozen_20260531.json
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results_frozen" / "phase_4s_seed_variance_frozen_20260531.json"

N_ATMS = 31
SEEDS = [1, 7, 13, 21, 42, 73, 99]
TT_HASH = "76013f9295fe036d980740994878c3be"
FORECAST_CSV_REL = "predictions/test_predictions_p0.55_s0.95.csv"
FORECAST_MD5 = "b9432a2eba76b887b49597cc705f0d8e"

WORKER = r"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
seed = int(sys.argv[2])
forecast_csv_rel = sys.argv[3]
expected_forecast_md5 = sys.argv[4]
expected_tt_hash = sys.argv[5]

sys.path.insert(0, str(ROOT))

from src.config import load_config
import src.sim.rolling_horizon as rh


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


forecast_csv = ROOT / forecast_csv_rel
if not forecast_csv.exists():
    raise FileNotFoundError(f"Forecast CSV not found: {forecast_csv}")
actual_forecast_md5 = md5_file(forecast_csv)
if actual_forecast_md5 != expected_forecast_md5:
    raise RuntimeError(
        f"Forecast CSV hash mismatch: {actual_forecast_md5}, expected {expected_forecast_md5}"
    )

cfg = load_config()
cfg["SIMULATION_DAYS"] = 30
cfg["PLANNING_HORIZON"] = 7
cfg["MIP_GAP"] = 0.05
cfg["TIME_LIMIT_SEC"] = 600
cfg["NUM_VEHICLES"] = 3
cfg["USE_HETEROGENEOUS_CAPACITY"] = True
cfg["STOCKOUT_PENALTY"] = 3000
cfg["SAFETY_FLOOR_PEN"] = 0.1
cfg["INITIAL_INV_LOW"] = 0.30
cfg["INITIAL_INV_HIGH"] = 0.50
cfg["SEED"] = seed
cfg["USE_REAL_DEMAND"] = True
cfg["REAL_DEMAND_CSV_PATH"] = str(forecast_csv)
cfg["SYMMETRIZE_TRAVEL_MATRIX"] = True
cfg["CPLEX_DETERMINISTIC"] = False

print("=" * 88)
print(f"RUN seed={seed}: frozen 30-day seed variance, mip_gap=0.05")
print("=" * 88)
print(f"Forecast hash verified: {actual_forecast_md5}")

t0 = time.time()
kpis, provenance = rh.run_simulation(cfg, map_generator=None, return_provenance=True)
sec = round(time.time() - t0, 1)

if provenance["travel_matrix_hash"] != expected_tt_hash:
    raise RuntimeError(
        f"Travel matrix hash mismatch: {provenance['travel_matrix_hash']}, expected {expected_tt_hash}"
    )
if provenance["forecast_csv_hash"] != expected_forecast_md5:
    raise RuntimeError(
        f"Forecast hash mismatch in provenance: {provenance['forecast_csv_hash']}, "
        f"expected {expected_forecast_md5}"
    )
if provenance["days"] != 30 or provenance["mip_gap"] != 0.05 or provenance["seed"] != seed:
    raise RuntimeError(f"Unexpected provenance for seed={seed}: {provenance}")

op_cost = (
    kpis["travel_cost"]
    + kpis["dispatch_cost"]
    + kpis["drop_fees"]
    + kpis["holding_cost"]
)
n_atm_days = cfg["SIMULATION_DAYS"] * 31

row = {
    "seed": seed,
    "label": f"seed_{seed}",
    "status": "complete",
    "forecast_csv_rel": forecast_csv_rel,
    "forecast_csv_hash_checked": actual_forecast_md5,
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


def load_existing() -> list[dict]:
    if not OUT.exists():
        return []
    data = json.loads(OUT.read_text(encoding="utf-8"))
    return data.get("results", [])


def metric_stats(values: list[float]) -> dict:
    if not values:
        return {}
    stats = {
        "mean": round(statistics.mean(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "range": round(max(values) - min(values), 2),
    }
    if len(values) >= 2:
        stats["sample_sd"] = round(statistics.stdev(values), 2)
    else:
        stats["sample_sd"] = None
    return stats


def aggregate(results: list[dict]) -> dict:
    complete = [r for r in results if r.get("status") == "complete"]
    by_seed = {int(r["seed"]): r for r in complete}
    stockouts = [by_seed[s]["summary"]["stockouts"] for s in SEEDS if s in by_seed]
    op_costs = [by_seed[s]["summary"]["op_cost"] for s in SEEDS if s in by_seed]
    reported = [by_seed[s]["summary"]["reported_total"] for s in SEEDS if s in by_seed]
    dispatches = [by_seed[s]["summary"]["dispatches"] for s in SEEDS if s in by_seed]
    service = [by_seed[s]["summary"]["service_level"] for s in SEEDS if s in by_seed]

    out = {
        "n_complete": len(complete),
        "completed_seeds": [s for s in SEEDS if s in by_seed],
        "stockouts": metric_stats(stockouts),
        "op_cost": metric_stats(op_costs),
        "reported_total": metric_stats(reported),
        "dispatches": metric_stats(dispatches),
        "service_level": metric_stats(service),
    }
    if 42 in by_seed and stockouts:
        mean_so = statistics.mean(stockouts)
        out["anchor_seed_42"] = {
            "stockouts": by_seed[42]["summary"]["stockouts"],
            "stockouts_delta_from_mean": round(by_seed[42]["summary"]["stockouts"] - mean_so, 2),
            "op_cost": by_seed[42]["summary"]["op_cost"],
        }
    return out


def write_payload(results: list[dict]) -> None:
    ordered = sorted(results, key=lambda r: int(r["seed"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "purpose": (
                    "Official frozen-OSM Phase 4.S 30-day multi-seed variance rerun "
                    "for the Proposed calibration cell."
                ),
                "interpretation_note": (
                    "This is a 30-day calibration/sensitivity variance run at mip_gap=0.05; "
                    "it is not the 73-day final headline evaluation."
                ),
                "locked_config": {
                    "travel_matrix": "frozen symmetric OSM",
                    "travel_matrix_hash": TT_HASH,
                    "forecast_csv": FORECAST_CSV_REL,
                    "forecast_csv_hash": FORECAST_MD5,
                    "cplex_mode": "legacy/default",
                    "mip_gap": 0.05,
                    "days": 30,
                    "planning_horizon": 7,
                    "num_vehicles": 3,
                    "use_heterogeneous_capacity": True,
                    "stockout_penalty": 3000,
                    "safety_floor_pen": 0.1,
                    "initial_inv_low": 0.30,
                    "initial_inv_high": 0.50,
                },
                "seed_grid": SEEDS,
                "results": ordered,
                "aggregate": aggregate(ordered),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_seed(seed: int) -> dict:
    print("\n" + "#" * 88)
    print(f"# START seed={seed}")
    print("#" * 88)

    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            WORKER,
            str(ROOT),
            str(seed),
            FORECAST_CSV_REL,
            FORECAST_MD5,
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


def print_summary(results: list[dict]) -> None:
    ordered = sorted(results, key=lambda r: int(r["seed"]))
    agg = aggregate(ordered)

    def fmt(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:,.2f}"

    print("\n" + "#" * 88)
    print("# SUMMARY: frozen 30-day seed variance")
    print("#" * 88)
    print(f"{'seed':>6} {'SO':>5} {'SL%':>8} {'op':>12} {'reported':>12} {'disp':>6}")
    for r in ordered:
        s = r["summary"]
        print(
            f"{r['seed']:>6} {s['stockouts']:>5} {s['service_level'] * 100:>7.2f}% "
            f"{s['op_cost']:>12,.2f} {s['reported_total']:>12,.2f} "
            f"{s['dispatches']:>6}"
        )

    if agg["n_complete"]:
        so = agg["stockouts"]
        op = agg["op_cost"]
        print(
            f"\nN={agg['n_complete']} | SO mean={so.get('mean')} sd={so.get('sample_sd')} "
            f"range=[{so.get('min')}, {so.get('max')}]"
        )
        print(
            f"Op cost mean={fmt(op.get('mean'))} sd={fmt(op.get('sample_sd'))} "
            f"range=[{fmt(op.get('min'))}, {fmt(op.get('max'))}]"
        )

    print(f"\nWrote {OUT}")


def main() -> None:
    started = time.time()
    results = load_existing()
    done = {int(r["seed"]) for r in results if r.get("status") == "complete"}

    print("=" * 88)
    print("RUN PHASE 4.S: frozen 30-day multi-seed variance, mip_gap=0.05")
    print("=" * 88)
    print(f"Output JSON   : {OUT}")
    print(f"Forecast CSV  : {FORECAST_CSV_REL}")
    print(f"Forecast hash : {FORECAST_MD5}")
    print(f"Frozen TT hash: {TT_HASH}")
    print(f"Seed grid     : {SEEDS}")
    print("Config        : days=30, mip_gap=0.05, hetero, nv=3, stockout=3000, sf=0.1")
    print("CPLEX mode    : legacy/default (non-deterministic)")
    print("=" * 88)
    print(f"Loaded {len(done)} completed seed(s); remaining {len(SEEDS) - len(done)} / {len(SEEDS)}")

    for seed in SEEDS:
        if seed in done:
            print(f"[SKIP] seed={seed}")
            continue
        row = run_seed(seed)
        results = [r for r in results if int(r["seed"]) != seed]
        results.append(row)
        write_payload(results)

    write_payload(results)
    print_summary(results)
    print(f"Total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
