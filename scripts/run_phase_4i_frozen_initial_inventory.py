"""
Official frozen-OSM Phase 4.I initial-inventory lower-bound sweep.

Does not mutate configs/optimize.yaml. Each cell runs in a fresh subprocess
with explicit config overrides. Resume-safe: if the output JSON already has
completed cells, they are skipped.

Output:
  docs/results_frozen/phase_4i_initial_inventory_frozen_20260530.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results_frozen" / "phase_4i_initial_inventory_frozen_20260530.json"

LOW_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
HIGH = 0.50
FORECAST_CSV_REL = "predictions/test_predictions_p0.55_s0.95.csv"
FORECAST_MD5 = "b9432a2eba76b887b49597cc705f0d8e"

WORKER = r"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
initial_low = float(sys.argv[2])
initial_high = float(sys.argv[3])
label = sys.argv[4]
forecast_csv_rel = sys.argv[5]
expected_md5 = sys.argv[6]

sys.path.insert(0, str(ROOT))

from src.config import load_config
import src.sim.rolling_horizon as rh


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


csv_path = ROOT / forecast_csv_rel
if not csv_path.exists():
    raise FileNotFoundError(f"Forecast CSV not found: {csv_path}")
actual_md5 = md5_file(csv_path)
if actual_md5 != expected_md5:
    raise RuntimeError(
        f"Forecast CSV hash mismatch for {forecast_csv_rel}: {actual_md5}, expected {expected_md5}"
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
cfg["INITIAL_INV_LOW"] = initial_low
cfg["INITIAL_INV_HIGH"] = initial_high
cfg["SEED"] = 42
cfg["USE_REAL_DEMAND"] = True
cfg["REAL_DEMAND_CSV_PATH"] = str(csv_path)
cfg["SYMMETRIZE_TRAVEL_MATRIX"] = True
cfg["CPLEX_DETERMINISTIC"] = False

print("=" * 88)
print(f"RUN {label}: initial_inventory=U({initial_low:.2f}, {initial_high:.2f}) x capacity")
print("=" * 88)
print(f"Forecast hash verified: {actual_md5}")

t0 = time.time()
kpis, provenance = rh.run_simulation(cfg, map_generator=None, return_provenance=True)
sec = round(time.time() - t0, 1)

op_cost = (
    kpis["travel_cost"]
    + kpis["dispatch_cost"]
    + kpis["drop_fees"]
    + kpis["holding_cost"]
)

row = {
    "label": label,
    "initial_inv_low": initial_low,
    "initial_inv_high": initial_high,
    "forecast_csv_rel": forecast_csv_rel,
    "forecast_csv_hash_checked": actual_md5,
    "compute_sec": sec,
    "stockouts": kpis["stockout_events"],
    "op_cost": round(op_cost, 2),
    "total_cost": round(op_cost + kpis["stockout_cost"], 2),
    "dispatches": kpis["total_dispatches"],
    "travel_cost": round(kpis["travel_cost"], 2),
    "dispatch_cost": round(kpis["dispatch_cost"], 2),
    "drop_fees": round(kpis["drop_fees"], 2),
    "holding_cost": round(kpis["holding_cost"], 2),
    "stockout_cost": round(kpis["stockout_cost"], 2),
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


def write_results(results: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "purpose": "Official frozen-OSM Phase 4.I initial-inventory lower-bound sweep for final thesis results.",
                "locked_config": {
                    "travel_matrix": "frozen symmetric OSM",
                    "travel_matrix_hash": "76013f9295fe036d980740994878c3be",
                    "forecast_csv": FORECAST_CSV_REL,
                    "forecast_csv_hash": FORECAST_MD5,
                    "cplex_mode": "legacy/default",
                    "mip_gap": 0.05,
                    "seed": 42,
                    "days": 30,
                    "planning_horizon": 7,
                    "num_vehicles": 3,
                    "use_heterogeneous_capacity": True,
                    "stockout_penalty": 3000,
                    "safety_floor_pen": 0.1,
                    "initial_inv_high": HIGH,
                },
                "initial_inv_low_grid": LOW_GRID,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_cell(low: float) -> dict:
    label = f"init_low_{low:.2f}"
    print("\n" + "#" * 88)
    print(f"# START {label}")
    print("#" * 88)

    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            WORKER,
            str(ROOT),
            str(low),
            str(HIGH),
            label,
            FORECAST_CSV_REL,
            FORECAST_MD5,
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


def main() -> None:
    started = time.time()
    results = load_existing()
    done = {round(float(r["initial_inv_low"]), 2) for r in results}

    print(f"Loaded {len(results)} completed cells from {OUT if OUT.exists() else '(none)'}")
    print(f"Remaining cells: {len(LOW_GRID) - len(done)} / {len(LOW_GRID)}")

    for low in LOW_GRID:
        low_key = round(float(low), 2)
        if low_key in done:
            print(f"[SKIP] initial_inv_low={low_key:.2f}")
            continue
        row = run_cell(low)
        results.append(row)
        write_results(results)

    print("\n" + "#" * 88)
    print("# SUMMARY")
    print("#" * 88)
    print(
        f"{'low':>6} {'high':>6} {'SO':>5} {'op':>12} {'total':>12} "
        f"{'holding':>12} {'disp':>5} {'forecast_hash':>34} {'tt_hash':>34}"
    )
    for r in sorted(results, key=lambda x: float(x["initial_inv_low"])):
        print(
            f"{r['initial_inv_low']:>6.2f} {r['initial_inv_high']:>6.2f} "
            f"{r['stockouts']:>5} {r['op_cost']:>12,.2f} "
            f"{r['total_cost']:>12,.2f} {r['holding_cost']:>12,.2f} "
            f"{r['dispatches']:>5} {r['provenance']['forecast_csv_hash']:>34} "
            f"{r['provenance']['travel_matrix_hash']:>34}"
        )

    print(f"\nWrote {OUT}")
    print(f"Total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
