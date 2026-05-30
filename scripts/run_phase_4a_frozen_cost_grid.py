"""
Official frozen-OSM Phase 4.A cost-parameter grid rerun.

Does not mutate configs/optimize.yaml. Each cell runs in a fresh subprocess
with explicit config overrides. Resume-safe: if the output JSON already has
completed cells, they are skipped.

Output:
  docs/results_frozen/phase_4a_cost_grid_frozen_20260530.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results_frozen" / "phase_4a_cost_grid_frozen_20260530.json"

STOCKOUT_GRID = [3000, 5000, 8000]
SAFETY_GRID = [0.001, 0.01, 0.1, 1.0]

WORKER = r"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
stockout_penalty = int(sys.argv[2])
safety_floor_pen = float(sys.argv[3])
label = sys.argv[4]

sys.path.insert(0, str(ROOT))

from src.config import load_config
import src.sim.rolling_horizon as rh

cfg = load_config()
cfg["SIMULATION_DAYS"] = 30
cfg["PLANNING_HORIZON"] = 7
cfg["MIP_GAP"] = 0.05
cfg["TIME_LIMIT_SEC"] = 600
cfg["NUM_VEHICLES"] = 3
cfg["USE_HETEROGENEOUS_CAPACITY"] = True
cfg["STOCKOUT_PENALTY"] = stockout_penalty
cfg["SAFETY_FLOOR_PEN"] = safety_floor_pen
cfg["INITIAL_INV_LOW"] = 0.30
cfg["INITIAL_INV_HIGH"] = 0.50
cfg["SEED"] = 42
cfg["USE_REAL_DEMAND"] = True
cfg["SYMMETRIZE_TRAVEL_MATRIX"] = True
cfg["CPLEX_DETERMINISTIC"] = False

assert cfg["REAL_DEMAND_CSV_PATH"].endswith("test_predictions_p0.55_s0.95.csv")

print("=" * 88)
print(f"RUN {label}: stockout_penalty={stockout_penalty}, safety_floor_pen={safety_floor_pen}")
print("=" * 88)

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
    "stockout_penalty": stockout_penalty,
    "safety_floor_pen": safety_floor_pen,
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
                "purpose": "Official frozen-OSM Phase 4.A cost-parameter grid rerun for final thesis results.",
                "locked_config": {
                    "travel_matrix": "frozen symmetric OSM",
                    "travel_matrix_hash": "76013f9295fe036d980740994878c3be",
                    "forecast_csv_hash": "b9432a2eba76b887b49597cc705f0d8e",
                    "cplex_mode": "legacy/default",
                    "mip_gap": 0.05,
                    "seed": 42,
                    "days": 30,
                    "planning_horizon": 7,
                    "num_vehicles": 3,
                    "use_heterogeneous_capacity": True,
                    "initial_inv_low": 0.30,
                    "initial_inv_high": 0.50,
                },
                "stockout_grid": STOCKOUT_GRID,
                "safety_floor_grid": SAFETY_GRID,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_cell(so: int, sf: float) -> dict:
    label = f"so{so}_sf{sf:g}"
    print("\n" + "#" * 88)
    print(f"# START {label}")
    print("#" * 88)

    p = subprocess.Popen(
        [sys.executable, "-c", WORKER, str(ROOT), str(so), str(sf), label],
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
    done = {(r["stockout_penalty"], float(r["safety_floor_pen"])) for r in results}

    cells = [(so, sf) for so in STOCKOUT_GRID for sf in SAFETY_GRID]
    print(f"Loaded {len(results)} completed cells from {OUT if OUT.exists() else '(none)'}")
    print(f"Remaining cells: {len(cells) - len(done)} / {len(cells)}")

    for so, sf in cells:
        if (so, float(sf)) in done:
            print(f"[SKIP] so={so}, sf={sf}")
            continue
        row = run_cell(so, sf)
        results.append(row)
        write_results(results)

    print("\n" + "#" * 88)
    print("# SUMMARY: stockouts")
    print("#" * 88)
    print(f"{'stockout':>10} " + " ".join(f"sf={sf:<8g}" for sf in SAFETY_GRID))
    for so in STOCKOUT_GRID:
        row = f"{so:>10} "
        for sf in SAFETY_GRID:
            hit = next(
                (
                    r for r in results
                    if r["stockout_penalty"] == so
                    and float(r["safety_floor_pen"]) == float(sf)
                ),
                None,
            )
            row += f"{hit['stockouts'] if hit else '?':>10}"
        print(row)

    print("\n# SUMMARY: operational cost")
    print(f"{'stockout':>10} " + " ".join(f"sf={sf:<12g}" for sf in SAFETY_GRID))
    for so in STOCKOUT_GRID:
        row = f"{so:>10} "
        for sf in SAFETY_GRID:
            hit = next(
                (
                    r for r in results
                    if r["stockout_penalty"] == so
                    and float(r["safety_floor_pen"]) == float(sf)
                ),
                None,
            )
            row += f"{hit['op_cost'] if hit else '?':>12}"
        print(row)

    print(f"\nWrote {OUT}")
    print(f"Total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
