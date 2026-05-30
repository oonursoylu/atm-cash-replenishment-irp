"""
Official frozen-OSM Phase 4.H capacity/fleet 2x2 rerun.

This script does not mutate configs/optimize.yaml. Each cell is run in a fresh
Python subprocess with explicit config overrides, using the frozen symmetric OSM
travel matrix and the locked legacy CPLEX reporting mode.

Outputs:
  docs/results_frozen/phase_4h_2x2_frozen_20260530.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results_frozen" / "phase_4h_2x2_frozen_20260530.json"


WORKER = r"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\onurs\Desktop\ma_2026_project")
sys.path.insert(0, str(ROOT))

from src.config import load_config
import src.sim.rolling_horizon as rh

use_hetero = sys.argv[1].lower() == "true"
num_vehicles = int(sys.argv[2])
label = sys.argv[3]

cfg = load_config()
cfg["SIMULATION_DAYS"] = 30
cfg["PLANNING_HORIZON"] = 7
cfg["MIP_GAP"] = 0.05
cfg["TIME_LIMIT_SEC"] = 600
cfg["NUM_VEHICLES"] = num_vehicles
cfg["USE_HETEROGENEOUS_CAPACITY"] = use_hetero
cfg["STOCKOUT_PENALTY"] = 3000
cfg["SAFETY_FLOOR_PEN"] = 0.1
cfg["INITIAL_INV_LOW"] = 0.30
cfg["INITIAL_INV_HIGH"] = 0.50
cfg["SEED"] = 42
cfg["USE_REAL_DEMAND"] = True
cfg["SYMMETRIZE_TRAVEL_MATRIX"] = True
cfg["CPLEX_DETERMINISTIC"] = False

assert cfg["REAL_DEMAND_CSV_PATH"].endswith("test_predictions_p0.55_s0.95.csv")

print("=" * 88)
print(f"RUN {label}: hetero={use_hetero}, nv={num_vehicles}, gap={cfg['MIP_GAP']}")
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
    "use_heterogeneous_capacity": use_hetero,
    "num_vehicles": num_vehicles,
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


CELLS = [
    (False, 2, "uniform_nv2"),
    (False, 3, "uniform_nv3"),
    (True, 2, "hetero_nv2"),
    (True, 3, "hetero_nv3"),
]


def run_cell(use_hetero: bool, nv: int, label: str) -> dict:
    print("\n" + "#" * 88)
    print(f"# START {label}")
    print("#" * 88)

    p = subprocess.Popen(
        [sys.executable, "-c", WORKER, str(use_hetero), str(nv), label],
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
    results: list[dict] = []
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for use_hetero, nv, label in CELLS:
        row = run_cell(use_hetero, nv, label)
        results.append(row)

        OUT.write_text(
            json.dumps(
                {
                    "purpose": (
                        "Official frozen-OSM Phase 4.H capacity/fleet 2x2 rerun "
                        "for final thesis results."
                    ),
                    "locked_config": {
                        "travel_matrix": "frozen symmetric OSM",
                        "travel_matrix_hash": "76013f9295fe036d980740994878c3be",
                        "forecast_csv_hash": "b9432a2eba76b887b49597cc705f0d8e",
                        "cplex_mode": "legacy/default",
                        "mip_gap": 0.05,
                        "seed": 42,
                        "days": 30,
                        "planning_horizon": 7,
                        "stockout_penalty": 3000,
                        "safety_floor_pen": 0.1,
                    },
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("\n" + "#" * 88)
    print("# SUMMARY")
    print("#" * 88)
    print(
        f"{'label':>14} {'cap':>8} {'nv':>3} {'SO':>5} {'op':>12} "
        f"{'total':>12} {'disp':>5} {'tt_hash':>34}"
    )

    for r in results:
        cap = "hetero" if r["use_heterogeneous_capacity"] else "uniform"
        print(
            f"{r['label']:>14} {cap:>8} {r['num_vehicles']:>3} "
            f"{r['stockouts']:>5} {r['op_cost']:>12,.2f} "
            f"{r['total_cost']:>12,.2f} {r['dispatches']:>5} "
            f"{r['provenance']['travel_matrix_hash']:>34}"
        )

    print(f"\nWrote {OUT}")
    print(f"Total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
