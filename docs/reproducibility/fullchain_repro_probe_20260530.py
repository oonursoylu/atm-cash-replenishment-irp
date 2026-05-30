import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\onurs\Desktop\ma_2026_project")
OUT = ROOT / "docs" / "fullchain_repro_probe_20260530.json"

WORKER = r"""
import json, sys, time
from pathlib import Path

ROOT = Path(r"C:\Users\onurs\Desktop\ma_2026_project")
sys.path.insert(0, str(ROOT))

from src.config import load_config
import src.sim.rolling_horizon as rh
import src.optim.irp_milp as irp

mode = sys.argv[1]
rep = int(sys.argv[2])

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
cfg["SEED"] = 42
cfg["USE_REAL_DEMAND"] = True
cfg["SYMMETRIZE_TRAVEL_MATRIX"] = True

assert cfg["REAL_DEMAND_CSV_PATH"].endswith("test_predictions_p0.55_s0.95.csv")

if mode == "legacy_cplex":
    def solve_single_horizon_legacy(sim_day, actual_inventory, master_data, cfg):
        return irp.solve_single_horizon(
            sim_day,
            actual_inventory,
            master_data,
            cfg,
            deterministic=False,
        )
    rh.solve_single_horizon = solve_single_horizon_legacy
elif mode == "det_cplex":
    pass
else:
    raise ValueError(mode)

t0 = time.time()
kpis, provenance = rh.run_simulation(cfg, map_generator=None, return_provenance=True)
sec = round(time.time() - t0, 1)

op = kpis["travel_cost"] + kpis["dispatch_cost"] + kpis["drop_fees"] + kpis["holding_cost"]

row = {
    "mode": mode,
    "rep": rep,
    "compute_sec": sec,
    "stockouts": kpis["stockout_events"],
    "op_cost": round(op, 2),
    "total_cost": round(op + kpis["stockout_cost"], 2),
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

def run_worker(mode, rep):
    print("\n" + "=" * 88)
    print(f"RUN {mode} rep={rep}")
    print("=" * 88)
    p = subprocess.run(
        [sys.executable, "-c", WORKER, mode, str(rep)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        raise SystemExit(p.returncode)
    start = p.stdout.index("JSON_RESULT_START") + len("JSON_RESULT_START")
    end = p.stdout.index("JSON_RESULT_END")
    return json.loads(p.stdout[start:end].strip())

plan = [
    ("legacy_cplex", 1),
    ("legacy_cplex", 2),
    ("det_cplex", 1),
    ("det_cplex", 2),
]

results = []
started = time.time()
for mode, rep in plan:
    row = run_worker(mode, rep)
    results.append(row)
    OUT.write_text(json.dumps({
        "purpose": "Full 30-day baseline cross-process reproducibility probe under frozen travel matrix.",
        "results": results,
    }, indent=2), encoding="utf-8")

print("\n" + "#" * 88)
print("# SUMMARY")
print("#" * 88)
print(f"{'mode':>14} {'rep':>3} {'SO':>5} {'op':>12} {'total':>12} {'disp':>5} {'tt_hash':>34}")
for r in results:
    print(
        f"{r['mode']:>14} {r['rep']:>3} {r['stockouts']:>5} "
        f"{r['op_cost']:>12,.2f} {r['total_cost']:>12,.2f} "
        f"{r['dispatches']:>5} {r['provenance']['travel_matrix_hash']:>34}"
    )

for mode in ["legacy_cplex", "det_cplex"]:
    rows = [r for r in results if r["mode"] == mode]
    same = (
        len(rows) == 2
        and rows[0]["stockouts"] == rows[1]["stockouts"]
        and rows[0]["op_cost"] == rows[1]["op_cost"]
        and rows[0]["total_cost"] == rows[1]["total_cost"]
        and rows[0]["dispatches"] == rows[1]["dispatches"]
        and rows[0]["provenance"]["travel_matrix_hash"] == rows[1]["provenance"]["travel_matrix_hash"]
    )
    print(f"{mode} cross_process_identical={same}")

print(f"\nWrote {OUT}")
print(f"Total elapsed: {(time.time() - started) / 60:.1f} min")
