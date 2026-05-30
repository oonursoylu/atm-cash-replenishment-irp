import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\onurs\Desktop\ma_2026_project")
OUT = ROOT / "docs" / "frozen_asymmetry_probe_20260530.json"

SYMMETRIC_REFS = {
    "legacy_cplex": {
        "stockouts": 72,
        "op_cost": 66840.51,
        "total_cost": 282840.51,
        "dispatches": 35,
        "travel_matrix_hash": "76013f9295fe036d980740994878c3be",
    },
    "det_cplex": {
        "stockouts": 59,
        "op_cost": 68601.59,
        "total_cost": 245601.59,
        "dispatches": 36,
        "travel_matrix_hash": "76013f9295fe036d980740994878c3be",
    },
}

WORKER = r"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\onurs\Desktop\ma_2026_project")
sys.path.insert(0, str(ROOT))

from src.config import load_config
import src.sim.rolling_horizon as rh
import src.optim.irp_milp as irp

mode = sys.argv[1]

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

# Critical condition for this probe:
# load data/frozen/travel_matrix_asymmetric.json
cfg["SYMMETRIZE_TRAVEL_MATRIX"] = False

assert cfg["REAL_DEMAND_CSV_PATH"].endswith("test_predictions_p0.55_s0.95.csv")

if mode == "legacy_cplex":
    deterministic = False
elif mode == "det_cplex":
    deterministic = True
else:
    raise ValueError(mode)

def solve_single_horizon_for_mode(sim_day, actual_inventory, master_data, cfg):
    return irp.solve_single_horizon(
        sim_day,
        actual_inventory,
        master_data,
        cfg,
        deterministic=deterministic,
    )

rh.solve_single_horizon = solve_single_horizon_for_mode

print("=" * 88)
print(f"RUN mode={mode} asymmetry probe deterministic={deterministic}")
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
    "mode": mode,
    "deterministic": deterministic,
    "symmetrize": cfg["SYMMETRIZE_TRAVEL_MATRIX"],
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

def run_worker(mode):
    print("\n" + "#" * 88)
    print(f"# START asymmetric {mode}")
    print("#" * 88)

    p = subprocess.Popen(
        [sys.executable, "-c", WORKER, mode],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines = []
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

started = time.time()
results = []

for mode in ["legacy_cplex", "det_cplex"]:
    row = run_worker(mode)
    results.append(row)
    OUT.write_text(
        json.dumps(
            {
                "purpose": "Frozen asymmetric travel-matrix baseline probe; compares asymmetric runs to existing frozen symmetric references.",
                "symmetric_refs": SYMMETRIC_REFS,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

print("\n" + "#" * 88)
print("# SUMMARY")
print("#" * 88)
print(f"{'mode':>14} {'matrix':>10} {'SO':>5} {'op':>12} {'total':>12} {'disp':>5} {'tt_hash':>34}")

for mode, ref in SYMMETRIC_REFS.items():
    print(
        f"{mode:>14} {'symmetric':>10} {ref['stockouts']:>5} "
        f"{ref['op_cost']:>12,.2f} {ref['total_cost']:>12,.2f} "
        f"{ref['dispatches']:>5} {ref['travel_matrix_hash']:>34}"
    )

for r in results:
    print(
        f"{r['mode']:>14} {'asymmetric':>10} {r['stockouts']:>5} "
        f"{r['op_cost']:>12,.2f} {r['total_cost']:>12,.2f} "
        f"{r['dispatches']:>5} {r['provenance']['travel_matrix_hash']:>34}"
    )

print("\nDELTA asymmetric - symmetric")
for r in results:
    ref = SYMMETRIC_REFS[r["mode"]]
    print(f"\n{r['mode']}:")
    print(f"  stockouts : {r['stockouts'] - ref['stockouts']}")
    print(f"  op_cost   : {r['op_cost'] - ref['op_cost']:,.2f}")
    print(f"  total_cost: {r['total_cost'] - ref['total_cost']:,.2f}")
    print(f"  dispatches: {r['dispatches'] - ref['dispatches']}")
    print(f"  asym_hash : {r['provenance']['travel_matrix_hash']}")

print(f"\nWrote {OUT}")
print(f"Total elapsed: {(time.time() - started) / 60:.1f} min")
