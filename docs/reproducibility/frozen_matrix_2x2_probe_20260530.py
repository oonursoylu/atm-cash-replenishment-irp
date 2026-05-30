import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\onurs\Desktop\ma_2026_project")
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.spatial import load_hardcoded_spatial
from src.data.travel import build_travel_matrix
import src.sim.rolling_horizon as rh

OUT_JSON = ROOT / "docs" / "frozen_matrix_2x2_probe_20260530.json"

def matrix_hash(tt):
    items = sorted((int(i), int(j), round(float(v), 6)) for (i, j), v in tt.items())
    return hashlib.md5(repr(items).encode()).hexdigest(), round(sum(v for _, _, v in items), 2)

def op_cost(k):
    return k["travel_cost"] + k["dispatch_cost"] + k["drop_fees"] + k["holding_cost"]

def make_cfg(base, hetero, nv):
    cfg = copy.deepcopy(base)
    cfg["SIMULATION_DAYS"] = 30
    cfg["PLANNING_HORIZON"] = 7
    cfg["MIP_GAP"] = 0.05
    cfg["TIME_LIMIT_SEC"] = 600
    cfg["NUM_VEHICLES"] = nv
    cfg["USE_HETEROGENEOUS_CAPACITY"] = hetero
    cfg["STOCKOUT_PENALTY"] = 3000
    cfg["SAFETY_FLOOR_PEN"] = 0.1
    cfg["INITIAL_INV_LOW"] = 0.30
    cfg["INITIAL_INV_HIGH"] = 0.50
    cfg["SEED"] = 42
    cfg["USE_REAL_DEMAND"] = True
    cfg["SYMMETRIZE_TRAVEL_MATRIX"] = True
    assert cfg["REAL_DEMAND_CSV_PATH"].endswith("test_predictions_p0.55_s0.95.csv")
    return cfg

base = load_config()
base = make_cfg(base, True, 3)

print("=" * 88)
print("FROZEN TRAVEL MATRIX 2x2 PROBE")
print("Build OSM matrix once, then reuse exact same tt for all cells.")
print("=" * 88)

sp = load_hardcoded_spatial()
frozen_tt, backend = build_travel_matrix(sp, base)
tt_hash, tt_sum = matrix_hash(frozen_tt)

print(f"frozen_backend={backend}")
print(f"frozen_tt_hash={tt_hash}")
print(f"frozen_tt_sum={tt_sum}")

def frozen_build_travel_matrix(_sp, _cfg):
    print(f"[FROZEN_TT] backend={backend} hash={tt_hash} sum={tt_sum}")
    return frozen_tt, backend

rh.build_travel_matrix = frozen_build_travel_matrix

cells = [
    ("baseline_first_hetero_nv3", True, 3),
    ("uniform_nv2", False, 2),
    ("uniform_nv3", False, 3),
    ("hetero_nv2", True, 2),
    ("baseline_last_hetero_nv3", True, 3),
]

out = {
    "meta": {
        "purpose": "Frozen travel-matrix 2x2 probe; no optimize.yaml mutation.",
        "frozen_backend": backend,
        "frozen_tt_hash": tt_hash,
        "frozen_tt_sum": tt_sum,
        "fixed": {
            "days": 30,
            "planning_horizon": 7,
            "mip_gap": 0.05,
            "time_limit_sec": 600,
            "stockout_penalty": 3000,
            "safety_floor_pen": 0.1,
            "initial_inventory_low_pct": 0.30,
            "initial_inventory_high_pct": 0.50,
            "seed": 42,
            "symmetrize": True,
            "csv": base["REAL_DEMAND_CSV_PATH"],
        },
    },
    "results": [],
}

started = time.time()
for idx, (label, hetero, nv) in enumerate(cells, 1):
    print("\n" + "=" * 88)
    print(f"[Cell {idx}/{len(cells)}] {label} | hetero={hetero} nv={nv}")
    print("=" * 88)

    cfg = make_cfg(base, hetero, nv)
    t0 = time.time()
    k = rh.run_simulation(cfg, map_generator=None)
    sec = round(time.time() - t0, 1)

    row = {
        "label": label,
        "use_heterogeneous_capacity": hetero,
        "num_vehicles": nv,
        "compute_sec": sec,
        "stockouts": k["stockout_events"],
        "op_cost": round(op_cost(k), 2),
        "total_cost": round(op_cost(k) + k["stockout_cost"], 2),
        "dispatches": k["total_dispatches"],
        "travel_cost": round(k["travel_cost"], 2),
        "dispatch_cost": round(k["dispatch_cost"], 2),
        "drop_fees": round(k["drop_fees"], 2),
        "holding_cost": round(k["holding_cost"], 2),
        "stockout_cost": round(k["stockout_cost"], 2),
        "frozen_tt_hash": tt_hash,
    }
    out["results"].append(row)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"RESULT {label}: SO={row['stockouts']} op={row['op_cost']:,.2f} total={row['total_cost']:,.2f} "
          f"dispatches={row['dispatches']} sec={sec}")

print("\n" + "#" * 88)
print("# SUMMARY")
print("#" * 88)
print(f"{'label':>28} {'SO':>5} {'op':>12} {'total':>12} {'disp':>6} {'travel':>9}")
for r in out["results"]:
    print(f"{r['label']:>28} {r['stockouts']:>5} {r['op_cost']:>12,.0f} {r['total_cost']:>12,.0f} "
          f"{r['dispatches']:>6} {r['travel_cost']:>9,.0f}")

first = out["results"][0]
last = out["results"][-1]
same = (
    first["stockouts"] == last["stockouts"]
    and abs(first["op_cost"] - last["op_cost"]) < 0.01
    and first["dispatches"] == last["dispatches"]
)
print("\nBASELINE FIRST vs LAST:")
print(f"  first: SO={first['stockouts']} op={first['op_cost']:,.2f} dispatches={first['dispatches']}")
print(f"  last : SO={last['stockouts']} op={last['op_cost']:,.2f} dispatches={last['dispatches']}")
print(f"  identical_under_frozen_matrix={same}")
print(f"\nWrote {OUT_JSON}")
print(f"Total elapsed: {(time.time() - started) / 60:.1f} min")
