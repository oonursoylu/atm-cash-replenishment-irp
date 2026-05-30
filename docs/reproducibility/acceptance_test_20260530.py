"""
TEMPORARY acceptance test for the reproducibility hardening. NO 30-day sim, NO
sweep. Resolves the "two controls changed together" confound by isolating the
travel-matrix fix from the CPLEX-determinism fix.

Procedure (cheap: 1 day-3 MILP solve per worker process):
  0. Pin the controlled cell (gap=0.05, nv=3, hetero, seed=42, s0.95 CSV) and
     LOAD the frozen travel matrix (so tau is identical in every process).
  1. Compute the day-3 ENTRY inventory ONCE (days 1-2 dispatch 0 vehicles, so it
     is deterministic) and assert it equals the known hash 7723e2e2...; save it
     to docs/day3_entry_state_20260530.json so workers reuse it (no re-solve).
  2. For each condition, solve the day-3 horizon MILP in TWO separate processes
     and compare objective + day-1 actions (stops/total_q/routes/qhash):
        (a) frozen matrix + CURRENT (non-deterministic) CPLEX   [deterministic=False]
        (b) frozen matrix + DETERMINISTIC CPLEX                 [deterministic=True]
  3. Attribute:
        (a) identical cross-process            -> frozen matrix alone is the fix;
                                                  CPLEX-det is pure hardening.
        (a) differs but (b) identical          -> both controls are necessary.
        neither identical                      -> determinism not yet achieved.

Run (from project root):  python scripts/_tmp_acceptance_test.py
(Internally re-invokes itself as: python scripts/_tmp_acceptance_test.py worker <a|b> <out.json>)
Safe to delete this script and docs/day3_entry_state_20260530.json afterwards.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.spatial import load_hardcoded_spatial, get_capacity_per_atm
from src.data.travel import load_frozen_matrix
from src.data.real_demand import load_real_demand
from src.optim.irp_milp import build_model, solve_model, extract_actions
from src.sim.rolling_horizon import _execute_day

DAY3_ENTRY_PATH = ROOT / "docs" / "day3_entry_state_20260530.json"
OUT_REPORT = ROOT / "docs" / "acceptance_test_20260530.json"
KNOWN_DAY3_ENTRY_HASH = "7723e2e27bac4bd8f118bebbdabe4382"
SIM_DAY = 3


def pinned_cfg() -> dict:
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
    assert cfg["REAL_DEMAND_CSV_PATH"].endswith("test_predictions_p0.55_s0.95.csv"), \
        cfg["REAL_DEMAND_CSV_PATH"]
    return cfg


def inv_sig(inv: dict) -> str:
    items = sorted((a, round(v, 6)) for a, v in inv.items())
    return hashlib.md5(repr(items).encode()).hexdigest()


def actions_sig(actions: dict) -> dict:
    q_items = sorted((a, round(q, 6)) for a, q in actions["q"].items())
    q_hash = hashlib.md5(repr(q_items).encode()).hexdigest()
    return {
        "stops": sum(1 for q in actions["q"].values() if q > 0),
        "total_q": round(sum(actions["q"].values()), 2),
        "routes": len(actions["routes"]),
        "qhash": q_hash,
    }


def _load_master_and_frozen(cfg):
    sp = load_hardcoded_spatial()
    cap = get_capacity_per_atm()
    frozen = load_frozen_matrix(sp, symmetrize=True)
    if frozen is None:
        raise SystemExit(
            "Frozen travel matrix not found. Run scripts/freeze_travel_matrix.py first."
        )
    tt, backend = frozen
    demand = load_real_demand(sp, cfg, cap)
    master = {
        "spatial": sp, "travel_time": tt,
        "d_mean": demand["d_mean"], "d_safety": demand["d_safety"],
        "capacity_per_atm": cap,
    }
    return sp, cap, tt, backend, demand, master


# --------------------------------------------------------------------------- #
# WORKER: solve day-3 once under the requested CPLEX condition, dump results.  #
# --------------------------------------------------------------------------- #
def run_worker(condition: str, out_path: str) -> int:
    cfg = pinned_cfg()
    _sp, _cap, tt, backend, _demand, master = _load_master_and_frozen(cfg)

    entry = json.loads(DAY3_ENTRY_PATH.read_text(encoding="utf-8"))
    inv = {a: float(v) for a, v in entry["inventory"].items()}
    assert inv_sig(inv) == entry["hash"], "day-3 entry state hash drifted on load"

    deterministic = (condition == "b")
    t0 = time.time()
    irp = build_model(SIM_DAY, dict(inv), master, cfg)
    sol = solve_model(irp, cfg, deterministic=deterministic)
    sd = irp.mdl.solve_details
    actions = extract_actions(irp, sol)
    result = {
        "condition": condition,
        "deterministic": deterministic,
        "pid": __import__("os").getpid(),
        "objective": (sol.objective_value if sol is not None else None),
        "mip_rel_gap": getattr(sd, "mip_relative_gap", None),
        "status": getattr(sd, "status", None),
        "solve_time_sec": round(time.time() - t0, 2),
        "frozen_tt_hash": entry.get("frozen_tt_hash"),
        "backend": backend,
        "actions": actions_sig(actions),
    }
    irp.mdl.end()
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[worker {condition} pid={result['pid']}] obj={result['objective']} "
          f"actions={result['actions']}")
    return 0


# --------------------------------------------------------------------------- #
# ORCHESTRATOR                                                                 #
# --------------------------------------------------------------------------- #
def compute_day3_entry() -> dict:
    cfg = pinned_cfg()
    _sp, _cap, tt, backend, demand, master = _load_master_and_frozen(cfg)
    from src.data.travel import matrix_content_hash
    frozen_hash = matrix_content_hash(tt)

    inv = demand["initial_inventory"].copy()
    print(f"  initial inventory hash : {inv_sig(inv)}")
    dummy = {k: 0 for k in ("travel_cost", "dispatch_cost", "drop_fees",
             "holding_cost", "stockout_cost", "stockout_events",
             "total_deliveries", "total_dispatches")}
    atms = sorted(master["spatial"]["atm_location"].keys())
    for d in (1, 2):
        _execute_day(d, inv, master, demand["actual_demand"], atms, tt, cfg, dummy)
    h = inv_sig(inv)
    print(f"  day-3 ENTRY inventory  : {h}")
    print(f"  expected (known)       : {KNOWN_DAY3_ENTRY_HASH}")
    match = (h == KNOWN_DAY3_ENTRY_HASH)
    print(f"  entry hash matches known: {match}")
    entry = {
        "hash": h, "known_hash": KNOWN_DAY3_ENTRY_HASH, "matches_known": match,
        "frozen_tt_hash": frozen_hash, "backend": backend,
        "inventory": {a: inv[a] for a in atms},
    }
    DAY3_ENTRY_PATH.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    print(f"  saved day-3 entry state -> {DAY3_ENTRY_PATH.relative_to(ROOT)}")
    return entry


def run_condition(condition: str, n_proc: int = 2) -> list[dict]:
    results = []
    for p in range(1, n_proc + 1):
        out = ROOT / "docs" / f"_acc_{condition}_{p}.json"
        print(f"\n  [condition {condition}] launching process {p}/{n_proc} ...")
        subprocess.run(
            [sys.executable, str(Path(__file__)), "worker", condition, str(out)],
            check=True, cwd=str(ROOT),
        )
        results.append(json.loads(out.read_text(encoding="utf-8")))
        out.unlink()
    return results


def cross_process_identical(results: list[dict]) -> bool:
    objs = {r["objective"] for r in results}
    sigs = {json.dumps(r["actions"], sort_keys=True) for r in results}
    return len(objs) == 1 and len(sigs) == 1


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        return run_worker(sys.argv[2], sys.argv[3])

    print("=" * 82)
    print("ACCEPTANCE TEST: isolate frozen-matrix fix vs CPLEX-determinism fix")
    print("=" * 82)
    print("\n[Step 1] Compute & pin day-3 entry state (frozen matrix, days 1-2 = 0 veh)")
    entry = compute_day3_entry()

    report = {"day3_entry": entry, "conditions": {}}
    for condition, desc in (("a", "frozen matrix + CURRENT (non-det) CPLEX"),
                            ("b", "frozen matrix + DETERMINISTIC CPLEX")):
        print("\n" + "=" * 82)
        print(f"[Step 2{condition}] {desc}  (2 separate processes)")
        print("=" * 82)
        results = run_condition(condition)
        identical = cross_process_identical(results)
        report["conditions"][condition] = {
            "desc": desc, "identical_cross_process": identical, "runs": results,
        }
        print(f"\n  >> condition ({condition}) cross-process identical: "
              f"{'YES (PASS)' if identical else 'NO (DIFFER)'}")
        for r in results:
            print(f"     pid={r['pid']} obj={r['objective']} "
                  f"stops={r['actions']['stops']} qhash={r['actions']['qhash']} "
                  f"t={r['solve_time_sec']}s")

    a_id = report["conditions"]["a"]["identical_cross_process"]
    b_id = report["conditions"]["b"]["identical_cross_process"]
    if a_id and b_id:
        attribution = ("Frozen matrix ALONE makes day-3 cross-process identical; "
                       "CPLEX-determinism is pure hardening (not required here).")
    elif (not a_id) and b_id:
        attribution = ("Frozen matrix is NOT sufficient alone; with deterministic "
                       "CPLEX the result is cross-process identical -> BOTH controls "
                       "are necessary.")
    elif (not a_id) and (not b_id):
        attribution = ("Even frozen matrix + deterministic CPLEX still differ "
                       "cross-process -> another non-determinism source remains.")
    else:
        attribution = ("Unexpected: (a) identical but (b) not -- determinism config "
                       "introduced variation; investigate.")
    report["attribution"] = attribution

    print("\n" + "#" * 82)
    print("# ATTRIBUTION")
    print("#" * 82)
    print(f"  (a) identical cross-process: {a_id}")
    print(f"  (b) identical cross-process: {b_id}")
    print(f"  => {attribution}")
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
