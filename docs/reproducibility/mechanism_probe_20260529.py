"""
TEMPORARY mechanism probe (NO 30-day sim, NO sweep). Two cheap tests:

A) RNG-leakage test: draw hetero initial inventory as if at sweep-position 1 vs 4,
   advancing the GLOBAL np.random + random RNGs in between, and check whether the
   inventory vector changes. (If unchanged -> inventory is position-independent ->
   RNG leakage refuted.)

B) CPLEX-determinism test: rebuild + solve the EXACT day-3 horizon MILP twice in the
   same process (day-3 entry inventory is deterministic because days 1-2 dispatch 0
   vehicles in every observed run) and compare objective + day-1 actions bit-for-bit.
   Day 3 is where the two 30-day runs first diverged (8 vs 9 stops).
"""

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.spatial import load_hardcoded_spatial, get_capacity_per_atm
from src.data.travel import build_travel_matrix
from src.data.real_demand import load_real_demand, _sample_initial_inventory
from src.optim.irp_milp import build_model, solve_model, extract_actions
from src.sim.rolling_horizon import _execute_day


def inv_sig(inv: dict) -> str:
    items = sorted((a, round(v, 6)) for a, v in inv.items())
    return hashlib.md5(repr(items).encode()).hexdigest()


def actions_sig(actions: dict) -> tuple:
    q_items = sorted((a, round(q, 6)) for a, q in actions["q"].items())
    q_hash = hashlib.md5(repr(q_items).encode()).hexdigest()
    stops = sum(1 for q in actions["q"].values() if q > 0)
    total_q = round(sum(actions["q"].values()), 2)
    n_routes = len(actions["routes"])
    return stops, total_q, n_routes, q_hash


def main() -> int:
    cfg = load_config()
    # Pin the controlled cell in the cfg dict (no YAML mutation needed).
    cfg["SIMULATION_DAYS"] = 30
    cfg["MIP_GAP"] = 0.05
    cfg["NUM_VEHICLES"] = 3
    cfg["USE_HETEROGENEOUS_CAPACITY"] = True

    sp = load_hardcoded_spatial()
    atms = sorted(sp["atm_location"].keys())
    cap = get_capacity_per_atm()  # heterogeneous tiers

    # ---------------- PART A: RNG-leakage test ----------------
    print("=" * 82)
    print("PART A  RNG-leakage test (hetero initial inventory at sweep-position 1 vs 4)")
    print("=" * 82)
    print("Inventory draw source: src/data/real_demand.py::_sample_initial_inventory")
    print("  rng = random.Random(seed)   # LOCAL instance, seed=cfg['SEED']=%r\n" % cfg["SEED"])
    sigs = []
    for cell in range(1, 5):
        # Simulate prior sweep cells consuming randomness from the GLOBAL RNGs.
        for _ in range(1000):
            np.random.random()
            random.random()
        inv = _sample_initial_inventory(atms, cap, seed=cfg["SEED"],
                                        low_pct=cfg["INITIAL_INV_LOW"],
                                        high_pct=cfg["INITIAL_INV_HIGH"])
        sig = inv_sig(inv)
        sigs.append(sig)
        print(f"  position {cell}: inv_hash={sig}  sum={sum(inv.values()):,.2f}  "
              f"first3={[round(inv[a], 2) for a in atms[:3]]}")
    print(f"\n  position 1 == position 4 ?  {sigs[0] == sigs[3]}  "
          f"(all identical: {len(set(sigs)) == 1})")

    # ---------------- PART B: CPLEX-determinism test ----------------
    print("\n" + "=" * 82)
    print("PART B  CPLEX-determinism test (rebuild+solve day-3 horizon TWICE, same process)")
    print("=" * 82)
    tt, _backend = build_travel_matrix(sp, cfg)
    demand = load_real_demand(sp, cfg, cap)
    master = {
        "spatial": sp, "travel_time": tt,
        "d_mean": demand["d_mean"], "d_safety": demand["d_safety"],
        "capacity_per_atm": cap,
    }
    inv = demand["initial_inventory"].copy()
    print(f"  initial inventory hash: {inv_sig(inv)}")

    # Deterministically advance to the day-3 entry state via days 1-2.
    dummy_kpis = {k: 0 for k in ("travel_cost", "dispatch_cost", "drop_fees",
                  "holding_cost", "stockout_cost", "stockout_events",
                  "total_deliveries", "total_dispatches")}
    for d in (1, 2):
        _execute_day(d, inv, master, demand["actual_demand"], atms, tt, cfg, dummy_kpis)
    day3_entry_hash = inv_sig(inv)
    print(f"  day-3 ENTRY inventory hash: {day3_entry_hash}  (deterministic; days 1-2 = 0 vehicles)")

    results = []
    for trial in (1, 2):
        irp = build_model(3, dict(inv), master, cfg)   # fresh model, identical inputs
        sol = solve_model(irp, cfg)
        sd = irp.mdl.solve_details
        obj = sol.objective_value if sol is not None else None
        gap = getattr(sd, "mip_relative_gap", None)
        status = getattr(sd, "status", None)
        stime = getattr(sd, "time", None)
        actions = extract_actions(irp, sol)
        asig = actions_sig(actions)
        irp.mdl.end()
        results.append({"obj": obj, "gap": gap, "status": status, "time": stime, "asig": asig})
        print(f"\n  Solve #{trial}:")
        print(f"    objective       = {obj}")
        print(f"    mip_rel_gap     = {gap}")
        print(f"    status          = {status}")
        print(f"    solve_time_sec  = {stime}")
        print(f"    day1 actions    = stops={asig[0]} total_q={asig[1]:,.0f} routes={asig[2]} qhash={asig[3]}")

    r1, r2 = results
    obj_same = (r1["obj"] == r2["obj"])
    act_same = (r1["asig"] == r2["asig"])
    print("\n" + "#" * 82)
    print(f"# day-3 objective bit-identical?  {obj_same}   (#1={r1['obj']!r}  #2={r2['obj']!r})")
    print(f"# day-3 day-1 actions identical?  {act_same}")
    print(f"#   #1 stops={r1['asig'][0]}  #2 stops={r2['asig'][0]}   "
          f"(the 30-day runs diverged here: 8 vs 9 stops)")
    if obj_same and act_same:
        print("#  => CPLEX DETERMINISTIC on this instance (same model -> same solution).")
    elif obj_same and not act_same:
        print("#  => DEGENERATE optima: same objective, DIFFERENT day-1 plan (cascades in sim).")
    else:
        print("#  => CPLEX NON-DETERMINISTIC: same model -> different objective.")
    print("#" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
