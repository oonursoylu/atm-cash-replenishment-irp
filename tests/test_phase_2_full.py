"""
End-to-end integration test for the refactored pipeline.

Runs the rolling-horizon simulation against the production config and
validates that (i) `load_config` produces every key the downstream modules
require, (ii) the simulation completes without solver-infeasibility, and
(iii) the returned KPI dictionary has the expected structure and value
ranges (non-negative costs, non-negative counts).

Requires CPLEX (docplex). Skips cleanly if not installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    print("=" * 82)
    print("END-TO-END PIPELINE INTEGRATION TEST")
    print("=" * 82)

    import importlib.util
    try:
        spec = importlib.util.find_spec("docplex.mp.model")
    except (ModuleNotFoundError, ValueError):
        spec = None
    if spec is None:
        print("[SKIP] docplex not installed; cannot run MILP. Test halted.")
        return

    from src.config import load_config
    from src.sim.rolling_horizon import run_simulation
    from src.viz.map import generate_and_save_map

    cfg = load_config()

    # Loader produced every key the downstream modules will index.
    required_keys = [
        "SIMULATION_DAYS", "PLANNING_HORIZON", "NUM_VEHICLES", "VEHICLE_CAPACITY",
        "ATM_CAPACITY", "INITIAL_INV_LOW", "INITIAL_INV_HIGH",
        "HOLDING_COST_PER_DAY", "DISPATCH_COST_PER_VEHICLE", "DROP_FEE_PER_ATM",
        "TRAVEL_COST_PER_MIN", "MIN_LOAD_PER_VISIT",
        "STOCKOUT_PENALTY", "SAFETY_FLOOR_PEN",
        "EOH_FIXED_FEE", "EOH_PEN_RATE",
        "ONSITE_FIXED_MIN", "OFFSITE_FIXED_MIN", "CASSETTE_COEF", "SHIFT_LIMIT_MIN",
        "USE_OSM", "URBAN_SPEED_KMH", "DETOUR_FACTOR",
        "MIP_GAP", "TIME_LIMIT_SEC", "USE_SYMMETRY_BREAKING",
        "SEED", "MAP_OUTPUT",
    ]
    for k in required_keys:
        assert k in cfg, f"Missing CONFIG key: {k}"
    print(f"[OK] config.load_config produced {len(cfg)} keys.")

    # Bridge (spatial, output_path) -> (spatial, cfg) callback shape.
    def _map_fn(spatial, cfg_inner):
        generate_and_save_map(spatial, cfg_inner["MAP_OUTPUT"])

    print("\n[RUN] Refactored pipeline:\n")
    kpis = run_simulation(cfg, map_generator=_map_fn)

    expected = {"travel_cost", "dispatch_cost", "drop_fees", "holding_cost",
                "stockout_cost", "stockout_events", "total_deliveries",
                "total_dispatches"}
    assert set(kpis) == expected, f"KPI keys mismatch: {set(kpis) ^ expected}"
    for cost_key in {"travel_cost", "dispatch_cost", "drop_fees",
                     "holding_cost", "stockout_cost"}:
        assert kpis[cost_key] >= 0, f"{cost_key} should be non-negative"
    assert kpis["stockout_events"] >= 0
    assert kpis["total_dispatches"] >= 0
    assert kpis["total_deliveries"] >= 0

    print("\n" + "=" * 82)
    print("PIPELINE INTEGRATION TEST OK")
    print("=" * 82)


if __name__ == "__main__":
    main()