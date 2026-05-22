"""
Data-layer regression test.

Exercises the three deterministic data modules (spatial, travel,
synthetic_demand) end-to-end and prints a small set of reference values
that should match the locked-in regression output exactly. Run from the
project root:

    python tests/test_phase_2a.py

Expected outputs (with SEED=42, baseline CONFIG):
    Spatial:       22 locations, 31 ATMs
    Haversine tt:  tt[(0,1)] = 26.7 min
    Synthetic dem: d_mean[Z0241002, t=1] = 15017.01
                   d_safety[Z0241002, t=1] = 36371.64
                   actual[Z0241002, t=1] = 16516.01
                   I0[Z0241002] = 177856.86
"""

import sys
from pathlib import Path

# Make the project root importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.spatial import load_hardcoded_spatial
from src.data.travel import build_travel_matrix
from src.data.synthetic_demand import generate_master_timeseries


# Minimal CONFIG subset to exercise the data-layer modules in isolation,
# decoupled from configs/optimize.yaml so this regression test does not
# break when YAML defaults are tuned.
CONFIG = {
    # Travel matrix
    "USE_OSM": False,                  # haversine fallback to avoid network dep
    "OSM_PLACE": "Kadikoy, Istanbul, Turkey",
    "OSM_NETWORK_DIST_M": 15_000,
    "URBAN_SPEED_KMH": 20,
    "DETOUR_FACTOR": 1.35,
    # Demand
    "SIMULATION_DAYS": 30,
    "PLANNING_HORIZON": 7,
    "ATM_CAPACITY": 400_000,
    "INITIAL_INV_LOW": 0.40,
    "INITIAL_INV_HIGH": 0.60,
    "SEED": 42,
}


def main() -> None:
    print("=" * 60)
    print("PHASE 2.A VALIDATION")
    print("=" * 60)

    # --- spatial.py ---
    sp = load_hardcoded_spatial()
    n_loc = sp["num_locations"]
    n_atms = len(sp["atm_location"])
    print(f"\n[spatial]  {n_loc} locations, {n_atms} ATMs")
    print(f"           depot:    {sp['location_name'][0]}")
    print(f"           loc 1:    type={sp['location_type'][1]}, "
          f"ATMs={sp['location_atms'][1]}")
    assert n_loc == 22 and n_atms == 31, "spatial counts off"

    # --- travel.py ---
    tt, backend = build_travel_matrix(sp, CONFIG)
    print(f"\n[travel]   backend={backend}, matrix size={len(tt)}")
    print(f"           tt[(0,1)] = {tt[(0, 1)]} min  "
          f"(depot -> {sp['location_name'][1].split(' (')[0]})")
    print(f"           tt[(1,0)] = {tt[(1, 0)]} min  (symmetric: "
          f"{tt[(0, 1)] == tt[(1, 0)]})")
    assert tt[(0, 1)] == 26.7, f"haversine tt mismatch: {tt[(0, 1)]}"

    # --- synthetic_demand.py ---
    dem = generate_master_timeseries(sp, CONFIG)
    a_test = "Z0241002"
    dm = dem["d_mean"][(a_test, 1)]
    ds = dem["d_safety"][(a_test, 1)]
    da = dem["actual_demand"][(a_test, 1)]
    i0 = dem["initial_inventory"][a_test]
    print(f"\n[demand]   {a_test}, t=1:")
    print(f"           d_mean        = {dm:.2f}")
    print(f"           d_safety      = {ds:.2f}")
    print(f"           actual_demand = {da:.2f}")
    print(f"           initial_inv   = {i0:.2f}")
    assert abs(dm - 15017.01) < 0.01, f"d_mean drift: {dm}"
    assert abs(ds - 36371.64) < 0.01, f"d_safety drift: {ds}"
    assert abs(da - 16516.01) < 0.01, f"actual drift: {da}"
    assert abs(i0 - 177856.86) < 0.01, f"I0 drift: {i0}"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — Phase 2.A modules behave identically to legacy.")
    print("=" * 60)


if __name__ == "__main__":
    main()