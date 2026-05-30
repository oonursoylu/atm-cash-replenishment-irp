"""
Rolling-horizon simulation: orchestrates setup, daily solve loops, KPI accumulation, and summary.
Map generation is an optional callback to decouple from visualization layer.
"""

import json
import time
from typing import Callable, TypedDict

from ..data.spatial import load_hardcoded_spatial, get_capacity_per_atm, SpatialData
from ..data.travel import build_travel_matrix
from ..data.synthetic_demand import generate_master_timeseries
from ..data.real_demand import load_real_demand
from ..optim.irp_milp import solve_single_horizon, MasterData
from ..provenance import build_provenance


class SimulationKPIs(TypedDict):
    """KPI accumulator. All values in TL except *_events / *_dispatches counts."""
    travel_cost: float
    dispatch_cost: float
    drop_fees: float
    holding_cost: float
    stockout_cost: float
    stockout_events: int
    total_deliveries: float
    total_dispatches: int


# Map generator signature: (spatial, cfg) -> None. None disables map output.
MapGenerator = Callable[[SpatialData, dict], None]


def _execute_day(
    sim_day: int,
    actual_inventory: dict[str, float],
    master_data: MasterData,
    actual_demand: dict[tuple[str, int], float],
    atms: list[str],
    tt: dict[tuple[int, int], float],
    cfg: dict,
    kpis: SimulationKPIs,
) -> bool:
    """
    Execute one simulation day: solve the rolling-horizon MILP, apply day-1
    decisions to actual_inventory in place, accumulate KPIs in place.

    Returns True if solved successfully, False if infeasible.
    KPI accumulation via in-place mutation as caller expects.
    """
    print(f"\n[DAY {sim_day}/{cfg['SIMULATION_DAYS']}] Solving 7-day horizon...")

    actions, sol = solve_single_horizon(sim_day, actual_inventory, master_data, cfg)
    if sol is None:
        print(f"  [!] INFEASIBLE on Day {sim_day}. Stopping simulation.")
        return False

    # Routing-side KPIs derived directly from the day-1 plan.
    daily_dispatches = len(actions["routes"])
    daily_stops = sum(1 for _a, qty in actions["q"].items() if qty > 0)
    daily_delivery_tl = sum(actions["q"].values())

    kpis["total_dispatches"] += daily_dispatches
    kpis["dispatch_cost"] += daily_dispatches * cfg["DISPATCH_COST_PER_VEHICLE"]
    kpis["drop_fees"] += daily_stops * cfg["DROP_FEE_PER_ATM"]
    kpis["total_deliveries"] += daily_delivery_tl

    day_travel_cost = 0.0
    for r in actions["routes"].values():
        day_travel_cost += sum(
            tt[(r[i], r[i + 1])] * cfg["TRAVEL_COST_PER_MIN"]
            for i in range(len(r) - 1)
        )
    kpis["travel_cost"] += day_travel_cost

    # Stockout cost uses the same Big-M penalty as the MILP objective.
    cap_per = master_data["capacity_per_atm"]
    daily_stockout_events = 0
    daily_holding_tl = 0.0

    for a in atms:
        delivered = actions["q"][a]
        demanded = actual_demand[(a, sim_day)]
        new_inv = actual_inventory[a] + delivered - demanded

        if new_inv < 0:
            daily_stockout_events += 1
            kpis["stockout_events"] += 1
            kpis["stockout_cost"] += cfg["STOCKOUT_PENALTY"]
            new_inv = 0

        new_inv = min(new_inv, cap_per[a])
        actual_inventory[a] = new_inv
        daily_holding_tl += new_inv * cfg["HOLDING_COST_PER_DAY"]

    kpis["holding_cost"] += daily_holding_tl

    print(f"  -> Executed: {daily_dispatches} vehicles, {daily_stops} stops, "
          f"{daily_delivery_tl:,.0f} TL loaded.")
    print(f"  -> Reality:  {daily_stockout_events} stockout events occurred today.")
    return True


def _print_summary(
    kpis: SimulationKPIs,
    cfg: dict,
    n_atms: int,
    elapsed_sec: float,
) -> None:
    """Final KPI summary report."""
    total_cost = (
        kpis["travel_cost"] + kpis["dispatch_cost"]
        + kpis["drop_fees"] + kpis["holding_cost"] + kpis["stockout_cost"]
    )

    print("\n" + "=" * 82)
    print(f"SIMULATION COMPLETE: {cfg['SIMULATION_DAYS']} DAYS EXECUTED")
    print("=" * 82)
    print(f"Total Compute Time : {elapsed_sec:.1f} seconds ({elapsed_sec / 60:.1f} mins)")

    print("\nOPERATIONAL KPIs (Actuals)")
    print(f"  Total Cost        : {total_cost:,.2f} TL")
    print(f"  Total Dispatches  : {kpis['total_dispatches']} vehicle shifts")
    print(f"  Total Cash Loaded : {kpis['total_deliveries']:,.0f} TL")
    print(f"  Stockout Events   : {kpis['stockout_events']} events "
          f"(over {cfg['SIMULATION_DAYS'] * n_atms} ATM-days)")

    print("\nCOST BREAKDOWN")
    print(f"  Travel Cost       : {kpis['travel_cost']:>12,.2f} TL")
    print(f"  Dispatch Cost     : {kpis['dispatch_cost']:>12,.2f} TL")
    print(f"  Drop Fees         : {kpis['drop_fees']:>12,.2f} TL")
    print(f"  Holding Cost      : {kpis['holding_cost']:>12,.2f} TL")
    print(f"  Stockout Penalties: {kpis['stockout_cost']:>12,.2f} TL")


def run_simulation(
    cfg: dict,
    *,
    map_generator: MapGenerator | None = None,
    return_provenance: bool = False,
) -> SimulationKPIs | tuple[SimulationKPIs, dict]:
    """
    Top-level rolling-horizon simulation.

    Setup: loads spatial topology, travel matrix, synthetic demand series.
    Loop:  solves a 7-day MILP each sim day, applies day-1 actions, advances inventory.
    Final: prints KPI summary and returns the accumulator.

    The optional map_generator callback is invoked once before the loop.
    Pass None to skip map output.

    Provenance: a reproducibility record (build_provenance) is always printed as a
    one-line `[PROVENANCE]` stamp. When return_provenance=True the function
    returns (kpis, provenance) so result writers can embed it; the default
    return shape (kpis only) is unchanged for existing callers.
    """
    print("=" * 82)
    print(f"STARTING {cfg['SIMULATION_DAYS']}-DAY ROLLING HORIZON SIMULATION")
    print(f"USING OPTIMAL PARAMS: {cfg['STOCKOUT_PENALTY']} TL Stockout Penalty "
          f"& {cfg['SAFETY_FLOOR_PEN']} TL/TL Safety Floor Penalty")
    if cfg["USE_HETEROGENEOUS_CAPACITY"]:
        print("CAPACITY MODE: heterogeneous (per-ATM tier from spatial.ATM_TIERS)")
    print("=" * 82)
    start_time = time.time()

    # Setup
    sp = load_hardcoded_spatial()
    tt, backend = build_travel_matrix(sp, cfg)
    atms = sorted(sp["atm_location"].keys())

    provenance = build_provenance(cfg, tt, backend)
    print(f"[PROVENANCE] {json.dumps(provenance)}")

    if cfg["USE_HETEROGENEOUS_CAPACITY"]:
        capacity_per_atm = get_capacity_per_atm()
    else:
        capacity_per_atm = {a: cfg["ATM_CAPACITY"] for a in atms}

    if cfg["USE_REAL_DEMAND"]:
        print(f"DEMAND SOURCE: real CSV ({cfg['REAL_DEMAND_CSV_PATH']})")
        demand = load_real_demand(sp, cfg, capacity_per_atm)
    else:
        print("DEMAND SOURCE: synthetic generator")
        demand = generate_master_timeseries(
            sp, cfg,
            capacity_per_atm=capacity_per_atm if cfg["USE_HETEROGENEOUS_CAPACITY"] else None,
        )
    master_data: MasterData = {
        "spatial": sp,
        "travel_time": tt,
        "d_mean": demand["d_mean"],
        "d_safety": demand["d_safety"],
        "capacity_per_atm": capacity_per_atm,
    }

    if map_generator is not None:
        map_generator(sp, cfg)

    actual_inventory = demand["initial_inventory"].copy()
    kpis: SimulationKPIs = {
        "travel_cost": 0.0, "dispatch_cost": 0.0, "drop_fees": 0.0,
        "holding_cost": 0.0, "stockout_cost": 0.0, "stockout_events": 0,
        "total_deliveries": 0.0, "total_dispatches": 0,
    }

    # Loop
    for sim_day in range(1, cfg["SIMULATION_DAYS"] + 1):
        ok = _execute_day(
            sim_day=sim_day,
            actual_inventory=actual_inventory,
            master_data=master_data,
            actual_demand=demand["actual_demand"],
            atms=atms,
            tt=tt,
            cfg=cfg,
            kpis=kpis,
        )
        if not ok:
            break

    elapsed = time.time() - start_time
    _print_summary(kpis, cfg, n_atms=len(atms), elapsed_sec=elapsed)
    if return_provenance:
        return kpis, provenance
    return kpis
