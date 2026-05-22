"""
Shared infrastructure for the B0/B1 replenishment baselines.

The baselines reuse the proposed system's instance (spatial topology, OSM
travel matrix, heterogeneous capacities, real-demand panels, SEED=42 initial
inventory) and reuse its realisation-and-accounting logic verbatim, so that
B0, B1, and the proposed XGBoost-IRP system are scored identically. Only the
daily decision rule differs between the three systems.

The accounting in `accumulate_day_kpis` mirrors
`src/sim/rolling_horizon.py::_execute_day` step for step; any divergence there
would invalidate the cross-system comparison.
"""

from __future__ import annotations

from typing import TypedDict

from ..data.spatial import (
    SpatialData,
    get_capacity_per_atm,
    load_hardcoded_spatial,
)
from ..data.travel import build_travel_matrix
from ..data.real_demand import load_real_demand


# --- types ------------------------------------------------------------------

class BaselineKPIs(TypedDict):
    """KPI accumulator. Same fields as sim.rolling_horizon.SimulationKPIs."""
    travel_cost: float
    dispatch_cost: float
    drop_fees: float
    holding_cost: float
    stockout_cost: float
    stockout_events: int
    total_deliveries: float
    total_dispatches: int


class DayPlan(TypedDict):
    """Same shape as optim.irp_milp.Day1Actions; the accounting code consumes
    a heuristic plan and a MILP plan identically."""
    q: dict[str, float]            # ATM ID -> cash delivered on this day
    routes: dict[int, list[int]]   # vehicle k -> [depot, loc, ..., depot]


class Instance(TypedDict):
    """Shared 73-day instance, identical to the proposed system's setup."""
    spatial: SpatialData
    travel_time: dict[tuple[int, int], float]
    atms: list[str]
    capacity_per_atm: dict[str, float]
    d_mean: dict[tuple[str, int], float]
    d_safety: dict[tuple[str, int], float]
    actual_demand: dict[tuple[str, int], float]
    initial_inventory: dict[str, float]


class BaselineRun(TypedDict):
    """Result of one baseline run over the full simulation window."""
    kpis: BaselineKPIs
    daily_log: list[dict]
    detail: dict


# --- instance setup ---------------------------------------------------------

def load_instance(cfg: dict) -> Instance:
    """Build the shared instance.

    Mirrors the setup block of rolling_horizon.run_simulation so the baselines
    see byte-identical capacities, travel times, demand panels, and the
    SEED-seeded initial inventory draw.
    """
    if not cfg["USE_REAL_DEMAND"]:
        raise ValueError("Baselines require USE_REAL_DEMAND=true.")

    sp = load_hardcoded_spatial()
    tt, _backend = build_travel_matrix(sp, cfg)
    atms = sorted(sp["atm_location"].keys())

    if cfg["USE_HETEROGENEOUS_CAPACITY"]:
        cap_per = get_capacity_per_atm()
    else:
        cap_per = {a: cfg["ATM_CAPACITY"] for a in atms}

    demand = load_real_demand(sp, cfg, cap_per)
    return {
        "spatial": sp,
        "travel_time": tt,
        "atms": atms,
        "capacity_per_atm": cap_per,
        "d_mean": demand["d_mean"],
        "d_safety": demand["d_safety"],
        "actual_demand": demand["actual_demand"],
        "initial_inventory": demand["initial_inventory"],
    }


def fresh_kpis() -> BaselineKPIs:
    """Zeroed KPI accumulator."""
    return {
        "travel_cost": 0.0,
        "dispatch_cost": 0.0,
        "drop_fees": 0.0,
        "holding_cost": 0.0,
        "stockout_cost": 0.0,
        "stockout_events": 0,
        "total_deliveries": 0.0,
        "total_dispatches": 0,
    }


# --- greedy router ----------------------------------------------------------

def greedy_route(
    loads: dict[str, float],
    atms: list[str],
    spatial: SpatialData,
    tt: dict[tuple[int, int], float],
    cfg: dict,
) -> DayPlan:
    """Location-level greedy nearest-neighbour router with multi-vehicle
    escalation.

    `loads` holds only the ATMs the decision rule scheduled for today, each
    with a positive, capacity-feasible load. ATMs are aggregated to their
    locations: a location enters the route once at least one of its ATMs is
    scheduled, and the visiting vehicle services every scheduled ATM there.

    Service time per location is a fixed component (onsite or offsite) plus a
    cassette component proportional to the cash dropped, matching the shift
    constraint in optim.irp_milp.build_model.

    A vehicle extends its route to the nearest unrouted location that keeps
    both the 8-hour shift budget (including the return-to-depot leg) and the
    vehicle cash capacity feasible. When no location qualifies the vehicle
    returns to the depot and the next vehicle starts, up to NUM_VEHICLES.
    Locations still unrouted after the last vehicle receive no delivery today.
    """
    depot = 0

    loc_atms: dict[int, list[str]] = {}
    for a, amt in loads.items():
        if amt <= 0:
            continue
        loc = spatial["atm_location"][a]
        loc_atms.setdefault(loc, []).append(a)

    loc_load = {
        loc: sum(loads[a] for a in members)
        for loc, members in loc_atms.items()
    }

    def service_time(loc: int) -> float:
        is_onsite = spatial["location_type"][loc] == "onsite"
        fixed = cfg["ONSITE_FIXED_MIN"] if is_onsite else cfg["OFFSITE_FIXED_MIN"]
        cassette = sum(cfg["CASSETTE_COEF"] * loads[a] for a in loc_atms[loc])
        return fixed + cassette

    shift = cfg["SHIFT_LIMIT_MIN"]
    vehicle_cap = cfg["VEHICLE_CAPACITY"]

    unrouted: set[int] = set(loc_atms.keys())
    routes: dict[int, list[int]] = {}

    for k in range(cfg["NUM_VEHICLES"]):
        if not unrouted:
            break
        route = [depot]
        cur = depot
        time_used = 0.0
        cash_used = 0.0
        while True:
            best_loc: int | None = None
            best_dist: float | None = None
            for loc in unrouted:
                dist = tt[(cur, loc)]
                if cash_used + loc_load[loc] > vehicle_cap:
                    continue
                round_trip = time_used + dist + service_time(loc) + tt[(loc, depot)]
                if round_trip > shift:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist, best_loc = dist, loc
            if best_loc is None:
                break
            time_used += best_dist + service_time(best_loc)
            cash_used += loc_load[best_loc]
            cur = best_loc
            route.append(best_loc)
            unrouted.discard(best_loc)
        if len(route) > 1:
            route.append(depot)
            routes[k] = route

    routed_locs = {loc for r in routes.values() for loc in r if loc != depot}
    q: dict[str, float] = {a: 0.0 for a in atms}
    for loc in routed_locs:
        for a in loc_atms[loc]:
            q[a] = loads[a]

    return {"q": q, "routes": routes}


# --- realisation and accounting ---------------------------------------------

def accumulate_day_kpis(
    plan: DayPlan,
    actual_inventory: dict[str, float],
    actual_demand: dict[tuple[str, int], float],
    sim_day: int,
    atms: list[str],
    tt: dict[tuple[int, int], float],
    capacity_per_atm: dict[str, float],
    cfg: dict,
    kpis: BaselineKPIs,
) -> dict:
    """Apply one day's plan: realise actual demand, advance inventory, and
    accumulate KPIs.

    Mirrors rolling_horizon._execute_day so the baselines are scored
    identically to the proposed system. Mutates `actual_inventory` and `kpis`
    in place; returns a per-day log record.
    """
    daily_dispatches = len(plan["routes"])
    daily_stops = sum(1 for _a, qty in plan["q"].items() if qty > 0)
    daily_delivery_tl = sum(plan["q"].values())

    kpis["total_dispatches"] += daily_dispatches
    kpis["dispatch_cost"] += daily_dispatches * cfg["DISPATCH_COST_PER_VEHICLE"]
    kpis["drop_fees"] += daily_stops * cfg["DROP_FEE_PER_ATM"]
    kpis["total_deliveries"] += daily_delivery_tl

    day_travel_cost = 0.0
    for r in plan["routes"].values():
        day_travel_cost += sum(
            tt[(r[i], r[i + 1])] * cfg["TRAVEL_COST_PER_MIN"]
            for i in range(len(r) - 1)
        )
    kpis["travel_cost"] += day_travel_cost

    daily_stockouts = 0
    daily_holding_tl = 0.0
    for a in atms:
        delivered = plan["q"][a]
        demanded = actual_demand[(a, sim_day)]
        new_inv = actual_inventory[a] + delivered - demanded
        if new_inv < 0:
            daily_stockouts += 1
            kpis["stockout_events"] += 1
            kpis["stockout_cost"] += cfg["STOCKOUT_PENALTY"]
            new_inv = 0
        new_inv = min(new_inv, capacity_per_atm[a])
        actual_inventory[a] = new_inv
        daily_holding_tl += new_inv * cfg["HOLDING_COST_PER_DAY"]
    kpis["holding_cost"] += daily_holding_tl

    return {
        "day": sim_day,
        "stockouts": daily_stockouts,
        "dispatches": daily_dispatches,
        "stops": daily_stops,
        "delivered": daily_delivery_tl,
        "travel": day_travel_cost,
        "holding": daily_holding_tl,
    }


def summarise(kpis: BaselineKPIs, n_atm_days: int) -> dict:
    """Bi-criteria summary. Operational cost is the sum of the four measurable
    components; reported total adds the shadow stockout penalty for
    transparency only."""
    op_cost = (
        kpis["travel_cost"]
        + kpis["dispatch_cost"]
        + kpis["drop_fees"]
        + kpis["holding_cost"]
    )
    return {
        "stockouts": kpis["stockout_events"],
        "service_level": 1.0 - kpis["stockout_events"] / n_atm_days,
        "op_cost": op_cost,
        "reported_total": op_cost + kpis["stockout_cost"],
        "dispatches": kpis["total_dispatches"],
        "travel": kpis["travel_cost"],
        "dispatch_cost": kpis["dispatch_cost"],
        "drop_fees": kpis["drop_fees"],
        "holding": kpis["holding_cost"],
        "total_deliveries": kpis["total_deliveries"],
    }