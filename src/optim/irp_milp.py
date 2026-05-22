"""
MILP for single-day rolling-horizon IRP: multi-vehicle routing with inventory balance,
capacity constraints, MTZ subtour elimination, and soft stockout penalties.
Orchestrates build_model (variables, constraints, objective), solve_model (CPLEX tuning),
extract_actions (solution -> dispatch dict).
"""

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from docplex.mp.model import Model

from ..data.spatial import SpatialData


# CPLEX status codes that indicate a usable integer solution is present.
# 101: optimal, 102: optimal-with-tolerance, 105: aborted-feasible,
# 107: time-limit-feasible, 113: solution-limit-feasible.
_FEASIBLE_STATUS_CODES = (101, 102, 105, 107, 113)


class IRPConfig(TypedDict):
    """CONFIG keys consumed by this module. Other keys are ignored."""
    PLANNING_HORIZON: int
    NUM_VEHICLES: int
    ATM_CAPACITY: float                 # uniform-mode default; unused by build_model
    VEHICLE_CAPACITY: float
    MIN_LOAD_PER_VISIT: float
    STOCKOUT_PENALTY: float
    EOH_FIXED_FEE: float
    EOH_PEN_RATE: float
    SAFETY_FLOOR_PEN: float
    DISPATCH_COST_PER_VEHICLE: float
    DROP_FEE_PER_ATM: float
    TRAVEL_COST_PER_MIN: float
    HOLDING_COST_PER_DAY: float
    ONSITE_FIXED_MIN: float
    OFFSITE_FIXED_MIN: float
    CASSETTE_COEF: float
    SHIFT_LIMIT_MIN: float
    MIP_GAP: float
    TIME_LIMIT_SEC: int
    USE_SYMMETRY_BREAKING: bool


class MasterData(TypedDict):
    """External inputs the optimiser depends on for one rolling-horizon solve."""
    spatial: SpatialData
    travel_time: dict[tuple[int, int], float]
    d_mean: dict[tuple[str, int], float]
    d_safety: dict[tuple[str, int], float]
    capacity_per_atm: dict[str, float]   # caller populates per-ATM cap


class Day1Actions(TypedDict):
    """First-day dispatch plan returned to the simulator."""
    q: dict[str, float]                  # ATM ID -> total cash delivered on day 1
    routes: dict[int, list[int]]         # vehicle k -> [depot, ..., depot]


@dataclass
class IRPModel:
    """
    Container holding the constructed CPLEX model plus references to every
    decision variable dict and the index sets needed to extract a solution.
    Rationale: the variable dicts are not retrievable from the Model object
    alone, so we keep the same Python references that built the constraints.
    """
    mdl: Model
    x: dict
    y: dict
    z: dict
    w: dict
    q: dict
    Inv: dict
    s: dict
    safety_slack: dict
    eoh_slack: dict
    is_stockout: dict
    is_eoh_short: dict
    u: dict
    T: list[int]
    K: list[int]
    L: list[int]
    I: list[str]
    depot: int


# ---- internal helpers ------------------------------------------------------

def _compute_demand_inputs(
    sim_day: int,
    master_data: MasterData,
    cfg: IRPConfig,
    I: list[str],
    T: list[int],
) -> tuple[dict, dict, dict]:
    """
    Slice forecast series for the current rolling window. Per-ATM capacity
    caps the forecast above a feasible level for each ATM individually.
    d_safe is the calibrated forecast safety quantile (alpha_safety) verbatim,
    with no multiplier applied.

    EOH target is the next-day safety quantile d_safe[T+1], ensuring
    continuity with the subsequent rolling window (Powell 2011, ADP
    post-decision state-value approximation).
    """
    cap_per = master_data["capacity_per_atm"]
    d_phys = {
        (a, t): min(master_data["d_mean"][(a, sim_day + t - 1)], cap_per[a])
        for a in I for t in T
    }
    d_safe = {
        (a, t): min(master_data["d_safety"][(a, sim_day + t - 1)], cap_per[a])
        for a in I for t in T
    }
    T_plus_1 = T[-1] + 1
    eoh = {
        a: min(
            master_data["d_safety"][(a, sim_day + T_plus_1 - 1)],
            cap_per[a],
        )
        for a in I
    }
    return d_phys, d_safe, eoh


# ---- public API: build / solve / extract -----------------------------------

def build_model(
    sim_day: int,
    actual_inventory: dict[str, float],
    master_data: MasterData,
    cfg: IRPConfig,
) -> IRPModel:
    """
    Construct the CPLEX MILP for one rolling-horizon window starting at sim_day.

    Decision variables: x (routing arcs), y (visits), z (deliveries),
    u (MTZ position), s (stockout slack), safety_slack, eoh_slack.
    Constraints: inventory balance, vehicle capacity, shift-time limit, MTZ
    subtour elimination, minimum load per visit, lexicographic symmetry breaking.
    Objective: travel + dispatch + drop fees + holding + safety-floor slack +
    stockout penalty + EOH penalty.

    No solve is performed here. The returned IRPModel holds the model plus
    references to all decision variables so the caller can solve and extract.
    """
    T = list(range(1, cfg["PLANNING_HORIZON"] + 1))
    K = list(range(cfg["NUM_VEHICLES"]))
    sp = master_data["spatial"]
    L = list(range(sp["num_locations"]))
    I = sorted(sp["atm_location"].keys())
    depot = 0

    L_visit = [l for l in L if l != depot]
    N = len(L_visit)
    onsite_locs = [l for l in L_visit if sp["location_type"][l] == "onsite"]
    offsite_locs = [l for l in L_visit if sp["location_type"][l] == "offsite"]

    cap_per = master_data["capacity_per_atm"]
    d_phys, d_safe, eoh = _compute_demand_inputs(sim_day, master_data, cfg, I, T)

    mdl = Model(name=f"IRP_Day_{sim_day}")

    # Decision variables.
    x = mdl.binary_var_dict(
        [(i, j, k, t) for i in L for j in L if i != j for k in K for t in T],
        name="x",
    )
    y = mdl.binary_var_dict(
        [(l, k, t) for l in L_visit for k in K for t in T], name="y"
    )
    z = mdl.binary_var_dict(
        [(a, k, t) for a in I for k in K for t in T], name="z"
    )
    w = mdl.binary_var_dict([(k, t) for k in K for t in T], name="w")
    q = mdl.continuous_var_dict(
        [(a, k, t) for a in I for k in K for t in T], lb=0, name="q"
    )
    # Per-ATM upper bound. Callable ub supported by docplex 2.20+.
    Inv = mdl.continuous_var_dict(
        [(a, t) for a in I for t in T], lb=0,
        ub=lambda key: cap_per[key[0]], name="Inv",
    )
    s = mdl.continuous_var_dict([(a, t) for a in I for t in T], lb=0, name="s")
    safety_slack = mdl.continuous_var_dict(
        [(a, t) for a in I for t in T], lb=0, name="safety_slack"
    )
    eoh_slack = mdl.continuous_var_dict(I, lb=0, name="eoh_slack")
    is_stockout = mdl.binary_var_dict(
        [(a, t) for a in I for t in T], name="is_stockout"
    )
    is_eoh_short = mdl.binary_var_dict(I, name="is_eoh_short")
    u = mdl.continuous_var_dict(
        [(l, k, t) for l in L_visit for k in K for t in T],
        lb=1, ub=N, name="u",
    )

    # Objective: travel + dispatch + drops + holding + stockout + EOH + safety.
    tt = master_data["travel_time"]
    travel = mdl.sum(
        tt[(i, j)] * cfg["TRAVEL_COST_PER_MIN"] * x[(i, j, k, t)]
        for i in L for j in L if i != j for k in K for t in T
    )
    dispatch = mdl.sum(cfg["DISPATCH_COST_PER_VEHICLE"] * w[(k, t)] for k in K for t in T)
    drops = mdl.sum(cfg["DROP_FEE_PER_ATM"] * z[(a, k, t)] for a in I for k in K for t in T)
    hold = mdl.sum(cfg["HOLDING_COST_PER_DAY"] * Inv[(a, t)] for a in I for t in T)
    stock = mdl.sum(cfg["STOCKOUT_PENALTY"] * is_stockout[(a, t)] for a in I for t in T)
    eoh_cost = mdl.sum(
        cfg["EOH_FIXED_FEE"] * is_eoh_short[a] + cfg["EOH_PEN_RATE"] * eoh_slack[a]
        for a in I
    )
    safety_cost = mdl.sum(
        cfg["SAFETY_FLOOR_PEN"] * safety_slack[(a, t)] for a in I for t in T
    )
    mdl.minimize(travel + dispatch + drops + hold + stock + eoh_cost + safety_cost)

    # Routing flow conservation and depot enter/exit.
    for k in K:
        for t in T:
            for l in L_visit:
                mdl.add_constraint(
                    mdl.sum(x[(i, l, k, t)] for i in L if i != l) == y[(l, k, t)]
                )
                mdl.add_constraint(
                    mdl.sum(x[(l, j, k, t)] for j in L if j != l) == y[(l, k, t)]
                )
            out_flow = mdl.sum(x[(depot, j, k, t)] for j in L if j != depot)
            in_flow = mdl.sum(x[(i, depot, k, t)] for i in L if i != depot)
            mdl.add_constraint(out_flow == w[(k, t)])
            mdl.add_constraint(in_flow == w[(k, t)])

    # ATM-to-location coupling and load bounds (per-ATM capacity).
    for a in I:
        l = sp["atm_location"][a]
        cap_a = cap_per[a]
        for k in K:
            for t in T:
                mdl.add_constraint(z[(a, k, t)] <= y[(l, k, t)])
                mdl.add_constraint(q[(a, k, t)] <= cap_a * z[(a, k, t)])
                mdl.add_constraint(q[(a, k, t)] >= cfg["MIN_LOAD_PER_VISIT"] * z[(a, k, t)])

    # Each non-depot location served by at most one vehicle per day.
    for l in L_visit:
        for t in T:
            mdl.add_constraint(mdl.sum(y[(l, k, t)] for k in K) <= 1)
            for k in K:
                mdl.add_constraint(y[(l, k, t)] <= w[(k, t)])

    # Inventory balance, stockout slack, and safety floor (per-ATM capacity).
    for a in I:
        cap_a = cap_per[a]
        for t in T:
            mdl.add_constraint(mdl.sum(z[(a, k, t)] for k in K) <= 1)
            prev = actual_inventory[a] if t == T[0] else Inv[(a, t - 1)]
            tot_in = mdl.sum(q[(a, k, t)] for k in K)
            mdl.add_constraint(Inv[(a, t)] == prev + tot_in - d_phys[(a, t)] + s[(a, t)])
            mdl.add_constraint(prev + tot_in <= cap_a)
            mdl.add_constraint(s[(a, t)] <= d_phys[(a, t)] * is_stockout[(a, t)])
            # Residual inventory must cover the learned tail gap (d_safe - d_mean).
            mdl.add_constraint(
                Inv[(a, t)] + safety_slack[(a, t)] >= d_safe[(a, t)] - d_phys[(a, t)]
            )

    # End-of-horizon target.
    T_end = T[-1]
    for a in I:
        mdl.add_constraint(Inv[(a, T_end)] + eoh_slack[a] >= eoh[a])
        mdl.add_constraint(eoh_slack[a] <= eoh[a] * is_eoh_short[a])

    # Vehicle capacity, shift-time budget, MTZ subtour elimination.
    for k in K:
        for t in T:
            mdl.add_constraint(mdl.sum(q[(a, k, t)] for a in I) <= cfg["VEHICLE_CAPACITY"])
            travel_expr = mdl.sum(tt[(i, j)] * x[(i, j, k, t)] for i in L for j in L if i != j)
            onsite_fixed = mdl.sum(cfg["ONSITE_FIXED_MIN"] * y[(l, k, t)] for l in onsite_locs)
            offsite_fixed = mdl.sum(cfg["OFFSITE_FIXED_MIN"] * y[(l, k, t)] for l in offsite_locs)
            cassette = mdl.sum(cfg["CASSETTE_COEF"] * q[(a, k, t)] for a in I)
            mdl.add_constraint(
                travel_expr + onsite_fixed + offsite_fixed + cassette <= cfg["SHIFT_LIMIT_MIN"]
            )
            for l in L_visit:
                for m in L_visit:
                    if l != m:
                        mdl.add_constraint(
                            u[(l, k, t)] - u[(m, k, t)] + N * x[(l, m, k, t)] <= N - 1
                        )

    if cfg["USE_SYMMETRY_BREAKING"]:
        for t in T:
            for k in range(len(K) - 1):
                mdl.add_constraint(w[(K[k], t)] >= w[(K[k + 1], t)])

    return IRPModel(
        mdl=mdl, x=x, y=y, z=z, w=w, q=q, Inv=Inv, s=s,
        safety_slack=safety_slack, eoh_slack=eoh_slack,
        is_stockout=is_stockout, is_eoh_short=is_eoh_short, u=u,
        T=T, K=K, L=L, I=I, depot=depot,
    )


def solve_model(irp: IRPModel, cfg: IRPConfig) -> Any:
    """
    Configure CPLEX and solve. Returns SolveSolution or None.
    Uses empirically-tuned heuristics (RINS, probe) to find good incumbents quickly.
    """
    mdl = irp.mdl
    mdl.parameters.mip.tolerances.mipgap = cfg["MIP_GAP"]
    mdl.parameters.timelimit = cfg["TIME_LIMIT_SEC"]
    mdl.parameters.emphasis.mip = 1
    mdl.parameters.mip.strategy.heuristicfreq = 20
    mdl.parameters.mip.strategy.rinsheur = 50
    mdl.parameters.mip.strategy.probe = 2
    return mdl.solve(log_output=False)


def extract_actions(irp: IRPModel, sol: Any) -> Day1Actions:
    """
    Read the day-1 dispatch decisions from a solved IRPModel.

    If sol is None or status is not feasible, returns zero-quantity actions
    so the simulator can index actions["q"][a] safely without branching.
    Binary threshold of 0.9 is used (instead of 0.5) because under a 5% MIP
    gap, integer values of 0.51 or 0.99 can occur and the tighter threshold
    avoids ambiguous route reconstruction.
    """
    actions: Day1Actions = {"q": {a: 0.0 for a in irp.I}, "routes": {}}

    has_solution = (sol is not None) and (
        irp.mdl.solve_details.status_code in _FEASIBLE_STATUS_CODES
    )
    if not has_solution:
        return actions

    for a in irp.I:
        actions["q"][a] = sum(irp.q[(a, k, 1)].solution_value for k in irp.K)

    for k in irp.K:
        used = any(
            irp.x[(i, j, k, 1)].solution_value > 0.9
            for i in irp.L for j in irp.L if i != j
        )
        if not used:
            continue
        route = [irp.depot]
        cur = irp.depot
        seen: set[int] = set()
        for _ in range(len(irp.L)):
            nxt = None
            for j in irp.L:
                if j != cur and irp.x[(cur, j, k, 1)].solution_value > 0.9:
                    nxt = j
                    break
            if nxt is None or nxt in seen:
                break
            if nxt == irp.depot:
                route.append(irp.depot)
                break
            route.append(nxt)
            seen.add(nxt)
            cur = nxt
        actions["routes"][k] = route

    return actions


def solve_single_horizon(
    sim_day: int,
    actual_inventory: dict[str, float],
    master_data: MasterData,
    cfg: IRPConfig,
) -> tuple[Day1Actions, Any]:
    """
    Legacy-compatible entry point. Builds, solves, extracts, and disposes of
    the CPLEX model in one call. Use this from the simulation loop; use
    build_model/solve_model/extract_actions directly when you need to inspect
    or modify the model between stages (e.g. recourse experiments).
    """
    irp = build_model(sim_day, actual_inventory, master_data, cfg)
    sol = solve_model(irp, cfg)
    actions = extract_actions(irp, sol)
    has_solution = (sol is not None) and (
        irp.mdl.solve_details.status_code in _FEASIBLE_STATUS_CODES
    )
    irp.mdl.end()
    return actions, (sol if has_solution else None)