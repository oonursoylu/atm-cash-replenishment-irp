"""
B1 -- XGBoost forecast + threshold replenishment with greedy routing.

B1 isolates the value of the forecasting layer. It reuses the v8 forecast
outputs already generated for the proposed system (d_mean at alpha_point,
d_safety at alpha_safety, from the prediction CSV loaded by real_demand.py)
but replaces the IRP MILP with a myopic threshold rule and the shared greedy
router.

Each day, for every ATM, B1 projects end-of-day inventory under a no-delivery
assumption using today's point forecast. When that projection falls below
tomorrow's safety forecast the ATM is scheduled, and the load lifts projected
end-of-day inventory to tomorrow's safety forecast plus a one-day
point-demand buffer:

    proj_eod = inventory - d_mean[a, t]
    trigger  : proj_eod < d_safety[a, t + 1]
    target   = d_safety[a, t + 1] + d_mean[a, t + 2]
    load     = target - proj_eod        (capped at C_a - inventory)

Forecast values are capped at per-ATM capacity, matching how the proposed
system feeds forecasts into the IRP (optim.irp_milp._compute_demand_inputs),
so B1 and the proposed system consume the forecast layer identically.
"""

from __future__ import annotations

from .common import (
    BaselineRun,
    Instance,
    accumulate_day_kpis,
    fresh_kpis,
    greedy_route,
)


def plan_day_b1(
    sim_day: int,
    inventory: dict[str, float],
    d_mean: dict[tuple[str, int], float],
    d_safety: dict[tuple[str, int], float],
    capacity_per_atm: dict[str, float],
    cfg: dict,
) -> dict[str, float]:
    """Return {ATM: load} for ATMs triggered today under the forecast
    threshold rule. A computed load below the minimum visit threshold is
    dropped to avoid micro-stops."""
    min_load = cfg["MIN_LOAD_PER_VISIT"]
    loads: dict[str, float] = {}
    for a, inv_a in inventory.items():
        cap = capacity_per_atm[a]
        dm_today = min(d_mean[(a, sim_day)], cap)
        ds_tomorrow = min(d_safety[(a, sim_day + 1)], cap)
        dm_after = min(d_mean[(a, sim_day + 2)], cap)
        proj_eod = inv_a - dm_today
        if proj_eod < ds_tomorrow:
            target = ds_tomorrow + dm_after
            raw = min(target - proj_eod, cap - inv_a)
            if raw >= min_load:
                loads[a] = raw
    return loads


def run_b1(cfg: dict, instance: Instance) -> BaselineRun:
    """Run B1 over the full simulation window."""
    atms = instance["atms"]
    cap_per = instance["capacity_per_atm"]
    tt = instance["travel_time"]
    sp = instance["spatial"]
    d_mean = instance["d_mean"]
    d_safety = instance["d_safety"]

    inventory = dict(instance["initial_inventory"])
    kpis = fresh_kpis()
    daily_log: list[dict] = []

    for sim_day in range(1, cfg["SIMULATION_DAYS"] + 1):
        loads = plan_day_b1(sim_day, inventory, d_mean, d_safety, cap_per, cfg)
        plan = greedy_route(loads, atms, sp, tt, cfg)
        rec = accumulate_day_kpis(
            plan, inventory, instance["actual_demand"], sim_day,
            atms, tt, cap_per, cfg, kpis,
        )
        daily_log.append(rec)

    return {"kpis": kpis, "daily_log": daily_log, "detail": {}}