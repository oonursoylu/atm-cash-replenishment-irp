"""
B0 -- static (s, S) order-up-to policy with greedy routing.

B0 is the forecast-free absolute baseline. Reorder point s and order-up-to
level S are computed once from a naive historical demand mean (no XGBoost
forecast), then held static across the simulation window. Each day every ATM
at or below its reorder point is scheduled for a same-day delivery that lifts
it toward S; scheduled ATMs are routed by the shared greedy router.

    s_a = max(2 * mu_a, MIN_LOAD_PER_VISIT)
    S_a = min(s_a + 5 * mu_a, 0.85 * C_a)

with mu_a the per-ATM mean daily withdrawal over positive-withdrawal days
strictly before the test window (see scripts/dump_train_means.py). The 2-day
reorder point covers one bridging day plus one day of safety, the 5-day
order-up-to span targets a roughly weekly delivery cycle, and the 0.85 * C_a
ceiling keeps the order-up-to level below the CIT cassette limit.
"""

from __future__ import annotations

from .common import (
    BaselineRun,
    Instance,
    accumulate_day_kpis,
    fresh_kpis,
    greedy_route,
)


def compute_ss_levels(
    train_means: dict[str, float],
    capacity_per_atm: dict[str, float],
    cfg: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute static per-ATM reorder point s and order-up-to level S."""
    s: dict[str, float] = {}
    S: dict[str, float] = {}
    for a, mu in train_means.items():
        cap = capacity_per_atm[a]
        s_a = max(2.0 * mu, cfg["MIN_LOAD_PER_VISIT"])
        S_a = min(s_a + 5.0 * mu, 0.85 * cap)
        s[a], S[a] = s_a, S_a
    return s, S


def plan_day_b0(
    inventory: dict[str, float],
    s: dict[str, float],
    S: dict[str, float],
    capacity_per_atm: dict[str, float],
    cfg: dict,
) -> dict[str, float]:
    """Return {ATM: load} for ATMs triggered today.

    An ATM triggers when its start-of-day inventory is at or below the reorder
    point; the load lifts it toward S, capped at physical capacity. A computed
    load below the minimum visit threshold is dropped to avoid micro-stops.
    """
    min_load = cfg["MIN_LOAD_PER_VISIT"]
    loads: dict[str, float] = {}
    for a, inv_a in inventory.items():
        if inv_a <= s[a]:
            raw = min(S[a] - inv_a, capacity_per_atm[a] - inv_a)
            if raw >= min_load:
                loads[a] = raw
    return loads


def run_b0(
    cfg: dict,
    instance: Instance,
    train_means: dict[str, float],
) -> BaselineRun:
    """Run B0 over the full simulation window."""
    atms = instance["atms"]
    cap_per = instance["capacity_per_atm"]
    tt = instance["travel_time"]
    sp = instance["spatial"]

    s, S = compute_ss_levels(train_means, cap_per, cfg)
    inventory = dict(instance["initial_inventory"])
    kpis = fresh_kpis()
    daily_log: list[dict] = []

    for sim_day in range(1, cfg["SIMULATION_DAYS"] + 1):
        loads = plan_day_b0(inventory, s, S, cap_per, cfg)
        plan = greedy_route(loads, atms, sp, tt, cfg)
        rec = accumulate_day_kpis(
            plan, inventory, instance["actual_demand"], sim_day,
            atms, tt, cap_per, cfg, kpis,
        )
        daily_log.append(rec)

    return {
        "kpis": kpis,
        "daily_log": daily_log,
        "detail": {"s_levels": s, "S_levels": S},
    }