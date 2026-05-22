"""
Phase 4.S — multi-seed variance estimation at the headline baseline cell.

Quantifies the single-cell MIP-gap-induced stockout/cost variance under
SEED=42 plus four alternative seeds. Provides empirical underpinning for
the headline figures used in Phase 4.A, 4.I, 4.F, and the final 73-day
headline run.

The headline cell is the symmetric-OSM Pareto-frontier best-service cell at
alpha=(0.55, 0.95), hetero capacity, nv=3, (stockout_penalty=3000,
safety_floor_pen=0.1), low=0.30, mip_gap=0.05, 30-day horizon.

Prior cell (SEED=42 -> 66 stockouts) reused from phase_4s_prior.json so only
4 new seeds are executed. Approx ~2.1h compute.

Pre-conditions in configs/optimize.yaml (pre-flight checked):
  real_demand_csv_path: predictions/test_predictions_p0.55_s0.95.csv
  use_heterogeneous_capacity: true
  num_vehicles: 3
  stockout_penalty: 3000
  safety_floor_pen: 0.1
  initial_inventory_low_pct: 0.30
  initial_inventory_high_pct: 0.50
  travel_matrix.symmetrize: true
  solver.mip_gap: 0.05

Run from project root:
    python scripts/sweep_4s_seedvar.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4s_raw.json")
PRIOR_JSON = Path("docs/phase_4s_prior.json")

SEED_GRID = [7, 19, 31, 42, 53]   # 42 reused from prior; 4 new
SWEEP_DAYS = 30

BASELINE_CHECKS = [
    (r'real_demand_csv_path:\s*["\']?predictions/test_predictions_p0\.55_s0\.95\.csv',
     "real_demand_csv_path must point to test_predictions_p0.55_s0.95.csv"),
    (r'^\s*stockout_penalty:\s*3000\b',
     "stockout_penalty must be 3000"),
    (r'^\s*safety_floor_pen:\s*0\.1\b',
     "safety_floor_pen must be 0.1"),
    (r'^\s*use_heterogeneous_capacity:\s*true\b',
     "use_heterogeneous_capacity must be true"),
    (r'^\s*num_vehicles:\s*3\b',
     "num_vehicles must be 3"),
    (r'^\s*initial_inventory_low_pct:\s*0\.30?\b',
     "initial_inventory_low_pct must be 0.30"),
    (r'^\s*initial_inventory_high_pct:\s*0\.50?\b',
     "initial_inventory_high_pct must be 0.50"),
    (r'^\s*symmetrize:\s*true\b',
     "travel_matrix.symmetrize must be true (legacy baseline)"),
    (r'^\s*mip_gap:\s*0\.05\b',
     "solver.mip_gap must be 0.05 (sweep noise floor target)"),
]


def check_baseline(cfg_text: str) -> list[str]:
    return [m for pat, m in BASELINE_CHECKS
            if not re.search(pat, cfg_text, flags=re.MULTILINE)]


def patch_yaml(text: str, **kwargs) -> str:
    out = text
    for key, val in kwargs.items():
        pat = rf"^(\s*{re.escape(key)}:\s+)\S+(.*)$"
        out = re.sub(pat, rf"\g<1>{val}\g<2>", out, count=1, flags=re.MULTILINE)
    return out


def parse_summary(stdout: str) -> dict:
    def f(pat, t=float):
        m = re.search(pat, stdout)
        return None if m is None else t(m.group(1).replace(",", ""))
    return {
        "total_cost":    f(r"Total Cost\s*:\s*([\d,.]+)\s*TL"),
        "stockouts":     f(r"Stockout Events\s*:\s*(\d+)", int),
        "dispatches":    f(r"Total Dispatches\s*:\s*(\d+)", int),
        "travel_cost":   f(r"Travel Cost\s*:\s*([\d,.]+)\s*TL"),
        "dispatch_cost": f(r"Dispatch Cost\s*:\s*([\d,.]+)\s*TL"),
        "drop_fees":     f(r"Drop Fees\s*:\s*([\d,.]+)\s*TL"),
        "holding_cost":  f(r"Holding Cost\s*:\s*([\d,.]+)\s*TL"),
        "stockout_cost": f(r"Stockout Penalties:\s*([\d,.]+)\s*TL"),
    }


def main():
    cfg_text = CFG_PATH.read_text(encoding="utf-8")

    issues = check_baseline(cfg_text)
    if issues:
        print("[!] Pre-flight check failed. Fix configs/optimize.yaml before running:")
        for i in issues:
            print(f"    - {i}")
        sys.exit(1)
    print("[OK] Pre-flight baseline check passed.")

    prior = []
    if PRIOR_JSON.exists():
        prior = json.loads(PRIOR_JSON.read_text())
        print(f"Loaded {len(prior)} prior cell(s) from {PRIOR_JSON}")

    prior_seeds = {int(p["seed"]) for p in prior}
    cells_to_run = [s for s in SEED_GRID if s not in prior_seeds]
    print(f"Will run {len(cells_to_run)} new cell(s), skipping {len(prior_seeds)} existing.")
    print(f"Total expected: N={len(SEED_GRID)} seeds.")

    backup = cfg_text
    results = list(prior)
    started = datetime.now()
    try:
        for i, seed in enumerate(cells_to_run, 1):
            print(f"\n{'='*60}\n[Cell {i}/{len(cells_to_run)}] seed={seed}\n{'='*60}")
            patched = patch_yaml(backup, days=SWEEP_DAYS, seed=seed)
            CFG_PATH.write_text(patched, encoding="utf-8")
            run = subprocess.run([sys.executable, "main.py"],
                                  capture_output=True, text=True, encoding="utf-8")
            if run.returncode != 0:
                parsed = {"error": run.stderr[-500:]}
                print(f"  [!] FAILED")
            else:
                parsed = parse_summary(run.stdout)
                print(f"  Cost={parsed['total_cost']:,.0f}  Stockouts={parsed['stockouts']}")
            results.append({"seed": seed, **parsed})
            OUT_JSON.write_text(json.dumps(results, indent=2))
    finally:
        CFG_PATH.write_text(backup, encoding="utf-8")
        print(f"\nRestored {CFG_PATH}")

    # Variance summary
    print(f"\n{'='*78}\nFinal sweep — headline baseline cell variance, N={len(SEED_GRID)}\n{'='*78}")
    print(f"{'seed':>8}{'stockouts':>12}{'op_cost':>14}{'total_cost':>14}{'dispatches':>12}")
    s_vals, op_vals, tot_vals, disp_vals = [], [], [], []
    for seed in SEED_GRID:
        r = next((x for x in results if int(x.get("seed", -1)) == seed), None)
        if r and "stockouts" in r:
            op_cost = (r["travel_cost"] + r["dispatch_cost"]
                       + r["drop_fees"] + r["holding_cost"])
            s_vals.append(r["stockouts"])
            op_vals.append(op_cost)
            tot_vals.append(r["total_cost"])
            disp_vals.append(r["dispatches"])
            print(f"{seed:>8}{r['stockouts']:>12}{op_cost:>14,.0f}"
                  f"{r['total_cost']:>14,.0f}{r['dispatches']:>12}")
        else:
            print(f"{seed:>8}{'?':>12}{'?':>14}{'?':>14}{'?':>12}")

    if len(s_vals) >= 2:
        print(f"\n{'metric':>14}{'mean':>14}{'std':>10}{'min':>10}{'max':>10}{'range':>10}")
        for name, vals in [("stockouts", s_vals), ("op_cost", op_vals),
                            ("total_cost", tot_vals), ("dispatches", disp_vals)]:
            mu = mean(vals); sd = stdev(vals); lo = min(vals); hi = max(vals)
            print(f"{name:>14}{mu:>14,.1f}{sd:>10,.1f}{lo:>10,.0f}{hi:>10,.0f}{hi-lo:>10,.0f}")
        s_mu, s_sd = mean(s_vals), stdev(s_vals)
        print(f"\nStockout variance: mean={s_mu:.1f}, std={s_sd:.1f}, "
              f"95% band ~[{s_mu - 2*s_sd:.0f}, {s_mu + 2*s_sd:.0f}]")

    print(f"\nElapsed: {datetime.now() - started}")


if __name__ == "__main__":
    main()
