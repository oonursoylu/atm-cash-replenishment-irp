"""
Phase 4.H — Fleet-size and capacity-mode ablation (clean re-run).

Full 2x2 grid: use_heterogeneous_capacity {false, true} x num_vehicles {2, 3}
at the final thesis cost parameters (stockout_penalty=3000, safety_floor_pen=0.1)
and the clean 50-trial alpha_safety=0.95 forecast.

Original Phase 4.H (2026-05-11) was contaminated: it consumed a forecast CSV
whose point model had been trained with only 3 Optuna trials. This re-run
validates Decision #6 (fleet size) and Decision #9 (capacity mode) under the
final operating configuration. Contaminated originals preserved in
docs/phase_4h_raw.json and docs/phase_4h_uniform_raw.json.

Pre-conditions in configs/optimize.yaml (pre-flight checked):
  real_demand_csv_path: predictions/test_predictions_p0.55_s0.95.csv
  stockout_penalty: 3000
  safety_floor_pen: 0.1
  initial_inventory_low_pct: 0.30
  initial_inventory_high_pct: 0.50
  symmetrize: true
  seed: 42
  planning_horizon: 7

Run from project root:
    python scripts/sweep_4h.py
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4h_clean_raw.json")

CAPACITY_GRID = [False, True]
VEHICLE_GRID = [2, 3]
SWEEP_DAYS = 30
SWEEP_MIP_GAP = 0.05

BASELINE_CHECKS = [
    (r'real_demand_csv_path:\s*["\']?predictions/test_predictions_p0\.55_s0\.95\.csv',
     "real_demand_csv_path must point to test_predictions_p0.55_s0.95.csv"),
    (r'^\s*stockout_penalty:\s*3000\b',
     "stockout_penalty must be 3000"),
    (r'^\s*safety_floor_pen:\s*0\.1\b',
     "safety_floor_pen must be 0.1"),
    (r'^\s*initial_inventory_low_pct:\s*0\.30?\b',
     "initial_inventory_low_pct must be 0.30"),
    (r'^\s*initial_inventory_high_pct:\s*0\.50?\b',
     "initial_inventory_high_pct must be 0.50"),
    (r'^\s*symmetrize:\s*true\b',
     "travel_matrix.symmetrize must be true"),
    (r'^\s*seed:\s*42\b',
     "reproducibility.seed must be 42"),
    (r'^\s*planning_horizon:\s*7\b',
     "simulation.planning_horizon must be 7"),
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

    cells = [(hetero, nv) for hetero in CAPACITY_GRID for nv in VEHICLE_GRID]
    n_cells = len(cells)

    backup = cfg_text
    results = []
    started = datetime.now()
    try:
        for i, (hetero, nv) in enumerate(cells, 1):
            cap_label = "hetero" if hetero else "uniform"
            print(f"\n{'='*60}")
            print(f"[Cell {i}/{n_cells}] {cap_label}, nv={nv}")
            print(f"{'='*60}")
            patched = patch_yaml(
                backup,
                days=SWEEP_DAYS,
                num_vehicles=nv,
                use_heterogeneous_capacity=str(hetero).lower(),
                mip_gap=SWEEP_MIP_GAP,
            )
            CFG_PATH.write_text(patched, encoding="utf-8")
            t0 = time.time()
            run = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True, text=True, encoding="utf-8",
            )
            compute_sec = round(time.time() - t0, 1)
            if run.returncode != 0:
                print(f"  [!] FAILED. stderr tail:\n{run.stderr[-500:]}")
                parsed = {"error": run.stderr[-1000:]}
            else:
                parsed = parse_summary(run.stdout)
                op_cost = (parsed["travel_cost"] + parsed["dispatch_cost"]
                           + parsed["drop_fees"] + parsed["holding_cost"])
                print(f"  Stockouts={parsed['stockouts']}  "
                      f"Op cost={op_cost:,.0f} TL  "
                      f"Total={parsed['total_cost']:,.0f} TL  "
                      f"Dispatches={parsed['dispatches']}  "
                      f"({compute_sec:.0f}s)")
            results.append({
                "use_heterogeneous_capacity": hetero,
                "num_vehicles": nv,
                "compute_sec": compute_sec,
                **parsed,
            })
            OUT_JSON.write_text(json.dumps(results, indent=2))
    finally:
        CFG_PATH.write_text(backup, encoding="utf-8")
        print(f"\nRestored {CFG_PATH}")

    print(f"\n{'='*72}")
    print(f"Phase 4.H clean re-run — 2x2 fleet-size x capacity-mode ablation")
    print(f"alpha_safety=0.95, stockout=3000, sf=0.1, gap=0.05, 30 days")
    print(f"{'='*72}")
    print(f"{'cap':>8}{'nv':>5}{'SO':>6}{'op_cost':>12}{'total':>12}"
          f"{'disp':>7}{'travel':>10}{'d_cost':>10}{'drop':>10}"
          f"{'hold':>10}{'sec':>8}")
    print("-" * 98)
    for hetero in CAPACITY_GRID:
        for nv in VEHICLE_GRID:
            r = next((x for x in results
                      if x["use_heterogeneous_capacity"] == hetero
                      and x["num_vehicles"] == nv), None)
            if r and "stockouts" in r:
                cap = "hetero" if hetero else "uniform"
                op = (r["travel_cost"] + r["dispatch_cost"]
                      + r["drop_fees"] + r["holding_cost"])
                print(f"{cap:>8}{nv:>5}{r['stockouts']:>6}{op:>12,.0f}"
                      f"{r['total_cost']:>12,.0f}{r['dispatches']:>7}"
                      f"{r['travel_cost']:>10,.0f}{r['dispatch_cost']:>10,.0f}"
                      f"{r['drop_fees']:>10,.0f}{r['holding_cost']:>10,.0f}"
                      f"{r['compute_sec']:>8.0f}")
    print(f"\nElapsed: {datetime.now() - started}")


if __name__ == "__main__":
    main()
