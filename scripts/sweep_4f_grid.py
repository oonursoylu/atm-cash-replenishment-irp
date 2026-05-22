"""
Phase 4.F — OSM symmetric vs asymmetric travel matrix ablation.

Decision Register #15: isolate the value of directional routing information.
Compares the asymmetric OSM matrix (baseline) against its symmetrized variant
(mean of (i,j) and (j,i) directions per pair).

Asymmetric baseline reused from Phase 4.I low=0.30 cell. Only 1 new cell
(symmetric=true) is run. ~30 min compute.

Pre-conditions in configs/optimize.yaml:
  real_demand_csv_path: predictions/test_predictions_p0.55_s0.95.csv
  use_heterogeneous_capacity: true
  num_vehicles: 3
  stockout_penalty: 3000
  safety_floor_pen: 0.1
  initial_inventory_low_pct: 0.30
  initial_inventory_high_pct: 0.50
  travel_matrix:
    symmetrize: false   <- driver toggles this between cells

Pre-conditions in source code:
  src/data/spatial.py (or wherever travel_time is built) must honour
  the travel_matrix.symmetrize flag — see Phase 4.F code-change instructions
  in the methodology doc.

Run from project root:
    python scripts/sweep_4f_grid.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4f_raw.json")
PRIOR_JSON = Path("docs/phase_4f_prior.json")

SYMMETRIZE_GRID = [False, True]
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
    (r'^\s*symmetrize:\s*(true|false)\b',
     "travel_matrix.symmetrize flag must exist in configs/optimize.yaml — "
     "ensure Phase 4.F code change has been applied to spatial.py and the "
     "flag is added under travel_matrix:"),
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
        print("[!] Pre-flight check failed. Fix configs/optimize.yaml (and code) before running:")
        for i in issues:
            print(f"    - {i}")
        sys.exit(1)
    print("[OK] Pre-flight baseline check passed.")

    prior = []
    if PRIOR_JSON.exists():
        prior = json.loads(PRIOR_JSON.read_text())
        print(f"Loaded {len(prior)} prior cell(s) from {PRIOR_JSON}")

    prior_flags = {bool(p["symmetrize"]) for p in prior}
    cells_to_run = [s for s in SYMMETRIZE_GRID if s not in prior_flags]
    print(f"Will run {len(cells_to_run)} new cell(s), skipping {len(prior_flags)} existing.")

    backup = cfg_text
    results = list(prior)
    started = datetime.now()
    try:
        for i, sym in enumerate(cells_to_run, 1):
            print(f"\n{'='*60}\n[Cell {i}/{len(cells_to_run)}] symmetrize={sym}\n{'='*60}")
            patched = patch_yaml(backup, days=SWEEP_DAYS, symmetrize=str(sym).lower())
            CFG_PATH.write_text(patched, encoding="utf-8")
            run = subprocess.run([sys.executable, "main.py"],
                                  capture_output=True, text=True, encoding="utf-8")
            if run.returncode != 0:
                parsed = {"error": run.stderr[-500:]}
                print(f"  [!] FAILED")
            else:
                parsed = parse_summary(run.stdout)
                print(f"  Cost={parsed['total_cost']:,.0f}  Stockouts={parsed['stockouts']}")
            results.append({"symmetrize": sym, **parsed})
            OUT_JSON.write_text(json.dumps(results, indent=2))
    finally:
        CFG_PATH.write_text(backup, encoding="utf-8")
        print(f"\nRestored {CFG_PATH}")

    print(f"\n{'='*72}\nFinal sweep — Decision #15 ablation\n{'='*72}")
    print(f"{'symmetrize':>14}{'stockouts':>12}{'op_cost':>14}{'total_cost':>14}{'travel':>10}")
    for sym in SYMMETRIZE_GRID:
        r = next((x for x in results if bool(x.get("symmetrize")) == sym), None)
        if r and "stockouts" in r:
            op_cost = (r["travel_cost"] + r["dispatch_cost"]
                       + r["drop_fees"] + r["holding_cost"])
            print(f"{str(sym):>14}{r['stockouts']:>12}{op_cost:>14,.0f}"
                  f"{r['total_cost']:>14,.0f}{r['travel_cost']:>10,.0f}")
        else:
            print(f"{str(sym):>14}{'?':>12}{'?':>14}{'?':>14}{'?':>10}")
    print(f"\nElapsed: {datetime.now() - started}")


if __name__ == "__main__":
    main()