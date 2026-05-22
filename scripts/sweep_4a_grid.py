"""
Phase 4.A — alpha_safety=0.95 hetero+nv=3 grid re-run.

Mirrors Phase 4.K's 12-cell hetero grid at the higher forecast safety quantile.
Tests whether (3000, 0.1) remains Pareto-dominant under more conservative
forecasts — directly addresses Decision Register #19 (forecast-IRP coupling).

Mini-4.A cell (3000, 0.1) at alpha_s=0.95 is reused from prior single-cell
what-if; results read from docs/phase_4a_prior.json if present.

Pre-conditions in configs/optimize.yaml:
  real_demand_csv_path: predictions/test_predictions_p0.55_s0.95.csv
  use_heterogeneous_capacity: true
  num_vehicles: 3

Run from project root:
    python scripts/sweep_4a_grid.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4a_raw.json")
PRIOR_JSON = Path("docs/phase_4a_prior.json")

STOCKOUT_GRID = [3000, 5000, 8000]
SAFETY_GRID = [0.001, 0.01, 0.1, 1.0]
SWEEP_DAYS = 30
REQUIRED_CSV_TOKEN = "test_predictions_p0.55_s0.95.csv"


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
    # Guard: this sweep is meaningful only at alpha_safety=0.95.
    if REQUIRED_CSV_TOKEN not in cfg_text:
        print(f"[!] {CFG_PATH} must reference {REQUIRED_CSV_TOKEN}. Aborting.")
        sys.exit(1)

    prior = []
    if PRIOR_JSON.exists():
        prior = json.loads(PRIOR_JSON.read_text())
        print(f"Loaded {len(prior)} prior cells from {PRIOR_JSON}")

    prior_keys = {(p["stockout_penalty"], p["safety_floor_pen"]) for p in prior}
    cells_to_run = [(so, sf) for so in STOCKOUT_GRID for sf in SAFETY_GRID
                    if (so, sf) not in prior_keys]
    print(f"Will run {len(cells_to_run)} new cells, skipping {len(prior_keys)} existing.")

    backup = cfg_text
    results = list(prior)
    started = datetime.now()
    try:
        for i, (so, sf) in enumerate(cells_to_run, 1):
            print(f"\n{'='*60}\n[Cell {i}/{len(cells_to_run)}] stockout={so}, sf={sf} (hetero+nv=3, alpha_s=0.95)\n{'='*60}")
            patched = patch_yaml(backup, days=SWEEP_DAYS, stockout_penalty=so, safety_floor_pen=sf)
            CFG_PATH.write_text(patched, encoding="utf-8")
            run = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True, encoding="utf-8")
            if run.returncode != 0:
                parsed = {"error": run.stderr[-500:]}
                print(f"  [!] FAILED")
            else:
                parsed = parse_summary(run.stdout)
                print(f"  Cost={parsed['total_cost']:,.0f}  Stockouts={parsed['stockouts']}")
            results.append({"stockout_penalty": so, "safety_floor_pen": sf, **parsed})
            OUT_JSON.write_text(json.dumps(results, indent=2))
    finally:
        CFG_PATH.write_text(backup, encoding="utf-8")
        print(f"\nRestored {CFG_PATH}")

    print(f"\n{'='*60}\nFinal 3x4 grid (stockouts) at alpha_safety=0.95\n{'='*60}")
    print(f"{'':>10}" + "".join(f"sf={sf:<8}" for sf in SAFETY_GRID))
    for so in STOCKOUT_GRID:
        row = f"so={so:<8}"
        for sf in SAFETY_GRID:
            match = next((r for r in results
                          if r["stockout_penalty"] == so and r["safety_floor_pen"] == sf), None)
            row += f"{match['stockouts'] if match and 'stockouts' in match else '?':>10}"
        print(row)
    print(f"\nElapsed: {datetime.now() - started}")


if __name__ == "__main__":
    main()