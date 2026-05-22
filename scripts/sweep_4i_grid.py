"""
Phase 4.I — initial inventory lower-bound sensitivity sweep.

Decision Register #10: validate the choice initial_inventory_low_pct=0.30.
Sweeps the lower bound of the U(low, high) initial inventory band over
{0.10, 0.20, 0.30, 0.40, 0.50} with high=0.50 fixed.

Cell low=0.50 yields a degenerate point distribution (no randomness),
included as the upper boundary of the sweep.

Baseline cell (low=0.30) reused from Phase 4.A (3000, 0.1) cell at alpha=0.95;
all other parameters identical to current optimize.yaml baseline.

Required state in configs/optimize.yaml (pre-flight checked):
  real_demand_csv_path: predictions/test_predictions_p0.55_s0.95.csv
  use_heterogeneous_capacity: true
  num_vehicles: 3
  stockout_penalty: 3000
  safety_floor_pen: 0.1
  initial_inventory_high_pct: 0.50

Run from project root:
    python scripts/sweep_4i_grid.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4i_raw.json")
PRIOR_JSON = Path("docs/phase_4i_prior.json")

LOW_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
SWEEP_DAYS = 30

# Pre-flight expectations. Each (regex, error_msg).
# Patterns tolerate 0.5 / 0.50 / 0.500 representations of decimals.
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
    (r'^\s*initial_inventory_high_pct:\s*0\.50?\b',
     "initial_inventory_high_pct must be 0.50"),
]


def check_baseline(cfg_text: str) -> list[str]:
    issues = []
    for pat, msg in BASELINE_CHECKS:
        if not re.search(pat, cfg_text, flags=re.MULTILINE):
            issues.append(msg)
    return issues


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

    prior_lows = {round(p["initial_inventory_low_pct"], 4) for p in prior}
    cells_to_run = [low for low in LOW_GRID if round(low, 4) not in prior_lows]
    print(f"Will run {len(cells_to_run)} new cells, skipping {len(prior_lows)} existing.")

    backup = cfg_text
    results = list(prior)
    started = datetime.now()
    try:
        for i, low in enumerate(cells_to_run, 1):
            print(f"\n{'='*60}\n[Cell {i}/{len(cells_to_run)}] initial_inventory_low_pct={low}\n{'='*60}")
            patched = patch_yaml(backup, days=SWEEP_DAYS, initial_inventory_low_pct=low)
            CFG_PATH.write_text(patched, encoding="utf-8")
            run = subprocess.run([sys.executable, "main.py"],
                                  capture_output=True, text=True, encoding="utf-8")
            if run.returncode != 0:
                parsed = {"error": run.stderr[-500:]}
                print(f"  [!] FAILED")
            else:
                parsed = parse_summary(run.stdout)
                print(f"  Cost={parsed['total_cost']:,.0f}  Stockouts={parsed['stockouts']}")
            results.append({"initial_inventory_low_pct": low, **parsed})
            OUT_JSON.write_text(json.dumps(results, indent=2))
    finally:
        CFG_PATH.write_text(backup, encoding="utf-8")
        print(f"\nRestored {CFG_PATH}")

    # Bi-criteria summary
    print(f"\n{'='*72}\nFinal sweep (alpha_safety=0.95, hetero+nv=3, stockout=3000, sf=0.1)\n{'='*72}")
    print(f"{'low':>10}{'stockouts':>12}{'op_cost':>14}{'total_cost':>14}{'dispatches':>14}")
    for low in LOW_GRID:
        r = next((x for x in results
                  if abs(x.get("initial_inventory_low_pct", -1) - low) < 1e-6), None)
        if r and "stockouts" in r:
            op_cost = (r["travel_cost"] + r["dispatch_cost"]
                       + r["drop_fees"] + r["holding_cost"])
            print(f"{low:>10.2f}{r['stockouts']:>12}{op_cost:>14,.0f}"
                  f"{r['total_cost']:>14,.0f}{r['dispatches']:>14}")
        else:
            print(f"{low:>10.2f}{'?':>12}{'?':>14}{'?':>14}{'?':>14}")
    print(f"\nElapsed: {datetime.now() - started}")


if __name__ == "__main__":
    main()