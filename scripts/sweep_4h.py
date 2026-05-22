"""
Phase 4.H — Vehicle-count ablation at validated cost parameters (5000, 0.01).

Tests num_vehicles ∈ {2, 3} at the capacity mode currently set in
configs/optimize.yaml. The hetero baseline (use_heterogeneous_capacity: true) is
written to docs/phase_4h_raw.json. To produce the uniform variant
(docs/phase_4h_uniform_raw.json), flip use_heterogeneous_capacity to false in
configs/optimize.yaml before running this script and rename the output file
afterwards; both raw JSONs in docs/ were generated this way.

Capacity-binding hypothesis: an earlier 73-day production-scale run with two
vehicles showed peak days (days 8, 11, 12) where both vehicles were fully
loaded with 22-28 stops, coinciding with stockout clusters. Adding a third
vehicle isolates whether IRP throughput is the limiting factor on those peaks.

Run from project root:
    python scripts/sweep_4h.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4h_raw.json")

VEHICLE_GRID = [2, 3]
SWEEP_DAYS = 30


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
    backup = CFG_PATH.read_text(encoding="utf-8")
    results = []
    started = datetime.now()
    try:
        for i, nv in enumerate(VEHICLE_GRID, 1):
            print(f"\n{'='*60}\n[Cell {i}/{len(VEHICLE_GRID)}] num_vehicles={nv}\n{'='*60}")
            patched = patch_yaml(backup, days=SWEEP_DAYS, num_vehicles=nv)
            CFG_PATH.write_text(patched, encoding="utf-8")
            run = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True, text=True, encoding="utf-8",
            )
            if run.returncode != 0:
                print(f"  [!] FAILED. stderr tail:\n{run.stderr[-500:]}")
                parsed = {"error": run.stderr[-1000:]}
            else:
                parsed = parse_summary(run.stdout)
                print(f"  Cost={parsed['total_cost']:,.0f} TL  "
                      f"Stockouts={parsed['stockouts']}  "
                      f"Dispatches={parsed['dispatches']}")
            results.append({"num_vehicles": nv, **parsed})
            OUT_JSON.write_text(json.dumps(results, indent=2))
    finally:
        CFG_PATH.write_text(backup, encoding="utf-8")
        print(f"\nRestored {CFG_PATH}")
    print(f"\nDone. {len(VEHICLE_GRID)} cells in {datetime.now() - started}")


if __name__ == "__main__":
    main()
