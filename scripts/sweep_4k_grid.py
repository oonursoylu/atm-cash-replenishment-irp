"""
Phase 4.K — Extended 12-cell hetero+nv=3 parameter grid.

Fills in the missing cells from Mini-4.C-hetero (which tested only 3 cells)
to give a symmetric hetero Pareto picture comparable to Phase 4.C uniform's
9-cell sweep. Adds sf=0.001 column to explore monotonic-extension behaviour.

Already-done cells from mini-4.C-hetero are SKIPPED to save compute; results
are read from docs/phase_4c_hetero_raw.json if present.

Run from project root (with optimize.yaml at hetero=true, nv=3):
    python scripts/sweep_4k_grid.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4k_raw.json")
PRIOR_JSON = Path("docs/phase_4c_hetero_raw.json")

STOCKOUT_GRID = [3000, 5000, 8000]
SAFETY_GRID = [0.001, 0.01, 0.1, 1.0]
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
    # Load prior mini-4.C-hetero results to skip duplicate cells
    prior = []
    if PRIOR_JSON.exists():
        prior = json.loads(PRIOR_JSON.read_text())
        print(f"Loaded {len(prior)} prior cells from {PRIOR_JSON}")

    prior_keys = {(p["stockout_penalty"], p["safety_floor_pen"]) for p in prior}
    cells_to_run = [(so, sf) for so in STOCKOUT_GRID for sf in SAFETY_GRID
                    if (so, sf) not in prior_keys]
    print(f"Will run {len(cells_to_run)} new cells, skipping {len(prior_keys)} existing.")

    backup = CFG_PATH.read_text(encoding="utf-8")
    results = list(prior)  # carry over prior cells
    started = datetime.now()
    try:
        for i, (so, sf) in enumerate(cells_to_run, 1):
            print(f"\n{'='*60}\n[Cell {i}/{len(cells_to_run)}] stockout={so}, sf={sf} (hetero+nv=3)\n{'='*60}")
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

    # Pretty print 3x4 grid
    print(f"\n{'='*60}\nFinal 3x4 grid (stockouts)\n{'='*60}")
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