"""
Phase 4.C-hetero — 3-cell mini-sweep to confirm (5000, 0.01) Pareto-optimality
under heterogeneous capacity + nv=3 (Phase 4.H validated configuration).

Tests three cells around the uniform-validated optimum to verify the
cost-parameter decision generalises under hetero mode.

Run from project root after config is set to:
    use_heterogeneous_capacity: true
    num_vehicles: 3

    python scripts/sweep_4c_hetero.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4c_hetero_raw.json")

# 3-cell mini-grid around uniform optimum (5000, 0.01)
CELLS = [
    (5000, 0.01),   # baseline (uniform optimum)
    (5000, 0.1),    # higher safety floor
    (3000, 0.01),   # lower stockout penalty
]
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
        for i, (so, sf) in enumerate(CELLS, 1):
            print(f"\n{'='*60}\n[Cell {i}/{len(CELLS)}] stockout={so}, safety_floor={sf} (hetero+nv=3)\n{'='*60}")
            patched = patch_yaml(
                backup,
                days=SWEEP_DAYS,
                stockout_penalty=so,
                safety_floor_pen=sf,
            )
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
                print(f"  Cost={parsed['total_cost']:,.0f} TL  Stockouts={parsed['stockouts']}")
            results.append({"stockout_penalty": so, "safety_floor_pen": sf, **parsed})
            OUT_JSON.write_text(json.dumps(results, indent=2))
    finally:
        CFG_PATH.write_text(backup, encoding="utf-8")
        print(f"\nRestored {CFG_PATH}")
    print(f"\nDone. {len(CELLS)} cells in {datetime.now() - started}")


if __name__ == "__main__":
    main()