"""
Phase 4.C — 2D sensitivity sweep over (stockout_penalty, safety_floor_pen).

Sweep runs at simulation horizon = 30 days (standard for sensitivity analyses);
the production-scale 73-day run is documented separately as the final headline
result.

The script back-up restores configs/optimize.yaml on exit (Ctrl+C safe via finally).
Results saved incrementally to docs/phase_4c_raw.json so crashes don't lose progress.

Run from project root:
    python scripts/sweep_4c.py
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CFG_PATH = Path("configs/optimize.yaml")
OUT_JSON = Path("docs/phase_4c_raw.json")

STOCKOUT_GRID = [3000, 5000, 8000]
SAFETY_GRID   = [0.01, 0.1, 1.0]
SWEEP_DAYS    = 30


def patch_yaml(text: str, **kwargs) -> str:
    """Regex-replace specific top-level scalar keys; preserves comments and layout."""
    out = text
    for key, val in kwargs.items():
        pat = rf"^(\s*{re.escape(key)}:\s+)\S+(.*)$"
        out = re.sub(pat, rf"\g<1>{val}\g<2>", out, count=1, flags=re.MULTILINE)
    return out


def parse_summary(stdout: str) -> dict:
    def f(pat, t=float):
        m = re.search(pat, stdout)
        if m is None:
            return None
        return t(m.group(1).replace(",", ""))
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
    total = len(STOCKOUT_GRID) * len(SAFETY_GRID)
    started = datetime.now()
    try:
        cell = 0
        for so in STOCKOUT_GRID:
            for sf in SAFETY_GRID:
                cell += 1
                print(f"\n{'='*60}\n[Cell {cell}/{total}] stockout={so}, safety_floor={sf}\n{'='*60}")
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
                results.append({
                    "stockout_penalty": so,
                    "safety_floor_pen": sf,
                    **parsed,
                })
                OUT_JSON.write_text(json.dumps(results, indent=2))
    finally:
        CFG_PATH.write_text(backup, encoding="utf-8")
        print(f"\nRestored {CFG_PATH}")

    elapsed = datetime.now() - started
    print(f"\nDone. {len(results)}/{total} cells in {elapsed}")
    print(f"Results: {OUT_JSON}")


if __name__ == "__main__":
    main()
