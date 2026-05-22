"""
Generate the point-forecast prediction CSV for the B2 (point + IRP) baseline.

B2 removes the safety quantile from the proposed system. This script copies
the quantile prediction CSV and overwrites the d_safety column with d_mean,
producing a forecast that emits a single point estimate per (ATM, day) cell.

Feeding d_safety = d_mean into the IRP collapses the safety-floor constraint
(Inv + safety_slack >= d_safe - d_phys becomes >= 0) and sets the
end-of-horizon target to the point forecast -- the faithful representation of
a point-forecast pipeline (B2 in the baseline ablation).

Output: predictions/test_predictions_point.csv

Usage (from project root):
    python scripts/make_point_forecast_csv.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_COLS = ("DATE", "CASHP_ID", "WITHDRWLS", "d_mean", "d_safety")


def main() -> None:
    with open(ROOT / "configs" / "optimize.yaml", encoding="utf-8") as f:
        ocfg = yaml.safe_load(f)

    src = ROOT / ocfg["data"]["real_demand_csv_path"]
    dst = ROOT / "predictions" / "test_predictions_point.csv"

    if not src.exists():
        raise FileNotFoundError(f"Quantile prediction CSV not found: {src}")

    df = pd.read_csv(src)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Prediction CSV missing columns {missing}; got {list(df.columns)}"
        )

    # The faithful point forecast: the safety quantile is replaced by the
    # point estimate. real_demand applies max(d_safety, d_mean) downstream,
    # so equality is preserved exactly.
    df["d_safety"] = df["d_mean"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"Wrote {dst}  ({len(df)} rows; d_safety := d_mean)")


if __name__ == "__main__":
    main()