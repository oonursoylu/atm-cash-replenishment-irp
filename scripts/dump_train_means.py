"""
One-off helper: dump per-ATM train-period mean daily withdrawal for B0.

B0 (static s, S) is forecast-free. Its (s, S) levels are anchored on a naive
historical mean rather than the XGBoost forecast. "Historical" here means
every observation strictly before the test window (DATE < split.test_start),
i.e. the same data the forecaster trained on, summarised by a plain mean over
positive-withdrawal days (WITHDRWLS > 0) so that ATM-inactive intervals do not
drag the mean down.

Output: data/train_atm_means.csv  (columns: CASHP_ID, train_mean_withdrawal)

Usage (from project root):
    python scripts/dump_train_means.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.spatial import load_hardcoded_spatial


def _load_raw(excel_path: Path, sheet: str) -> pd.DataFrame:
    """Load the raw ATM withdrawal table. Prefer the forecast pipeline's own
    loader so the column schema matches; fall back to a plain read."""
    try:
        from src.forecast.xgboost_v8 import load_atm_excel
        return load_atm_excel(str(excel_path), sheet)
    except Exception as exc:  # noqa: BLE001
        print(f"[INFO] load_atm_excel unavailable ({exc}); using pd.read_excel.")
        return pd.read_excel(excel_path, sheet_name=sheet)


def main() -> None:
    with open(ROOT / "configs" / "forecast.yaml", encoding="utf-8") as f:
        fcfg = yaml.safe_load(f)

    excel_path = ROOT / fcfg["data"]["excel_path"]
    sheet = fcfg["data"]["sheet_name"]
    test_start = pd.Timestamp(fcfg["split"]["test_start"])

    df = _load_raw(excel_path, sheet)
    for col in ("DATE", "CASHP_ID", "WITHDRWLS"):
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' not found in raw data; "
                f"columns present: {list(df.columns)}"
            )
    df["DATE"] = pd.to_datetime(df["DATE"])

    atms = sorted(load_hardcoded_spatial()["atm_location"].keys())
    pre_test = df[(df["DATE"] < test_start) & (df["CASHP_ID"].isin(atms))]
    active = pre_test[pre_test["WITHDRWLS"] > 0]

    means = active.groupby("CASHP_ID")["WITHDRWLS"].mean().reindex(atms)

    missing = means[means.isna()].index.tolist()
    if missing:
        fill = float(active["WITHDRWLS"].mean())
        print(f"[WARN] no pre-test active rows for {missing}; "
              f"filling with global mean {fill:,.0f}")
        means = means.fillna(fill)

    out = means.rename("train_mean_withdrawal").reset_index()
    out_path = ROOT / "data" / "train_atm_means.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path}  ({len(out)} ATMs)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()