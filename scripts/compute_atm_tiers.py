"""
Compute per-ATM capacity tier (low/mid/high) via tertile clustering on
train-period active-day mean withdrawal. Output is intended to be pasted
into src/data/spatial.py as a static ATM_TIERS dict.

Train-period only to avoid data leakage from val/test into clustering.
Active-day mean (WITHDRWLS > 0) used to neutralize censored / inactive days.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXCEL_PATH = "data/2024-12-09_ATM_Branch_Data.xlsx"
TRAIN_END = pd.Timestamp("2007-10-01")  # exclusive — val starts here

TIER_CAPACITY = {"low": 250_000, "mid": 400_000, "high": 500_000}


def main() -> None:
    df = pd.read_excel(EXCEL_PATH, sheet_name="ATM")
    df["DATE"] = pd.to_datetime(df["DATE"])

    train = df[df["DATE"] < TRAIN_END]
    active = train[train["WITHDRWLS"] > 0]

    per_atm = active.groupby("CASHP_ID").agg(
        n_active=("WITHDRWLS", "size"),
        mean=("WITHDRWLS", "mean"),
        max=("WITHDRWLS", "max"),
    ).sort_values("mean")

    q33 = per_atm["mean"].quantile(1 / 3)
    q67 = per_atm["mean"].quantile(2 / 3)

    def assign(m: float) -> str:
        if m <= q33: return "low"
        if m <= q67: return "mid"
        return "high"

    per_atm["tier"] = per_atm["mean"].map(assign)

    print(f"Train period: < {TRAIN_END.date()}, {len(per_atm)} ATMs")
    print(f"Tertile thresholds: q33 = {q33/1000:.1f}K, q67 = {q67/1000:.1f}K\n")

    print(f"{'CASHP_ID':<12} {'n_active':>10} {'mean (K)':>10} {'max (K)':>10} {'tier':>6}")
    print("-" * 54)
    for atm_id, row in per_atm.iterrows():
        print(f"{atm_id:<12} {row['n_active']:>10.0f} "
              f"{row['mean']/1000:>10.1f} {row['max']/1000:>10.1f} {row['tier']:>6}")

    print("\nFeasibility (max-day vs cap):")
    for tier, cap in TIER_CAPACITY.items():
        sub = per_atm[per_atm["tier"] == tier]
        max_day = sub["max"].max()
        margin = (cap - max_day) / cap * 100
        print(f"  {tier:>4}: n={len(sub):>2}, max-day = {max_day/1000:>6.1f}K, "
              f"cap = {cap/1000:.0f}K, headroom = {margin:>4.0f}%")

    print("\n# ============== PASTE INTO src/data/spatial.py ==============")
    print("ATM_TIERS: dict[str, str] = {")
    for atm_id, row in per_atm.iterrows():
        print(f"    {atm_id!r:<14}: {row['tier']!r},")
    print("}")
    print("\nTIER_CAPACITY: dict[str, int] = {")
    for tier, cap in TIER_CAPACITY.items():
        print(f"    {tier!r:<8}: {cap:>7_d},")
    print("}")


if __name__ == "__main__":
    main()