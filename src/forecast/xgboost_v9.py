"""
XGBoost forecasting module v9 — extends v8 with feature engineering improvements.

Changes from v8:
- ATM-level target encoding (mean, std, p90 from train period)
- Cluster-tier one-hot from spatial.ATM_TIERS
- AGI regime-shift features (is_post_AGI, agi_effect_strength)
- Continuous day_of_month (complementing dom_* dummies)
- Consolidated is_payday_window flag
- Optuna search space extended in train_models.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.spatial import ATM_TIERS


# =============================================================================
# Section 1 — Excel I/O
# =============================================================================

def load_atm_excel(excel_path: str | Path, sheet_name: str = "ATM") -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["WITHDRWLS"] = df["WITHDRWLS"].astype("float64")
    return df.sort_values(["CASHP_ID", "DATE"]).reset_index(drop=True)


# =============================================================================
# Section 2 — Calendar features
# =============================================================================

SPECIAL_DAYS: list[str] = [
    "2006-01-01", "2006-01-10", "2006-01-11", "2006-01-12", "2006-01-13",
    "2006-04-23", "2006-05-19", "2006-08-30", "2006-10-23", "2006-10-24",
    "2006-10-25", "2006-10-29", "2006-12-31",
    "2007-01-01", "2007-01-02", "2007-01-03", "2007-04-23", "2007-05-19",
    "2007-08-30", "2007-10-12", "2007-10-13", "2007-10-14", "2007-10-29",
    "2007-12-20", "2007-12-21", "2007-12-22", "2007-12-23", "2007-12-31",
    "2008-01-01", "2008-04-23", "2008-05-19", "2008-08-30", "2008-09-30",
    "2008-10-01", "2008-10-02", "2008-10-29", "2008-12-08", "2008-12-09",
    "2008-12-10", "2008-12-11", "2008-12-31",
]

HALF_DAYS: list[str] = [
    "2006-01-09", "2006-10-22", "2006-12-30",
    "2007-10-11", "2007-12-19",
    "2008-09-29", "2008-12-07",
]

AGI_START_DATE = pd.Timestamp("2008-01-01")
AGI_RAMP_DAYS = 30


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar dummies + holiday/payday + AGI regime + continuous day_of_month.

    AGI trap: training window (<2007-10-01) has is_post_AGI=0 throughout, so the
    model cannot learn this signal from train data. Kept for test-set distribution
    shift visibility; reported as known limitation.
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    dom = pd.get_dummies(df["DATE"].dt.day,        prefix="dom",   drop_first=True)
    mon = pd.get_dummies(df["DATE"].dt.month,      prefix="month", drop_first=True)
    dow = pd.get_dummies(df["DATE"].dt.dayofweek,  prefix="dow",   drop_first=True)
    df = pd.concat([df, dom, mon, dow], axis=1)

    df["day_of_month"] = df["DATE"].dt.day.astype(int)

    official = pd.to_datetime(SPECIAL_DAYS)
    half = pd.to_datetime(HALF_DAYS)
    df["day_official_holiday"] = df["DATE"].isin(official).astype(int)
    df["day_half_day"]         = df["DATE"].isin(half).astype(int)
    df["day_normal_day"]       = ((df["day_official_holiday"] == 0) &
                                  (df["day_half_day"] == 0)).astype(int)

    df["is_payday_15"] = df["DATE"].dt.day.isin([15, 16, 17, 18]).astype(int)
    df["is_payday_1"]  = df["DATE"].dt.day.isin([1, 2, 3, 4]).astype(int)
    df["is_month_end"] = df["DATE"].dt.day.isin([28, 29, 30, 31]).astype(int)
    df["is_payday_window"] = ((df["is_payday_15"] == 1) |
                              (df["is_payday_1"] == 1)).astype(int)

    days_since_agi = (df["DATE"] - AGI_START_DATE).dt.days
    df["is_post_AGI"] = (days_since_agi >= 0).astype(int)
    df["agi_effect_strength"] = np.minimum(
        np.maximum(days_since_agi, 0) / AGI_RAMP_DAYS, 1.0
    )

    holidays_arr = np.array(sorted(pd.to_datetime(SPECIAL_DAYS + HALF_DAYS).values))

    def _days_to_next(d: pd.Timestamp) -> int:
        future = holidays_arr[holidays_arr >= d.to_datetime64()]
        return min(14, int((future[0] - d.to_datetime64()) / np.timedelta64(1, "D"))) if len(future) else 14

    def _days_since_prev(d: pd.Timestamp) -> int:
        past = holidays_arr[holidays_arr <= d.to_datetime64()]
        return min(14, int((d.to_datetime64() - past[-1]) / np.timedelta64(1, "D"))) if len(past) else 14

    df["days_to_next_holiday"]    = df["DATE"].apply(_days_to_next).astype(int)
    df["days_since_prev_holiday"] = df["DATE"].apply(_days_since_prev).astype(int)

    return df


# =============================================================================
# Section 2A — Target encoding
# =============================================================================

def compute_atm_target_stats(
    train_df: pd.DataFrame,
    atm_tiers: dict[str, str] = ATM_TIERS,
) -> dict[str, dict]:
    """Per-ATM, per-tier and global statistics from training data.

    Caller must pass `filter_zeros`-cleaned train period only; passing
    raw or full-history data leaks future information into encoding.

    Returns nested dict with three resolution levels (atm, tier, global)
    so `apply_target_encoding` can fall back hierarchically for ATMs
    unseen in train — Micci-Barreca (2001) rationale, here a hard
    fallback rather than weighted blend.
    """
    if train_df.empty:
        raise ValueError("train_df is empty; cannot compute encoding stats")

    def _stats(series: pd.Series) -> dict[str, float]:
        return {
            "mean": float(series.mean()),
            "std":  float(series.std()) if len(series) > 1 else 0.0,
            "p90":  float(series.quantile(0.90)),
        }

    per_atm = {
        atm_id: _stats(s)
        for atm_id, s in train_df.groupby("CASHP_ID")["WITHDRWLS"]
    }

    tier_col = train_df["CASHP_ID"].map(atm_tiers)
    train_tiered = train_df.assign(_tier=tier_col).dropna(subset=["_tier"])
    per_tier = {
        tier: _stats(g["WITHDRWLS"])
        for tier, g in train_tiered.groupby("_tier")
    }

    global_stats = _stats(train_df["WITHDRWLS"])

    return {"per_atm": per_atm, "per_tier": per_tier, "global": global_stats}


def apply_target_encoding(
    df: pd.DataFrame,
    atm_stats: dict,
    atm_tiers: dict[str, str] = ATM_TIERS,
) -> pd.DataFrame:
    """Append target encoding columns via three-level fallback.

    Resolution order per ATM: per-ATM stats → per-tier stats → global.
    Mapping is built once over unique IDs then vectorised via Series.map
    to avoid row-wise apply on full panel.
    """
    df = df.copy()
    per_atm  = atm_stats["per_atm"]
    per_tier = atm_stats["per_tier"]
    global_  = atm_stats["global"]

    resolved: dict[str, dict[str, float]] = {}
    for atm_id in df["CASHP_ID"].unique():
        if atm_id in per_atm:
            resolved[atm_id] = per_atm[atm_id]
            continue
        tier = atm_tiers.get(atm_id)
        if tier is not None and tier in per_tier:
            resolved[atm_id] = per_tier[tier]
        else:
            resolved[atm_id] = global_

    for stat_key, col in (
        ("mean", "target_enc_mean"),
        ("std",  "target_enc_std"),
        ("p90",  "target_enc_p90"),
    ):
        df[col] = df["CASHP_ID"].map(
            {atm: resolved[atm][stat_key] for atm in resolved}
        )

    return df


# =============================================================================
# Section 2B — Cluster-tier features
# =============================================================================

def add_cluster_tier_features(
    df: pd.DataFrame,
    atm_tiers: dict[str, str] = ATM_TIERS,
) -> pd.DataFrame:
    """One-hot encode cluster tier (low/mid/high) per ATM.

    Tiers sourced from src.data.spatial.ATM_TIERS — same authoritative
    mapping the IRP uses for capacity assignment, so forecast and
    optimisation layers stay aligned.
    """
    df = df.copy()
    tier = df["CASHP_ID"].map(atm_tiers)
    df["cluster_tier_low"]  = (tier == "low").astype(int)
    df["cluster_tier_mid"]  = (tier == "mid").astype(int)
    df["cluster_tier_high"] = (tier == "high").astype(int)
    return df


# =============================================================================
# Section 3 — Lag and rolling features
# =============================================================================

def build_lag_rolling_features(df: pd.DataFrame, seq_len: int = 30) -> pd.DataFrame:
    base_lags  = (1, 2, 3, 7, 14, 28, 30)
    base_rolls = (7, 14, 28)
    lags  = [l for l in base_lags  if l <= seq_len]
    rolls = [r for r in base_rolls if r <= seq_len]

    parts = []
    for _, group in df.groupby("CASHP_ID"):
        g = group.sort_values("DATE").copy()
        for lag in lags:
            g[f"lag_{lag}"] = g["WITHDRWLS"].shift(lag)
        for r in rolls:
            g[f"roll_mean_{r}"] = g["WITHDRWLS"].shift(1).rolling(r).mean()
            g[f"roll_std_{r}"]  = g["WITHDRWLS"].shift(1).rolling(r).std()
        parts.append(g)

    return pd.concat(parts).dropna().reset_index(drop=True)


# =============================================================================
# Section 4 — Zero handling
# =============================================================================

def _drop_long_zero_runs(group: pd.DataFrame, threshold: int) -> pd.DataFrame:
    mask = np.ones(len(group), dtype=bool)
    zero_count = 0
    for i, val in enumerate(group["WITHDRWLS"].values):
        if val == 0:
            zero_count += 1
        else:
            if zero_count >= threshold:
                mask[i - zero_count:i] = False
            zero_count = 0
    if zero_count >= threshold:
        mask[len(group) - zero_count:] = False
    return group[mask]


def filter_zeros(
    df: pd.DataFrame,
    consecutive_zero_threshold: int = 14,
    drop_isolated_zeros: bool = True,
) -> pd.DataFrame:
    parts = [_drop_long_zero_runs(g, consecutive_zero_threshold)
             for _, g in df.groupby("CASHP_ID")]
    out = pd.concat(parts, ignore_index=True)
    if drop_isolated_zeros:
        out = out[out["WITHDRWLS"] > 0]
    return out.reset_index(drop=True)


# =============================================================================
# Section 5 — Temporal split
# =============================================================================

def split_train_val_test(
    df: pd.DataFrame,
    val_start: str | pd.Timestamp = "2007-10-01",
    test_start: str | pd.Timestamp = "2007-12-08",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_ts = pd.Timestamp(val_start)
    test_ts = pd.Timestamp(test_start)
    train = df[df["DATE"] < val_ts].reset_index(drop=True)
    val   = df[(df["DATE"] >= val_ts) & (df["DATE"] < test_ts)].reset_index(drop=True)
    test  = df[df["DATE"] >= test_ts].reset_index(drop=True)
    return train, val, test


# =============================================================================
# Section 6 — Sample weighting
# =============================================================================

def compute_sample_weights(
    df: pd.DataFrame,
    weight_clip: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    atm_means = df.groupby("CASHP_ID")["WITHDRWLS"].mean()
    global_mean = df["WITHDRWLS"].mean()
    weights_map = np.clip(atm_means / global_mean, *weight_clip)
    return df["CASHP_ID"].map(weights_map).values


# =============================================================================
# Section 7 — Quantile model fit
# =============================================================================

def fit_quantile_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    params: dict,
    alpha: float,
    sample_weights: np.ndarray | None = None,
    seed: int = 42,
) -> xgb.XGBRegressor:
    full_params = {
        **params,
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "random_state": seed,
        "tree_method": "hist",
    }
    model = xgb.XGBRegressor(**full_params)
    model.fit(X, y, sample_weight=sample_weights, verbose=False)
    return model


# =============================================================================
# Section 8 — Conformal calibration
# =============================================================================

def compute_conformal_shift(
    model: xgb.XGBRegressor,
    X_cal: pd.DataFrame,
    y_cal_actual: np.ndarray,
    target_quantile: float,
) -> float:
    raw_pred = np.expm1(model.predict(X_cal))
    residuals = y_cal_actual - raw_pred
    return float(np.quantile(residuals, target_quantile))


def apply_conformal(predictions: np.ndarray, shift: float) -> np.ndarray:
    return np.maximum(predictions + shift, 0.0)


# =============================================================================
# Section 9 — Bias correction
# =============================================================================

def compute_static_bias(
    model: xgb.XGBRegressor,
    X_cal: pd.DataFrame,
    y_cal_actual: np.ndarray,
    cashp_ids: pd.Series,
) -> dict[str, float]:
    pred = np.expm1(model.predict(X_cal))
    df = pd.DataFrame({
        "CASHP_ID": cashp_ids.values,
        "resid": y_cal_actual - pred,
    })
    return df.groupby("CASHP_ID")["resid"].mean().to_dict()


def apply_rolling_bias_correction(
    predictions_df: pd.DataFrame,
    static_bias: dict[str, float],
    window: int = 14,
    shrinkage: float = 0.7,
    cold_start_min_obs: int = 3,
) -> pd.DataFrame:
    out_rows = []
    for atm_id, atm_df in predictions_df.groupby("CASHP_ID"):
        atm_df = atm_df.sort_values("DATE")
        actuals: list[float] = []
        preds: list[float] = []
        static = static_bias.get(atm_id, 0.0)
        for _, row in atm_df.iterrows():
            d_mean_raw   = float(row["d_mean_raw"])
            d_safety_raw = float(row["d_safety_raw"])
            actual       = float(row["actual"])

            if len(actuals) < cold_start_min_obs:
                correction = static
            else:
                w_act = np.array(actuals[-window:])
                w_pre = np.array(preds[-window:])
                rolling_resid = float((w_act - w_pre).mean())
                correction = shrinkage * rolling_resid + (1.0 - shrinkage) * static

            d_mean   = max(d_mean_raw + correction, 0.0)
            d_safety = max(d_safety_raw, d_mean)

            row_out = row.to_dict()
            row_out["correction"] = correction
            row_out["d_mean"]     = d_mean
            row_out["d_safety"]   = d_safety
            out_rows.append(row_out)

            actuals.append(actual)
            preds.append(d_mean_raw)
    return pd.DataFrame(out_rows)


# =============================================================================
# Section 10 — Calibrated batch prediction
# =============================================================================

def predict_with_calibration(
    point_model: xgb.XGBRegressor,
    safety_model: xgb.XGBRegressor,
    X: pd.DataFrame,
    shift_safety: float,
    cashp_ids: pd.Series,
    dates: pd.Series,
    actuals: np.ndarray | None = None,
) -> pd.DataFrame:
    d_mean_raw   = np.expm1(point_model.predict(X))
    d_safety_raw = apply_conformal(np.expm1(safety_model.predict(X)), shift_safety)
    out = pd.DataFrame({
        "CASHP_ID":     cashp_ids.values,
        "DATE":         dates.values,
        "d_mean_raw":   d_mean_raw,
        "d_safety_raw": d_safety_raw,
    })
    if actuals is not None:
        out["actual"] = actuals
    return out