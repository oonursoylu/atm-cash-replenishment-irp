# ATM Demand Forecasting using XGBoost — Version 8.
# Standalone v8 forecast baseline (Optuna search + three-stage calibration +
# adaptive rolling bias correction). The production version of this code lives
# in src/forecast/xgboost_v8.py; this file is preserved as the reference run.
#
# Adaptive bias correction (v8 addition over v7):
#   At each test day t, the per-ATM bias is computed from residuals over the
#   rolling window [t-14, t-1], shrunk toward the cross-ATM rolling mean with
#   coefficient 0.7 (James-Stein style). For t < 14, the static training-period
#   bias is used as fallback. This lets the forecast self-adapt to distribution
#   shift (December holidays, year-end, the January 2008 AGI/wage adjustment)
#   with at most a 14-day lag, without violating the no-look-ahead constraint.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Paths are resolved relative to the project root so the script runs unchanged
# on any machine. The production pipeline (src/forecast/xgboost_v8.py) reads
# these locations from configs/forecast.yaml; the legacy script hardcodes them.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = str(PROJECT_ROOT / "data" / "2024-12-09_ATM_Branch_Data.xlsx")
OUTPUT_PATH = str(PROJECT_ROOT / "predictions" / "xgboost_forecast_results.csv")
SEED = 42
ZERO_RUN_THRESHOLD = 14
SEQ_LEN = 30
N_TRIALS = 50

# 3-way temporal split
VAL_START = "2007-10-01"
TEST_START = "2007-12-08"

TAU_MEAN = 0.55
TAU_SAFETY = 0.90
SIM_END = "2008-02-13"

# Adaptive correction parameters (NEW in v8)
ROLLING_WINDOW_DAYS = 14   # rolling residual window for bias correction
SHRINKAGE_LAMBDA = 0.7     # weight on per-ATM rolling bias; (1-lambda) on global

SPECIAL_DAYS = [
    '2006-01-01', '2006-01-10', '2006-01-11', '2006-01-12', '2006-01-13', '2006-04-23', '2006-05-19', '2006-08-30', '2006-10-23', '2006-10-24', '2006-10-25', '2006-10-29', '2006-12-31',
    '2007-01-01', '2007-01-02', '2007-01-03', '2007-04-23', '2007-05-19', '2007-08-30', '2007-10-12', '2007-10-13', '2007-10-14', '2007-10-29', '2007-12-20', '2007-12-21', '2007-12-22', '2007-12-23', '2007-12-31',
    '2008-01-01', '2008-04-23', '2008-05-19', '2008-08-30', '2008-09-30', '2008-10-01', '2008-10-02', '2008-10-29', '2008-12-08', '2008-12-09', '2008-12-10', '2008-12-11', '2008-12-31'
]

HALF_DAYS = [
    '2006-01-09', '2006-10-22', '2006-12-30', '2007-10-11', '2007-12-19', '2008-09-29', '2008-12-07'
]

np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------
# SECTION 1: FEATURES (unchanged from v7)
# ---------------------------------------------------------------------
def create_calendar_and_holiday_features(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["day_of_month"] = df["DATE"].dt.day
    df["month"] = df["DATE"].dt.month
    df["day_of_week"] = df["DATE"].dt.dayofweek

    dom_dummies = pd.get_dummies(df["day_of_month"], prefix="dom", drop_first=True)
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow", drop_first=True)

    df = df.drop(columns=["day_of_month", "month", "day_of_week"])
    df = pd.concat([df, dom_dummies, month_dummies, dow_dummies], axis=1)

    official = pd.to_datetime(SPECIAL_DAYS)
    half = pd.to_datetime(HALF_DAYS)

    df["day_official_holiday"] = df["DATE"].isin(official).astype(int)
    df["day_half_day"] = df["DATE"].isin(half).astype(int)
    df["day_normal_day"] = ((df["day_official_holiday"] == 0) &
                            (df["day_half_day"] == 0)).astype(int)

    df["is_payday_15"] = df["DATE"].dt.day.isin([15, 16, 17, 18]).astype(int)
    df["is_payday_1"] = df["DATE"].dt.day.isin([1, 2, 3, 4]).astype(int)
    df["is_month_end"] = df["DATE"].dt.day.isin([28, 29, 30, 31]).astype(int)

    holidays_arr = np.array(sorted(pd.to_datetime(SPECIAL_DAYS + HALF_DAYS).values))

    def days_to_next(d):
        future = holidays_arr[holidays_arr >= d.to_datetime64()]
        if len(future) == 0:
            return 14
        return min(14, int((future[0] - d.to_datetime64()) / np.timedelta64(1, 'D')))

    def days_since_prev(d):
        past = holidays_arr[holidays_arr <= d.to_datetime64()]
        if len(past) == 0:
            return 14
        return min(14, int((d.to_datetime64() - past[-1]) / np.timedelta64(1, 'D')))

    df["days_to_next_holiday"] = df["DATE"].apply(days_to_next).astype(int)
    df["days_since_prev_holiday"] = df["DATE"].apply(days_since_prev).astype(int)

    return df

def create_lag_and_rolling_features(df, window):
    base_lags = [1, 2, 3, 7, 14, 28, 30]
    base_rolls = [7, 14, 28]
    lags = [lag for lag in base_lags if lag <= window]
    rolls = [r for r in base_rolls if r <= window]

    groups = []
    for atm_id, group in df.groupby("CASHP_ID"):
        group = group.sort_values("DATE").copy()
        for lag in lags:
            group[f"lag_{lag}"] = group["WITHDRWLS"].shift(lag)
        for r in rolls:
            group[f"roll_mean_{r}"] = group["WITHDRWLS"].shift(1).rolling(r).mean()
            group[f"roll_std_{r}"] = group["WITHDRWLS"].shift(1).rolling(r).std()
        groups.append(group)

    df_feat = pd.concat(groups).reset_index(drop=True)
    return df_feat.dropna().reset_index(drop=True)

# ---------------------------------------------------------------------
# SECTION 2: ZERO HANDLING (unchanged from v7)
# ---------------------------------------------------------------------
def drop_long_zero_runs(group):
    mask = np.ones(len(group), dtype=bool)
    zero_count = 0
    for i, val in enumerate(group["WITHDRWLS"].values):
        if val == 0:
            zero_count += 1
        else:
            if zero_count >= ZERO_RUN_THRESHOLD:
                mask[i - zero_count:i] = False
            zero_count = 0
    if zero_count >= ZERO_RUN_THRESHOLD:
        mask[len(group) - zero_count:] = False
    return group[mask]

def apply_zero_handling(df):
    parts = []
    for atm_id, group in df.groupby("CASHP_ID"):
        cleaned = drop_long_zero_runs(group)
        parts.append(cleaned)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["WITHDRWLS"] > 0]
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------
# SECTION 3: METRICS
# ---------------------------------------------------------------------
def calculate_smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) /
                   (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))

# ---------------------------------------------------------------------
# SECTION 4: ADAPTIVE BIAS CORRECTION (NEW in v8)
# ---------------------------------------------------------------------
def compute_adaptive_corrections(test_df, static_per_atm_bias, static_global_bias,
                                  window_days=14, shrinkage=0.7):
    """For each (ATM, date) in test_df, compute the bias correction from
    the previous `window_days` of realised residuals. Cold-start days
    (insufficient history) fall back to the static per-ATM correction.

    Inputs:
        test_df              : DataFrame sorted by DATE with columns
                               [DATE, CASHP_ID, WITHDRWLS, d_mean_raw]
        static_per_atm_bias  : Series indexed by CASHP_ID (training-period bias)
        static_global_bias   : float (training-period global bias)
        window_days          : rolling lookback in days
        shrinkage            : weight on per-ATM rolling bias (0..1)

    Returns:
        ndarray of correction values aligned with test_df rows
    """
    test_df = test_df.sort_values(["DATE", "CASHP_ID"]).reset_index(drop=True)

    # Pre-compute per-row residual at d_mean_raw vs WITHDRWLS
    # (residual = actual - raw_pred, what we WOULD ADD to make it correct)
    test_df["_resid_raw"] = test_df["WITHDRWLS"] - test_df["d_mean_raw"]

    n = len(test_df)
    corrections = np.zeros(n)
    cold_start_count = 0
    rolling_used_count = 0

    # Build a lookup: for each (atm, date), find residuals in [date-window, date-1]
    # We index by date for fast filtering.
    test_df["_dt"] = pd.to_datetime(test_df["DATE"])

    # For efficient retrieval, group residual records by ATM
    atm_history = {atm: g[["_dt", "_resid_raw"]].sort_values("_dt").reset_index(drop=True)
                   for atm, g in test_df.groupby("CASHP_ID")}

    # Global rolling history (across all ATMs)
    global_history = test_df[["_dt", "_resid_raw"]].sort_values("_dt").reset_index(drop=True)

    for i in range(n):
        atm = test_df.at[i, "CASHP_ID"]
        d = test_df.at[i, "_dt"]
        window_start = d - pd.Timedelta(days=window_days)

        # Per-ATM rolling residuals strictly BEFORE today
        ah = atm_history[atm]
        ah_window = ah[(ah["_dt"] >= window_start) & (ah["_dt"] < d)]
        n_atm = len(ah_window)

        if n_atm < 3:
            # Cold start for this ATM: use static training-period correction
            corrections[i] = static_per_atm_bias.get(atm, static_global_bias)
            cold_start_count += 1
            continue

        atm_rolling_bias = ah_window["_resid_raw"].mean()

        # Global rolling residuals (across all ATMs) for shrinkage target
        gh_window = global_history[(global_history["_dt"] >= window_start) &
                                    (global_history["_dt"] < d)]
        global_rolling_bias = gh_window["_resid_raw"].mean() if len(gh_window) > 0 else static_global_bias

        # Shrinkage: lambda * per-ATM + (1-lambda) * global
        corrections[i] = shrinkage * atm_rolling_bias + (1 - shrinkage) * global_rolling_bias
        rolling_used_count += 1

    return corrections, cold_start_count, rolling_used_count

# ---------------------------------------------------------------------
# SECTION 5: PIPELINE
# ---------------------------------------------------------------------
def run_xgboost_pipeline():
    print("=" * 75)
    print("XGBOOST V8: v7 + ADAPTIVE ROLLING BIAS CORRECTION")
    print(f"  Rolling window={ROLLING_WINDOW_DAYS}d  shrinkage λ={SHRINKAGE_LAMBDA}")
    print("=" * 75)

    print(f"[1/8] Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("ERROR: File not found.")
        return

    atm_df = pd.read_excel(DATA_PATH, sheet_name="ATM")

    print("[2/8] Building features (calendar, distance-to-holiday, paydays, lags)...")
    df_feat = create_calendar_and_holiday_features(atm_df)
    df_feat = create_lag_and_rolling_features(df_feat, SEQ_LEN)

    print("[3/8] Zero handling (v3 mode)...")
    n_before = len(df_feat)
    df_clean = apply_zero_handling(df_feat)
    n_after = len(df_clean)
    print(f"      -> rows: {n_before:,} -> {n_after:,}")

    print(f"[4/8] 3-way temporal split:")
    val_start = pd.Timestamp(VAL_START)
    test_start = pd.Timestamp(TEST_START)
    df_clean = df_clean.sort_values(["CASHP_ID", "DATE"]).reset_index(drop=True)

    train_df = df_clean[df_clean["DATE"] < val_start].reset_index(drop=True)
    val_df = df_clean[(df_clean["DATE"] >= val_start) &
                      (df_clean["DATE"] < test_start)].reset_index(drop=True)
    test_df = df_clean[df_clean["DATE"] >= test_start].reset_index(drop=True)

    print(f"      -> Train:   {len(train_df):,} rows  "
          f"({train_df['DATE'].min().date()} -> {train_df['DATE'].max().date()})")
    print(f"      -> Val:     {len(val_df):,} rows  "
          f"({val_df['DATE'].min().date()} -> {val_df['DATE'].max().date()})")
    print(f"      -> Test:    {len(test_df):,} rows  "
          f"({test_df['DATE'].min().date()} -> {test_df['DATE'].max().date()})")

    exclude_cols = ["CASHP_ID", "DATE", "WITHDRWLS", "cluster"]
    features = [c for c in df_clean.columns if c not in exclude_cols]

    y_train_log = np.log1p(train_df["WITHDRWLS"])
    X_train = train_df[features]
    X_val = val_df[features]
    X_test = test_df[features]

    atm_means = train_df.groupby("CASHP_ID")["WITHDRWLS"].mean()
    global_mean = train_df["WITHDRWLS"].mean()
    weights_map = np.clip(atm_means / global_mean, 0.5, 2.0)
    sample_weights = train_df["CASHP_ID"].map(weights_map).values
    print(f"      -> sample weight range (clipped): "
          f"[{sample_weights.min():.2f}, {sample_weights.max():.2f}]")

    # --- STUDY 1: ASYMMETRIC POINT (pinball@TAU_MEAN) ---
    print(f"\n[5/8] Optuna tuning ({N_TRIALS} trials each)")
    print(f"  Study 1/2: ASYMMETRIC point (pinball@{TAU_MEAN} on Val)...")

    def objective_mean(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "reg:quantileerror",
            "quantile_alpha": TAU_MEAN,
            "random_state": SEED,
            "tree_method": "hist",
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train_log, sample_weight=sample_weights, verbose=False)
        y_pred_raw = np.expm1(model.predict(X_val))
        return pinball_loss(val_df["WITHDRWLS"].values, y_pred_raw, alpha=TAU_MEAN)

    study_mean = optuna.create_study(direction="minimize")
    study_mean.optimize(objective_mean, n_trials=N_TRIALS)
    best_params_mean = study_mean.best_trial.params
    print(f"            -> Best pinball@{TAU_MEAN} on Val: {study_mean.best_value:.2f}")

    # --- STUDY 2: SAFETY QUANTILE ---
    print(f"  Study 2/2: SAFETY (pinball@{TAU_SAFETY} on Val)...")

    def objective_q90(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "reg:quantileerror",
            "quantile_alpha": TAU_SAFETY,
            "random_state": SEED,
            "tree_method": "hist",
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train_log, sample_weight=sample_weights, verbose=False)
        y_pred_raw = np.expm1(model.predict(X_val))
        return pinball_loss(val_df["WITHDRWLS"].values, y_pred_raw, alpha=TAU_SAFETY)

    study_q90 = optuna.create_study(direction="minimize")
    study_q90.optimize(objective_q90, n_trials=N_TRIALS)
    best_params_q90 = study_q90.best_trial.params
    print(f"            -> Best pinball@{TAU_SAFETY} on Val: {study_q90.best_value:.2f}")

    # --- FINAL TRAINING (Train+Val) ---
    print(f"\n[6/8] Final training on Train+Val...")
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    X_full = full_train_df[features]
    y_full_log = np.log1p(full_train_df["WITHDRWLS"])
    full_atm_means = full_train_df.groupby("CASHP_ID")["WITHDRWLS"].mean()
    full_global_mean = full_train_df["WITHDRWLS"].mean()
    full_weights_map = np.clip(full_atm_means / full_global_mean, 0.5, 2.0)
    full_sample_weights = full_train_df["CASHP_ID"].map(full_weights_map).values

    best_params_mean["objective"] = "reg:quantileerror"
    best_params_mean["quantile_alpha"] = TAU_MEAN
    best_params_mean["random_state"] = SEED
    best_params_mean["tree_method"] = "hist"
    model_mean = xgb.XGBRegressor(**best_params_mean)
    model_mean.fit(X_full, y_full_log, sample_weight=full_sample_weights)

    best_params_q90["objective"] = "reg:quantileerror"
    best_params_q90["quantile_alpha"] = TAU_SAFETY
    best_params_q90["random_state"] = SEED
    best_params_q90["tree_method"] = "hist"
    model_q90 = xgb.XGBRegressor(**best_params_q90)
    model_q90.fit(X_full, y_full_log, sample_weight=full_sample_weights)

    # Static per-ATM bias from training data (cold-start fallback)
    full_pred_mean = np.expm1(model_mean.predict(X_full))
    bias_df = pd.DataFrame({
        "CASHP_ID": full_train_df["CASHP_ID"].values,
        "actual": full_train_df["WITHDRWLS"].values,
        "pred": full_pred_mean,
    })
    bias_df["resid"] = bias_df["actual"] - bias_df["pred"]
    static_per_atm_bias = bias_df.groupby("CASHP_ID")["resid"].mean()
    static_global_bias = bias_df["resid"].mean()
    print(f"      -> Static (cold-start) global bias: {static_global_bias:+,.0f} TL")
    print(f"      -> Static per-ATM spread: "
          f"min={static_per_atm_bias.min():+,.0f}  "
          f"max={static_per_atm_bias.max():+,.0f}")

    # Conformal calibration shift from VAL set
    val_pred_q90_pre = np.expm1(model_q90.predict(X_val))
    val_residuals_q90 = val_df["WITHDRWLS"].values - val_pred_q90_pre
    raw_val_cov = (val_residuals_q90 <= 0).mean() * 100
    calibration_shift = float(np.quantile(val_residuals_q90, TAU_SAFETY))
    print(f"      -> Val raw τ={TAU_SAFETY} coverage (pre-shift): {raw_val_cov:.1f}%")
    print(f"      -> Conformal shift: {calibration_shift:+,.0f} TL")

    # --- TEST RAW PREDICTIONS ---
    test_df["d_mean_raw"] = np.maximum(np.expm1(model_mean.predict(X_test)), 0)
    test_df["d_safety_raw"] = np.maximum(
        np.expm1(model_q90.predict(X_test)) + calibration_shift,
        test_df["d_mean_raw"]
    )

    # --- ADAPTIVE BIAS CORRECTION ---
    print(f"\n[7/8] Computing adaptive rolling bias corrections "
          f"(window={ROLLING_WINDOW_DAYS}d, λ={SHRINKAGE_LAMBDA})...")
    corrections, cold_n, roll_n = compute_adaptive_corrections(
        test_df.copy(), static_per_atm_bias, static_global_bias,
        window_days=ROLLING_WINDOW_DAYS, shrinkage=SHRINKAGE_LAMBDA
    )
    print(f"      -> Cold-start applications (static): {cold_n:,}")
    print(f"      -> Rolling applications (adaptive):  {roll_n:,}")
    print(f"      -> Correction stats: mean={corrections.mean():+,.0f}  "
          f"std={corrections.std():,.0f}  "
          f"min={corrections.min():+,.0f}  max={corrections.max():+,.0f}")

    # Sort test_df to match the order corrections were computed in
    test_df = test_df.sort_values(["DATE", "CASHP_ID"]).reset_index(drop=True)
    test_df["correction"] = corrections
    test_df["d_mean"] = np.maximum(test_df["d_mean_raw"] + test_df["correction"], 0)
    test_df["d_safety"] = np.maximum(test_df["d_safety_raw"] + test_df["correction"],
                                     test_df["d_mean"])

    # --- DIAGNOSTICS: ablation between v7-style (static) and v8 (adaptive) ---
    print(f"\n[8/8] Test set diagnostics:")
    test_actual = test_df["WITHDRWLS"].values

    # v7-style (static correction) for comparison on the same test set
    v7style_correction = test_df["CASHP_ID"].map(static_per_atm_bias).fillna(static_global_bias).values
    test_df["d_mean_v7style"] = np.maximum(test_df["d_mean_raw"] + v7style_correction, 0)

    bias_v7 = (test_df["d_mean_v7style"].values - test_actual).mean()
    smape_v7 = calculate_smape(test_actual, test_df["d_mean_v7style"].values)
    bias_v8 = (test_df["d_mean"].values - test_actual).mean()
    smape_v8 = calculate_smape(test_actual, test_df["d_mean"].values)
    cov_v8 = (test_actual <= test_df["d_safety"].values).mean() * 100
    cov_v8_buf = (test_actual <= test_df["d_safety"].values * 1.25).mean() * 100

    print(f"  v7-style static correction:  bias={bias_v7:+,.0f}  SMAPE={smape_v7:.2f}%")
    print(f"  v8 adaptive correction:      bias={bias_v8:+,.0f}  SMAPE={smape_v8:.2f}%")
    print(f"  v8 τ={TAU_SAFETY} raw coverage:    {cov_v8:.1f}%")
    print(f"  v8 τ={TAU_SAFETY} buffered (1.25): {cov_v8_buf:.1f}%")

    # 60-day simulation window
    sim_start = pd.Timestamp(TEST_START)
    sim_end = pd.Timestamp(SIM_END)
    win_df = test_df[(test_df["DATE"] >= sim_start) & (test_df["DATE"] <= sim_end)]
    if len(win_df) > 0:
        wa = win_df["WITHDRWLS"].values
        wd = win_df["d_mean"].values
        ws = win_df["d_safety"].values
        bias_w = (wd - wa).mean()
        smape_w = calculate_smape(wa, wd)
        cov_w = (wa <= ws).mean() * 100
        cov_w_buf = (wa <= ws * 1.25).mean() * 100
        print(f"\n[DIAG-WINDOW] Sim window {sim_start.date()} -> {sim_end.date()} "
              f"({len(win_df):,} rows):")
        print(f"  bias={bias_w:+,.0f}  SMAPE={smape_w:.2f}%  "
              f"raw_cov={cov_w:.1f}%  buf_cov={cov_w_buf:.1f}%")
        print(f"  v7 reference on same window:  bias=-4,669  SMAPE=41.71%  "
              f"raw_cov=76.2%  buf_cov=90.0%")
        print(f"  v6 reference on same window:  bias=-6,560  SMAPE=39.77%  "
              f"raw_cov=78.2%  buf_cov=90.1%")

    # Per-ATM bias (top 5)
    print(f"\n[DIAG] Per-ATM bias on test set (top 5 worst, v8 adaptive):")
    test_df["resid"] = test_df["d_mean"] - test_df["WITHDRWLS"]
    per_atm_test_bias = (test_df.groupby("CASHP_ID")["resid"].mean()
                         .sort_values(key=lambda s: -s.abs()).head(5))
    for atm_id, b in per_atm_test_bias.items():
        # Compare with v7-style on the same ATM
        v7_b = (test_df[test_df["CASHP_ID"] == atm_id]["d_mean_v7style"]
                - test_df[test_df["CASHP_ID"] == atm_id]["WITHDRWLS"]).mean()
        delta = b - v7_b
        print(f"    {atm_id:<14} v8 bias = {b:+,.0f} TL  "
              f"(v7-style: {v7_b:+,.0f}; Δ {delta:+,.0f})")

    # --- Adaptation trajectory: did corrections change as test progressed? ---
    print(f"\n[DIAG] Correction trajectory (mean correction by week of test):")
    test_df["_week"] = ((test_df["DATE"] - sim_start).dt.days // 7).clip(lower=0)
    weekly_corr = test_df.groupby("_week")["correction"].agg(['mean', 'std', 'count'])
    for w, row in weekly_corr.head(10).iterrows():
        print(f"    Week {int(w):2d}: mean correction={row['mean']:+,.0f}  "
              f"std={row['std']:,.0f}  n={int(row['count'])}")

    # --- EXPORT ---
    export_df = test_df[["DATE", "CASHP_ID", "WITHDRWLS", "d_mean", "d_safety"]]
    export_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSuccess! Forecasts saved to: {OUTPUT_PATH}")
    print(f"  Rows: {len(export_df):,} | "
          f"ATMs: {export_df['CASHP_ID'].nunique()} | "
          f"Days: {export_df['DATE'].nunique()}")

if __name__ == "__main__":
    run_xgboost_pipeline()
