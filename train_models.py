"""
Forecast orchestrator: Optuna search -> Stage-1/2/3 fit-calibrate-refit -> test prediction + bias correction.
Supports v8 (default) and v9 forecast modules with optional ablation flags (--no-agi, --ml-only).

Usage: python train_models.py --config configs/forecast.yaml [--forecast-version v9] [--no-agi] [--ml-only]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml


EXCLUDE_COLS = ["CASHP_ID", "DATE", "WITHDRWLS", "cluster"]

# v9 calendar additions over v8 (for --ml-only ablation)
V9_CALENDAR_FEATURES = (
    "day_of_month", "is_payday_window", "is_post_AGI", "agi_effect_strength",
)


# =============================================================================
# Forecast module loader — keeps v8 frozen, v9 opt-in
# =============================================================================

def _get_forecast_module(version: str):
    """Lazy import; v8 path never touches v9 code (frozen guarantee)."""
    if version == "v9":
        from src.forecast import xgboost_v9 as fc
        return fc
    from src.forecast import xgboost_v8 as fc
    return fc


# =============================================================================
# Optuna search space — v9 adds 3 regularisation params
# =============================================================================

def _suggest_params(trial: optuna.Trial, forecast_version: str) -> dict:
    p = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    if forecast_version == "v9":
        # Counterbalance richer feature set with explicit regularisation
        p["gamma"]      = trial.suggest_float("gamma", 0.0, 5.0)
        p["reg_alpha"]  = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    return p


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return np.maximum(alpha * diff, (alpha - 1.0) * diff).mean()


def _make_objective(fc, X_tr, y_tr_log, w_tr, X_va, y_va_actual,
                    alpha, seed, forecast_version):
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, forecast_version)
        model = fc.fit_quantile_model(X_tr, y_tr_log, params, alpha,
                                       sample_weights=w_tr, seed=seed)
        pred = np.expm1(model.predict(X_va))
        return _pinball_loss(y_va_actual, pred, alpha)
    return objective


# =============================================================================
# Stage helpers
# =============================================================================

def _run_optuna(fc, X_tr: np.ndarray, y_tr_log: np.ndarray, w_tr: np.ndarray,
                X_va: np.ndarray, y_va_actual: np.ndarray,
                alpha: float, n_trials: int, seed: int, forecast_version: str,
                logger: logging.Logger, label: str) -> dict:
    logger.info(f"Optuna [{label}] alpha={alpha} ({n_trials} trials, {forecast_version})")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_make_objective(fc, X_tr, y_tr_log, w_tr, X_va, y_va_actual,
                                    alpha, seed, forecast_version),
                   n_trials=n_trials, show_progress_bar=False)
    logger.info(f"  best val pinball: {study.best_value:.4f}")
    return study.best_params


def _stage_1_holdout_fit(fc, X_tr: np.ndarray, y_tr_log: np.ndarray, w_tr: np.ndarray,
                         params_p: dict, params_s: dict,
                         alpha_p: float, alpha_s: float, seed: int, logger: logging.Logger):
    logger.info("Stage-1: train-only fit (for hold-out calibration)")
    m_p = fc.fit_quantile_model(X_tr, y_tr_log, params_p, alpha_p,
                                 sample_weights=w_tr, seed=seed)
    m_s = fc.fit_quantile_model(X_tr, y_tr_log, params_s, alpha_s,
                                 sample_weights=w_tr, seed=seed)
    return m_p, m_s


def _stage_2_calibrate(fc, m_p_h, m_s_h, X_va: np.ndarray, y_va_actual: np.ndarray,
                       va_cashp: np.ndarray, alpha_s: float, logger: logging.Logger):
    logger.info("Stage-2: conformal shift + static bias on hold-out val")
    shift = fc.compute_conformal_shift(m_s_h, X_va, y_va_actual, target_quantile=alpha_s)
    static = fc.compute_static_bias(m_p_h, X_va, y_va_actual, va_cashp)
    bias_vals = list(static.values())
    logger.info(f"  shift_safety={shift:.1f}  static_bias n={len(static)} "
                f"mean={np.mean(bias_vals):.0f} range=[{min(bias_vals):.0f}, {max(bias_vals):.0f}]")
    return shift, static


def _stage_3_refit(fc, trva: pd.DataFrame, features: list, params_p: dict, params_s: dict,
                   alpha_p: float, alpha_s: float, seed: int, logger: logging.Logger):
    logger.info("Stage-3: production refit on train+val")
    X_trva = trva[features]
    y_trva_log = np.log1p(trva["WITHDRWLS"])
    w_trva = fc.compute_sample_weights(trva)
    m_p = fc.fit_quantile_model(X_trva, y_trva_log, params_p, alpha_p,
                                 sample_weights=w_trva, seed=seed)
    m_s = fc.fit_quantile_model(X_trva, y_trva_log, params_s, alpha_s,
                                 sample_weights=w_trva, seed=seed)
    return m_p, m_s


# =============================================================================
# v9-specific feature pipeline branches
# =============================================================================

def _apply_v9_pre_split_features(fc, df: pd.DataFrame) -> pd.DataFrame:
    """Cluster-tier one-hot — split-independent, applied before lag/rolling."""
    return fc.add_cluster_tier_features(df)


def _apply_v9_target_encoding(fc, tr, va, te):
    """Train-only stats; deterministic dict applied to all splits."""
    stats = fc.compute_atm_target_stats(tr)
    tr = fc.apply_target_encoding(tr, stats)
    va = fc.apply_target_encoding(va, stats)
    te = fc.apply_target_encoding(te, stats)
    return tr, va, te, stats


# =============================================================================
# Main orchestrator
# =============================================================================

def run(config: dict, args: argparse.Namespace, logger: logging.Logger) -> Path:
    """
    Forecast training orchestrator: Optuna search → 3-stage calibration → test prediction.

    Pipeline:
    1. Load + feature engineer ATM data (lag, rolling, calendar, zero-handling)
    2. Split train/val/test by date
    3. Parallel Optuna for point (α=0.55) and safety (α=0.95) quantiles
    4. Stage-1: hold-out fit on training data
    5. Stage-2: conformal calibration (shift + static bias) on validation
    6. Stage-3: production refit on train+val
    7. Inference: predict with calibration + rolling bias correction on test

    Args:
        config: Forecast config dict (data paths, split dates, Optuna settings)
        args: CLI args (forecast-version, alpha, trials, seed, ablations)
        logger: Logger instance

    Returns:
        Path to saved joblib artifact (trained models + calibration metadata)
    """
    alpha_p = args.alpha_point  if args.alpha_point  is not None else config["quantiles"]["alpha_point"]
    alpha_s = args.alpha_safety if args.alpha_safety is not None else config["quantiles"]["alpha_safety"]
    seed    = args.seed         if args.seed         is not None else config["seed"]
    forecast_version = args.forecast_version

    if args.trials is not None:
        n_trials = args.trials
    elif args.quick:
        n_trials = config["optuna"]["trials_quick"]
    else:
        n_trials = config["optuna"]["trials"]

    fc = _get_forecast_module(forecast_version)
    is_v9 = forecast_version == "v9"
    apply_v9_features = is_v9 and not args.ml_only
    ver_suffix = "" if forecast_version == "v8" else f"_{forecast_version}"

    models_dir = Path(config["paths"]["models_dir"]); models_dir.mkdir(parents=True, exist_ok=True)
    preds_dir  = Path(config["paths"]["predictions_dir"]); preds_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run config: forecast={forecast_version}, alpha_p={alpha_p}, "
                f"alpha_s={alpha_s}, trials={n_trials}, seed={seed}, "
                f"no_agi={args.no_agi}, ml_only={args.ml_only}")

    # ------- Pipeline through split -------
    logger.info("Loading + feature engineering")
    df = fc.load_atm_excel(config["data"]["excel_path"], config["data"]["sheet_name"])
    df = fc.build_calendar_features(df)
    if apply_v9_features:
        df = _apply_v9_pre_split_features(fc, df)
    df = fc.build_lag_rolling_features(df)
    df = fc.filter_zeros(df,
                          consecutive_zero_threshold=config["zero_handling"]["consecutive_zero_threshold"],
                          drop_isolated_zeros=config["zero_handling"]["drop_isolated_zeros"])
    tr, va, te = fc.split_train_val_test(df,
                                          val_start=config["split"]["val_start"],
                                          test_start=config["split"]["test_start"])
    if apply_v9_features:
        tr, va, te, _enc_stats = _apply_v9_target_encoding(fc, tr, va, te)
    logger.info(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    features = [c for c in tr.columns if c not in EXCLUDE_COLS]
    if args.no_agi:
        features = [c for c in features if c not in ("is_post_AGI", "agi_effect_strength")]
    if args.ml_only:
        features = [c for c in features if c not in V9_CALENDAR_FEATURES]
    logger.info(f"  feature count: {len(features)}")
    X_tr = tr[features]; y_tr_log = np.log1p(tr["WITHDRWLS"])
    X_va = va[features]; y_va_actual = va["WITHDRWLS"].values
    X_te = te[features]; y_te_actual = te["WITHDRWLS"].values
    w_tr = fc.compute_sample_weights(tr,
                                      weight_clip=(config["sample_weights"]["clip_low"],
                                                   config["sample_weights"]["clip_high"]))

    # ------- Optuna search -------
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_p = _run_optuna(fc, X_tr, y_tr_log, w_tr, X_va, y_va_actual,
                          alpha_p, n_trials, seed, forecast_version, logger, "point")
    best_s = _run_optuna(fc, X_tr, y_tr_log, w_tr, X_va, y_va_actual,
                          alpha_s, n_trials, seed, forecast_version, logger, "safety")
    try:
        with open(models_dir / f"best_params_point{ver_suffix}.json",  "w") as f:
            json.dump(best_p, f, indent=2)
        with open(models_dir / f"best_params_safety{ver_suffix}.json", "w") as f:
            json.dump(best_s, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save best_params JSON: {e}")
        raise

    # ------- Stage-1 / 2 / 3 -------
    m_p_h, m_s_h = _stage_1_holdout_fit(fc, X_tr, y_tr_log, w_tr, best_p, best_s,
                                         alpha_p, alpha_s, seed, logger)
    shift_safety, static_bias = _stage_2_calibrate(fc, m_p_h, m_s_h, X_va, y_va_actual,
                                                     va["CASHP_ID"], alpha_s, logger)
    trva = pd.concat([tr, va], ignore_index=True)
    m_p, m_s = _stage_3_refit(fc, trva, features, best_p, best_s,
                               alpha_p, alpha_s, seed, logger)

    # ------- Persist -------
    artifacts = {
        "point_model":      m_p,
        "safety_model":     m_s,
        "shift_safety":     shift_safety,
        "static_bias":      static_bias,
        "feature_cols":     features,
        "alpha_point":      alpha_p,
        "alpha_safety":     alpha_s,
        "seed":             seed,
        "forecast_version": forecast_version,
        "no_agi":           args.no_agi,
        "ml_only":          args.ml_only,
        "trained_at":       datetime.now().isoformat(),
    }
    artifacts_path = models_dir / f"{forecast_version}_artifacts_p{alpha_p}_s{alpha_s}.joblib"
    try:
        joblib.dump(artifacts, artifacts_path)
        logger.info(f"Saved artifacts: {artifacts_path}")
    except IOError as e:
        logger.error(f"Failed to save artifacts to {artifacts_path}: {e}")
        raise

    # ------- Test prediction + bias correction + CSV -------
    logger.info("Test prediction + bias correction")
    preds_raw = fc.predict_with_calibration(
        m_p, m_s, X_te, shift_safety,
        cashp_ids=te["CASHP_ID"], dates=te["DATE"], actuals=y_te_actual,
    )
    bc = config["bias_correction"]
    preds_corrected = fc.apply_rolling_bias_correction(
        preds_raw, static_bias,
        window=bc["window"], shrinkage=bc["shrinkage"],
        cold_start_min_obs=bc["cold_start_min_obs"],
    )

    csv_df = (preds_corrected[["DATE", "CASHP_ID", "actual", "d_mean", "d_safety"]]
              .rename(columns={"actual": "WITHDRWLS"}))
    csv_path = preds_dir / f"test_predictions{ver_suffix}_p{alpha_p}_s{alpha_s}.csv"
    try:
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions: {csv_path}")
    except IOError as e:
        logger.error(f"Failed to save predictions to {csv_path}: {e}")
        raise

    # ------- Coverage diagnostics -------
    cov_p = (y_te_actual <= preds_corrected["d_mean"]).mean()
    cov_s = (y_te_actual <= preds_corrected["d_safety"]).mean()
    logger.info(f"Test coverage: d_mean={cov_p:.3f} (target {alpha_p}) | "
                f"d_safety={cov_s:.3f} (target {alpha_s})")

    return artifacts_path


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast orchestrator — v8 (default) or v9.")
    p.add_argument("--config",            type=Path, required=True)
    p.add_argument("--forecast-version",  choices=["v8", "v9"], default="v8",
                   help="Forecast module to use (default: v8 for backward compat)")
    p.add_argument("--alpha-point",       type=float, default=None)
    p.add_argument("--alpha-safety",      type=float, default=None)
    p.add_argument("--trials",            type=int,   default=None)
    p.add_argument("--quick",             action="store_true")
    p.add_argument("--no-agi",            action="store_true",
                   help="Drop AGI features (regime-shift trap diagnostic)")
    p.add_argument("--ml-only",           action="store_true",
                   help="Drop ALL v9 features, keep extended 9-D Optuna (search-space ablation)")
    p.add_argument("--seed",              type=int,   default=None)
    args = p.parse_args()
    if args.ml_only and args.forecast_version != "v9":
        p.error("--ml-only requires --forecast-version v9 (extended Optuna search needs v9 path)")
    return args


def setup_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_models_{timestamp}.log"
    logger = logging.getLogger("train_models")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler();        sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    logger = setup_logger(Path(config["paths"]["logs_dir"]))
    run(config, args, logger)


if __name__ == "__main__":
    main()