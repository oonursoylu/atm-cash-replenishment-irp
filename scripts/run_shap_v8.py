"""
SHAP interpretability analysis on the v8 forecast artifacts.

Uses XGBoost's native TreeSHAP via Booster.predict(pred_contribs=True), which
bypasses the SHAP library's model-introspection layer and avoids the
XGBoost 2.x + SHAP 0.49 base_score parsing bug (scientific-notation float
strings such as '1.0812169E1' cannot be parsed by SHAP's _set_xgboost_model_attributes).

The SHAP library is still used for plotting, since plotting consumes a plain
numpy array of contribution values without re-introspecting the model.

Outputs:
- outputs/shap/shap_v8_point_bar.png       — top-20 mean|SHAP| bar plot
- outputs/shap/shap_v8_point_beeswarm.png  — top-20 beeswarm plot
- outputs/shap/shap_v8_point_top20.csv     — top-20 features ranked numerically
- outputs/shap/shap_v8_safety_*            — equivalent for safety model

Run from project root:
    python scripts/run_shap_v8.py
    python scripts/run_shap_v8.py --sample 1000
"""

import sys
from pathlib import Path

# Make project root importable when running as `python scripts/run_shap_v8.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml

from src.forecast import xgboost_v8 as fc


def load_artifacts(path: Path) -> dict:
    arts = joblib.load(path)
    print(f"  alpha_point={arts['alpha_point']}, alpha_safety={arts['alpha_safety']}, "
          f"seed={arts['seed']}, version={arts['forecast_version']}")
    print(f"  feature count: {len(arts['feature_cols'])}")
    print(f"  xgboost runtime: {xgb.__version__}, shap runtime: {shap.__version__}")
    return arts


def build_test_features(forecast_cfg: dict, feature_cols: list) -> pd.DataFrame:
    df = fc.load_atm_excel(forecast_cfg["data"]["excel_path"],
                            forecast_cfg["data"]["sheet_name"])
    df = fc.build_calendar_features(df)
    df = fc.build_lag_rolling_features(df)
    df = fc.filter_zeros(
        df,
        consecutive_zero_threshold=forecast_cfg["zero_handling"]["consecutive_zero_threshold"],
        drop_isolated_zeros=forecast_cfg["zero_handling"]["drop_isolated_zeros"],
    )
    _tr, _va, te = fc.split_train_val_test(
        df,
        val_start=forecast_cfg["split"]["val_start"],
        test_start=forecast_cfg["split"]["test_start"],
    )
    missing = [c for c in feature_cols if c not in te.columns]
    if missing:
        raise RuntimeError(f"Test features missing columns from artifact: {missing[:5]}...")
    return te[feature_cols]


def compute_shap_native(model: xgb.XGBRegressor, X: pd.DataFrame, sample_size: int | None):
    """
    Compute SHAP values via XGBoost's native TreeSHAP, returning (shap_values, X_sample).

    Booster.predict(pred_contribs=True) returns an (n_rows, n_features + 1) array;
    the last column is the bias term (base_score-equivalent) which we drop.
    """
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    # Extract the underlying Booster from the sklearn wrapper
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_sample, feature_names=list(X_sample.columns))
    contribs = booster.predict(dmatrix, pred_contribs=True)

    # Drop bias column (last)
    shap_values = contribs[:, :-1]
    assert shap_values.shape == (len(X_sample), X_sample.shape[1]), \
        f"shape mismatch: {shap_values.shape} vs expected ({len(X_sample)}, {X_sample.shape[1]})"

    return shap_values, X_sample


def top_n(shap_values: np.ndarray, X: pd.DataFrame, n: int) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False)
              .head(n)
              .reset_index(drop=True))


def make_plots(shap_values: np.ndarray, X: pd.DataFrame, label: str, outdir: Path) -> None:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"v8 {label} model — top 20 features by mean |SHAP|")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_v8_{label}_bar.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title(f"v8 {label} model — SHAP value distribution (top 20)")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_v8_{label}_beeswarm.png", dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="models/v8_artifacts_p0.55_s0.95.joblib")
    parser.add_argument("--forecast-config", default="configs/forecast.yaml")
    parser.add_argument("--outdir", default="outputs/shap")
    parser.add_argument("--sample", type=int, default=2000,
                         help="Subsample size for SHAP compute (set 0 or negative for all rows)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading artifacts from {args.artifacts}")
    arts = load_artifacts(Path(args.artifacts))

    print(f"\n[2/3] Building test features via v8 pipeline")
    forecast_cfg = yaml.safe_load(open(args.forecast_config))
    X_test = build_test_features(forecast_cfg, arts["feature_cols"])
    print(f"  test rows: {len(X_test):,}")

    sample_size = args.sample if args.sample > 0 else None
    if sample_size:
        print(f"  SHAP will run on a {sample_size}-row subsample (random_state=42)")
    else:
        print(f"  SHAP will run on full test set")

    print(f"\n[3/3] Computing SHAP per model (native XGBoost TreeSHAP)")
    for label, model_key, alpha in [
        ("point", "point_model", arts["alpha_point"]),
        ("safety", "safety_model", arts["alpha_safety"]),
    ]:
        print(f"\n  --- {label} model (alpha={alpha}) ---")
        shap_values, X_sample = compute_shap_native(arts[model_key], X_test, sample_size)
        top20 = top_n(shap_values, X_sample, n=20)
        top20.to_csv(outdir / f"shap_v8_{label}_top20.csv", index=False)
        print(f"  Top 10 features by mean |SHAP|:")
        for _, row in top20.head(10).iterrows():
            print(f"    {row['feature']:<40} {row['mean_abs_shap']:>14.4f}")
        make_plots(shap_values, X_sample, label, outdir)
        print(f"  Saved: shap_v8_{label}_bar.png, shap_v8_{label}_beeswarm.png, shap_v8_{label}_top20.csv")

    print(f"\n[OK] All SHAP outputs in {outdir}/")


if __name__ == "__main__":
    main()