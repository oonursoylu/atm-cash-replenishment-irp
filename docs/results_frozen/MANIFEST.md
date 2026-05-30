# Frozen IRP Results Manifest

Date started: 2026-05-30

## Purpose

This folder will contain the canonical frozen-OSM IRP rerun results used for the final thesis text.

Older result files in `docs/phase_*.json`, `docs/baseline_results.json`, and `docs/headline_73day_*` were produced before the OSM travel matrix was frozen. They are retained as historical/pre-frozen artifacts, but they should not be used as final thesis evidence unless explicitly copied or regenerated here with frozen provenance.

## Locked Configuration

See `../CONFIG_LOCK.md`.

Canonical reporting setup:
- Travel matrix: frozen symmetric OSM
- Travel matrix hash: `76013f9295fe036d980740994878c3be`
- Forecast baseline hash: `b9432a2eba76b887b49597cc705f0d8e`
- CPLEX mode: legacy/default
- Calibration gap: 0.05
- Headline gap: 0.02
- Seed: 42

## Result Index

| Thesis section | Experiment | Frozen result file | Status | Notes |
|---|---|---|---|---|
| 4.2.2 | Capacity/fleet 2x2 | TBD | pending | Existing probe available in `docs/reproducibility/`, but official frozen result still pending. |
| 4.2.3 | Cost parameter grid | TBD | pending | Must be rerun under frozen symmetric matrix. |
| 4.3.2 / 4.4 | Alpha safety sweep | TBD | pending | Requires s0.90/s0.99 forecast hash checks first. |
| 4.2.4 | Initial inventory sweep | TBD | pending | Must be rerun under frozen symmetric matrix. |
| 4.4.1 | Multi-seed variance | TBD | pending | Must be rerun under frozen symmetric matrix if time allows. |
| 4.5.6 | Baseline comparison | TBD | pending | Must be rerun or revalidated under frozen matrix. |
| 4.5 | 73-day headline | TBD | pending | Run last, after calibration decisions are confirmed. |

## Writing Rule

Final thesis claims should be written from this manifest and the frozen result JSONs, not from the pre-frozen root `docs/phase_*.json` files.

## Forecast Inputs for Alpha Sweep

| alpha_safety | CSV | MD5 hash | Attestation |
|---:|---|---|---|
| 0.90 | `predictions/test_predictions_p0.55_s0.9.csv` | `4ed26d7f262d7503d8220386d32aad80` | v8, 50-trial Optuna, seed 42, log `train_models_20260526_225747.log` |
| 0.95 | `predictions/test_predictions_p0.55_s0.95.csv` | `b9432a2eba76b887b49597cc705f0d8e` | v8, 50-trial Optuna, seed 42, log `train_models_20260526_230631.log` |
| 0.99 | `predictions/test_predictions_p0.55_s0.99.csv` | `e23806961b4c2cb7290d7dbf3905305f` | v8, 50-trial Optuna, seed 42, log `train_models_20260526_231250.log` |

The active alpha-sweep CSVs are distinct from the earlier reduced-trial artifacts. Final alpha-sweep IRP reruns should use only these hashes.
