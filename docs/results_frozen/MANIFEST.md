# Frozen IRP Results Manifest

Date started: 2026-05-30

## Purpose

This folder contains the canonical frozen-OSM IRP rerun results used for the final thesis text.

Older result files previously kept in root `docs/` were produced before the OSM travel matrix was frozen. They are retained under `docs/archive/pre_frozen_osm/results/` as historical/pre-frozen artifacts, but they should not be used as final thesis evidence unless explicitly regenerated with frozen provenance.

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
| 4.2.2 | Capacity/fleet 2x2 | `phase_4h_2x2_frozen_20260530.json` | complete | Frozen result: uniform-nv3 has lowest stockouts (70), hetero-nv3 has lowest operational cost (66,840.51 TL). |
| 4.2.3 | Cost parameter grid | `phase_4a_cost_grid_frozen_20260530.json` | complete | Frozen grid complete: `3000/0.1` is the cost-disciplined knee (72 SO / 66,840.51 TL); `3000/1.0` is the higher-service alternative (69 SO / 70,062.79 TL). Old inverse-penalty and U-shape claims are weakened/withdrawn. |
| 4.3.2 / 4.4 | Alpha safety sweep | `phase_4_alpha_safety_sweep_frozen_20260530.json` | complete | Frozen result confirms monotone service-cost trade-off: alpha 0.90 -> 84 SO / 64,577.18 TL op; 0.95 -> 72 SO / 66,840.51 TL op; 0.99 -> 34 SO / 79,617.33 TL op. Alpha 0.95 remains the cost-disciplined policy point; 0.99 is the high-service alternative. |
| 4.2.4 | Initial inventory sweep | `phase_4i_initial_inventory_frozen_20260530.json` | complete | Frozen result: lows 0.10/0.20/0.30/0.40/0.50 give 79/79/72/73/71 SO and 66,247.06/67,287.43/66,840.51/68,012.25/69,219.36 TL op. Old sharp V-shape is weakened: `0.30` remains the cost-disciplined operating band, while `0.50` is a higher-inventory alternative with only 1 fewer SO and higher holding/op cost. |
| 4.4.1 | Multi-seed variance | `phase_4s_seed_variance_frozen_20260531.json` | complete | Frozen 30-day seed variance at the Proposed calibration cell, gap 0.05, N=7 seeds {1,7,13,21,42,73,99}: mean 74.86 SO, sigma 2.97 SO, sigma op cost 549.53 TL; seed 42 is representative at -0.96 sigma. |
| 4.5.6 | Baseline comparison | `phase_4_baseline_comparison_frozen_20260531.json` + `phase_4j_proposed_headline_frozen_20260531.json` | complete | Frozen 73-day component-attribution comparison: B0 52 SO / 229,354.51 TL op; B1 25 SO / 215,726.01 TL op; B2 252 SO / 134,386.70 TL op; Proposed 120 SO / 156,718.75 TL op. Service levels should be computed from stockouts over 2,263 ATM-days. |
| 4.5 | 73-day headline | `phase_4j_proposed_headline_frozen_20260531.json` | complete | Frozen headline: Proposed 120 SO / 94.70% SL / 156,718.75 TL op / 516,718.75 TL reported total over 73 days, gap 0.02, seed 42, legacy CPLEX. |
| Appendix / 4.4 | High-service alpha 0.99 variant | `phase_4j_high_service_alpha099_frozen_20260601.json` | complete | Frozen 73-day appendix variant, not the headline: alpha_safety 0.99 gives 51 SO / 97.75% SL / 187,355.41 TL op / 340,355.41 TL reported total, gap 0.02, seed 42. Compared with the alpha 0.95 headline, it avoids 69 SO at +30,636.66 TL op cost; the increment is more than fully accounted for by holding (+31,239.64 TL), partly offset by small net reductions in dispatch/drop costs and a small travel increase. |

## Writing Rule

Final thesis claims should be written from this manifest and the frozen result JSONs, not from the pre-frozen archive files.

## Forecast Inputs for Alpha Sweep

| alpha_safety | CSV | MD5 hash | Attestation |
|---:|---|---|---|
| 0.90 | `predictions/test_predictions_p0.55_s0.9.csv` | `4ed26d7f262d7503d8220386d32aad80` | v8, 50-trial Optuna, seed 42, log `train_models_20260526_225747.log` |
| 0.95 | `predictions/test_predictions_p0.55_s0.95.csv` | `b9432a2eba76b887b49597cc705f0d8e` | v8, 50-trial Optuna, seed 42, log `train_models_20260526_230631.log` |
| 0.99 | `predictions/test_predictions_p0.55_s0.99.csv` | `e23806961b4c2cb7290d7dbf3905305f` | v8, 50-trial Optuna, seed 42, log `train_models_20260526_231250.log` |

The active alpha-sweep CSVs are distinct from the earlier reduced-trial artifacts. Final alpha-sweep IRP reruns should use only these hashes.


