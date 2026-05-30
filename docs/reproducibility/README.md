# IRP Reproducibility Audit Evidence

Date: 2026-05-30

## Purpose

This folder preserves the evidence used to diagnose and fix the IRP reproducibility issue.

The main issue was not forecast RNG or initial inventory RNG. The issue was that the live OSM travel-time matrix could differ slightly across separate Python processes. Since the sweep scripts run IRP cells in separate subprocesses, some older IRP calibration cells may have been solved under slightly different travel matrices.

## Locked Reporting Configuration

See `../CONFIG_LOCK.md`.

Main reporting configuration:
- Travel matrix: frozen symmetric OSM matrix
- Travel matrix hash: `76013f9295fe036d980740994878c3be`
- Asymmetric ablation hash: `bdbc52b4269920118507228965a3d4bf`
- Forecast CSV hash: `b9432a2eba76b887b49597cc705f0d8e`
- CPLEX mode: legacy/default, documented as reproducible on the 8-core environment
- Calibration gap: 0.05
- Headline gap: 0.02
- Seed: 42

## Evidence Summary

1. Initial inventory RNG was not the source.
   - Initial inventory hash was identical across sweep positions.
   - Day-3 entry state hash was deterministic.

2. Live OSM travel matrix drift was observed.
   - Separate Python processes produced multiple OSM matrix hashes.
   - The same process was stable.
   - This made cross-sweep comparisons vulnerable when every sweep cell was run in a separate subprocess.

3. Frozen matrix fixed reproducibility.
   - With frozen travel matrix, baseline-first and baseline-last matched.
   - Full 30-day legacy runs were cross-process identical.
   - Full 30-day deterministic CPLEX runs were also cross-process identical.

4. CPLEX deterministic mode was not used as the reporting default.
   - It changed the selected within-gap solution materially.
   - Therefore it remains opt-in.
   - Reporting default remains legacy CPLEX with frozen matrix.

5. Symmetric/asymmetric finding was revised.
   - Asymmetric matrix produced more stockouts in both solver modes.
   - Operational-cost direction was solver-mode dependent.
   - Therefore the symmetric matrix decision remains supported, but the old strong cost-tradeoff wording should be softened.

## Files

- `acceptance_test_20260530.json` / `.py`: day-3 cross-process acceptance test.
- `fullchain_repro_probe_20260530.json` / `.py`: full 30-day cross-process reproducibility probe.
- `frozen_matrix_2x2_probe_20260530.json` / `.py`: frozen 2x2 calibration check.
- `frozen_asymmetry_probe_20260530.json` / `.py`: frozen symmetric vs asymmetric check.
- `day3_entry_state_20260530.json`: saved deterministic day-3 entry state.
- `mechanism_probe_20260529.py`: mechanism probe for RNG and solver behavior.
- `cplex_introspect_20260530.py`: CPLEX parallel/threads introspection.
