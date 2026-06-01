# Final 73-Day Headline Run Results

This file summarizes the final 73-day rolling-horizon simulation. The raw terminal output is kept separately in `docs/headline_73day_terminal_output.md`.

## Run Setup

| Item | Value |
|---|---:|
| Simulation length | 73 days |
| ATM-days | 2,263 |
| Planning horizon | 7 days |
| Demand input | Real forecast CSV |
| Forecast point quantile | 0.55 |
| Forecast safety quantile | 0.95 |
| Vehicles | 3 |
| ATM capacity mode | Heterogeneous tiers |
| ATM capacities | 250,000 / 400,000 / 500,000 TL |
| Vehicle capacity | 1,500,000 TL |
| Stockout penalty | 3,000 TL per event |
| Safety-floor penalty | 0.1 TL / TL / day |
| Initial inventory | U(0.30, 0.50) x ATM capacity |
| MIP gap | 0.02 |
| Seed | 42 |
| Compute time | 15,229.0 seconds = 253.8 minutes |

## Main Results

| Metric | Value |
|---|---:|
| Stockout events | 121 |
| Service level | 94.65% |
| Total dispatches | 87 vehicle shifts |
| Average dispatches | 1.19 vehicle shifts/day |
| Total cash loaded | 95,514,012 TL |
| Operational cost | 156,295.27 TL |
| Reported total cost, including stockout penalty | 519,295.27 TL |

The operational cost is the sum of travel cost, dispatch cost, drop fees, and holding cost. The reported total cost additionally includes the stockout penalty. For interpretation, the two main outcomes are therefore stockouts and operational cost.

## Cost Breakdown

| Component | Value |
|---|---:|
| Travel cost | 4,436.16 TL |
| Dispatch cost | 26,100.00 TL |
| Drop fees | 50,950.00 TL |
| Holding cost | 74,809.11 TL |
| Operational cost | 156,295.27 TL |
| Stockout penalties | 363,000.00 TL |
| Reported total cost | 519,295.27 TL |

The stockout penalty is:

```text
121 stockouts x 3,000 TL = 363,000 TL
```

The operational cost is:

```text
4,436.16 + 26,100.00 + 50,950.00 + 74,809.11 = 156,295.27 TL
```

## Stockout Pattern Over Time

| Period | Days | Stockouts | Comment |
|---|---:|---:|---|
| Early buffer period | 1-6 | 0 | Initial cash levels cover demand |
| Catch-up period | 7-13 | 65 | Inventory falls and the model sends larger deliveries |
| Stabilized period | 14-73 | 56 | The rolling-horizon policy becomes more stable |
| Total | 1-73 | 121 | Matches the terminal output |

The catch-up period creates 65 of the 121 stockouts. After day 13, the run has 56 stockouts over 1,860 ATM-days, which gives a steady-state service level of about 97.0%.

## Comparison With The 30-Day Calibration Run

| Metric | 30-day run | 73-day run |
|---|---:|---:|
| Stockouts | 66 | 121 |
| Stockouts per day | 2.20 | 1.66 |
| Operational cost | 70,059 TL | 156,295 TL |
| Operational cost per day | 2,335 TL | 2,141 TL |
| Reported total cost | 268,059 TL | 519,295 TL |
| Reported total cost per day | 8,935 TL | 7,114 TL |
| Dispatches | 37 | 87 |
| Dispatches per day | 1.23 | 1.19 |

The longer run has lower per-day stockouts and lower per-day operational cost. This is mainly because the early catch-up period is spread over a longer horizon.

## Fleet Use

All three vehicles are used on four days in the terminal output: days 11, 12, 13, and 60. The full fleet is therefore used mainly during the catch-up period, with one later high-demand day.

Most days use one or two vehicles. Some days use no vehicle because the current inventory is enough for that day.

## Notes

- The result is based on one 73-day run with seed 42.
- The full-window service level is 94.65%.
- The stabilized period after day 13 reaches about 97.0% service level.
- The terminal output contains a few small display artefacts such as `-0 TL loaded`; these do not affect the final KPI totals.
- The raw terminal output is preserved in `docs/headline_73day_terminal_output.md`.
