# October 2026 neural price-curve pilots

The MLP and temporal convolution were fitted to the same frozen input snapshot,
annual eligibility window, inner validation boundary and October development
products as the ridge pilot in `price_full_curve_ridge_pilot_2026.md`. All eight
auction and feature-set comparisons were checked for identical product starts,
actual prices and reference prices across the three pilots. No November labels
were used. The annual sample has 363/353/360/362 days for DA/IDA1/IDA2/IDA3;
October has 30/29/30/31 days. A fixed daily output excludes October 25 for the
full-day products; IDA1 October 8 was cancelled. The frozen features predate the
DST correction and missing inputs are filled using training medians with flags.

The MLP uses a single ReLU hidden layer, with predetermined candidates of 32
units and L2 alpha 0.1 or 64 units and alpha 1.0. It uses Adam, batch size 32,
training MSE on per-output standardized spreads, a maximum of 250 epochs and
training-loss tolerance stopping. Preprocessing and target scaling are fitted
on each training split. Candidate selection uses chronological inner MAE; the
chosen configuration is refitted on eligible annual days. DA refits and the
IDA3 external refit reached the 250-epoch limit with convergence warnings.

The temporal model uses two one-dimensional convolutions with kernel width five
and dilations one and two across ordered quarter-hour products. Channels include
past auction-price curves, already cleared curves, delivery-hour sine and cosine,
and prior hourly curves and calendar indicators broadcast to each time step.
The external variant adds NORD forecasts, gas and French previous-day quantity
channels. MSE on globally standardized spreads is minimized using Adam; widths
16 and 32 are compared using inner MAE, with up to 80 epochs and patience 10.
The selected width and epoch count are refitted on annual data. Scaling and
missing-value replacement are fitted on training data only. Both pilots use
seed 20261112 and one CPU thread, no GPU, same gate and assumed prior-day data
availability as the ridge pilot, and no same-day realized target-derived inputs.

| Auction | Reference | Ridge core | MLP core | Temporal core | Ridge + external | MLP + external | Temporal + external |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DA | 21.36 | 18.24 | 20.65 | 23.44 | 17.92 | 19.79 | 23.85 |
| IDA1 | 8.82 | 8.92 | 9.69 | 9.24 | 8.96 | 11.94 | 9.00 |
| IDA2 | 10.20 | 10.17 | 11.34 | 10.11 | 10.32 | 11.92 | 10.26 |
| IDA3 | 14.59 | 16.35 | 18.79 | 15.10 | 16.38 | 24.14 | 21.73 |

Values are October MAE in EUR/MWh. The reference is the preceding published DA
price for DA and latest available cleared auction price for the IDAs. October is
a development set, not a final independent test, and these two configurations
per architecture do not constitute a comprehensive model search.

Reproduce with `python -m scripts.pilot_curve_mlp --run --output NEW_DIRECTORY`
and `python -m scripts.pilot_curve_temporal --run --output NEW_DIRECTORY` in the
bess environment. Detailed run reports and saved artifacts are under
`outputs/price_all_auctions_curve_mlp_pilot` and
`outputs/price_curve_temporal_pilot_v2`.
