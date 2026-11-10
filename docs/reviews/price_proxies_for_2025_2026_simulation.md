# Price proxies for 2025–2026 battery simulations

## Native products and price features

The saved search inputs have hourly DA prices through September 2025 and
quarter-hourly DA prices from October 2025. They have quarter-hourly intraday
auction prices from January 2025 onward. In 2024, the records labeled IDA in
the snapshot are hourly; the identity and gate times of the records before
the launch of pan-European IDAs on 13 June 2024 require verification.

The hourly IDA search uses native hourly IDA targets and the matching hourly DA
price, provided the configured DA publication is before the IDA decision. It
models IDA1 minus DA, IDA2 minus IDA1, and IDA3 minus IDA2. For the mixed
January–September 2025 regime, the quarter-hourly IDA search carries each DA
hour's price over the four quarters it covers. IDA targets and controls remain
quarter-hourly. Historical quarter-hourly price inputs requested by an hourly
model are averaged only if all four source quarters are present. These are
feature-alignment rules, not a joint model trained across resolutions.

## Candidates versus models

The old search prepared eight target groups, two resolutions for each auction,
and 92,160 candidate inner fits. The new full-year search has four quarter-hour
target groups and at most 78,480 candidate inner fits. Candidate fits compare
feature subsets, model families and hyperparameters; they are not all required
for deployment. Each selected cohort/availability-class winner is an additional
alternative, so the runner can save more than one winner per target group.

For a 2025–2026 simulator a minimal single-policy design needs five distinct
target/resolution specifications: hourly DA, quarter-hourly DA, and one
quarter-hourly model for each of the three IDAs. A model specification can be
refitted at multiple earlier cutoffs, creating multiple model vintages. Old
hourly IDA models are historical diagnostics and are not required to predict
quarter-hourly IDAs in 2025–2026.

## What remains for a battery study

The full-year quarter-hour search trains on October 2025–September 2026 and
validates later. Its models do not provide out-of-sample predictions for their
own training period. Likewise, selecting a historical DA model using a 2025
validation score means that period is a development period, not an untouched
test. A time-consistent replay needs specifications selected using only earlier
information, plus model fit cutoffs preceding each evaluation decision. Early
2025 is a cold start for quarter-hour IDAs and needs an explicit baseline or
bootstrap policy, not future-label training. Define the policy and coverage
before claiming a continuous 2025–2026 out-of-sample study.

The search models predict each IDA at its own gate using actual earlier
clearing prices as features and spread anchors. At the DA gate those same-day
prices have not cleared: the simulator needs a separate forecast of future
auctions at that earlier cutoff. A possible design is to propagate predicted
earlier-auction prices into later equations, then replace them with actual
prices after publication. The complete feature vector, including neighboring
products and price lags, must be reconstructed for each cutoff. This recursion
is an additional modeling assumption and must be tested for distribution shift.

The continuous simulator in this checkout currently builds planning curves
from actual auction prices and thus represents a price-oracle benchmark; the
new search is not wired to it as a historical, multi-gate forecast provider.
The separate realtime-MPC checkout already has a price-forecast provider for
its own `CausalPriceForecaster` artifacts. The search's Ridge/joblib artifacts
use a different feature contract and are not directly compatible with it. The
existing fitted scenario-tree utility is a separate interface and does not by
itself deploy these search models. A forecast-price replay needs a provider
that emits each auction's remaining price curve at each decision cutoff, records
training/feature provenance, handles native DA/IDA order resolution, and keeps
the actual clearing curve solely for publication and realized accounting.

The authorized same-day French/Swiss realized-data experiments are retrospective.
The Italian–Swiss DA spread includes the target DA price and must be reported
as outcome-derived, never as a normal DA proxy. Missing data, publication
assumptions and periods without causal training coverage must be visible in the
study output rather than filled with future prices. The user also requested a
separate intentionally in-sample preliminary baseline; its interpretation and
launcher are described in `price_in_sample_baseline_2026.md`.
