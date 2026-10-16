# Prepared forecasts into the optimizer

`optimization.forecast_inputs.load_forecast_inputs(config)` loads prepared
CSV/Parquet tables and saved JSON artifacts into the existing `OptimizationInput`.
`build_forecast_inputs(config, pv_forecasts, auction_inputs, known_results=None,
accepted_trades=None)` accepts the same tables and loaded model objects in memory.
Use `configs/stochastic_prepared.example.yaml`, fill its explicit equipment,
physical-state, trading and data fields, then run:

```bash
PYTHONPATH=src python -m bess_pv_trading optimize --config your_prepared_config.yaml
```

This boundary performs no acquisition or model fitting. Acquisition and later
preprocessing use the DuckDB catalog and should export these canonical tables
with their snapshot IDs/checksums retained as additional provenance columns.
The loader hashes every supplied file. Missing products, unknown information
clocks and incompatible native durations are rejected without filling.

All offered quantities are assumed to execute at their auction clearing price.
There are no bid-price curves or acceptance/rejection scenarios. Before the MCP
is published, `pending_position_mwh` contains guaranteed committed quantities;
these are fixed positions in every residual scenario. Supply explicit zeros if
no quantity was submitted. After publication, retain the same quantities and
write their actual MCP cashflow once to the priced trade ledger. The legacy
`accepted_trades_path` name refers to this already executed, priced ledger.
The optimizer reports incremental new trading/dispatch profit; prior committed
cashflow belongs to the external realized ledger and must be added once when
reporting total realized profit. This adapter prepares one solve; the existing
event replay manages submission, publication and realized accounting.

The `optimization.forecast_inputs` section contains `delivery_day`, an aware
`decision_cutoff`, optional `residual_start`, `pv_forecast_path`, `auction_inputs`,
explicit `price_distribution` (`point` or `gaussian`), and an explicit settlement
point-price study assumption. Existing `plant`, `battery`, `state`, `trading`,
`degradation`, `solver` and `outputs` sections apply. State timestamp must equal
the exact residual boundary. A later solve carries actual stored energy,
cumulative aging and Xu segment history when applicable. Market calendar
validity remains bounded; this adapter introduces no hourly DA dispatch mapping.

Each PV member has one row per native delivery interval:

| Columns | Meaning |
|---|---|
| `farm_id`, `member_id`, `member_origin` | Explicit farm and forecast-member identity; realized/oracle origins are rejected. |
| `member_probability` | Positive constant within each member; member weights sum to one. |
| `forecast_cycle_time`, `forecast_available_at` | Aware source and availability clocks, both no later than the current decision. |
| `interval_start_utc`, `interval_end_utc`, `pv_ac_mw` | Exact native products and finite nonnegative AC MW. |
| `input_eligibility`, `review_status` | Every row eligible with no review exclusion. Optional review exclusion flags must be false. |
| `model_fit_cutoff` | Required unless `pv_model_fit_status=unfitted_explicit_assumption` is explicitly configured. |

PV member labels remain latent in the information tree. Completed interval PV
values distinguish histories only at interval end plus configurable telemetry
delay. Battery controls precede the interval; automatic curtailment follows the
same end-of-interval observation convention as the oracle. Joint scenario mass
is PV-member probability times price-path probability. Price draws are shared
across members as an explicit independent product measure; dependence between
regional prices and PV members is not estimated here.

An auction input chooses exactly one of these forms, always declaring
`native_interval_minutes: 15`:

| Input form | Required files and columns |
|---|---|
| Supplied absolute mean | `mean_curve_path`: `product_start`, `product_end`, `mean_price_eur_mwh`, `source_time`, `available_at`, `fit_cutoff`, `origin`, `zone`. |
| Saved DA point model | `point_model_path`: `LinearPriceForecast` JSON; `features_path`: exact native future products, its named predictors, `feature_available_at`, `source_time`, `zone`. |
| Saved IDA conditional spread | `spread_model_path`: `ConditionalSpreadPriceForecast` JSON; `features_path`: exact eligible products, its static predictors, `static_feature_available_at`, `source_time`, `zone`. |

Future features contain no realized target. Model fits and static source inputs
must be available by the current solve. An IDA frame may retain its intended
future `forecast_cutoff`; this does not assert that future ancestor prices are
already known. At every simulated auction result node the spread model rebuilds
predecessor-price features from sampled or known ancestor curves, then returns
an absolute mean. Earlier actual future-auction labels are never inserted into
static features. The explicit native-scope declaration must match the model's
training regime; pooled point artifacts do not independently encode that scope.

Gaussian mode additionally requires an auction-scoped `calibration_path` for
each unknown auction. Separate calibrated bias is added to the point mean and
innovations have zero mean and calibrated temporal error covariance. `branches`
is the count at each unknown result, so two branches at all four auctions yield
up to 16 price paths for each PV member. A configurable joint-leaf limit bounds
tree size. Point mode explicitly uses zero innovations. Unverified saved models,
features or Gaussian calibrations require `allow_retrospective: true`; all
prepared study outputs retain `operational_eligible: false`. These switches do
not establish historical publication evidence or trading performance.

`known_results_path` contains `auction`, `product_start`, `product_end`,
`price_eur_mwh`, `published_at`, `zone` (`NORD`). Recorded publication must be no
later than the solve and overrides the calendar's result clock. Supply no
forecast entry for a known auction. Every enabled auction whose result is
already published must have its complete actual curve. IDA3 contains only its
48 noon-to-midnight products. Rectangular scenario coefficients outside that
mask are inert zeros and cannot be traded; they are never observed price labels.

The priced ledger contains `auction`, `submission_time`, `publication_time`,
`interval_id`, `quantity_mwh`, `price_eur_per_mwh`, `trade_id`. Submission cannot
follow the gate; publication equals the known result clock; residual-product
prices must match the actual curve. Quantities stay fixed. A later solve can
read full-day tables and selects its exact remaining products, while retaining
the original UTC product IDs and already priced ledger cashflows.

Offline evidence is deliberately synthetic. The helper
`tests/unit/test_forecast_inputs.py::write_prepared_case(directory)` exports
canonical files and an executable four-auction/two-PV configuration. Its optional
`gaussian=True` mode exports a small DA-only four-leaf Gaussian fixture. Tests
exercise latent PV information, branch-conditional intraday means, known prices,
committed positions, residual state, IDA3 masks, future-information rejection,
file hashes, an actual licensed MILP solve and the real `optimize` CLI route.
