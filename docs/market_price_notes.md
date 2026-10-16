# Market and price implementation evidence

`markets.py` implements one local delivery day in Europe/Rome and stores UTC interval boundaries. The ordinary/full-calendar tests cover 92, 96 and 100 products. Historical quarter-hour backtests are bounded to 1 October 2025 through 31 August 2026. Native hourly DA products before October 2025 are excluded. Rule revisions 13 and 14 use their event dates; auction times are unchanged by revision 14.

Publication and submission are different events. IDA results use the documented deadlines, DA uses an explicitly labeled 13:30 study assumption. `schedule(..., recorded_results={...}, cancelled=(...))` accepts actual publications or missing/cancelled auctions. No missing price is synthesized. All results clocks are local and converted with the timezone database, including DST.

`data/market_prices.py` imports canonical CSV or explicitly mapped XML. Required columns are `provider,dataset,zone,auction,product_start,product_end,price_eur_mwh,published_at,publication_provenance,revision`. Timestamp strings require explicit offsets. Negative prices survive. Repeated product/revision records, missing prices, nonpositive or unsupported durations and ambiguous revision ordering fail validation. Separate completeness reports list gaps. This is a manual normalization boundary, not an automatic decoder for every native GME archive format.

Cache identity includes provider, dataset, zone, product, date range, vintage/revision and schema version. Cache retrieval time is metadata and never replaces `published_at`. Imports check caches before source files. The shared `data/api.py` pipeline now retrieves documented ENTSO-E day-ahead prices and load/generation/solar/wind forecasts, plus Terna load observations/forecasts, explicit MGP/MSD load forecasts, MSD wind forecasts and national actual renewable energy by source. See [API pipeline](api_pipeline.md) and [Terna adapter](terna_api.md) for supported schemas, configuration and the cache-first `download-data` command. Environment secrets or hidden interactive prompts are used only after cache misses; credentials and token responses are excluded from saved request provenance. Mocked downloads and CLI dry-runs are tested; no live authenticated retrieval was executed.

Provider normalization preserves raw source units, reference labels and retrieval time. Unverified interval bounds, publication times, revision archives and historical coverage remain explicit. Source document creation is not publication, and date-only publication labels do not gain an invented clock. Downloaded fundamentals with unknown availability remain ineligible for causal historical features. No verified ENTSO-E or Terna IDA-clearing-price subset is configured. The GME API adapter is now implemented with offline documented-format tests, while user-run authenticated coverage is pending. Terna imbalance-price retrieval remains blocked by conflicting official route documentation; historical settlement economics also remain unresolved. Actual auction labels and independent publication/settlement evidence are still needed for a historical trading evaluation.

GME's verified API guide provides the missing IDA source: `PublicMarketResults`,
`ME_ZonalPrices`, segments `MI-A1`, `MI-A2` and `MI-A3`, followed by selection of
the NORD zone. The API uses a GME-issued user ID and password, distinct from
ENTSO-E tokens and Terna Key/Secret credentials. GME's public market-results
pages also expose download interfaces. The new acquisition/parser adapter uses
DuckDB snapshots and is described in [GME acquisition](gme_api.md). No retrieved
DA dataset supplies these IDA targets.

`forecasting/prices.py` provides NumPy Ridge with training-only centering/scaling, a DA level model and IDA spreads relative to the previous auction. The pooled product regression uses cyclic hour/weekday/month features and one-day/seven-day same-clock price lags. Joins use local date, clock and repeated-hour fold, rather than shifting 96 rows across DST. Previous-auction values enter only after their publication at the row's decision cutoff. Unmatched/missing lags stay missing and their rows are omitted, with counts. This causal interface omits regional fundamentals and local-to-regional solar proxies. Ridge alpha is configured, not tuned on validation data.

`forecasting/fundamental_experiment.py` adds a separate retrospective comparison
of those price/calendar features with ENTSO-E regional load, solar and onshore
wind forecasts. It requires complete interval coverage, duration-averages MW
into each native price product, freezes the joined inputs and evaluates all
models on identical chronological holdout rows. The January prototype trains
on 8–24 January 2025 and evaluates 25–31 January. Retrieved forecast revisions
do not establish historical availability; this prototype does not promote
unverified fundamentals into causal trading inputs or residual scenario fits.

The strict causal interface obtains residuals from expanding daily rolling-origin fits whose labels were available by each fold's cutoff. The final model is fitted by an explicit cutoff. Whole-vector Gaussian sampling uses shrunk temporal covariance; the independent Gaussian mode is explicitly named. Bootstrap draws complete residual trajectories. Equal-length calibration excludes differently sized DST product days unless calibrated separately. These causal algorithms have synthetic-fixture evidence. Separately, the cached actual DA prices now support retrospective empirical studies, with hourly and quarter-hour regimes evaluated individually; original historical publication vintages remain unverified. The executable synthetic CSV fixture described below supplies demonstration IDA models, not real IDA observations.

`scenarios.py` builds a finite tree at actual publication, bid and pre-interval control events. DA prices are levels and each IDA is the preceding price plus its conditional spread and equation-specific innovation. There is no added second cross-auction residual. Each observed DA result retains all unresolved IDA descendants. Known results remain fixed on reoptimization; their publication must precede its cutoff. Controls and bid ownership follow observable histories, and latent future branch labels never enter earlier owners. Current weather is one frozen deterministic vintage. The synthetic residual is a whole-day level offset with explicit branch masses. The fitted interface checks model/calibration/feature cutoffs and uses eligible Ridge models with per-auction OOS distributions; conditional innovations across auction equations are an explicit independence approximation.

The appropriate real-input mode is **stochastic-price / deterministic-PV**. Imbalance prices in these interfaces are explicit fixed point estimates, not future realized historical values. Their economic validity for Italy remains unresolved. Real-market evaluation remains blocked until actual auction labels, recorded/justified historical availability, sufficient OOS history, and verified settlement economics are supplied. Passing fixture tests is evidence of algorithm behavior, not calibrated historical forecasting skill.

## Rolling linear models and Gaussian paths

`study-linear-prices` uses `forecasting/linear_price_study.py`,
`price_features.py`, `price_pipeline.py` and `gaussian_prices.py`. The feature
builder adds exact same-clock lags, complete prior-day summaries and price
profiles, consecutive rolling history, calendar categories/interactions and
aligned load/solar/wind forecasts. Holiday dates must be explicitly configured.
Residual load means forecast load minus forecast solar and wind; it is kept
separate from a model using all three constituent predictors. Historical actual
load/generation and Terna MSD wind are not substituted for DA forecasts.

Candidates predict the price level or its difference from yesterday's same-slot
price. Feature sets, ridge strengths and rolling training windows are selected
using validation MAE only. Baselines can win that selection. The selected
specification is then fixed; coefficients may be refitted weekly using strictly
earlier delivery days. A later calibration period supplies chronological
out-of-sample errors. A delivery-maturity gap separates calibration from the
final test forecast issuance. The final test is excluded from model selection.
The default static uncertainty fit also excludes all final-test errors. The
separate adaptive Gaussian study fixes its window/shrinkage settings using
inner calibration days, then may update bias and covariance from complete
earlier test days matured before each new issuance cutoff. These online updates
do not reselect hyperparameters using test outcomes. Some broad-study test dates overlap earlier simple
price experiments, so this is not a fresh prospective validation claim.

For product vector \(p_d\), the sampled trajectory is
\(p_d^{(s)}=\hat p_d+b+\epsilon_d^{(s)}\), where
\(\epsilon_d^{(s)}\sim N(0,\Sigma)\). The static fit estimates slot bias \(b\)
and covariance only on calibration errors. The adaptive fit uses its frozen
policy and earlier complete mature errors; a D-1 noon issuance excludes the
entire D-1 delivery day's errors.
The covariance mixes the centered error sample covariance with its diagonal,
plus a declared variance floor. Raw price variance includes predictable price
variation and is not the noise variance. Bias correction and zero-mean noise
remain distinct in the saved artifacts. Samples preserve negative prices.

Native durations and complete civil-day grids, including repeated-hour folds,
define separate calibrations. Unsupported DST layouts are reported rather than
interpolated. Final-test scores include MAE/RMSE, marginal coverage and interval
widths, CRPS, Gaussian likelihood, and consecutive-product ramp uncertainty.
These diagnostics can reveal poor Gaussian tail calibration; passing code tests
alone does not establish forecasting reliability.

`gaussian_price_tree` returns the existing `ScenarioBundle` and `InformationTree`
used by the optimizer, with shared bids before publication and physical recourse
after prices become observable. It accepts fixed absolute means or
`conditional_means` callbacks that receive read-only sampled/known ancestor
clearing curves. A callback can return the preceding auction level plus a
linear spread; this propagates dependence between auctions while innovations
remain independent across equations and temporally correlated within each
auction. `fitted_price_tree` remains available for its earlier causal model
interface. A DA-only schedule needs no invented IDA model. Calibrations are
explicitly scoped to their auction. IDA3 uses its actual 48 quarter-hour products
from local noon to midnight; inert rectangular-array placeholders outside its
trading mask are not observed prices or calibration labels. The adapter checks
model, feature and calibration cutoffs and rejects unverified historical
vintages unless `allow_retrospective=True` is explicitly supplied.

`forecasting/intraday_price_features.py` reads native GME IDA labels and the
explicitly selected DA source from DuckDB. It constructs IDA1-minus-DA,
IDA2-minus-IDA1 and IDA3-minus-IDA2 targets with own-history lags and available
earlier-auction prices. `build_future_intraday_price_features` needs no own
future target; `ConditionalSpreadPriceForecast.predict_absolute` rebuilds its
ancestor-price features for each scenario branch. The study's assumed auction
result clocks remain separate from unknown source publication timestamps, and
fitting requires `allow_assumed_availability=True`. Historical hourly DA prices
may enter a quarter-hour IDA regression only as explicitly marked containing-hour
features; the observed IDA targets retain native quarter-hour boundaries. These
interfaces are fixture-tested and remain operationally ineligible pending actual
GME history and verified historical availability.

## Frozen September–October check

`configs/price_forward_check.yaml` references a saved protocol that fixes the
selected by-hour linear specification, seven-day coefficient refit schedule and
42-day Gaussian window with 0.3 shrinkage before examining the additional prices.
The additional period is 1 September through 16 October 2026; the download end
`2026-10-17` is exclusive. Run the price-only download, then stop acquisition
before opening its DuckDB database for the check:

```bash
PYTHONPATH=src python -m bess_pv_trading download-data --config configs/entsoe_price_extension.yaml --interactive
PYTHONPATH=src python -m bess_pv_trading check-frozen-prices --config configs/price_forward_check.yaml
```

The check reads completed, checksum-verified `data/api/market.duckdb` snapshots
and writes a new run under `outputs/price_forward_checks/`. Incomplete requested
coverage produces a waiting status, rather than a scored partial holdout. The
protocol retains source-study hashes, the previously inspected cutoff and
baseline definitions. Coefficient refits use strictly earlier eligible labels;
Gaussian updates use earlier complete mature out-of-sample errors. The new
period selects neither model hyperparameters nor covariance settings. This
adds a frozen evaluation period while historical publication/vintage evidence
remains unverified; it does not establish executable historical trading.

For a future day whose target is unknown,
`build_future_linear_price_features(history_native, future_products, ...)`
requires strictly earlier observed price history and a complete future native
grid. Aligned forecast columns may accompany the future products. It rejects
actual-price target columns and leaves unavailable history missing. Use the
saved feature settings with `LinearPriceForecast.load(...).predict(features)`;
then supply those means, feature availability and the point-model fit cutoff
to the Gaussian tree adapter. This interface does not establish missing source
publication vintages. The final saved point model belongs to its own most
recent refit; earlier holdout dates must use their recorded predictions or
appropriately dated refit artifacts.

`forecasting/price_oracle.py:build_perfect_foresight_scenarios` provides the
requested perfect-price-information benchmark. Each PV member receives the
same actual clearing-price trajectory, with no Gaussian price sampling.
Members and weights remain explicit, and member identity is not revealed as
extra information. Bids and battery commands remain shared until completed PV
observations distinguish histories. Configurable delayed observations keep
decisions shared longer. The joint benchmark currently has one PV member;
unit and solver fixtures also exercise multiple members.

Actual price tables must match native product boundaries, and every tradable
auction requires its own actual prices. Missing IDA prices are not replaced by
DA prices. This is a non-executable oracle, intentionally using future actual
prices as objective coefficients; it never certifies them as published before
the decision cutoff. For multiple PV members, automatic curtailment can react
at interval end while battery controls remain pre-interval. Its value is a
price-information upper bound only for policies evaluated on the same prices,
PV members, probabilities, physical constraints, observation and curtailment
conventions, and settlement inputs. No real-farm profit claim follows from the
synthetic solver fixtures.

The first January forecast comparison held out 25–31 January 2025. Prices
averaged EUR 140.38/MWh (range 109.22–184.48). The load/solar/wind ridge model had
MAE 9.64/MWh and maximum absolute error 33.50/MWh; yesterday's same-hour benchmark
had MAE 8.86/MWh. MAE is the average absolute error, not the worst error. One week
does not establish stable superiority or a calibrated scenario model.

Runnable examples (existing `bess` environment, no installation):

```bash
PYTHONPATH=src python -m bess_pv_trading import-market-data --config configs/italy_nord.example.yaml
PYTHONPATH=src python -m bess_pv_trading train-price-models --config configs/italy_nord.example.yaml
PYTHONPATH=src python -m pytest -q tests/unit/test_markets.py tests/unit/test_prices.py
```

The first two commands intentionally give actionable missing-input errors until actual manual price files exist. See [sources.md](sources.md) for consulted official references and exact supported claims.


A runnable import/training demonstration is provided in `configs/synthetic_prices.yaml` and `tests/fixtures/synthetic_market_prices.csv`. It contains 288 explicitly synthetic records: 18 days, four auctions, four selected products per auction. It is deliberately sparse and the full requested-day coverage report identifies 5,760 missing products. These gaps are reported, never filled, and this fixture is not a full-day market backtest. Regenerate with `PYTHONPATH=src python tests/fixtures/generate_market_fixture.py` (seed 29).

```bash
PYTHONPATH=src python -m bess_pv_trading import-market-data --config configs/synthetic_prices.yaml
PYTHONPATH=src python -m bess_pv_trading train-price-models --config configs/synthetic_prices.yaml
```

Both commands were executed successfully in the existing bess environment. Artifacts are grouped under `outputs/market_price_fixture/`: canonical CSV and coverage JSON, cache request/retrieval manifests, four model JSON files, four out-of-sample residual CSVs, and `models/training_manifest.json`. Each demonstrated auction model uses 44 lag-complete training rows, omits 28 initial rows without week lags, and produces 28 rolling-origin OOS residual rows. Input provenance explicitly records `synthetic_fixture` and its synthetic dataset ID. This demonstrates execution and causal mechanics, not historical forecasting performance.

Equal observed sibling price vectors are combined into one outcome with summed probability **before** future draws are generated. The controller never sees residual-draw indices. A zero-scale residual tree therefore has one leaf; duplicate bootstrap draws cannot create artificial observable distinctions.


## Prepared-input initial optimization

`optimization/prepared.py:load_prepared_inputs` connects the exported PV CSV and a separately supplied point-price CSV to the shared optimizer. This route supports one farm/member, a complete ordinary/DST delivery grid, and an initial cutoff no later than the DA submission. Future planned DA/IDA decisions are not accepted trades. No historical replay is implied. The template `configs/prepared_optimization.example.yaml` leaves absent input paths and equipment values null; it fails with an actionable missing-input error until populated.

The PV CSV contract matches the actual forecast export: `farm_id,member_id,member_origin,member_probability,forecast_cycle_time,forecast_available_at,interval_start_utc,interval_end_utc,pv_ac_mw,input_eligibility,review_status`. Exactly one source vintage and member with probability one must cover the complete local delivery day. AC MW is preserved. Missing/negative PV, model-invalid intervals and review exclusions fail validation; source/availability clocks must precede the cutoff. An optional PV model fit cutoff is checked, or the configuration must explicitly acknowledge an unfitted PV-output assumption. Ex-post AC observation columns are not used as forecasts.

The price CSV requires `auction,product_start,price_eur_mwh,available_at,origin`, plus per-row `source_time,fit_cutoff` or explicit common provenance timestamps in the optimization configuration. Optional `product_end` and `zone` are validated. Each auction must cover exactly its eligible quarter-hour products; IDA3 supplies only the second half of the day. Every source, fit and availability timestamp must be eligible at the initial cutoff. Future realized/oracle labels are rejected; negative forecast prices remain valid. No missing price is filled. Ineligible morning IDA3 coefficients are predecessor placeholders behind a false trading mask, not invented observed products.

The resulting single-leaf information tree uses the supplied point trajectories with zero residual uncertainty and records origin as `supplied point-price / deterministic-PV`; it is not labeled synthetic merely because a common zero-innovation tree builder is reused. Signed imbalance settlement requires an explicit configured point price, prior availability and `study_assumption` status. Configured plant/BESS/POI values, known delivery-start energy, commercial bounds, imbalance penalty and throughput aging assumptions are required. This is a deterministic integrated planning interface with supplied economics, not a validated Italian historical settlement adapter.
