# Staged auction-price feature search

This workflow screens features and model families for Italian NORD DA, IDA1,
IDA2 and IDA3 prices. It uses the existing native quarter-hour product,
calendar, earlier-auction and cancellation-aware spread definitions. It does
not modify the battery simulation. Run the data downloader to completion before
preparing a fresh snapshot: DuckDB is shared and must not have another writer.

## Study design

- The annual training window is 1 October 2025 through 30 September 2026.
  The latest 20% of eligible training days forms the inner selection tail;
  earlier eligible days fit the candidates. October and November 2026 are
  reserved for subsequent development validation and test, respectively.
- The mandatory core comprises price history, price curves, calendar and
  holiday features, earlier cleared auction prices and cancellation flags.
  IDAs predict a spread to the latest earlier auction that cleared for the
  same delivery product; the clearing price is added back for scoring.
- Four external blocks contain NORD load/generation forecasts, gas, French
  previous-day quantities and Swiss previous-day quantities. Historical
  availability of the previous-day data is an explicit research assumption.
  The Italian–Swiss DA spread and other target-derived price features are
  excluded.
- NORD inputs are target-interval demand, total generation, solar and wind
  forecasts. Gas inputs are the PSV index at D-1 and D-7. French inputs are
  previous-day nuclear and hydro generation, France–Italy commercial
  exchange, and the source balance `value`. Swiss inputs are previous-day
  generation, consumption and physical exports to Italy and France, plus the
  reservoir level lagged by one week.
- The screen compares core alone and core plus each individual block for the
  nine existing model families. Up to four reproducible parameter settings
  are considered per family and exact feature set. The four leading families
  per auction advance to the block-combination and individual-feature
  ablation stages. Main comparisons use the common rows complete across the
  available previous-day blocks, even when a candidate omits one of them.
  Coverage reports show the missing-data impact of individual features and
  block combinations relative to the core-only rows.
- After the main winner is selected, a separate retrospective comparison adds
  same-delivery-day realized quantities on matched rows. These values can be
  observed after auction gate closure; the resulting model is a retrospective
  proxy, not a gate-available forecast. The candidate pool consists of Swiss
  generation, consumption and physical exports to Italy and France, plus
  French physical exchanges, nuclear, hydro and total generation, and observed
  consumption. Entirely unavailable quantities are recorded and omitted.

The search optimizes the inner selection score. Its winning inner score is
selection-dependent, and is not an independent October or November test.
Training losses differ by model family; scores use MAE, WAPE, RMSE and error
relative to the existing price reference. WAPE is 100 times the sum of
absolute errors divided by the sum of absolute actual prices.

The nine families are Ridge, Elastic Net, random forest, Extra Trees,
histogram gradient boosting, LightGBM, CatBoost, XGBoost and a feedforward
MLP. Their objectives and sampling distributions are specified in the
[feature and model protocol](price_search_protocol_and_features.md#model-families-and-searched-hyperparameters);
the staged workflow uses four seeded draws per product and family, shared
across all feature sets, rather than the 20 draws described for the old search.
Linear models and the MLP fit their scaling on training rows only. No price
calibration or probabilistic scenarios are performed. The other sections of
that protocol describe the exact base features and external fields; its old
training dates and retrospective commercial block do not apply here.

With all four blocks and all four auctions eligible, the stages schedule at
most about 1,300 fit requests before checkpoint reuse: 720 in screening, up to
256 combinations, up to 240 single-feature ablations, up to 64 refinement fits
and eight paired retrospective fits. Some requests are already checkpointed by
earlier stages. Actual counts depend on source coverage and selected features.
The annual export is a separate serial job of up to 12 fits.

## Run on this machine

Use a new output directory and wait for the downloader to stop before preparing
the snapshot. Preparation reads the catalog but does not fit models. It freezes
the source selections, feature arrays and provenance for the search; later
catalog updates do not change that snapshot. Its configured source window ends
on 1 January 2027 so that December 2026 can be included when it becomes
available; the October validation and November test boundaries are unchanged.

```bash
cd /home/rghetti/bess_pv_trading
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.search_price_staged \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --output outputs/price_staged_search_full_year --workers 30 --prepare-only
```

Inspect the preparation output and coverage before launching the search in
tmux. The run uses CPU workers and caps native numerical-library threads at
one per worker. Change `--workers` for the available shared-machine capacity.

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.search_price_staged \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --output outputs/price_staged_search_full_year --workers 30 --run
```

The search checkpoints completed fits. Resume with the same command and output
directory; interrupted fits may need to run again. If a fit has failed, inspect
the failure and add `--retry-failed` to retry it. The stages write JSON results
and per-auction CSV tables and figures under `reports/`. Preserve the old search
directory separately: legacy fits are not automatically transferred to the new
snapshot. Its status file may still say `running` after an interruption. The
matrix reader now supports an optional later export horizon, changing its code
fingerprint; resuming the old search requires its original code revision and
environment, even though its checkpoint files are unchanged.

After the staged search finishes, explicitly refit the selected variants:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.refit_staged_price_proxies \
  --search outputs/price_staged_search_full_year \
  --output outputs/price_staged_proxies_2026 --run
```

The export fits up to three models per auction: the main winner using every
eligible selected-feature row, the retrospective variant, and a main-feature
comparator fitted on exactly the retrospective variant's rows. The matched
comparison is descriptive and in sample. The annual refit uses eligible rows
from the complete annual training window.
Saved predictions label rows used for fitting. The prepared matrices contain
observed target products, so price files cover usable observed rows only;
they do not manufacture prices for cancelled auctions or unavailable future
products. Daily coverage files identify missing and partial days. These files
are not yet a drop-in forecast provider for the existing MPC simulator.
