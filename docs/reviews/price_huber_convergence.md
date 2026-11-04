# Robust price-regression convergence

The first Huber searches used standardized predictors and unscaled targets.
The day-ahead search completed 56 fits, but its log records iteration-limit
warnings from the Huber optimizer. Its Huber scores should therefore be
treated as provisional, not as converged model comparisons. A subsequent
single day-ahead interaction-Huber probe standardized the target and increased
the limit to 1,000 iterations; it still emitted a convergence warning. No full
target-scaled Huber search has been launched on the strength of that probe.

The first IDA Huber search completed 144 fits. Its part of the shared job log
does not show a convergence warning, although the old worker did not attach
warnings to individual candidate records. On its shared validation sample,
the lowest Huber WAPE was 4.89% for IDA1, 5.48% for IDA2 and 9.18% for IDA3.
The lowest Lasso WAPE in the same search was 4.74%, 5.49% and 9.18%
respectively. These are minimum scores chosen on the evaluation period, not
independent test results; availability clocks are assumed and retrospective.

The current worker saves convergence warnings per candidate and excludes
warned candidates from subsequent ranking. Earlier manifests lack that
per-candidate evidence, so absence of a warning field in an old result does
not certify convergence. Source results and logs are in
`outputs/energy_price_forecasts/nord_da_2026_huber_search_v1/`,
`outputs/energy_price_forecasts/nord_ida_2026_huber_search_v1/`, and the
local job log `/tmp/queue_price_huber.log`.
