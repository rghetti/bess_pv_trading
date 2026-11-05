# Prior-day Swiss border context for DA prices

This exploratory CPU comparison adds four 24-hour-shifted Swissgrid source
fields to DA price curves: the commercial Swiss–Italy schedule, intraday net
transfer capacity in each direction, and the Italian-NORD versus Swiss spot
price spread. The schedule is a commercial measure, not measured physical
flow. Source-day midnight publication is a retrospective placeholder; actual
historical release and revisions are unverified. A fixed elapsed 24-hour shift
does not always preserve local clock time through daylight-saving transitions.

The 2026 border archive produces a short training period: both arms use 4,237
training products and 4,992 matched March–April selection products across 52
complete days. Five linear and Extra Trees candidates per arm completed with
no recorded failures or convergence warnings. Both arms' best candidate was
the same Elastic Net.

| Features | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| Price curves only | 17.822 | 13.422% |
| Price curves and Swiss border context | 18.313 | 13.792% |

The paired MAE gain of the border model was -0.491 EUR/MWh. A 5,000-resample
independent-delivery-day bootstrap with seed 173 gives a 95% percentile
interval of -0.760 to -0.199 EUR/MWh. It does not account for temporal
dependence, model selection or repeated inspection, and does not establish
general uselessness of cross-border data. Two of the five matched candidate
pairs had lower selection MAE with border inputs, but neither beat the best
price-only model. The predeclared later replay was therefore skipped. Input
snapshot hashes, candidates, common-day exclusions and predictions are in
`outputs/energy_price_forecasts/nord_da_2026_swiss_border_price_early_v1/`.
