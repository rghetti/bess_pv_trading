# DA nearest-neighbor price analogues

This CPU search compared standardized nearest-neighbor predictions against an
interaction-ridge reference for quarter-hourly DA prices. Features comprise
recent DA price curves and fundamentals, with and without previous IDA prices.
Analogue candidates average 25, 100 or 400 historical products using distance
or uniform weights. Training ends in January 2026; candidate ranking covers the
previously examined February–April period. Exact features, standardization and
parameters are recorded in `price_model_search_da_previous_idas_analog.yaml`
and its manifest. Historical forecast and publication timing are partly assumed.

With previous IDA features, the best analogue scored 15.743 EUR/MWh MAE and
12.282% WAPE; interaction ridge scored 12.121 EUR/MWh and 9.456%. Both cover
the same 7,868 products. On the 2,296 products in local hours 10:00–16:59,
the analogue scored 24.286 EUR/MWh and 22.334% WAPE versus 17.474 EUR/MWh
and 16.069% for ridge. The saved predictions have identical timestamps and
actual prices. All ten fits completed without recorded failures. Because the
analogue is weaker overall and on these daytime products, this grid was not
promoted to a later rolling evaluation. This does not rule out other distance
metrics, feature weighting or analogue designs.
