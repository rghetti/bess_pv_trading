# Regularized DA neural search

Eight CPU MLP fits were evaluated for native quarter-hourly DA prices from
February through April 2026, after training through January. The four two-layer
architectures and regularization settings are defined in
`configs/research/price_model_search_da_regularized_mlp.yaml`. Inputs and the
target are standardized using training data. Training minimizes squared error;
early stopping uses a random 15% split within the historical training set, with
15 iterations of patience. Reported scores are in EUR/MWh and WAPE rather than
the training objective. One feature set includes previous IDA prices, while the
other includes DA lagged prices, daily curves and fundamentals.

The best MLP used the DA-curve fundamentals without previous IDA inputs and
reached 10.098% WAPE and 12.944 EUR/MWh MAE. The matched interaction-ridge
reference using the same features reached 9.638% and 12.354 EUR/MWh. The saved
prediction files have identical timestamps and actuals for all 7,868 products.
The search recorded no failures or convergence warnings. This is a comparison of
specific configurations on a previously examined validation period, not proof
that neural models cannot improve the results. Some historical publication times
and source forecast vintages remain assumptions.
