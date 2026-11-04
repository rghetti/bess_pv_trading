# Regularized IDA neural search

Twenty-four CPU MLP fits cover IDA1–3, four two-layer architectures and
regularization settings, and direct-price versus spread-to-cleared-DA targets.
The models minimize squared error on standardized targets, with standardized
inputs and early stopping on a random 15% split within the historical training
set. They train through January 2026 and evaluate February–April 2026. Exact
features and hyperparameters are in
`configs/research/price_model_search_ida_regularized_mlp.yaml`.

| Auction | Products | Best MLP WAPE | Best MLP MAE EUR/MWh | Best earlier Extra Trees WAPE | Best earlier Extra Trees MAE EUR/MWh |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 8,444 | 5.876% | 7.483 | 4.741% | 6.037 |
| IDA2 | 8,252 | 6.160% | 7.903 | 5.477% | 7.027 |
| IDA3 | 4,128 | 10.647% | 13.382 | 9.411% | 11.828 |

The saved prediction files for each pair have identical timestamps and actual
prices. All 24 neural fits completed without recorded failures or convergence
warnings. Each minimum was selected on this same validation period, which has
also informed other studies; these scores are exploratory, not unbiased final
model performance. Publication timing and historical forecast vintage assumptions
remain in the input dataset. This finite neural search does not establish that
other neural methods cannot outperform the references.
