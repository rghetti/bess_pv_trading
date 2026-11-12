# October 2026 intraday price diagnostic

The inputs are the completed annual refits in `outputs/price_staged_proxies_2026`.
October rows were not used for training. Results use the saved snapshot and its
actual earlier-cleared auction prices; availability assumptions and cancellation
fallbacks remain those of the staged study. Reproduce the descriptive analysis
with `python -m scripts.analyze_october_intraday_prices` in the bess environment.
Tables and hour plots are written to `outputs/price_october_intraday_analysis`.

IDA2 covers 2,788 quarter-hours on 29 days, with MAE 10.74 EUR/MWh and WAPE
4.55%; the preceding-price reference has MAE 10.78 EUR/MWh on the same rows.
The refitted elastic net predicts a constant 0.811 EUR/MWh spread over that
reference for all October observations. Its worst day is 25 October, MAE 28.01.
October 26 and 27 have targets and anchors but their saved matrices are missing
the historical 02:00 curve from the autumn clock change.

IDA3 covers 1,488 quarter-hours on 31 days, with MAE 14.97 EUR/MWh and WAPE
6.15%; reference MAE is 14.59 EUR/MWh. Mean signed error is -5.72 EUR/MWh.
The model predicts a mean spread of -1.77 EUR/MWh, with standard deviation 2.26.
Mean hourly MAE is highest at 12:00 (18.49 EUR/MWh versus 18.12 for the
reference) and lowest at 23:00 (8.20 versus 7.59). Sundays have MAE 21.80,
reference 21.32, and bias -17.44 over four days. October 25 has MAE 48.71.
For the highest actual-price quartile, bias is -11.27 and MAE 18.08 versus
17.13 for the reference. These subgroups are descriptive and small.

For IDA3, when the observed price exceeds the reference by 20–50 EUR/MWh,
model MAE is 31.74 versus 29.94 for the reference (203 products); when it is
20–50 below, model MAE is 27.11 versus 29.92 (128 products). These movement
groups use the realized price and therefore are diagnostics, not features.
Patterns suggest the selected spread models struggle with upward moves and that
the IDA3 negative correction hurts on Sundays; they do not establish a cause.

The curve builder handles the autumn repeated hour as the mean of both complete
sets of products, requiring the latest publication of those products. For the
nonexistent spring hour it interpolates the adjacent complete local-hour means,
requiring both to be published. Other incomplete hours remain missing. This
changes the feature implementation and its fingerprint; existing models and
frozen matrices keep their original values and cannot silently be resumed as if
the inputs matched. Regenerating compatible matrices and predictions requires a
separate user-run workflow; the saved October scores above remain unchanged.
