# December 2025: retrospective forecast versus perfect auction prices

Completed 1–31 December 2025 (Europe/Rome), four farms, nine profiles per arm: 72 independent continuous trajectories. Differences are forecast minus oracle.

The forecast model is a frozen ridge model per auction with fit cutoff 30 September 2025, with causally updated price lags. The perfect-price arm knows actual auction curves in planning; both arms use actual settlements and independent continuous trajectories.

| Farm | Profile | Forecast profit EUR | Oracle profit EUR | Difference EUR | Forecast final SOC % | Oracle final SOC % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | DA | 1601.85 | 1758.54 | -156.69 | 7.00 | 7.00 |
| 1 | DA+IDA1 | 2279.58 | 5468.78 | -3189.20 | 7.00 | 7.00 |
| 1 | DA+IDA1+IDA2 | 2130.73 | 7848.02 | -5717.29 | 7.00 | 7.00 |
| 1 | DA+IDA1+IDA2+IDA3 | 2640.69 | 10115.24 | -7474.55 | 7.00 | 22.81 |
| 1 | DA+IDA1+IDA3 | 2498.39 | 7503.31 | -5004.92 | 7.00 | 49.94 |
| 1 | DA+IDA2 | 2083.72 | 6181.52 | -4097.80 | 9.65 | 7.00 |
| 1 | DA+IDA2+IDA3 | 2265.09 | 8011.70 | -5746.61 | 7.00 | 22.81 |
| 1 | DA+IDA3 | 2074.18 | 4708.98 | -2634.80 | 7.00 | 22.81 |
| 1 | sequential_all_auctions | 1720.44 | 1954.43 | -233.99 | 7.00 | 39.96 |
| 345 | DA | 5027.26 | 5134.96 | -107.70 | 7.27 | 7.27 |
| 345 | DA+IDA1 | 5881.96 | 10870.55 | -4988.58 | 7.26 | 7.26 |
| 345 | DA+IDA1+IDA2 | 5587.26 | 14690.33 | -9103.07 | 7.27 | 7.27 |
| 345 | DA+IDA1+IDA2+IDA3 | 6350.90 | 18298.10 | -11947.19 | 7.30 | 19.16 |
| 345 | DA+IDA1+IDA3 | 6234.31 | 14120.37 | -7886.06 | 7.30 | 29.17 |
| 345 | DA+IDA2 | 5531.86 | 11952.04 | -6420.17 | 7.67 | 7.27 |
| 345 | DA+IDA2+IDA3 | 5794.45 | 14971.94 | -9177.49 | 7.30 | 19.16 |
| 345 | DA+IDA3 | 5524.74 | 9700.07 | -4175.33 | 7.30 | 24.10 |
| 345 | sequential_all_auctions | 4898.99 | 5122.85 | -223.86 | 7.30 | 38.15 |
| 901 | DA | 13747.73 | 14023.06 | -275.33 | 35.33 | 35.33 |
| 901 | DA+IDA1 | 15373.69 | 29417.31 | -14043.62 | 86.08 | 86.08 |
| 901 | DA+IDA1+IDA2 | 15396.51 | 40469.43 | -25072.92 | 69.30 | 74.19 |
| 901 | DA+IDA1+IDA2+IDA3 | 17504.86 | 50347.80 | -32842.94 | 89.88 | 93.00 |
| 901 | DA+IDA1+IDA3 | 16611.72 | 38664.95 | -22053.23 | 80.57 | 68.72 |
| 901 | DA+IDA2 | 15321.05 | 32944.41 | -17623.36 | 69.60 | 72.15 |
| 901 | DA+IDA2+IDA3 | 15905.32 | 41102.10 | -25196.78 | 89.88 | 93.00 |
| 901 | DA+IDA3 | 14977.72 | 26708.14 | -11730.42 | 89.88 | 91.25 |
| 901 | sequential_all_auctions | 13566.23 | 14429.13 | -862.90 | 89.88 | 93.00 |
| 910 | DA | 27699.51 | 28385.13 | -685.61 | 62.30 | 62.30 |
| 910 | DA+IDA1 | 31130.75 | 52732.41 | -21601.66 | 86.81 | 86.80 |
| 910 | DA+IDA1+IDA2 | 30017.59 | 69584.06 | -39566.47 | 55.42 | 59.24 |
| 910 | DA+IDA1+IDA2+IDA3 | 33396.72 | 85266.22 | -51869.50 | 87.90 | 84.94 |
| 910 | DA+IDA1+IDA3 | 32177.92 | 66658.11 | -34480.19 | 90.80 | 93.00 |
| 910 | DA+IDA2 | 29932.09 | 57944.41 | -28012.32 | 55.71 | 60.41 |
| 910 | DA+IDA2+IDA3 | 30680.41 | 70417.17 | -39736.75 | 82.40 | 84.04 |
| 910 | DA+IDA3 | 29753.33 | 47962.15 | -18208.83 | 90.80 | 93.00 |
| 910 | sequential_all_auctions | 26929.40 | 28222.00 | -1292.60 | 86.06 | 84.94 |

Net profit = auction revenue + signed imbalance settlement − degradation − import fees − transaction fees − charged imbalance penalty. Objective values and illustrative penalties are not realized cash. The configured charged imbalance penalty is zero.

## Farm 1: profit components

F = forecast, O = oracle, Δ = F − O. Monetary columns are EUR; absolute imbalance is MWh.

| Profile | Arm | Auction revenue | Settlement | Degradation | Import fee | Trading fee | Imbalance penalty | Absolute imbalance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DA | F | 4580.705 | -2869.354 | 109.503 | 0.000 | 0.000 | 0.000 | 22.471 |
| DA | O | 4746.920 | -2785.369 | 203.011 | 0.000 | 0.000 | 0.000 | 22.469 |
| DA | Δ | -166.215 | -83.985 | -93.508 | 0.000 | 0.000 | 0.000 | 0.002 |
| DA+IDA1 | F | 4979.870 | -2646.301 | 53.987 | 0.000 | 0.000 | 0.000 | 21.708 |
| DA+IDA1 | O | 8203.103 | -2630.579 | 103.742 | 0.000 | 0.000 | 0.000 | 21.725 |
| DA+IDA1 | Δ | -3223.233 | -15.723 | -49.755 | 0.000 | 0.000 | 0.000 | -0.017 |
| DA+IDA1+IDA2 | F | 4750.753 | -2547.875 | 72.143 | 0.000 | 0.000 | 0.000 | 20.980 |
| DA+IDA1+IDA2 | O | 10549.559 | -2565.770 | 135.765 | 0.000 | 0.000 | 0.000 | 20.951 |
| DA+IDA1+IDA2 | Δ | -5798.806 | 17.895 | -63.622 | 0.000 | 0.000 | 0.000 | 0.028 |
| DA+IDA1+IDA2+IDA3 | F | 5217.458 | -2494.491 | 82.278 | 0.000 | 0.000 | 0.000 | 20.361 |
| DA+IDA1+IDA2+IDA3 | O | 12766.232 | -2505.350 | 145.644 | 0.000 | 0.000 | 0.000 | 20.392 |
| DA+IDA1+IDA2+IDA3 | Δ | -7548.775 | 10.860 | -63.366 | 0.000 | 0.000 | 0.000 | -0.031 |
| DA+IDA1+IDA3 | F | 5028.704 | -2474.096 | 56.214 | 0.000 | 0.000 | 0.000 | 20.184 |
| DA+IDA1+IDA3 | O | 10085.082 | -2441.366 | 140.405 | 0.000 | 0.000 | 0.000 | 20.406 |
| DA+IDA1+IDA3 | Δ | -5056.378 | -32.730 | -84.191 | 0.000 | 0.000 | 0.000 | -0.222 |
| DA+IDA2 | F | 4719.189 | -2578.191 | 57.273 | 0.000 | 0.000 | 0.000 | 20.931 |
| DA+IDA2 | O | 8850.227 | -2549.854 | 118.850 | 0.000 | 0.000 | 0.000 | 20.936 |
| DA+IDA2 | Δ | -4131.039 | -28.337 | -61.577 | 0.000 | 0.000 | 0.000 | -0.005 |
| DA+IDA2+IDA3 | F | 4831.011 | -2496.829 | 69.095 | 0.000 | 0.000 | 0.000 | 20.304 |
| DA+IDA2+IDA3 | O | 10658.486 | -2500.403 | 146.383 | 0.000 | 0.000 | 0.000 | 20.351 |
| DA+IDA2+IDA3 | Δ | -5827.475 | 3.574 | -77.287 | 0.000 | 0.000 | 0.000 | -0.047 |
| DA+IDA3 | F | 4696.485 | -2508.023 | 114.284 | 0.000 | 0.000 | 0.000 | 20.386 |
| DA+IDA3 | O | 7378.847 | -2487.833 | 182.033 | 0.000 | 0.000 | 0.000 | 20.369 |
| DA+IDA3 | Δ | -2682.362 | -20.190 | -67.749 | 0.000 | 0.000 | 0.000 | 0.016 |
| sequential_all_auctions | F | 4365.263 | -2487.197 | 157.621 | 0.000 | 0.000 | 0.000 | 20.332 |
| sequential_all_auctions | O | 4713.576 | -2544.980 | 214.162 | 0.000 | 0.000 | 0.000 | 20.791 |
| sequential_all_auctions | Δ | -348.314 | 57.783 | -56.542 | 0.000 | 0.000 | 0.000 | -0.460 |

| Profile | Forecast fallbacks | Oracle fallbacks | Forecast final energy MWh | Oracle final energy MWh |
| --- | ---: | ---: | ---: | ---: |
| DA | 0 | 0 | 0.052500 | 0.052500 |
| DA+IDA1 | 0 | 0 | 0.052500 | 0.052500 |
| DA+IDA1+IDA2 | 0 | 0 | 0.052500 | 0.052500 |
| DA+IDA1+IDA2+IDA3 | 0 | 0 | 0.052500 | 0.171085 |
| DA+IDA1+IDA3 | 0 | 0 | 0.052500 | 0.374585 |
| DA+IDA2 | 0 | 0 | 0.072376 | 0.052500 |
| DA+IDA2+IDA3 | 0 | 0 | 0.052500 | 0.171085 |
| DA+IDA3 | 0 | 0 | 0.052500 | 0.171085 |
| sequential_all_auctions | 0 | 0 | 0.052500 | 0.299674 |
## Farm 345: profit components

F = forecast, O = oracle, Δ = F − O. Monetary columns are EUR; absolute imbalance is MWh.

| Profile | Arm | Auction revenue | Settlement | Degradation | Import fee | Trading fee | Imbalance penalty | Absolute imbalance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DA | F | 11647.202 | -6453.733 | 166.212 | 0.000 | 0.000 | 0.000 | 53.392 |
| DA | O | 11877.253 | -6465.810 | 276.486 | 0.000 | 0.000 | 0.000 | 53.624 |
| DA | Δ | -230.051 | 12.078 | -110.274 | 0.000 | 0.000 | 0.000 | -0.231 |
| DA+IDA1 | F | 12397.212 | -6398.261 | 116.987 | 0.000 | 0.000 | 0.000 | 52.977 |
| DA+IDA1 | O | 17417.034 | -6368.067 | 178.421 | 0.000 | 0.000 | 0.000 | 53.050 |
| DA+IDA1 | Δ | -5019.822 | -30.194 | -61.435 | 0.000 | 0.000 | 0.000 | -0.073 |
| DA+IDA1+IDA2 | F | 11826.223 | -6097.252 | 141.713 | 0.000 | 0.000 | 0.000 | 51.387 |
| DA+IDA1+IDA2 | O | 21018.400 | -6120.167 | 207.902 | 0.000 | 0.000 | 0.000 | 51.397 |
| DA+IDA1+IDA2 | Δ | -9192.177 | 22.915 | -66.189 | 0.000 | 0.000 | 0.000 | -0.010 |
| DA+IDA1+IDA2+IDA3 | F | 12771.314 | -6271.909 | 148.501 | 0.000 | 0.000 | 0.000 | 51.511 |
| DA+IDA1+IDA2+IDA3 | O | 24829.545 | -6310.869 | 220.579 | 0.000 | 0.000 | 0.000 | 52.246 |
| DA+IDA1+IDA2+IDA3 | Δ | -12058.231 | 38.960 | -72.078 | 0.000 | 0.000 | 0.000 | -0.735 |
| DA+IDA1+IDA3 | F | 12635.474 | -6279.096 | 122.071 | 0.000 | 0.000 | 0.000 | 51.986 |
| DA+IDA1+IDA3 | O | 20672.126 | -6359.807 | 191.953 | 0.000 | 0.000 | 0.000 | 52.913 |
| DA+IDA1+IDA3 | Δ | -8036.652 | 80.711 | -69.883 | 0.000 | 0.000 | 0.000 | -0.928 |
| DA+IDA2 | F | 11795.311 | -6129.248 | 134.200 | 0.000 | 0.000 | 0.000 | 51.229 |
| DA+IDA2 | O | 18254.725 | -6107.954 | 194.733 | 0.000 | 0.000 | 0.000 | 51.228 |
| DA+IDA2 | Δ | -6459.414 | -21.294 | -60.533 | 0.000 | 0.000 | 0.000 | 0.001 |
| DA+IDA2+IDA3 | F | 12182.279 | -6242.802 | 145.023 | 0.000 | 0.000 | 0.000 | 51.393 |
| DA+IDA2+IDA3 | O | 21384.349 | -6189.697 | 222.708 | 0.000 | 0.000 | 0.000 | 51.733 |
| DA+IDA2+IDA3 | Δ | -9202.070 | -53.105 | -77.685 | 0.000 | 0.000 | 0.000 | -0.339 |
| DA+IDA3 | F | 12146.574 | -6446.543 | 175.289 | 0.000 | 0.000 | 0.000 | 52.972 |
| DA+IDA3 | O | 16344.764 | -6387.485 | 257.204 | 0.000 | 0.000 | 0.000 | 52.862 |
| DA+IDA3 | Δ | -4198.190 | -59.058 | -81.915 | 0.000 | 0.000 | 0.000 | 0.110 |
| sequential_all_auctions | F | 11418.123 | -6277.769 | 241.365 | 0.000 | 0.000 | 0.000 | 51.707 |
| sequential_all_auctions | O | 11848.713 | -6440.218 | 285.648 | 0.000 | 0.000 | 0.000 | 52.405 |
| sequential_all_auctions | Δ | -430.589 | 162.449 | -44.284 | 0.000 | 0.000 | 0.000 | -0.698 |

| Profile | Forecast fallbacks | Oracle fallbacks | Forecast final energy MWh | Oracle final energy MWh |
| --- | ---: | ---: | ---: | ---: |
| DA | 0 | 0 | 0.072723 | 0.072723 |
| DA+IDA1 | 0 | 0 | 0.072615 | 0.072615 |
| DA+IDA1+IDA2 | 0 | 0 | 0.072702 | 0.072702 |
| DA+IDA1+IDA2+IDA3 | 0 | 0 | 0.072974 | 0.191562 |
| DA+IDA1+IDA3 | 0 | 0 | 0.072974 | 0.291656 |
| DA+IDA2 | 0 | 0 | 0.076685 | 0.072721 |
| DA+IDA2+IDA3 | 0 | 0 | 0.072975 | 0.191562 |
| DA+IDA3 | 0 | 0 | 0.073011 | 0.240969 |
| sequential_all_auctions | 0 | 0 | 0.072974 | 0.381458 |
## Farm 901: profit components

F = forecast, O = oracle, Δ = F − O. Monetary columns are EUR; absolute imbalance is MWh.

| Profile | Arm | Auction revenue | Settlement | Degradation | Import fee | Trading fee | Imbalance penalty | Absolute imbalance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DA | F | 19276.755 | -4488.807 | 1040.214 | 0.000 | 0.000 | 0.000 | 33.983 |
| DA | O | 19584.791 | -4321.928 | 1239.803 | 0.000 | 0.000 | 0.000 | 33.388 |
| DA | Δ | -308.036 | -166.879 | -199.589 | 0.000 | 0.000 | 0.000 | 0.595 |
| DA+IDA1 | F | 21345.508 | -4857.329 | 1114.489 | 0.000 | 0.000 | 0.000 | 34.267 |
| DA+IDA1 | O | 35414.964 | -4715.901 | 1281.757 | 0.000 | 0.000 | 0.000 | 34.010 |
| DA+IDA1 | Δ | -14069.456 | -141.428 | -167.267 | 0.000 | 0.000 | 0.000 | 0.257 |
| DA+IDA1+IDA2 | F | 20526.179 | -4013.977 | 1115.688 | 0.000 | 0.000 | 0.000 | 29.429 |
| DA+IDA1+IDA2 | O | 45736.367 | -3999.153 | 1267.784 | 0.000 | 0.000 | 0.000 | 29.592 |
| DA+IDA1+IDA2 | Δ | -25210.188 | -14.824 | -152.097 | 0.000 | 0.000 | 0.000 | -0.163 |
| DA+IDA1+IDA2+IDA3 | F | 23006.095 | -4415.360 | 1085.874 | 0.000 | 0.000 | 0.000 | 30.463 |
| DA+IDA1+IDA2+IDA3 | O | 56021.228 | -4411.180 | 1262.249 | 0.000 | 0.000 | 0.000 | 32.120 |
| DA+IDA1+IDA2+IDA3 | Δ | -33015.134 | -4.180 | -176.375 | 0.000 | 0.000 | 0.000 | -1.656 |
| DA+IDA1+IDA3 | F | 22068.372 | -4304.178 | 1152.475 | 0.000 | 0.000 | 0.000 | 30.094 |
| DA+IDA1+IDA3 | O | 44348.814 | -4437.713 | 1246.152 | 0.000 | 0.000 | 0.000 | 31.978 |
| DA+IDA1+IDA3 | Δ | -22280.442 | 133.535 | -93.677 | 0.000 | 0.000 | 0.000 | -1.884 |
| DA+IDA2 | F | 20428.326 | -3995.686 | 1111.587 | 0.000 | 0.000 | 0.000 | 29.122 |
| DA+IDA2 | O | 38270.398 | -4068.472 | 1257.513 | 0.000 | 0.000 | 0.000 | 29.440 |
| DA+IDA2 | Δ | -17842.072 | 72.786 | -145.926 | 0.000 | 0.000 | 0.000 | -0.318 |
| DA+IDA2+IDA3 | F | 21435.811 | -4439.734 | 1090.756 | 0.000 | 0.000 | 0.000 | 31.085 |
| DA+IDA2+IDA3 | O | 46620.598 | -4303.195 | 1215.306 | 0.000 | 0.000 | 0.000 | 31.313 |
| DA+IDA2+IDA3 | Δ | -25184.786 | -136.539 | -124.550 | 0.000 | 0.000 | 0.000 | -0.228 |
| DA+IDA3 | F | 20636.310 | -4466.922 | 1191.666 | 0.000 | 0.000 | 0.000 | 32.036 |
| DA+IDA3 | O | 32271.353 | -4189.705 | 1373.510 | 0.000 | 0.000 | 0.000 | 31.606 |
| DA+IDA3 | Δ | -11635.043 | -277.217 | -181.844 | 0.000 | 0.000 | 0.000 | 0.430 |
| sequential_all_auctions | F | 19332.857 | -4554.815 | 1211.814 | 0.000 | 0.000 | 0.000 | 32.267 |
| sequential_all_auctions | O | 20333.678 | -4513.638 | 1390.914 | 0.000 | 0.000 | 0.000 | 32.557 |
| sequential_all_auctions | Δ | -1000.821 | -41.176 | -179.099 | 0.000 | 0.000 | 0.000 | -0.290 |

| Profile | Forecast fallbacks | Oracle fallbacks | Forecast final energy MWh | Oracle final energy MWh |
| --- | ---: | ---: | ---: | ---: |
| DA | 0 | 0 | 0.706625 | 0.706625 |
| DA+IDA1 | 0 | 0 | 1.721585 | 1.721585 |
| DA+IDA1+IDA2 | 0 | 0 | 1.385980 | 1.483788 |
| DA+IDA1+IDA2+IDA3 | 0 | 0 | 1.797661 | 1.860000 |
| DA+IDA1+IDA3 | 0 | 0 | 1.611489 | 1.374318 |
| DA+IDA2 | 0 | 0 | 1.391939 | 1.443025 |
| DA+IDA2+IDA3 | 0 | 0 | 1.797661 | 1.860000 |
| DA+IDA3 | 0 | 0 | 1.797661 | 1.824978 |
| sequential_all_auctions | 0 | 0 | 1.797661 | 1.860000 |
## Farm 910: profit components

F = forecast, O = oracle, Δ = F − O. Monetary columns are EUR; absolute imbalance is MWh.

| Profile | Arm | Auction revenue | Settlement | Degradation | Import fee | Trading fee | Imbalance penalty | Absolute imbalance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DA | F | 40750.733 | -11604.282 | 1446.937 | 0.000 | 0.000 | 0.000 | 90.773 |
| DA | O | 41664.846 | -11330.858 | 1948.860 | 0.000 | 0.000 | 0.000 | 90.891 |
| DA | Δ | -914.112 | -273.424 | -501.923 | 0.000 | 0.000 | 0.000 | -0.118 |
| DA+IDA1 | F | 43045.252 | -10557.016 | 1357.489 | 0.000 | 0.000 | 0.000 | 83.613 |
| DA+IDA1 | O | 65186.667 | -10643.373 | 1810.884 | 0.000 | 0.000 | 0.000 | 82.773 |
| DA+IDA1 | Δ | -22141.414 | 86.357 | -453.394 | 0.000 | 0.000 | 0.000 | 0.840 |
| DA+IDA1+IDA2 | F | 41693.244 | -10270.765 | 1404.891 | 0.000 | 0.000 | 0.000 | 78.311 |
| DA+IDA1+IDA2 | O | 81721.160 | -10462.707 | 1674.394 | 0.000 | 0.000 | 0.000 | 78.432 |
| DA+IDA1+IDA2 | Δ | -40027.916 | 191.942 | -269.504 | 0.000 | 0.000 | 0.000 | -0.121 |
| DA+IDA1+IDA2+IDA3 | F | 44882.879 | -10088.505 | 1397.653 | 0.000 | 0.000 | 0.000 | 77.374 |
| DA+IDA1+IDA2+IDA3 | O | 97294.678 | -10289.124 | 1739.336 | 0.000 | 0.000 | 0.000 | 77.862 |
| DA+IDA1+IDA2+IDA3 | Δ | -52411.799 | 200.619 | -341.683 | 0.000 | 0.000 | 0.000 | -0.487 |
| DA+IDA1+IDA3 | F | 44359.190 | -10833.231 | 1348.038 | 0.000 | 0.000 | 0.000 | 82.829 |
| DA+IDA1+IDA3 | O | 78794.186 | -10474.190 | 1661.890 | 0.000 | 0.000 | 0.000 | 81.113 |
| DA+IDA1+IDA3 | Δ | -34434.996 | -359.041 | -313.852 | 0.000 | 0.000 | 0.000 | 1.716 |
| DA+IDA2 | F | 41444.578 | -10063.717 | 1448.774 | 0.000 | 0.000 | 0.000 | 77.769 |
| DA+IDA2 | O | 69743.676 | -10100.437 | 1698.829 | 0.000 | 0.000 | 0.000 | 77.919 |
| DA+IDA2 | Δ | -28299.098 | 36.720 | -250.055 | 0.000 | 0.000 | 0.000 | -0.150 |
| DA+IDA2+IDA3 | F | 42284.429 | -10233.319 | 1370.698 | 0.000 | 0.000 | 0.000 | 77.591 |
| DA+IDA2+IDA3 | O | 82429.159 | -10307.673 | 1704.320 | 0.000 | 0.000 | 0.000 | 77.361 |
| DA+IDA2+IDA3 | Δ | -40144.731 | 74.354 | -333.622 | 0.000 | 0.000 | 0.000 | 0.230 |
| DA+IDA3 | F | 42436.259 | -11160.135 | 1522.796 | 0.000 | 0.000 | 0.000 | 87.487 |
| DA+IDA3 | O | 60432.903 | -10576.611 | 1894.137 | 0.000 | 0.000 | 0.000 | 83.944 |
| DA+IDA3 | Δ | -17996.644 | -583.523 | -371.340 | 0.000 | 0.000 | 0.000 | 3.544 |
| sequential_all_auctions | F | 39018.254 | -10382.224 | 1706.629 | 0.000 | 0.000 | 0.000 | 78.961 |
| sequential_all_auctions | O | 40869.119 | -10627.264 | 2019.857 | 0.000 | 0.000 | 0.000 | 79.343 |
| sequential_all_auctions | Δ | -1850.866 | 245.040 | -313.228 | 0.000 | 0.000 | 0.000 | -0.382 |

| Profile | Forecast fallbacks | Oracle fallbacks | Forecast final energy MWh | Oracle final energy MWh |
| --- | ---: | ---: | ---: | ---: |
| DA | 0 | 0 | 3.115148 | 3.115148 |
| DA+IDA1 | 0 | 0 | 4.340261 | 4.340227 |
| DA+IDA1+IDA2 | 0 | 0 | 2.770953 | 2.961848 |
| DA+IDA1+IDA2+IDA3 | 0 | 0 | 4.395104 | 4.247164 |
| DA+IDA1+IDA3 | 0 | 0 | 4.540121 | 4.650000 |
| DA+IDA2 | 0 | 0 | 2.785291 | 3.020741 |
| DA+IDA2+IDA3 | 0 | 0 | 4.119783 | 4.201925 |
| DA+IDA3 | 0 | 0 | 4.540121 | 4.650000 |
| sequential_all_auctions | 0 | 0 | 4.303150 | 4.247164 |

## Solver diagnostics

| Farm | Arm | Active attempts | Status counts | Fallbacks | Maximum reported relative gap |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | forecast | 739 | optimal: 739 | 0 | 9.22311e-05 |
| 1 | oracle | 739 | optimal: 739 | 0 | 9.59511e-05 |
| 345 | forecast | 739 | optimal: 739 | 0 | 9.56527e-05 |
| 345 | oracle | 739 | optimal: 739 | 0 | 9.57884e-05 |
| 901 | forecast | 739 | optimal: 739 | 0 | 9.55351e-05 |
| 901 | oracle | 739 | optimal: 739 | 0 | 9.83889e-05 |
| 910 | forecast | 739 | optimal: 739 | 0 | 9.82354e-05 |
| 910 | oracle | 739 | optimal: 739 | 0 | 9.80999e-05 |

Independent ledger audit passed for 72 trajectories: 5912 actual auction-price curve checks, 2232 realized PV day checks and 72 full settlement-price checks. The audit also reconstructs auction and imbalance cashflows. [Ledger validation evidence](ledger_validation.json).

## Assumptions and validation

Execution-target source counts below are quarter-hours per farm, shared by all profiles and both arms; night-policy counts are listed separately from metered values.

| Farm | Target source | Quarter-hours |
| --- | --- | ---: |
| 1 | metered_ac | 990 |
| 1 | monthly_night_zero | 1581 |
| 1 | pvlib_historical_ghi | 405 |
| 345 | metered_ac | 1346 |
| 345 | monthly_night_zero | 1581 |
| 345 | pvlib_retained_exclusion | 49 |
| 901 | metered_ac | 1395 |
| 901 | monthly_night_zero | 1581 |
| 910 | metered_ac | 1395 |
| 910 | monthly_night_zero | 1581 |

The model produces point forecasts, not stochastic price scenarios. PV uses unfitted pvlib conversion and the reviewed counterfactual execution targets; forecasts follow the configured night-zero and six-hour weather availability rules. The comparison is retrospective; price publication times and some source availability are assumed. Final SOC is shown because remaining inventory has no cash value in profit.

IDA1 for delivery 10 December was documented as cancelled and skipped in both arms. No missing auction prices were replaced by zero. Cancellation knowledge is treated as known throughout that delivery day's planning, including earlier gates; this is a retrospective scheduling assumption.

All eight profiles contain DA; the sequential reference uses all available auctions but plans only the current auction. The oracle is perfect in auction prices only, with the same updating PV forecasts, execution targets and daily planning horizon. It is not a global perfect-control upper bound. Each profile carries SOC, accepted commitments, degradation and fee accounting across midnight. The final-day optimization targets 50% SOC; execution can deviate.

Model features are price lags at 24/48/168/336 hours and seven calendar terms, with standardized ridge alpha 20. Missing nominal lags use older published same-clock observations within the configured age limit. No models were trained or selected on December profits. Artifact creation and source retrieval occurred retrospectively; fit cutoffs describe data eligibility, not proof of historical model deployment. Some pre-October hourly training labels were repeated within quarters. No residual scenarios or probability calibration enter optimization.

Battery sizes, symmetric grid limits, proxy locations, generic throughput aging and operator fee assumptions are illustrative. Curated PV replacements can use retrospective historical GHI and are counterfactual rather than metered production. Availability uses assumed source publication clocks; the comparison does not reconstruct historical revisions.

The model assumes price taking and guaranteed full execution, including offsetting trades across auctions. The tracking controller uses realized interval-average PV; it is not a tested live telemetry or trading system. These assumptions and perfect knowledge of auction curves can materially affect the size of the oracle advantage.

Validation: complete unique profile/date sets; matching saved physical settings, weather, realized PV, clearing and settlement inputs; run/preflight fingerprints; daily-to-monthly cash reconciliation; continuous daily battery boundaries. Runtime reporting also reconciles executed trade revenue and fees and checks interval-level battery continuity. See each trajectory's JSON solver diagnostics for status, gaps and fallback details.

Compute: at most two farms concurrently, one solver and one native numerical thread per worker, 120 seconds per auction solve, relative MIP gap 1e-4, no absolute gap, no GPUs. Per-date dispatch plots are disabled (zero dates); monthly comparison and daily-profit distribution plots are retained.

Source manifests and validation evidence: [report_sources.json](report_sources.json). Tables: [paired_monthly.csv](paired_monthly.csv), [daily_all_farms.csv](daily_all_farms.csv).
