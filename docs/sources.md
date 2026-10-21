# Consulted authoritative sources

Access date: 16 October 2026. Local farm semantics come from the user's implementation prompt and inspected store, not public market or pvlib documentation. Each entry records the statement used; a linked page does not imply an operational download adapter.

| Source | Supported statement and implementation use |
|---|---|
| [GME Technical Rule 03 rev. 13, in force 29 September 2025](https://www.mercatoelettrico.org/Portals/0/Documents/en-US/20250930DTF03rev13MPE_En.pdf) | Sections 2–3: DA gate D−1 12:00; provisional/final results no earlier than 12:55/13:05; IDA gates D−1 15:00, D−1 22:00, D 10:00; IDA results by 15:30, 22:30, 10:30. IDA3 trades noon onward. `markets.py` separates all events. |
| [GME Technical Rule 03 rev. 14, in force 13 January 2026](https://gme.mercatoelettrico.org/Portals/0/Documents/en-US/20260113DTF03rev14MPE_en.pdf) | Sections 2–3 retain the auction times and windows above. Later continuous-market/nomination changes are outside this implementation. Rule provenance switches by event date, including an auction crossing the revision boundary. |
| [GME quarter-hour MGP go-live notice, 29 September 2025](https://www.mercatoelettrico.org/en-us/Home/NoticesandAnnouncements/NoticesandAnnouncementsME?id=7101) | Quarter-hour MGP delivery begins 1 October 2025. Earlier hourly-to-quarter-hour commercial mapping is unimplemented, so the calendar rejects earlier delivery days. |
| [ENTSO-E Intraday Auctions on SIDC](https://www.entsoe.eu/network_codes/cacm/implementation/ida/) | Confirms three IDA gate times and delivery windows, European start on 13 June 2024, and actual cancellation notices. Missing/cancelled results must be represented explicitly. |
| [GME MGP manual downloads](https://gme.mercatoelettrico.org/en-us/Home/Results/Electricity/MGP/Download/Download) | Describes XML archives and market codes MGP, MI-A1, MI-A2, MI-A3. Manual CSV/XML import requires explicit canonical field mapping and timestamp semantics; no unverified native export layout is guessed. |
| [GME API Service technical guide](https://gme.mercatoelettrico.org/Portals/0/Documents/en-US/20251015Manuale_tecnico_API_En.pdf) | Verified pages 5–7 and 49–50 document the API base, username/password authentication, Bearer requests, base64 ZIP JSON responses and `ME_ZonalPrices` for `MGP`, `MI-A1`, `MI-A2`, `MI-A3`. `data/gme.py` and the shared downloader implement this contract with offline documented-format tests; authenticated response/schema coverage remains pending the user's GME access. |
| [GME API service access](https://www.mercatoelettrico.org/en-us/Home/APIService) and [registration form](https://api.mercatoelettrico.org/users/RegistrationForm/RegistrationRequest) | GME issues separate user ID/password credentials after the documented registration and signed-document process. Existing ENTSO-E/Terna credentials do not establish GME access. The service terms describe free access subject to their conditions; no account registration or document submission was performed by the agent. |
| [ENTSO-E Transparency Platform extraction guide](https://transparency.entsoe.eu/content/static_content/download?path=%2FStatic+content%2Fweb+api%2FIG-for-TP-data-extraction-process.pdf) | Official guide identifies day-ahead price document A44, with identical in/out domains. This alone does not establish IDA price coverage or historical publication timestamps; no IDA substitution is implemented. The older HTML Guide URL failed retrieval during inspection. |
| [Terna developer catalog](https://developer.terna.it/docs/read/apis_catalog) | `data/terna.py` implements verified load/forecast and renewable-generation subsets through the shared cache-first API pipeline. Public schemas were consulted; tests use mocks and documented fixtures. No live authenticated retrieval, complete historic coverage, DA/IDA prices, or settlement-law verification is claimed. |
| [Terna access-token instructions](https://developer.terna.it/docs/read/Access_Token) and [request examples](https://developer.terna.it/docs/read/How_to_make_a_request) | Form-encoded POST to `/public-api/access-token` uses client ID, client secret and the client-credentials grant. Bearer token response includes an expiry; the caller keeps credentials and tokens outside request/cache provenance. No undocumented scope parameter is added. |
| [Terna Total Load](https://developer.terna.it/docs/read/apis_catalog/load/Total_Load) | Verified `/load/v2.0/total-load` parameters and response fields provide actual and forecast MW by English-named bidding zone. Reference labels, timezone and offsets are retained; neither interval bounds nor historical forecast publication times are inferred. |
| [Terna Load Forecast](https://developer.terna.it/docs/read/apis_catalog/market/Load_Forecast) | Verified `/market/v1.0/input/forecast-load` schema supports explicit MGP/MSD session forecasts. The documented general production schedule does not prove when a returned historical revision was available. |
| [Terna Wind Production Forecast](https://developer.terna.it/docs/read/apis_catalog/generation/Wind_Production_Forecast) | Verified `/generation/v2.0/wind-production-forecast` schema is for MSD1 through MSD6. Forecast creation (`prevision_date`) and publication labels remain separate; a publication date without a clock does not establish historical availability. |
| [Terna Renewable Generation](https://developer.terna.it/docs/read/apis_catalog/generation/Renewable_Generation) | Verified `/generation/v2.0/renewable-generation` returns national actual energy by source in GWh. It is not treated as a NORD solar forecast, and no unverified duration is used to convert energy to MW. |
| [Terna Daily Prices](https://developer.terna.it/docs/read/apis_catalog/fees/Daily_Prices) and [Preliminary Prices](https://developer.terna.it/docs/read/apis_catalog/fees/Preliminary_Prices) | Each page conflicts internally between its principal `/fees/v1.0/...` route and `/ifees/v1.0/market/...` cURL example. Imbalance-price downloads remain explicitly blocked pending endpoint and settlement-regime verification; these fields are not substituted for IDA prices. |

The configured DA final-results clock of 13:30 is a conservative **study assumption**, because the official rule provides an earliest final-publication time rather than a guaranteed deadline. Recorded timestamps take precedence; a late or cancelled auction is not made available from an ordinary schedule alone. No historical imbalance law, tariff exemption or settlement coefficient has been established by these sources.

## Linear and probabilistic price forecasting

| Source | Supported statement and implementation use |
|---|---|
| [Lago, Marcjasz, De Schutter and Weron (2021), author manuscript](https://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/21_011.pdf) | Regularized autoregressive price models use multiple previous daily price profiles, weekday indicators and exogenous forecasts. The paper motivates strong seasonal baselines, chronological evaluation and longer testing. Our compact ridge candidates are inspired by these features, rather than an exact reproduction of LEAR's lasso model. |
| [Jordan, Krueger and Lerch, Evaluating probabilistic forecasts with scoringRules](https://arxiv.org/abs/1709.04743) | Proper scoring rules including Gaussian CRPS assess forecast distributions; point MAE alone cannot validate probabilistic paths. The implementation computes analytic Gaussian CRPS alongside coverage, likelihood and temporal ramp diagnostics. |

## Optimization and aging

| Source | Supported statement and implementation use |
|---|---|
| [Gurobi Multiple Scenarios](https://docs.gurobi.com/projects/optimizer/en/current/features/multiscenario.html) | Related-model scenario solves do not replace explicitly shared stochastic decisions. The implementation builds one deterministic equivalent with node-owned bids and physical controls. |
| [Gurobi optimization status codes](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html) | Distinguishes optimal, infeasible, unbounded, ambiguous and time-limit outcomes; ambiguous status can be diagnosed with dual reductions disabled. Extraction additionally requires an incumbent. |
| [Xu et al., original author preprint, version 2](https://arxiv.org/html/1707.04567v2) | Implemented `xu_segmented`: Eq. (5) marginal depth costs, Eq. (7) discharge-only aging and Eqs. (8)–(15) segmented-energy balances. Priority binaries explicitly implement the shallow-first charge/discharge policy of Appendix Eqs. (37)–(40); Eq. (39) initializes once and segment history is carried thereafter. Eq. (24) stress defaults and the configured total EUR pack cost are illustrative, not farm calibration; nonzero self discharge is unsupported. The constant-throughput baseline remains available. Published DOI supplied in the task: [10.1109/TPWRS.2017.2733339](https://doi.org/10.1109/TPWRS.2017.2733339). |

The Gurobi model-attributes web page returned HTTP 429 during consultation.
Available attributes are queried defensively from installed gurobipy 13.0.3;
failed page retrieval is not represented as a consulted reference. The original
Xu author preprint was read; publisher metadata could not be retrieved, so the
bibliography does not invent journal volume, pages or publication year.

## Weather-to-PV conversion

| Source | Supported statement and implementation use |
|---|---|
| [pvlib nomenclature](https://pvlib-python.readthedocs.io/en/stable/user_guide/extras/nomenclature.html) | North-zero, clockwise azimuth convention used by source-angle conversion. Source east-positive/south-zero meaning comes from the task, not this public documentation. |
| [pvlib irradiance DNI](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.dni.html) | Derivation from GHI, DHI and zenith, with horizon thresholds. Horizontal direct radiation is not passed directly as DNI; the model explicitly uses an 85-degree zero-DNI threshold. |
| [pvlib single-axis tracking](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.tracking.singleaxis.html) | Separate explicit axis tilt/azimuth, rotation limit, backtracking and GCR parameters. This does not establish anonymous farms' actual geometry or rear-side gain. |
| [pvlib PVWatts DC](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html) | DC output from effective irradiance, cell temperature, reference watts and temperature coefficient; implementation derives reference capacity from AC farm rating using `pv_output_efficiency`, converts kW to watts and model output to MW, then applies the configured DC-to-AC and transformer factors and AC forecast cap. |

All entries above were accessed on 16 October 2026. The documentation was used
alongside installed pvlib 0.16.1. Farm measured-AC semantics, hourly forecast-GHI
labels, direct-radiation provenance, and policy parameters are user-supplied or
locally inspected inputs; public library documentation does not verify them.
