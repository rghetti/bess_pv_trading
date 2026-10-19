# Project Guidance

## Project overview

<!-- Describe the purpose, scope, and main components of this project. -->
- We are developing a multi-stage scenario-based stochastic milp optimization model in gurobi that
participates in the day ahead market and the three intraday auctions in europe (and in the continuous
 market as well in the future), to maximize profits on the energy market by making use of our assets and reoptimizing
  when new informations and forecasts are available.
- The asset we assume to own is a pv farm with a li-ion bess that shares the grid interconnection with the grid;
in the objective function we will deploy a penalty on battery degradation to preserve its useful life. My current
favourite model is the one presented  here https://www.researchgate.net/publication/318460951_Factoring_the_Cycle_Aging_Cost_of_Batteries_Participating_in_Electricity_Markets ;
it keeps the formulation linear and is a good approximation of more complex functions
- We have metadata and time series weather forecast data for 4 anonymous plants in nothern italy participating in IT-North bidding zone (for now position is milano)
- In the future we will have ensemble forecasts for weather that we will use to generate pv forecasts as probabilistic scenarios for the milp; build the problem
having that in mind vut also adapt it to work on the current data structures.
- For now we have only one forecast so what we need to implement right now is a model using pvlib functions and the data we have to translate the rolling forecasts in pv prediction of our farms
- The structure of the data is 2d or 3d with the form (farms, cycles, leadtime) meaning each farm has
a certain number of overlayed time series covering until lead time in the future.
Inspect the data structure if needed; sometimes the data is flattened on a single dimension
- The ghi is the satellite ground truth data and is most likely reliable
- Absolute first priority is the data prepocessing and cleaning:
 we need to impelement a data cleaning structure where we compare the forecasts
  features and historical power time-series to see which of them is more
  reliable/clearly wrong especially wrt to the time of the day or if it suddenly drops
  to zero and then recovers (most likely due to faults)
- The total power field is the total installed capacity (in kW)
- Tracking type id corresponds to: 0=still; 1=tracking monoaxial; 3=tracking monoaxial bifacial, and needs to be taken into account when using the forecasts
- Power and total power are at the AC side of the plant
- Some metadata isn't available like the grid interconnection limits etc so we need to think of reasonable values
related to the size of the farm in question.
- We need to fetch the data from entso e and Terna for the prices of the auctions and imbalances
 but i have an api token for both,
so whenever writing scripts that download data: if not cached, dowload the required fields directly from the website,
prompting me to input clied id and/or secret (invisible on the cli), otherwise cache them with timestamps/date as keys.
- We will implement a prototype linear regression model on the price forecasts (we will have a perfect foresight benchmark on the side) using classical features like recent past prices, regional load, wind and solar forecasts for the day (available in TERNA?) etc + gaussian noise residual. We will deploy it in our objective function where for each member of the ensemble we will draw a number of price forecast and optimize over each.
- Phisical constrainst include: POI limits, BESS capacity and maximum charge/discharge power, no simultaneous
 charge/discharge (via binary variable), soc box bound, inverter limits etc. we suppose the PV farm can curtail eletricity, give it to
 the BESS for charging or sell it to the grid via the POI. There is no local load, the battery either charges importing from the grid or the PV or it discahrges to the grid for selling. make the bess charecteristics configurable via a yaml config, as well as optimization parameters (number of threads, walltime...)
- Since we are not a consumer we don't need to pay an additional fee for grid usage when importing power to charge the battery
- Since we already model the uncertainty of the prices for the day-ahead market and the IDA prices, the objective function at each step already takes into account possible trading opportunities, but reoptimizes anyway the the mcp is
revealed, letting us recodition future unknown prices on the new information, and new more accurate forecasts are available.
- special care needs to be given to tile alignement of the time series data and availability of the data, ie usually the entire time series is known 4 hours after the cycle time beginning (if the forecasts starts from
00:00 then it is available at 04:00); this is important when aligning information with gate closure of each
auction (check carefully when each of the relevant time windows are during the day). usually the cycles start
 6 hours apart from the previous one (every 6 hours they are regenerated).
- The constraints need to be valid for each possible scenario and the objective funciton is the average
 over all the generated objective functions.
- tilt and azimuth are in degrees and 0 azimuth is south facing
- market actions in the studied period are every 15 min, so each day is 96 decision while the measurements are usually hourly so when aligning the data
we need to linearly interpolate it
- As a first experiment, the time horizon of the problem is every market product of the next day; future experiments will consider longer rolling horizons.
- Check every information I gave you against authoritive sources online when in doubt.
## Code style guidelines

<!-- Document language-specific conventions, formatting, naming, and design preferences. -->
- Use instructive names for the variables and include at each code block, on top of a docstring, also a short description of what that chunk of code does (coarsely, not at every line)
- The working env is bess, activated by conda activate bess; do not ever change it or download
packages in it without my explicit permission
- Never hard code particular values we agreed on, make scripts as much as possible configurable from the outside via yaml files
- Commit and push only when asked by me; the remote repo is called bess_pv_trading.
- This is the working directory. The farm time series Zarr and static-parameter Parquet files are in `data/farms data/`; later price downloads also belong under `data/`.
- Prefer reusing available code when possible, use short and concise commands rather than long and convoluted
blocks. Always try to vectorize operations when libraries and data structures allow it.
- We have access on this machine to 48 cores and 4 GPUs (they are shared though so check usage with htop and nvtop or similar readable to you if executing something big), so try to parallelize tasks when possible.
- Generate meaningful plots of relevant quantities and group them by runs/ examined starting date of the experiment. This is true for the cached data as well:
group it by day or by a recognizable id
- always give exaustive commit comments!
- Default solver optimality tolerance is a relative MIP gap of `1e-4` (0.01%). Do not configure an absolute MIP gap.
- Before launching an experiment, explain its purpose, data, model/features, evaluation split, baselines and saved outputs to the user. This is an overview, not a request for approval unless approval is explicitly required.
- The perfect-foresight benchmark uses the same actual auction-price curve for all PV/weather members on a delivery day. Forecast price scenarios may differ to represent uncertainty before clearing; do not add price noise to the oracle.
- API acquisition and dataset readers use `data/api/market.duckdb`. Build new workflows on the DuckDB catalog and retain snapshot IDs/checksums; the file cache is legacy compatibility. Coordinate with an active downloader before opening the database from another process.
- Auctions use price-taking, guaranteed full execution of every submitted quantity. Do not model bid-price curves, acceptance probabilities or rejection. Submitted quantities become fixed commitments immediately; until price publication they are unpriced commitments. At publication, book the actual quantity times the clearing price once, update future forecasts and reoptimize without changing those commitments. Keep forecast objective values separate from realized cash flow.
