# Configuration layout

- `api_experiments/` contains provider, download, and API-cache configurations.
- `price_experiments/` contains price-data imports, price forecasting, and price-model studies.
- `full_experiments/` contains end-to-end optimization or replay configurations. Put new complete studies here, with all study dates, data sources, asset and inverter parameters, initial/terminal state, trading limits, degradation assumptions, solver settings, and output paths specified in YAML.
- Configurations at this level remain shared data, PV-farm, audit, and reporting settings used by more than one experiment.

Paths in commands and documentation should use these directories so each experiment's purpose is clear.
