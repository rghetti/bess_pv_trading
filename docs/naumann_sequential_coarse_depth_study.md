# Sequential fixed-rate coarse-depth comparison

The user requested sequential auction planning and seven Gurobi threads. This
study supersedes the interrupted integrated-depth comparison. Farm 1, January
1–3 2026, uses `current_auction_only`: each auction optimizes its own orders
against fixed earlier commitments without anticipating future auction decisions.
It is a different policy from integrated planning, not just a solver setting.

Three cases use 2, 3 and 4 depth segments, respectively; each uses fixed 0.5C
cycling stress, no variable C-rate aging, SOC-dependent calendar aging, fresh
SOH, chronological midnight updates, 50% planned daily terminal SOC and the grid
tracking controller. The 2- and 3-segment curves are coarse approximations and
must not be assumed accurate. Physical operating power limits remain active.
The 2025 stationary-storage pack price is an economic proxy, not a quote.

Weather forecasts are the latest eligible vintages under the six-hour lag
assumption. PV models are retrospectively selected and actual clearing prices
are used as an oracle. Curated realized PV, including reviewed substitutions, is
the execution input. Missing auction products are skipped only if entirely
absent; partial missing products fail preflight. There is no training, fitting,
or train/test split; these three winter days do not establish annual performance.

Each solve uses seven Gurobi threads, 240 seconds, a relative MIP gap of 1e-4,
solver feasibility and integrality tolerances of 1e-5, and ordinary solution
validation of 1e-4. Carry exposures and memory across days and preserve raw
solver errors; tolerance-sized replay/physical command corrections are recorded.
Run cases sequentially in separate output directories. Record statuses,
violations, optimality gaps, model size, runtime, daily SOH, calendar and cycle
losses, market cash, wear cost, throughput and imbalance. Use independent
rainflow on complete executed paths with unfinished half-cycles and analytical
versus segmented depth stress at 0.5C. Compare tradeoffs without claiming that
any simulated trajectory is an empirical calibration of a commercial battery.
