# Reviewed neural PV forecasts in continuous replay

The four validation-selected model bundles from the original checkout are
copied under `models/pv/reviewed_ghi_2026_selected`; their source manifest and
file checksums identify the BiGRU, Transformer, BiLSTM and TCN used for farms
1, 345, 901 and 910. `configs/research/neural_pv_research.yaml` enables these
models for a future retrospective comparison and uses the reviewed execution
corpus. The larger PV model search is in progress; do not launch the matched
closed-loop benchmark until the user indicates that its models are ready.

The inference adapter uses `forecast_ghi` and forecast weather inputs, not
observed historical GHI or observed AC power. It applies the selected bundle's
training preprocessing, verifies its declared model checksums, bounds AC output
by the farm rating and enforces the configured monthly night-zero policy.
Models are reused within a worker only while their file signatures remain
unchanged. Missing selected model files fail research-study input checks.

At each auction or physical MPC cutoff, deterministic PV uses the latest
weather cycle available under the configured six-hour release-lag assumption.
When `cycle_updates: latest_available` is selected, ensemble paths likewise
switch to the latest available cycle that covers the requested remaining
intervals. Initial and later cycles retain their source, issue time, assumed
availability, member IDs, conversion and output hash. Newer ensemble cycles
are converted once for their available remaining interval range and reused;
later forecasts must never replace historical executed decisions. The existing
pvlib benchmark remains pinned to its original configuration.

An offline input check on farm 1 and 15 January 2026 verified 50 neural-converted
members: the 05:45 UTC request selected 14 January 00 UTC, and the 06:00 UTC
request selected 15 January 00 UTC as its six-hour assumed availability began.
This is a preflight, not a closed-loop optimization result. The deterministic
provider similarly checks latest-vintage routing in unit tests.

These weights were selected in November 2026. Their training used historical
GHI and reviewed targets, while validation used the preceding day's 00 UTC
forecast; later-cycle performance is a separate evaluation. Historical runs
with these weights and current reviewed synthetic replacements are retrospective,
not evidence that the model or revised target existed at the time. The 2025
model-training period must not be presented as untouched PV test data.
