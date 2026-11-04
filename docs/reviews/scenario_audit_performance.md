# Scenario audit alignment

The independent auditor aligns the scenario and interval index once per MPC
horizon rather than pivoting each physical quantity separately. Duplicate
scenario/interval pairs still fail rather than being aggregated. Missing values
remain missing and are reported by the existing residual checks.

On the completed reviewed January farm 1 `adaptive_ecc` controller, the old and
new implementations produced identical audit results for all 192 conditional
horizons, with zero violations. One sequential comparison on the shared machine
measured 31.644 seconds before and 13.696 seconds after. These are individual
wall-clock samples under shared load, not a general speed guarantee. The running
January batch and its first independent audit used their pinned earlier code.

The focused independent-auditor test suite passed all eight tests, including
conditional histories spanning file batches and rows supplied in an order
different from the recorded scenario order.
