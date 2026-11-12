# Continuous-study numerical consistency audit

The January 2026 integrated run failed after a validated solve when the next
optimization input checked carried accepted and pending positions with a fixed
1e-8 MWh POI allowance. The solution validator used the configured 1e-4
physical allowance. The failed event was after the latest daily checkpoint, so
the exact offending position is unavailable in the saved artifacts; a small
numerical residual is plausible, but is not proven. The preceding IDA1 solve
reported a maximum Gurobi violation of 2.41121e-5 in model units.

The input check now uses the same configured physical absolute tolerance as
the solution validator, or twice the configured solver feasibility tolerance
if larger. It leaves the accepted and pending positions unchanged and keeps the
optimization's POI bounds unchanged. An excess above the allowance fails with
the interval, signed position, excess and tolerance. Every validated solve
records the largest cumulative POI excess, and each input records the carried
position excess, even with compact validation diagnostics.

The audit followed input construction, solver-result validation, carried
commitments, health replay, physical execution, day-boundary checkpointing,
resume and output accounting. These checks intentionally do not all share a
single number: integrality uses the solver integrality tolerance; monetary
accounting uses separate euro and relative tolerances; aging exposure residuals
are scaled to the solver's row units; and memory/state checks protect invariants
of numerically reconstructed, physically executed commands rather than raw
solver variables. Execution logs any tolerance-sized command projection and
preserves the internal energy rate. Checkpoint restore retains its configuration
and input identity checks. Reporting uses the configured numerical tolerances.

The independent Naumann bank replay still has a strict 1e-7 MWh residual guard,
and analytical executed-health invariants retain tighter numerical checks.
They are not proof that no other edge case exists. If one fails, retain the
failing residual and trajectory before changing the threshold; blindly using a
single 1e-4 value for squared exposures, euro costs, fractional health and MWh
would hide errors of different physical scales. In particular, accepting a
small carried POI residual does not guarantee that a later solve is feasible.

The CLI supports `--resume` for a checkpointed continuous experiment without
editing its configuration. Identity checks still require the original settings
and inputs. A resumed study rolls back work after its most recent checkpoint;
the changed validator means the continuation is not a byte-identical rerun of
the earlier code. The user must launch experiments themselves.
