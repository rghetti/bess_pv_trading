# K25 continuous replay checkpoint storage

This analysis describes the completed `deterministic_six_workers_v2` batch,
not the newer daily hard-terminal-SOC run. Sizes are approximate binary MiB of
pretty-printed JSON; the representative checkpoint is farm 1, equal-weight MPC,
at `runs/farm_1/january_to_august__farm_1__deterministic_original/continuous/controllers/deterministic_mpc/json/checkpoint.json`.
It is 568 MiB, and the batch's six trajectory checkpoints total about 3.01 GiB.

`ContinuousReplay.save_checkpoint` atomically replaces one cumulative snapshot
after the local-midnight settlement handoff. It does not create a separate full
checkpoint for each day. The checkpoint includes both the state needed to
continue and all accumulated records currently needed to finalize the reports;
it is serialized again each day. A low-level replay can resume from it after
checking configuration and input fingerprints. The K25 queue does not currently
offer automatic batch resume.

| Field | Approximate size | Purpose and retention |
| --- | ---: | --- |
| `mpc_state` | 148 MiB | The latest plan and outstanding action are live controller state. Most of the size is accumulated MPC solve diagnostics (about 112 MiB) and issued actions (about 36 MiB), also needed for final reports. The completed horizon rows have already been moved into daily Parquet partitions; their archive index must remain verifiable. |
| `forecast_inputs` | 118 MiB | Forecast-input and provenance records collected for reporting and ensemble diagnostics. Past records are historical, but the existing finalization code reads the complete list. |
| `plan_snapshots` | 86 MiB | Past auction plans used for expected-dispatch reports; snapshots for relevant current and future delivery days can also affect planning and provisional IDA3 positions. |
| `solves` | 60 MiB | Auction solve diagnostics and validation results. The retained attempt identities are also used to reconcile the separate attempt log when resuming. |
| `fee_records` | 48 MiB | Per-trade fee details; unpublished auction results can still need them when booked. Older records also support audit and reconciliation. |
| `records` | 41 MiB | Realized per-interval dispatch and settlement. The final dispatch table and financial reports currently require all intervals. |
| `accepted_trades` | 27 MiB | Executed ledger, including fixed commitments and realized cashflows. Relevant current/future trades are live inputs; the full history currently supports final fee reconciliation and revenue reporting. |
| `control_plan` | 23 MiB | Planned actions by product. Current/future actions affect control; historical entries can be separated only after checking all readers. |
| `event_trace` | 10 MiB | Audit history and an aggregate retained-submission count used by the final summary. |

Other fields include physical battery/degradation state, submitted but possibly
unpriced commitments, auction publication status, the transaction-fee state,
pending control, projections, initial expected values, event cursor and
identity. In particular, submitted quantities cannot be discarded just because
their prices have not yet been published. Projections include unresolved future
boundaries as well as historical report data.

## Safe direction for a smaller checkpoint

1. Keep a compact live snapshot containing the physical state, event cursor,
   relevant current/future commitments and plans, fee state, unresolved
   projections, and the controller's latest plan and outstanding action.
2. Archive completed historical records incrementally, for example in immutable
   daily Parquet partitions, with a manifest recording paths, row counts and
   checksums. MPC horizon diagnostics already use a version of this pattern.
3. Publish the archive and checkpoint atomically in the right order. Resume
   must verify partitions and restore enough history for ledger reconciliation,
   attempt reconciliation, reporting and any still-live cross-day events.
4. Compare uninterrupted and interrupted/resumed replays, including auction
   commitments, dispatch, battery and fee states, and final reports. Confirm
   crash handling does not duplicate or omit rows.

This is a design analysis, not an implemented checkpoint migration. The
existing checkpoints and completed run outputs have not been removed or changed.
