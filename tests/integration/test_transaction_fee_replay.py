"""Executed fees reconcile across submissions, publications and residual solves."""
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.simulation.synthetic import _replay_strategy, synthetic_inputs
from bess_pv_trading.transaction_fees import account_trade_fees, transaction_fees_from_config


@pytest.mark.parametrize("mode", ["none", "flat", "gme"])
def test_executed_fee_ledger_and_causal_monthly_state_reconcile(tmp_path, mode):
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))
    config = load_config(f"configs/full_experiments/synthetic_fees_{mode}.yaml")
    config["outputs"]["plots"] = False
    # The flat run also solves in the submission/publication gap: committed fees
    # must be sunk before the matching auction price becomes known.
    config["execution"]["reoptimize_physical_controls"] = mode == "flat"
    fixture = synthetic_inputs(config)
    fixture["realized_imbalance_prices_eur_per_mwh"] = np.full(len(fixture["grid"].starts), 60.0)
    destination = tmp_path / mode
    summary = _replay_strategy(config, fixture, "deterministic_integrated", destination)
    ledger = pd.read_csv(destination / "ledger.csv")
    dispatch = pd.read_csv(destination / "dispatch.csv")
    solves = json.loads((destination / "solves.json").read_text())
    publications = json.loads((destination / "price_publications.json").read_text())
    settings = transaction_fees_from_config(config)
    _, _, fees = account_trade_fees(settings, settings.initial_state, ledger.to_dict("records"))
    total = float(ledger.transaction_fee_eur.sum())
    assert total == pytest.approx(fees["transaction_fee_eur"])
    assert dispatch.transaction_fee_eur.sum() == pytest.approx(total)
    assert summary["transaction_fees"]["gross_traded_mwh"] == pytest.approx(ledger.quantity_mwh.abs().sum())
    assert summary["transaction_fees"]["validation"]["valid"]
    expected = ledger.cashflow_eur.sum() + dispatch.realized_imbalance_cashflow_eur.sum()
    expected -= dispatch.degradation_eur.sum() + dispatch.import_fee_eur.sum() + total
    assert summary["realized_profit_after_imbalance_eur"] == pytest.approx(expected)
    assert summary["initial_expected_transaction_fees_eur"] == solves[0]["expected_transaction_fees_eur"]
    assert all(row["publication_booked_transaction_fee_eur"] == 0 for row in publications)
    submission = pd.to_datetime(ledger.submission_time, utc=True)
    for record in solves:
        at = pd.Timestamp(record["cutoff"])
        executed = submission < at if record["solve_trigger"] == "auction_submission" else submission <= at
        incurred = ledger.loc[executed, "transaction_fee_eur"].sum()
        assert record["sunk_transaction_fees_eur"] == pytest.approx(incurred)
    if mode == "flat":
        assert total == pytest.approx(0.04 * ledger.quantity_mwh.abs().sum())
        last = fixture["schedule"][-1]
        assert any(record["solve_trigger"] == "physical_control" and "IDA3" not in record["known_auctions"]
                   for record in solves if last.submission_at <= pd.Timestamp(record["cutoff"]) < last.results_at)
    elif mode == "none":
        assert total == 0
