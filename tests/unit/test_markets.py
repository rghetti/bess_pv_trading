"""Auction gates, publication times and DST-aware delivery products."""
import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.markets import AuctionCalendar


def test_calendar_dst_products_gates_and_results_are_separate():
    calendar = AuctionCalendar()
    assert len(calendar.delivery_grid("2026-03-29")) == 92
    assert len(calendar.delivery_grid("2025-10-26")) == 100
    ordinary = calendar.delivery_grid("2025-10-20")
    assert len(ordinary) == 96
    assert np.allclose(ordinary.delta_hours, .25)
    assert len(ordinary.to_domain().interval_ids) == 96
    summer = calendar.schedule("2025-10-20")
    winter = calendar.schedule("2025-10-28")
    assert summer[0].gate_at.hour == 10
    assert winter[0].gate_at.hour == 11
    assert summer[0].submission_at == summer[0].gate_at - pd.Timedelta(minutes=5)
    assert "study_assumption" in summer[0].result_provenance
    assert summer[0].results_at > summer[0].gate_at
    assert len(summer[3].eligible_intervals) == 48
    assert all(ordinary.starts[t].tz_convert("Europe/Rome").hour >= 12 for t in summer[3].eligible_intervals)


def test_historical_validity_missing_results_cancellation():
    calendar = AuctionCalendar()
    with pytest.raises(ValueError, match="outside configured auction-calendar validity"):
        calendar.delivery_grid("2025-09-30")
    events = calendar.schedule("2025-10-02", recorded_results={"DA": "2025-10-01T12:00:00Z", "IDA1": None}, cancelled=("IDA2",))
    assert events[0].result_provenance == "recorded_publication"
    assert not events[1].available and events[1].results_at is None
    assert not events[2].available
    boundary = calendar.schedule("2026-01-13")
    assert boundary[0].rule_version == "GME-DTF03-rev13"
    assert boundary[3].rule_version == "GME-DTF03-rev14"
    with pytest.raises(ValueError, match="explicit timezone"):
        calendar.schedule("2025-10-02", recorded_results={"DA": "2025-10-01T13:30:00"})
