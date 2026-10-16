"""Information publication, cutoffs, and event mutation tests."""

import pandas as pd
import pytest

from bess_pv_trading.simulation.events import Event, EventEngine, EventKind
from bess_pv_trading.config import ConfigurationError, validate_config


def test_publications_before_coincident_control_and_results_do_not_resubmit():
    instant = pd.Timestamp("2026-06-02T08:00Z")
    seen = []
    events = [Event(instant, k) for k in reversed(list(EventKind))]
    engine = EventEngine(events)
    engine.run({k: lambda event: seen.append(event.kind) for k in EventKind})
    assert seen == list(EventKind)
    assert len(engine.trace) == 6


def test_cannot_insert_an_earlier_event():
    now = pd.Timestamp("2026-06-02T08:00Z")
    engine = EventEngine([Event(now, EventKind.PHYSICAL_CONTROL)])
    with pytest.raises(ValueError, match="before the current"):
        engine.run({EventKind.PHYSICAL_CONTROL:
                    lambda event: engine.schedule(Event(now, EventKind.AUCTION_RESULTS))})


def test_config_rejects_nonfinite_efficiency_and_invalid_solver():
    for config in ({"battery": {"converter_charge_efficiency": 1.1}},
                   {"solver": {"threads": 0}}, {"solver": {"mip_gap": -1}},
                   {"degradation": {"model": "xu"}}):
        with pytest.raises(ConfigurationError):
            validate_config(config)


def test_config_prevents_credentials_entering_saved_snapshots():
    with pytest.raises(ConfigurationError,match="environment credentials") as raised:
        validate_config({"provider":{"client_secret":"fixture_private_value"}})
    assert "fixture_private_value" not in str(raised.value)


def test_solver_default_is_relative_gap_only():
    from dataclasses import fields
    from bess_pv_trading.domain import SolverSettings
    assert SolverSettings().mip_gap == 1e-4
    assert "mip_gap_abs" not in {item.name for item in fields(SolverSettings)}
    with pytest.raises(ConfigurationError,match="relative MIP gap only"):
        validate_config({"solver":{"mip_gap_abs":1e-6}})
