"""Preserve failed validation evidence while omitting successful residual maps."""

from bess_pv_trading.domain import ValidationReport
from bess_pv_trading.simulation.validation_diagnostics import auction_validation_record, mpc_validation_record


def test_auction_pass_omits_residuals_but_keeps_warning_and_objective():
    """Successful checks still record status and nonblocking warning messages."""
    report = ValidationReport(True, {"energy": 1e-10}, 12.0, warnings=("numerical warning",))
    saved = auction_validation_record(report, "failures_only")
    assert saved["valid"] is True
    assert saved["warnings"] == ["numerical warning"]
    assert saved["expected_objective_eur"] == 12.0
    assert "max_errors" not in saved
    assert auction_validation_record(report, "full")["max_errors"] == {"energy": 1e-10}


def test_auction_failure_retains_all_residuals():
    """A rejected solution retains the evidence needed to diagnose the failure."""
    report = ValidationReport(False, {"energy": 0.1}, 12.0, violations=("energy: 0.1",))
    saved = auction_validation_record(report, "failures_only")
    assert saved["max_errors"] == {"energy": 0.1}
    assert saved["violations"] == ("energy: 0.1",)


def test_mpc_success_filters_only_residuals_without_changing_source():
    """Operating components and status remain while duplicate validation detail disappears."""
    original = {"status": "optimal", "physical_residuals": {"energy": 0.0},
                "component_residuals": {"wear": 0.0}, "component_validation_allowances": {"wear": 1e-6},
                "recomputed_components": {"wear": 3.0}, "components": {"wear": 3.0}}
    saved = mpc_validation_record(original, "failures_only", has_plan=True)
    assert saved["validation_passed"]
    assert saved["components"] == {"wear": 3.0}
    assert "physical_residuals" not in saved
    assert "component_residuals" not in saved
    assert "component_validation_allowances" not in saved
    assert "recomputed_components" not in saved
    assert original["physical_residuals"] == {"energy": 0.0}
    assert mpc_validation_record(original, "full", has_plan=True) == original


def test_mpc_failures_and_rejected_refinements_keep_details():
    """Both a failed solve and a rejected secondary validation keep residuals."""
    failed = {"status": "invalid_or_failed_solve", "failure_reason": "bad residual",
              "physical_residuals": {"energy": 0.1}}
    assert mpc_validation_record(failed, "failures_only", has_plan=False) == failed
    refined = {"status": "imbalance_tiebreak_rejected", "failure_reason": "bad candidate",
               "component_residuals": {"wear": 0.2}}
    assert mpc_validation_record(refined, "failures_only", has_plan=True) == refined
    captured_failure = {"status": "optimal", "physical_residuals": {"energy": 0.0},
                        "failed_validation_residuals": {"energy": 0.1}}
    assert mpc_validation_record(captured_failure, "failures_only", has_plan=True) == captured_failure
