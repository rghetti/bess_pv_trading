"""Control persisted validation detail without changing any validation checks."""

from dataclasses import asdict


def auction_validation_record(validation, mode):
    """Keep full failed checks and a small status and warnings record on success."""
    if mode == "full" or not validation.valid:
        return asdict(validation)
    return {"valid": True, "warnings": list(validation.warnings),
            "expected_objective_eur": validation.expected_objective_eur,
            "residuals_omitted_after_pass": True}


def mpc_validation_record(diagnostics, mode, *, has_plan):
    """Drop successful residual maps while retaining failures and solver outcomes."""
    record = dict(diagnostics)
    failure = (not has_plan or record.get("failure_reason") or record.get("failed_validation_residuals")
               or record.get("oracle_retry_initial_rejection")
               or str(record.get("status", "")).endswith("_rejected"))
    if mode == "failures_only" and not failure and "physical_residuals" in record:
        # Recomputed components duplicate the retained physical component
        # totals; the individual residual and tolerance maps are audit detail.
        for name in ("physical_residuals", "component_residuals", "component_validation_allowances",
                     "recomputed_components"):
            record.pop(name, None)
        record["validation_passed"] = True
        record["validation_residuals_omitted_after_pass"] = True
    return record
