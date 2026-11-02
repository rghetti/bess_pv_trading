"""Resolve January participation study lengths without changing source YAMLs."""
from copy import deepcopy


def january_participation_days(config, days):
    """Return the requested first January days with a distinct resumable run ID."""
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 31:
        raise ValueError("January participation days must be an integer from 1 to 31")
    result = deepcopy(config)
    settings = result["continuous_optimization"]
    if (str(settings.get("start_date")) != "2026-01-01"
            or str(settings.get("end_date")) != "2026-01-10"
            or "_ten_days_" not in str(settings.get("run_id", ""))
            or not result.get("continuous_participation_study", {}).get("enabled", False)):
        raise ValueError("January day count requires a ten-day participation study configuration")
    # Keep the old ten-day configuration exactly identical to existing checkpoints.
    if days != 10:
        settings["end_date"] = f"2026-01-{days:02d}"
        settings["run_id"] = settings["run_id"].replace("_ten_days_", f"_days_{days:02d}_", 1)
    return result
