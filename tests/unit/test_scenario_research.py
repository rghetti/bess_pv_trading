"""Probability, dependence, information and reduction contracts on small finite laws."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bess_pv_trading.scenario_research import (
    ReductionSettings, ScenarioTrajectories, energy_score, observable_history_nodes,
    reduce_scenarios, transform_dependence, validate_history_nodes, variogram_score, weighted_crps,
)


def bundle_for(pv, weights=None, prices=False):
    """Coupled prices make accidental destruction of same-time dependence visible."""
    pv = np.asarray(pv, dtype=float)
    weights = np.full(len(pv), 1 / len(pv)) if weights is None else np.asarray(weights)
    positive = 100 * pv + 3 if prices else None
    negative = -50 * pv - 7 if prices else None
    return ScenarioTrajectories(pv, weights, tuple(f"s{index}" for index in range(len(pv))), positive, negative)


def test_joint_trajectory_inputs_are_copied_frozen_and_validated():
    pv = np.array([[1.0, 2.0], [3.0, 4.0]])
    provenance = {"source": {"snapshots": ["one"]}}
    bundle = replace(bundle_for(pv), provenance=provenance)
    pv[0, 0] = 99
    provenance["source"]["snapshots"].append("two")
    assert bundle.pv_ac_mw[0, 0] == 1
    assert bundle.provenance["source"]["snapshots"] == ("one",)
    with pytest.raises(ValueError):
        bundle.pv_ac_mw[0, 0] = 10
    with pytest.raises(TypeError):
        bundle.provenance["source"]["snapshots"] = ()
    with pytest.raises(ValueError, match="weights"):
        replace(bundle, weights=np.array([0.5, 0.6]))
    with pytest.raises(ValueError, match="match"):
        replace(bundle, positive_prices_eur_per_mwh=np.ones((2, 1)))
    with pytest.raises(ValueError, match="requires"):
        replace(bundle, negative_prices_eur_per_mwh=np.ones((2, 2)))


def test_shuffle_preserves_joint_interval_law_and_changes_temporal_pairing():
    bundle = bundle_for(np.arange(24).reshape(4, 6), prices=True)
    shuffled = transform_dependence(bundle, "independent_shuffle", seed=5)
    assert not np.array_equal(shuffled.pv_ac_mw, bundle.pv_ac_mw)
    np.testing.assert_array_equal(np.sort(shuffled.pv_ac_mw, axis=0), np.sort(bundle.pv_ac_mw, axis=0))
    np.testing.assert_array_equal(shuffled.positive_prices_eur_per_mwh, 100 * shuffled.pv_ac_mw + 3)
    np.testing.assert_array_equal(shuffled.negative_prices_eur_per_mwh, -50 * shuffled.pv_ac_mw - 7)
    np.testing.assert_array_equal(shuffled.pv_ac_mw,
        transform_dependence(bundle, "independent_shuffle", seed=5).pv_ac_mw)
    assert transform_dependence(bundle, "raw") is bundle
    with pytest.raises(ValueError, match="equal"):
        transform_dependence(replace(bundle, weights=np.array([0.1, 0.2, 0.3, 0.4])), "independent_shuffle")


def test_ecc_keeps_calibrated_marginals_and_template_ranks():
    bundle = bundle_for([[3, 1], [1, 3], [2, 2]], prices=True)
    calibrated = np.array([[8, 7], [6, 9], [7, 8]])
    coupled = transform_dependence(bundle, "ecc", calibrated_pv_marginals=calibrated)
    np.testing.assert_array_equal(np.sort(coupled.pv_ac_mw, axis=0), np.sort(calibrated, axis=0))
    np.testing.assert_array_equal(np.argsort(coupled.pv_ac_mw, axis=0), np.argsort(bundle.pv_ac_mw, axis=0))
    np.testing.assert_array_equal(coupled.positive_prices_eur_per_mwh, bundle.positive_prices_eur_per_mwh)
    unchanged = transform_dependence(bundle, "ecc", calibrated_pv_marginals=bundle.pv_ac_mw)
    assert unchanged.provenance["dependence_transform"]["unchanged_pv"] is True
    reverse = transform_dependence(bundle, "ecc", calibrated_pv_marginals=calibrated,
        reference_pv=4 - bundle.pv_ac_mw)
    assert not np.array_equal(reverse.pv_ac_mw, coupled.pv_ac_mw)
    with pytest.raises(ValueError, match="requires"):
        transform_dependence(bundle, "ecc")
    with pytest.raises(ValueError, match="only"):
        transform_dependence(bundle, "raw", calibrated_pv_marginals=calibrated)


def test_ecc_ties_reproducible_without_claiming_unidentified_raw_dependence():
    bundle = bundle_for(np.ones((5, 3)))
    samples = np.arange(15).reshape(5, 3)
    first = transform_dependence(bundle, "ecc", seed=7, calibrated_pv_marginals=samples)
    second = transform_dependence(bundle, "ecc", seed=7, calibrated_pv_marginals=samples)
    np.testing.assert_array_equal(first.pv_ac_mw, second.pv_ac_mw)
    np.testing.assert_array_equal(np.sort(first.pv_ac_mw, axis=0), np.sort(samples, axis=0))
    assert first.provenance["dependence_transform"]["ties"] == "seeded_random"


def test_history_partitions_never_reveal_current_or_future_values():
    pv = np.array([[1, 2, 3, 4], [1, 2, 9, 5], [1, 8, 3, 2], [5, 2, 3, 4]])
    nodes = observable_history_nodes(pv)
    assert len(set(nodes[:, 0])) == 1
    assert nodes[0, 1] == nodes[1, 1] == nodes[2, 1]
    assert nodes[3, 1] != nodes[0, 1]
    assert nodes[0, 2] == nodes[1, 2]
    assert nodes[0, 2] != nodes[2, 2]
    assert nodes[0, 3] != nodes[1, 3]
    # Changing a suffix cannot alter any decisions preceding its revelation.
    changed = pv.copy()
    changed[:, 2:] += 100
    np.testing.assert_array_equal(nodes[:, :3], observable_history_nodes(changed)[:, :3])
    assert not nodes.flags.writeable
    np.testing.assert_array_equal(nodes, validate_history_nodes(nodes, pv))
    anticipated = nodes.copy()
    anticipated[1, 1] = 99
    with pytest.raises(ValueError, match="indistinguishable"):
        validate_history_nodes(anticipated, pv)
    merged = nodes.copy()
    merged[:, 2] = 42
    with pytest.raises(ValueError, match="merge"):
        validate_history_nodes(merged, pv)


def test_coarse_bins_branch_stages_and_lags_limit_recourse_information():
    pv = np.array([[0.11, 0.12, 1, 1], [0.19, 0.18, 2, 2], [0.25, 0.26, 3, 3]])
    coarse = observable_history_nodes(pv, observation_quantum_mw=0.2)
    assert coarse[0, 1] == coarse[1, 1]
    assert coarse[0, 2] == coarse[1, 2]
    assert coarse[0, 1] != coarse[2, 1]
    delayed = observable_history_nodes(pv, revelation_lag_intervals=2)
    assert len(set(delayed[:, 1])) == 1
    assert len(set(delayed[:, 2])) == 3
    sparse = observable_history_nodes(pv, branch_intervals=(2,))
    assert len(set(sparse[:, 1])) == 1
    np.testing.assert_array_equal(sparse[:, 2], sparse[:, 3])
    np.testing.assert_array_equal(observable_history_nodes(pv, branch_intervals=()), np.zeros(pv.shape))
    with pytest.raises(ValueError, match="positive"):
        observable_history_nodes(pv, revelation_lag_intervals=0)


def test_additional_observed_features_are_delayed_and_quantized():
    pv = np.ones((3, 3))
    prices = np.array([[[10], [100], [50]], [[11], [200], [0]], [[20], [300], [-10]]])
    nodes = observable_history_nodes(pv, observed_features=prices, feature_quantum=[5])
    assert len(set(nodes[:, 0])) == 1
    assert nodes[0, 1] == nodes[1, 1]
    assert nodes[0, 1] != nodes[2, 1]
    assert nodes[0, 2] != nodes[1, 2]


def test_reduction_retains_actual_members_and_transports_every_probability():
    bundle = bundle_for([[0, 0], [0.1, 0.2], [3, 4], [3.1, 3.9]], [0.1, 0.2, 0.3, 0.4], prices=True)
    result = reduce_scenarios(bundle, 2, battery_capacity_mwh=0.5, poi_export_mw=2, poi_import_mw=1)
    np.testing.assert_array_equal(result.bundle.pv_ac_mw, bundle.pv_ac_mw[result.selected_indices])
    np.testing.assert_array_equal(result.bundle.positive_prices_eur_per_mwh,
        bundle.positive_prices_eur_per_mwh[result.selected_indices])
    np.testing.assert_allclose(result.bundle.weights, np.bincount(result.assignment, bundle.weights, minlength=2))
    assert result.bundle.weights.sum() == pytest.approx(1)
    assert all(result.bundle.weights > 0)
    assert result.bundle.provenance["scenario_reduction"]["milp_value_loss_bound"] is False
    repeated = reduce_scenarios(bundle, 2, battery_capacity_mwh=0.5, poi_export_mw=2, poi_import_mw=1)
    np.testing.assert_array_equal(result.selected_indices, repeated.selected_indices)
    np.testing.assert_array_equal(result.assignment, repeated.assignment)
    assert not result.assignment.flags.writeable


def test_duplicate_members_retain_mass_and_zero_weight_members_are_not_selected():
    bundle = bundle_for([[1, 1], [1, 1], [20, 20]], [0.4, 0.6, 0])
    result = reduce_scenarios(bundle, 2)
    np.testing.assert_array_equal(np.sort(result.selected_indices), [0, 1])
    assert result.weighted_distance == pytest.approx(0, abs=1e-7)
    assert sorted(result.bundle.weights) == [0.4, 0.6]
    with pytest.raises(ValueError, match="positive-weight"):
        reduce_scenarios(bundle, 3)


def test_reduction_feature_weights_change_medoid_when_cost_exposure_differs():
    bundle = bundle_for([[0, 0], [1, 1], [2, 2]])
    prices = np.array([[0, 0], [0, 0], [1000, 1000]])
    bundle = replace(bundle, positive_prices_eur_per_mwh=prices)
    physical = reduce_scenarios(bundle, 1, settings=ReductionSettings(
        cumulative_deficit_weight=0, cumulative_balance_weight=0, saturation_weight=0,
        poi_weight=0, settlement_weight=0))
    economic = reduce_scenarios(bundle, 1, settings=ReductionSettings(
        pv_weight=0, cumulative_deficit_weight=0, cumulative_balance_weight=0, saturation_weight=0,
        poi_weight=0, settlement_weight=1))
    assert physical.selected_indices[0] == 1
    assert economic.selected_indices[0] == 0
    assert economic.bundle.provenance["scenario_reduction"]["active_features"] == ("uncontrolled_settlement",)


def test_candidate_blocks_produce_same_reduction_and_full_count_loses_no_mass():
    bundle = bundle_for(np.arange(35).reshape(7, 5) % 11, prices=True)
    small = reduce_scenarios(bundle, 3, settings=ReductionSettings(distance_block_size=1))
    large = reduce_scenarios(bundle, 3, settings=ReductionSettings(distance_block_size=100))
    np.testing.assert_array_equal(small.selected_indices, large.selected_indices)
    np.testing.assert_array_equal(small.assignment, large.assignment)
    full = reduce_scenarios(bundle, 7)
    assert full.weighted_distance == pytest.approx(0, abs=1e-6)
    np.testing.assert_array_equal(np.sort(full.selected_indices), np.arange(7))


def test_weighted_scores_match_direct_finite_distribution_formulas():
    samples = np.array([[0, 1, 2], [1, 2, 4], [3, 4, 9]], dtype=float)
    observation = np.array([1, 3, 5])
    weights = np.array([0.2, 0.3, 0.5])
    expected_crps = weights @ np.abs(samples - observation)
    expected_energy = weights @ np.linalg.norm(samples - observation, axis=1)
    for first in range(3):
        for second in range(3):
            expected_crps -= weights[first] * weights[second] * np.abs(samples[first] - samples[second]) / 2
            expected_energy -= weights[first] * weights[second] * np.linalg.norm(samples[first] - samples[second]) / 2
    np.testing.assert_allclose(weighted_crps(samples, observation, weights), expected_crps)
    assert energy_score(samples, observation, weights, block_size=1) == pytest.approx(expected_energy)
    expected_variogram = sum((abs(observation[t] - observation[u]) ** 0.5
        - weights @ np.abs(samples[:, t] - samples[:, u]) ** 0.5) ** 2
        for t in range(3) for u in range(t + 1, 3))
    assert variogram_score(samples, observation, weights) == pytest.approx(expected_variogram)


def test_marginal_crps_is_invariant_to_shuffle_but_joint_scores_can_differ():
    bundle = bundle_for([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    shuffled = transform_dependence(bundle, "independent_shuffle", seed=4)
    observed = np.array([1, 1, 1])
    np.testing.assert_allclose(weighted_crps(bundle.pv_ac_mw, observed), weighted_crps(shuffled.pv_ac_mw, observed))
    assert energy_score(bundle.pv_ac_mw, observed) != pytest.approx(energy_score(shuffled.pv_ac_mw, observed))
    assert variogram_score(bundle.pv_ac_mw, observed) == 0
    assert variogram_score(shuffled.pv_ac_mw, observed) > 0
    assert variogram_score(np.ones((3, 1)), [1]) == 0


@pytest.mark.parametrize("arguments", [
    {"distance_block_size": True}, {"power_scale_mw": 0}, {"pv_weight": -1}, {"settlement_weight": np.nan},
])
def test_invalid_reduction_settings_fail(arguments):
    with pytest.raises(ValueError):
        ReductionSettings(**arguments)


def test_open_loop_mpc_per_lead_relabeling_is_a_negative_control():
    """A solver-level invariant prevents claiming temporal value in shared controls."""
    from bess_pv_trading.domain import (
        BatteryParameters, DegradationSettings, DeliveryGrid, PhysicalState, PlantParameters, SolverSettings,
    )
    from bess_pv_trading.optimization.mpc import MpcInput, MpcSettings, solve_mpc

    bundle = bundle_for([[0, 0, 0.5], [0.2, 0.5, 0], [0.5, 0, 0.2]])
    shuffled = transform_dependence(bundle, "independent_shuffle", seed=3)
    start = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    starts = tuple(start + timedelta(minutes=15 * index) for index in range(3))
    grid = DeliveryGrid(tuple(stamp.isoformat() for stamp in starts), starts,
        tuple(stamp + timedelta(minutes=15) for stamp in starts))
    battery = BatteryParameters(capacity_mwh=1, max_power_mw=1, converter_charge_efficiency=1,
        converter_discharge_efficiency=1, energy_min_mwh=0, energy_max_mwh=1)
    inputs = MpcInput(start, grid, PlantParameters(1, 1, 1, battery), PhysicalState(start, 0.1),
        np.array([0.1, 0.1, 0.1]), bundle.pv_ac_mw, bundle.scenario_ids, bundle.weights,
        DegradationSettings(1), policy="ensemble_mpc", settings=MpcSettings(terminal_inventory_mode="none"))
    solver = SolverSettings(threads=1, feasibility_tolerance=1e-8, integrality_tolerance=1e-8)
    original_result = solve_mpc(inputs, solver)
    shuffled_result = solve_mpc(replace(inputs, pv_ac_mw=shuffled.pv_ac_mw), solver)
    assert original_result.plan is not None, original_result.diagnostics
    assert shuffled_result.plan is not None, shuffled_result.diagnostics
    assert original_result.diagnostics["planning_objective_value"] == pytest.approx(
        shuffled_result.diagnostics["planning_objective_value"], abs=1e-6)
