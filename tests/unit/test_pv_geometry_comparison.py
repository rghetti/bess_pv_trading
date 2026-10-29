"""Verify isolated geometry scenarios retain the fixed temperature contract."""

from types import SimpleNamespace

import pytest

from bess_pv_trading.config import load_config
from scripts.compare_pv_geometry_factorial import factorial_plan


@pytest.mark.parametrize('farm_id,tracking_id,combinations,unique', [(901, 3, 512, 320), (910, 1, 32, 32)])
def test_factorial_plan_enumerates_and_deduplicates_at_fixed_temperature(farm_id, tracking_id, combinations, unique):
    config = load_config('configs/pv_experiments/geometry_factorial_legacy_baseline.yaml')
    capacity = float(config['farms'][str(farm_id)]['ac_capacity_kw'])
    farm = SimpleNamespace(farm_id=farm_id, ac_capacity_kw=capacity,
                           ac_capacity_mw=capacity / 1000, tracking_type_id=tracking_id)
    candidate = {'farm_id': farm_id, 'candidate': {'location': 'verona', 'transposition': 'haydavies',
        'gcr': .5, 'max_angle': 45., 'backtrack': True, 'bifacial_enabled': tracking_id == 3,
        'albedo': .35 if tracking_id == 3 else .2,
        'bifaciality': .9 if tracking_id == 3 else .8,
        'height_to_pitch': .5 if tracking_id == 3 else .4},
        'parameters': {'coefficient': .7, 'u0': 300., 'u1': 80., 'gamma': -.002}}
    plan = factorial_plan(config, farm, candidate)
    assert len(plan['cases']) == combinations
    assert len(plan['unique_models']) == unique
    assert plan['fixed_temperature'] == {'u0': 25., 'u1': 6.84, 'gamma': -.004}
    assert set(plan['cases'][0]['choices'].values()) == {'old'}
    assert set(plan['cases'][-1]['choices'].values()) == {'new'}
    baseline = plan['unique_models'][plan['cases'][0]['unique_id']]['settings']
    changed = plan['unique_models'][plan['cases'][-1]['unique_id']]['settings']
    assert (baseline['location']['latitude'], baseline['location']['longitude']) == (45.4642, 9.19)
    assert (changed['location']['latitude'], changed['location']['longitude']) == (45.4384, 10.9916)
    assert changed['tracker']['gcr'] == .5
    assert changed['output']['calibrated_coefficient'] == .7
    for model in plan['unique_models']:
        assert model['settings']['temperature']['u0'] == 25.
        assert model['settings']['temperature']['u1'] == 6.84
        assert model['settings']['dc']['gamma_pdc_per_c'] == -.004


@pytest.mark.parametrize('farm_id,coordinates', [(901, (45.6495, 13.7768)), (910, (45.4384, 10.9916))])
def test_standard_farm_location_and_temperature(farm_id, coordinates):
    config = load_config('configs/standard.yaml', overrides={'farm_id': farm_id})
    location = config['pv']['location']
    assert (location['latitude'], location['longitude']) == coordinates
    assert location['location_status'] == 'proxy'
    assert config['pv']['temperature']['u0'] == 25.
    assert config['pv']['temperature']['u1'] == 6.84
    assert config['pv']['dc']['gamma_pdc_per_c'] == -.004
