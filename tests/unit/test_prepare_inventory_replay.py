"""Verify portable controller configurations are relocated without execution."""
import importlib.util
from pathlib import Path

import pytest


def preparer():
    """Load the standalone preparation script for focused unit checks."""
    script = Path(__file__).resolve().parents[2] / 'scripts/prepare_inventory_replay.py'
    spec = importlib.util.spec_from_file_location('portable_preparer', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record():
    """Supply one minimal historical case with an enabled recovery policy."""
    return {'repository': '/old/repository', 'farm': '1', 'status': 'complete', 'mode': 'continuous',
        'resolved_config': {'_project_root': '/old/repository',
            'continuous_optimization': {'output_root': '/old/repository/outputs/old', 'run_id': 'old', 'resume': True},
            'experiment_outputs': {'overwrite_existing': True},
            'realtime_mpc_comparison': {'mode': 'continuous', 'controllers': ['track_committed_grid','deterministic_mpc']},
            'realtime_mpc': {'provisional_ida3_mode': 'ida2_plan'},
            'execution': {'mpc_curtailment_recovery': True},
            'pv': {'forecast_replacement': {'enabled': False}},
            '_pv_shared_settings': {'forecast_replacement': {'enabled': False}},
            'inputs': {'zarr': '/old/repository/data/forecast.zarr'}}}


def test_prepare_rebases_and_pins_new_model(tmp_path):
    """New model, safe run identity and original controller settings survive."""
    module = preparer()
    old = record()
    model = {'name': 'new', 'kind': 'reviewed_hpo', 'artifact_path': 'models/farm_1/result.json',
             'available_at_utc': '2026-11-06T00:00:00Z'}
    result = module.prepare(old, tmp_path, 'new_run', {'1': model}, True)
    assert old['resolved_config']['continuous_optimization']['run_id'] == 'old'
    assert result['_project_root'] == str(tmp_path)
    assert result['inputs']['zarr'] == str(tmp_path / 'data/forecast.zarr')
    assert result['continuous_optimization']['run_id'] == 'new_run'
    assert not result['continuous_optimization']['resume']
    assert not result['experiment_outputs']['overwrite_existing']
    assert result['realtime_mpc']['provisional_ida3_mode'] == 'ida2_plan'
    assert result['execution']['mpc_curtailment_recovery']
    assert result['_pv_shared_settings']['forecast_replacement']['models_by_farm']['1'] == model
    assert result['include'] == '../standard.yaml'


def test_prepare_rejects_incomplete_or_frozen_records(tmp_path):
    """A failed or frozen source cannot silently become a continuous rerun."""
    module = preparer()
    item = record()
    item['status'] = 'failed'
    with pytest.raises(ValueError, match='did not complete'):
        module.prepare(item, tmp_path, 'new_run', None, False)
    item['status'] = 'complete'
    item['mode'] = 'frozen_commitments'
    with pytest.raises(ValueError, match='continuous comparisons'):
        module.prepare(item, tmp_path, 'new_run', None, False)


def test_cli_writes_numeric_tolerances_as_yaml_numbers(tmp_path):
    """Scientific-notation tolerances remain numeric after YAML parsing."""
    import gzip
    import json
    import subprocess
    import sys
    import yaml

    item = record()
    item.update(batch='study', job_id='job')
    item['resolved_config']['solver'] = {'feasibility_tolerance': 1e-8}
    inventory = tmp_path / 'inventory.json.gz'
    with gzip.open(inventory, 'wt') as stream:
        json.dump({'research_jobs': [item], 'main_checkout_modes': []}, stream)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/standard.yaml').write_text('{}\n')
    script = Path(__file__).resolve().parents[2] / 'scripts/prepare_inventory_replay.py'
    subprocess.run([sys.executable, str(script), '--inventory', str(inventory), '--root', str(tmp_path),
        '--id', 'research:study:job', '--run-id', 'safe_run', '--keep-original-model'], check=True)
    output = tmp_path / 'configs/research/portable_safe_run.yaml'
    result = yaml.safe_load(output.read_text())
    assert isinstance(result['solver']['feasibility_tolerance'], float)
    assert result['solver']['feasibility_tolerance'] == 1e-8
