"""Chronology, target isolation, network contracts and CPU artifact reproducibility."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from bess_pv_trading.config import load_config
from bess_pv_trading.data.contracts import FarmMetadata
from bess_pv_trading.forecasting.pv_neural_data import (
    feature_frame, fit_preprocessor, pack_sequences, transform_features,
    unpack_predictions, validate_neural_config)
from bess_pv_trading.forecasting.pv_neural_models import PVNetwork
from bess_pv_trading.forecasting.pv_neural_training import (
    fit_network, load_network, predict_network, save_network, tensor_pack)


@pytest.fixture
def neural_case():
    config = load_config("configs/pv_experiments/neural_validation.yaml")
    section = deepcopy(config["pv_neural"])
    section.update(max_epochs=4, patience=3, batch_size=2, prediction_batch_size=2)
    candidate = {**section["candidates"][0], "width": 8, "depth": 2}
    frames = []
    for split, day in zip(("train", "validation", "test"), ("2025-01-05", "2025-02-03", "2025-02-10")):
        times = pd.date_range(day+" 10:00", periods=12, freq="15min", tz="UTC")
        part = pd.DataFrame({"interval_start_utc": times, "delivery_day": day,
            "auction": "DA", "forecast_cycle_utc": times[0]-pd.Timedelta(days=1),
            "lead_hours": np.arange(12)/4+24, "solar_zenith_deg": np.linspace(60, 40, 12),
            "solar_azimuth_deg": np.linspace(120, 190, 12), "split": split,
            "eligible": True, "daylight": True, "measured_ac_mw": np.linspace(.1, .6, 12)})
        for name in config["forecast_mapping"]:
            part[name] = np.linspace(1, 2, len(part))
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    return config, section, candidate, frame


def test_neural_features_are_independent_of_targets_and_fitted_pv_parameters(neural_case):
    _, section, _, frame = neural_case
    before = feature_frame(frame, "weather", section["delivery_timezone"])
    changed = frame.copy()
    changed["measured_ac_mw"] = 1000
    changed["baseline_mw"] = -1000
    changed["u0"] = -10
    changed["calibrated_coefficient"] = 99
    pd.testing.assert_frame_equal(before, feature_frame(changed, "weather", section["delivery_timezone"]))
    assert not {"measured_ac_mw", "baseline_mw", "u0", "calibrated_coefficient"} & set(before)


def test_neural_imputation_and_scaling_use_training_only(neural_case):
    _, section, _, frame = neural_case
    features = feature_frame(frame, "core", section["delivery_timezone"])
    mask = (frame.split == "train").to_numpy()
    original = fit_preprocessor(features, mask, section)
    modified = features.copy()
    modified.loc[~mask, :] = np.nan
    assert fit_preprocessor(modified, mask, section) == original
    transformed = transform_features(modified, original)
    assert np.isfinite(transformed).all()
    assert (transformed[~mask, features.shape[1]:] == 1).all()


@pytest.mark.parametrize("architecture", ["mlp", "residual", "tcn"])
def test_network_architectures_have_finite_gradients_and_physical_outputs(neural_case, architecture):
    _, section, candidate, frame = neural_case
    candidate.update(architecture=architecture, kernel_size=3)
    features = feature_frame(frame, "core", section["delivery_timezone"])
    preprocessor = fit_preprocessor(features, (frame.split == "train").to_numpy(), section)
    packed = pack_sequences(frame, transform_features(features, preprocessor), 1.)
    tensors = tensor_pack(packed)
    network = PVNetwork(packed["x"].shape[-1], candidate, section)
    prediction = network(tensors["x"], tensors["month"], tensors["auction"], tensors["daylight"])
    assert bool(((prediction >= 0) & (prediction <= 1)).all())
    prediction.sum().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in network.parameters())
    np.testing.assert_allclose(unpack_predictions(packed, prediction.detach().numpy(), len(frame), 1.),
                               prediction.detach().numpy().reshape(-1))


@pytest.mark.parametrize("architecture", ["mlp", "residual", "tcn"])
def test_future_test_targets_cannot_change_training_or_validation_selection(neural_case, architecture):
    _, section, candidate, frame = neural_case
    candidate.update(architecture=architecture, kernel_size=3)
    features = feature_frame(frame, "core", section["delivery_timezone"])
    preprocessor = fit_preprocessor(features, (frame.split == "train").to_numpy(), section)
    packed = pack_sequences(frame, transform_features(features, preprocessor), 1.)
    altered = deepcopy(packed)
    altered["y"][altered["split"] == "test"] = 999
    torch.set_num_threads(1)
    original, diagnostic = fit_network(packed, candidate, section, seed=section["seed"])
    changed, other = fit_network(altered, candidate, section, seed=section["seed"])
    assert diagnostic["validation_nrmse"] == other["validation_nrmse"]
    assert diagnostic["best_epoch"] == other["best_epoch"]
    for key in original.state_dict():
        torch.testing.assert_close(original.state_dict()[key], changed.state_dict()[key], rtol=0, atol=0)


def test_neural_weights_reload_and_farm_checksum_contract(neural_case, tmp_path):
    config, section, candidate, frame = neural_case
    features = feature_frame(frame, "core", section["delivery_timezone"])
    preprocessor = fit_preprocessor(features, (frame.split == "train").to_numpy(), section)
    packed = pack_sequences(frame, transform_features(features, preprocessor), 1.)
    model, diagnostic = fit_network(packed, candidate, section, seed=section["seed"])
    farm_id = config["farm_id"]
    farm = FarmMetadata(farm_id, "NORD", config["farms"][str(farm_id)]["ac_capacity_kw"], 1, 0, 0, 0)
    path = tmp_path/"network.pt"
    save_network(model, path, {"candidate": candidate, "preprocessor": preprocessor,
        "neural_settings": section, "diagnostics": diagnostic, "farm_metadata": asdict(farm)})
    restored, _ = load_network(path, farm_id=farm_id)
    np.testing.assert_array_equal(predict_network(model, tensor_pack(packed), section["prediction_batch_size"]),
                                  predict_network(restored, tensor_pack(packed), section["prediction_batch_size"]))
    from bess_pv_trading.forecasting.pv_neural_predict import predict_neural_frame
    future_free = predict_neural_frame(frame.drop(columns="measured_ac_mw"), path, farm_id=farm_id)
    expected = unpack_predictions(packed,
        predict_network(model, tensor_pack(packed), section["prediction_batch_size"]), len(frame), farm.ac_capacity_mw)
    np.testing.assert_array_equal(future_free, expected)
    with pytest.raises(ValueError, match="different farm"):
        load_network(path, farm_id=-1)
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_network(path, farm_id=farm_id)


def test_neural_config_rejects_nonchronological_splits_and_gpu(neural_case):
    config, _, _, _ = neural_case
    validate_neural_config(config)
    invalid = deepcopy(config)
    invalid["pv_neural"]["validation_end_day_exclusive"] = invalid["pv_neural"]["start_day"]
    with pytest.raises(ValueError, match="increasing"):
        validate_neural_config(invalid)
    invalid = deepcopy(config)
    invalid["pv_neural"]["device"] = "cuda"
    with pytest.raises(ValueError, match="CPU"):
        validate_neural_config(invalid)


def test_boundary_targets_are_embargoed_before_the_next_first_da(neural_case, monkeypatch, tmp_path):
    import bess_pv_trading.forecasting.pv_neural_data as module
    config, _, _, _ = neural_case
    config["pv_neural"].update(minimum_training_days=1, minimum_validation_days=1, minimum_test_days=1)
    dates = ["2025-01-30", "2025-01-31", "2025-02-06", "2025-02-07", "2025-02-10"]
    starts = pd.to_datetime([day+"T12:00Z" for day in dates])
    frame = pd.DataFrame({"delivery_day": dates, "eligible": True, "daylight": True,
                          "target_available_at_utc": starts+pd.Timedelta(minutes=15)})
    monkeypatch.setattr(module, "prepare_corpus", lambda *a, **k: (frame.copy(), None, {}, {}))
    prepared, _, _, _ = module.prepare_neural_corpus(config, config["farm_id"], tmp_path)
    assert prepared.eligible.tolist() == [True, False, True, False, True]


@pytest.mark.parametrize("day,quarters", [("2025-03-30", 92), ("2025-10-26", 100)])
def test_neural_sequences_preserve_dst_and_ida3_noon(day, quarters):
    from bess_pv_trading.markets import AuctionCalendar
    config = load_config("configs/pv_experiments/neural_2025_2026.yaml")
    calendar = AuctionCalendar.from_config(config)
    grid = calendar.delivery_grid(day)
    rows = []
    for event in calendar.schedule(day):
        times = grid.starts[list(event.eligible_intervals)]
        rows.append(pd.DataFrame({"interval_start_utc": times, "delivery_day": day,
            "forecast_cycle_utc": event.submission_at-pd.Timedelta(hours=12),
            "auction": event.auction, "split": "train", "measured_ac_mw": 0,
            "eligible": True, "daylight": True}))
    frame = pd.concat(rows, ignore_index=True)
    packed = pack_sequences(frame, np.ones((len(frame), 2), dtype=np.float32), 1.)
    assert sorted((packed["indices"] >= 0).sum(axis=1)) == sorted([quarters]*3+[48])
    ida3 = frame.loc[frame.auction == "IDA3", "interval_start_utc"].iloc[0]
    assert ida3.tz_convert(config["market"]["timezone"]).hour == 12


def test_new_weather_features_need_neither_proxy_geometry_nor_auction(neural_case):
    from bess_pv_trading.forecasting.pv_forecast_search import weather_features
    config, _, _, frame = neural_case
    original = weather_features(frame, 'weather', config['pv_neural']['delivery_timezone'])
    changed = frame.drop(columns=['solar_zenith_deg', 'solar_azimuth_deg']).copy()
    changed['auction'] = 'IDA3'
    changed['latitude'] = np.nan
    changed['longitude'] = np.nan
    changed['measured_ac_mw'] = np.nan
    changed['satellite_ghi_w_m2'] = np.nan
    pd.testing.assert_frame_equal(original,weather_features(changed,'weather',config['pv_neural']['delivery_timezone']))


def test_disabled_auction_embeddings_make_predictions_auction_invariant(neural_case):
    _,section,candidate,frame = neural_case
    assert section['auction_embedding_dim']==0
    features=feature_frame(frame,'core',section['delivery_timezone'])
    pre=fit_preprocessor(features,(frame.split=='train').to_numpy(),section)
    packed=pack_sequences(frame,transform_features(features,pre),1.)
    tensors=tensor_pack(packed)
    model=PVNetwork(packed['x'].shape[-1],candidate,section).eval()
    assert model.auction_embedding is None
    with torch.no_grad():
        first=model(tensors['x'],tensors['month'],tensors['auction'],tensors['daylight'])
        second=model(tensors['x'],tensors['month'],torch.ones_like(tensors['auction']),tensors['daylight'])
    torch.testing.assert_close(first,second,rtol=0,atol=0)


@pytest.mark.parametrize('architecture',['residual_mlp','mixture_experts','bigru','transformer'])
def test_new_architectures_have_bounded_outputs_and_finite_gradients(neural_case,architecture):
    from bess_pv_trading.forecasting.pv_forecast_search import weather_features
    from bess_pv_trading.forecasting.pv_residual_networks import WeatherResidualNetwork
    config,section,_,frame=neural_case
    features=weather_features(frame,'core',section['delivery_timezone'])
    pre=fit_preprocessor(features,(frame.split=='train').to_numpy(),section)
    packed=pack_sequences(frame,transform_features(features,pre),1.)
    tensors=tensor_pack(packed)
    spec={key:values[0] for key,values in config['pv_forecast_search']['neural_space'].items()}
    spec['architecture']=architecture
    model=WeatherResidualNetwork(packed['x'].shape[-1],spec)
    valid=torch.as_tensor(packed['indices']>=0)
    prediction=model(tensors['x'],tensors['y'],tensors['daylight'],valid)
    assert ((prediction>=0)&(prediction<=1)).all()
    prediction.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_online_gain_does_not_read_future_observations(neural_case):
    from bess_pv_trading.forecasting.pv_online_bias import completed_day_statistics,causal_gain_prediction
    config,section,_,frame=neural_case
    frame=frame.copy()
    duration=pd.Timedelta(minutes=config['source']['historical_interval_minutes'])
    frame['target_available_at_utc']=frame.interval_start_utc+duration
    frame['decision_cutoff_utc']=frame.forecast_cycle_utc
    prediction=frame.measured_ac_mw.to_numpy().copy()
    options=config['pv_online_bias']
    def compute(data):
        history=completed_day_statistics(data,prediction,section['delivery_timezone'])
        return causal_gain_prediction(data,prediction,history,window_days=max(options['window_days']),
            minimum_days=min(options['minimum_days']),gain_bounds=options['gain_bounds'])
    before,_=compute(frame)
    changed=frame.copy()
    future=changed.split.eq('test')
    changed.loc[future,'measured_ac_mw']=-changed.loc[future,'measured_ac_mw']
    after,_=compute(changed)
    np.testing.assert_array_equal(before[~future],after[~future])


def test_calibration_counts_overlapping_forecasts_once_and_waits_for_midnight(neural_case):
    from bess_pv_trading.forecasting.pv_online_bias import completed_day_statistics
    config,section,_,frame=neural_case
    frame=frame.copy();frame['target_available_at_utc']=frame.interval_start_utc
    doubled=pd.concat([frame,frame.assign(auction='IDA3')],ignore_index=True)
    result=completed_day_statistics(doubled,doubled.measured_ac_mw.to_numpy(),section['delivery_timezone'])
    assert result.quarters.sum()==len(frame)
    for _,row in result.iterrows():
        boundary=(pd.Timestamp(row.delivery_day).tz_localize(section['delivery_timezone'])+pd.DateOffset(days=1)).tz_convert('UTC')
        assert row.available>=boundary


def test_completed_day_calibration_preserves_dst_midnight_boundaries():
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.forecasting.pv_online_bias import completed_day_statistics
    config=load_config('tests/fixtures/configs/continuous_cases.yaml')
    config['market']['valid_to']=max(case['delivery_day'] for case in config['test_case']['dst_cases'])
    calendar=AuctionCalendar.from_config(config)
    duration=pd.Timedelta(minutes=config['source']['historical_interval_minutes'])
    timezone=config['market']['timezone']
    for case in config['test_case']['dst_cases']:
        starts=calendar.delivery_grid(case['delivery_day']).starts
        frame=pd.DataFrame({'delivery_day':case['delivery_day'],'interval_start_utc':starts,
            'target_available_at_utc':starts+duration,'measured_ac_mw':np.ones(len(starts))})
        history=completed_day_statistics(frame,frame.measured_ac_mw.to_numpy(),timezone)
        midnight=pd.Timestamp(case['delivery_day']).tz_localize(timezone)+pd.DateOffset(days=1)
        assert history.quarters.iloc[0]==case['expected_physical_intervals']
        assert history.available.iloc[0]==midnight.tz_convert('UTC')


@pytest.mark.parametrize('architecture',['direct_tcn','residual_tcn'])
def test_market_free_tcn_output_contract(neural_case,architecture):
    from bess_pv_trading.forecasting.pv_forecast_search import weather_features
    from bess_pv_trading.forecasting.pv_residual_networks import WeatherResidualNetwork
    config,section,_,frame=neural_case
    features=weather_features(frame,'core',section['delivery_timezone'])
    pre=fit_preprocessor(features,(frame.split=='train').to_numpy(),section)
    packed=pack_sequences(frame,transform_features(features,pre),1.)
    tensors=tensor_pack(packed)
    spec={key:values[0] for key,values in config['pv_forecast_search']['neural_space'].items()}
    spec['architecture']=architecture;model=WeatherResidualNetwork(packed['x'].shape[-1],spec)
    result=model(tensors['x'],tensors['y'],tensors['daylight'],torch.as_tensor(packed['indices']>=0))
    assert torch.isfinite(result).all()
    assert ((result>=0)&(result<=1)).all()
    result.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_bundle_preserves_convolution_padding_for_single_day_inference(neural_case):
    from bess_pv_trading.forecasting.pv_forecast_bundle import _pad_sequence_contract
    from bess_pv_trading.forecasting.pv_forecast_search import weather_features
    from bess_pv_trading.forecasting.pv_residual_networks import WeatherResidualNetwork
    config,section,_,frame=neural_case
    features=weather_features(frame,'core',section['delivery_timezone'])
    pre=fit_preprocessor(features,(frame.split=='train').to_numpy(),section)
    x=transform_features(features,pre)
    original=pack_sequences(frame,x,1.)
    # A training corpus containing an autumn DST day has a longer padded axis.
    length=original['indices'].shape[1]+int(60/config['source']['historical_interval_minutes'])
    full=_pad_sequence_contract(original,length)
    subset=frame.split.eq('test').to_numpy()
    single=_pad_sequence_contract(pack_sequences(frame.loc[subset].reset_index(drop=True),x[subset],1.),length)
    spec={key:values[0] for key,values in config['pv_forecast_search']['neural_space'].items()}
    spec['architecture']='direct_tcn'
    torch.manual_seed(section['seed'])
    model=WeatherResidualNetwork(x.shape[1],spec).eval()
    # Exercise a learned head rather than the constant zero-initialized head.
    torch.nn.init.normal_(model.head.weight)
    def predict(packed):
        with torch.no_grad():
            return model(torch.as_tensor(packed['x']),torch.zeros(packed['indices'].shape),
                torch.as_tensor(packed['daylight']),torch.as_tensor(packed['indices']>=0)).numpy()
    full_result=unpack_predictions(full,predict(full),len(frame),1.)[subset]
    single_result=unpack_predictions(single,predict(single),int(subset.sum()),1.)
    np.testing.assert_allclose(single_result,full_result,rtol=0,atol=1e-6)
    assert (single['indices'][:,original['indices'].shape[1]:]==-1).all()
    with pytest.raises(ValueError,match='exceeds'):
        _pad_sequence_contract(original,original['indices'].shape[1]-1)
