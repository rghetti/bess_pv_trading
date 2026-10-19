"""Read independent battery reference cases without coupling them to study tuning."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from bess_pv_trading.config import load_config
from bess_pv_trading.domain import battery_parameters_from_config


@lru_cache
def _cases():
    return load_config(Path(__file__).resolve().parents[1] / "fixtures/configs/battery_cases.yaml")


def case_config(name):
    """Return an isolated case so negative tests cannot mutate another case."""
    return deepcopy(_cases()[name])


def battery_config(name):
    return case_config("batteries")[name]


def configured_battery(name="unit"):
    return battery_parameters_from_config(battery_config(name))
