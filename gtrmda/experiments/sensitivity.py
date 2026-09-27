from __future__ import annotations

from copy import deepcopy


GRID = {
    "demonstrations": [0, 2, 4, 8, 16],
    "budget": [1, 2, 4, 8, 16],
    "hops": [1, 2, 3, 4, 5],
    "lambda_cf": [0.0, 0.1, 0.2, 0.5, 1.0],
}


def sensitivity_config(base: dict, parameter: str, value) -> dict:
    if parameter not in GRID:
        raise ValueError(f"Unknown sensitivity parameter: {parameter}")
    config = deepcopy(base)
    section, key = {
        "demonstrations": ("data", "demonstrations"),
        "budget": ("evaluation", "default_budget"),
        "hops": ("data", "hops"),
        "lambda_cf": ("training", "lambda_cf"),
    }[parameter]
    config[section][key] = value
    return config

