from __future__ import annotations

from copy import deepcopy


ABLATIONS = {
    "full": {},
    "without_relation_semantics": {"zero_relation_features": True},
    "without_in_context_demonstrations": {"demonstrations": 0},
    "without_test_time_verifier": {"uniform_aggregation": True},
    "without_counterfactual_loss": {"lambda_cf": 0.0},
    "single_trajectory": {"budget": 1},
}


def ablation_config(base: dict, name: str) -> tuple[dict, dict]:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {name}")
    config = deepcopy(base)
    flags = dict(ABLATIONS[name])
    if "demonstrations" in flags:
        config["data"]["demonstrations"] = flags["demonstrations"]
    if "lambda_cf" in flags:
        config["training"]["lambda_cf"] = flags["lambda_cf"]
    if "budget" in flags:
        config["evaluation"]["default_budget"] = flags["budget"]
    return config, flags

