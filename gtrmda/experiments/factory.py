from __future__ import annotations

import torch

from gtrmda.data.io import DatasetBundle
from gtrmda.data.prompt import DemonstrationSampler, QueryPromptBuilder
from gtrmda.methodology.model import GTRMDA


def relation_endpoint_types(bundle: DatasetBundle) -> tuple[torch.Tensor, torch.Tensor]:
    graph = bundle.graph
    heads = [graph.type_to_idx[relation.head_type] for relation in graph.relations]
    tails = [graph.type_to_idx[relation.tail_type] for relation in graph.relations]
    return torch.tensor(heads), torch.tensor(tails)


def build_model(bundle: DatasetBundle, config: dict) -> GTRMDA:
    model_config = config["model"]
    heads, tails = relation_endpoint_types(bundle)
    return GTRMDA(
        feature_dim=model_config["feature_dim"],
        relation_feature_dim=model_config["relation_feature_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_node_types=max(model_config["num_node_types"], len(bundle.graph.type_names)),
        relation_head_types=heads,
        relation_tail_types=tails,
        relation_layers=model_config["relation_layers"],
        entity_layers=model_config["entity_layers"],
        verifier_hidden_dim=model_config["verifier_hidden_dim"],
        dropout=model_config["dropout"],
    )


def build_prompt_components(bundle: DatasetBundle, config: dict):
    data_config = config["data"]
    builder = QueryPromptBuilder(
        bundle.graph,
        bundle.entity_features,
        hops=data_config["hops"],
        max_neighbors=data_config["max_neighbors"],
        always_excluded_relations=set(data_config["target_aliases"]),
    )
    return builder, DemonstrationSampler(builder, seed=config["seed"])


def move_demonstrations(demonstrations, device):
    for item in demonstrations:
        item.prompt = item.prompt.to(device)
    return demonstrations

