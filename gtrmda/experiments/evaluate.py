from __future__ import annotations

import pandas as pd
import torch

from .factory import move_demonstrations
from .metrics import binary_metrics, ranking_metrics


@torch.no_grad()
def evaluate_query_table(
    model,
    bundle,
    builder,
    demonstration_sampler,
    config: dict,
    device: torch.device,
    query_table: pd.DataFrame | None = None,
    budget: int | None = None,
    uniform_aggregation: bool = False,
    candidate_diseases: set[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    table = bundle.queries if query_table is None else query_table
    relation = config["data"]["target_relation"]
    relation_features = torch.from_numpy(bundle.relation_features).float().to(device)
    demonstrations = move_demonstrations(
        demonstration_sampler.sample(config["data"]["demonstrations"], relation), device
    )
    budget = budget or config["evaluation"]["default_budget"]
    relation_spec = bundle.graph.relations[bundle.graph.relation_to_idx[relation]]
    diseases = sorted(candidate_diseases or {
        entity.id for entity in bundle.graph.entities
        if entity.type == relation_spec.tail_type
    })
    mirnas = sorted(set(table["mirna"]))
    known_labels = {
        (str(row.mirna), str(row.disease)): int(row.label)
        for row in table.itertuples(index=False)
    }
    records = []
    pair_index = 0
    for mirna in mirnas:
        for disease in diseases:
            pair_index += 1
            label = known_labels.get((mirna, disease), 0)
            prompt = builder.build(
                mirna, relation, disease, seed=config["seed"] + pair_index
            ).to(device)
            result = model.infer(
                prompt, relation_features, demonstrations,
                budget=budget, seed=config["seed"] + pair_index,
                temperature=config["evaluation"]["verifier_temperature"],
                uniform_aggregation=uniform_aggregation,
            )
            records.append({
                "mirna": mirna,
                "disease": disease,
                "label": label,
                "observed_label": int((mirna, disease) in known_labels),
                "score": float(result.score.cpu()),
                "mean_verifier": float(torch.stack(
                    [torch.sigmoid(item.verifier_logit) for item in result.trajectories]
                ).mean().cpu()),
            })
    predictions = pd.DataFrame(records)
    metrics = binary_metrics(predictions["label"], predictions["score"])

    ranks = []
    positives = predictions[predictions["label"] == 1]
    for positive in positives.itertuples(index=False):
        other_positives = {
            item.disease for item in positives.itertuples(index=False)
            if item.mirna == positive.mirna and item.disease != positive.disease
        }
        candidates = predictions[
            (predictions["mirna"] == positive.mirna)
            & (~predictions["disease"].isin(other_positives))
        ].sort_values("score", ascending=False)
        rank = int(candidates.reset_index(drop=True).index[
            candidates.reset_index(drop=True)["disease"] == positive.disease
        ][0]) + 1
        ranks.append(rank)
    metrics.update(ranking_metrics(
        ranks, hits=tuple(config["evaluation"]["hits"]),
        ndcg_k=config["evaluation"]["ndcg_k"],
    ))
    return metrics, predictions
