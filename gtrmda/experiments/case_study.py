from __future__ import annotations

import torch

from .factory import move_demonstrations


@torch.no_grad()
def explain_query(model, bundle, builder, sampler, config, mirna, disease, device):
    model.eval()
    relation = config["data"]["target_relation"]
    prompt = builder.build(mirna, relation, disease, seed=config["seed"]).to(device)
    demonstrations = move_demonstrations(
        sampler.sample(config["data"]["demonstrations"], relation), device
    )
    relation_features = torch.from_numpy(bundle.relation_features).float().to(device)
    result = model.infer(
        prompt, relation_features, demonstrations,
        budget=config["evaluation"]["default_budget"], seed=config["seed"],
    )
    evidence = []
    for rank, trajectory_index in enumerate(
        torch.argsort(result.aggregation_weights, descending=True).tolist(), start=1
    ):
        trajectory = result.trajectories[trajectory_index]
        edges = []
        for edge_index in trajectory.selected_edges.tolist():
            src = int(prompt.edge_index[0, edge_index])
            dst = int(prompt.edge_index[1, edge_index])
            relation_id = bundle.graph.relations[int(prompt.edge_types[edge_index])].id
            edges.append({
                "head": prompt.node_ids[src], "relation": relation_id,
                "tail": prompt.node_ids[dst],
                "attention": float(trajectory.edge_weights[edge_index].cpu())
                if trajectory.edge_weights.numel() else 0.0,
            })
        evidence.append({
            "rank": rank,
            "trajectory": trajectory_index,
            "raw_score": float(torch.sigmoid(trajectory.score).cpu()),
            "aggregation_weight": float(result.aggregation_weights[trajectory_index].cpu()),
            "verifier_features": trajectory.verifier_features.cpu().tolist(),
            "edges": edges,
        })
    return {"mirna": mirna, "disease": disease, "score": float(result.score.cpu()),
            "evidence": evidence}
