from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gtrmda.data.schema import GraphPrompt, LabeledPrompt

from .encoders import PromptNodeEncoder
from .entity_reasoner import EntityGraphReasoner
from .in_context import GraphInContextConditioner
from .relation_encoder import RelationGraphEncoder
from .verifier import EvidenceVerifier, verifier_features


@dataclass
class Trajectory:
    score: torch.Tensor
    verifier_logit: torch.Tensor | None
    edge_weights: torch.Tensor
    selected_edges: torch.Tensor
    verifier_features: torch.Tensor | None = None


@dataclass
class InferenceResult:
    score: torch.Tensor
    trajectories: list[Trajectory]
    aggregation_weights: torch.Tensor


class GTRMDA(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        relation_feature_dim: int,
        hidden_dim: int,
        num_node_types: int,
        relation_head_types: torch.Tensor,
        relation_tail_types: torch.Tensor,
        relation_layers: int = 2,
        entity_layers: int = 4,
        verifier_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_encoder = PromptNodeEncoder(
            feature_dim, hidden_dim, num_node_types, dropout=dropout
        )
        self.relation_encoder = RelationGraphEncoder(
            relation_feature_dim, hidden_dim, num_node_types, relation_layers, dropout
        )
        self.context = GraphInContextConditioner(hidden_dim, dropout)
        self.reasoner = EntityGraphReasoner(hidden_dim, entity_layers, dropout)
        self.verifier = EvidenceVerifier(verifier_hidden_dim, dropout)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("relation_head_types", relation_head_types.long())
        self.register_buffer("relation_tail_types", relation_tail_types.long())

    def encode_relations(self, relation_features: torch.Tensor) -> torch.Tensor:
        return self.relation_encoder(
            relation_features, self.relation_head_types, self.relation_tail_types
        )

    def condition_relation(
        self,
        prompt: GraphPrompt,
        demonstrations: list[LabeledPrompt],
        relation_states: torch.Tensor,
    ) -> torch.Tensor:
        query_relation = relation_states[prompt.query_relation]
        if not demonstrations:
            return query_relation
        graph_states, demo_relations, labels = [], [], []
        for demonstration in demonstrations:
            demo_prompt = demonstration.prompt
            node_states = self.node_encoder(demo_prompt)
            graph_states.append(node_states.mean(dim=0))
            demo_relations.append(relation_states[demo_prompt.query_relation])
            labels.append(demonstration.label)
        return self.context(
            query_relation,
            torch.stack(graph_states),
            torch.stack(demo_relations),
            torch.tensor(labels, device=query_relation.device),
        )

    def forward(
        self,
        prompt: GraphPrompt,
        relation_features: torch.Tensor,
        demonstrations: list[LabeledPrompt] | None = None,
        max_layers: int | None = None,
    ) -> Trajectory:
        relation_states = self.encode_relations(relation_features)
        conditioned = self.condition_relation(
            prompt, demonstrations or [], relation_states
        )
        initial = self.node_encoder(prompt)
        score, _, edge_weights = self.reasoner(
            initial, prompt, relation_states, conditioned, max_layers=max_layers
        )
        selected = torch.argsort(edge_weights, descending=True)[:8]
        return Trajectory(score, None, edge_weights, selected)

    def _structural_validity(self, prompt: GraphPrompt) -> torch.Tensor:
        if prompt.num_edges == 0:
            return prompt.node_features.new_zeros(())
        src, dst = prompt.edge_index
        src_types = prompt.node_types[src]
        dst_types = prompt.node_types[dst]
        head_types = self.relation_head_types[prompt.edge_types]
        tail_types = self.relation_tail_types[prompt.edge_types]
        forward = (src_types == head_types) & (dst_types == tail_types)
        inverse = (src_types == tail_types) & (dst_types == head_types)
        return (forward | inverse).float().mean()

    def infer(
        self,
        prompt: GraphPrompt,
        relation_features: torch.Tensor,
        demonstrations: list[LabeledPrompt] | None = None,
        budget: int = 8,
        seed: int = 2026,
        temperature: float | None = None,
        uniform_aggregation: bool = False,
    ) -> InferenceResult:
        trajectories = []
        generator = torch.Generator(device=prompt.edge_types.device).manual_seed(seed)
        for index in range(budget):
            if prompt.num_edges:
                retain_fraction = 0.75 + 0.25 * ((index % 4) / 3)
                count = max(1, int(round(prompt.num_edges * retain_fraction)))
                selected = torch.randperm(
                    prompt.num_edges, generator=generator, device=prompt.edge_types.device
                )[:count]
                sampled = prompt.select_edges(selected)
            else:
                selected = torch.empty(0, dtype=torch.long, device=prompt.edge_types.device)
                sampled = prompt
            depth = 1 + index % len(self.reasoner.layers)
            trajectory = self.forward(sampled, relation_features, demonstrations, max_layers=depth)
            if sampled.num_edges:
                key_count = max(1, min(8, sampled.num_edges // 8 or 1))
                key = trajectory.selected_edges[:key_count]
                random_order = torch.randperm(
                    sampled.num_edges, generator=generator, device=sampled.edge_types.device
                )
                random_edges = random_order[:key_count]
                score_without_key = self.forward(
                    sampled.without_edges(key), relation_features, demonstrations, max_layers=depth
                ).score
                score_without_random = self.forward(
                    sampled.without_edges(random_edges), relation_features, demonstrations,
                    max_layers=depth,
                ).score
            else:
                score_without_key = trajectory.score
                score_without_random = trajectory.score
            structural = self._structural_validity(sampled)
            provenance = (
                (sampled.edge_provenance > 0).float().mean()
                if sampled.num_edges else structural.new_zeros(())
            )
            features = verifier_features(
                structural, provenance, trajectory.score,
                score_without_key, score_without_random,
            )
            verifier_logit = self.verifier(features)
            full_edge_weights = torch.zeros(
                prompt.num_edges, device=trajectory.edge_weights.device
            )
            if selected.numel():
                full_edge_weights[selected] = trajectory.edge_weights
            trajectories.append(
                Trajectory(
                    trajectory.score, verifier_logit, full_edge_weights,
                    selected[trajectory.selected_edges] if selected.numel() else selected,
                    features,
                )
            )
        logits = torch.stack([item.verifier_logit for item in trajectories])
        tau = temperature if temperature is not None else self.temperature.clamp_min(0.05)
        weights = (
            torch.full_like(logits, 1.0 / len(trajectories))
            if uniform_aggregation else torch.softmax(logits / tau, dim=0)
        )
        probabilities = torch.stack([torch.sigmoid(item.score) for item in trajectories])
        final_score = torch.sum(weights * probabilities)
        return InferenceResult(final_score, trajectories, weights)
