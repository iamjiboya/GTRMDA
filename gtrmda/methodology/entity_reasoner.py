from __future__ import annotations

import torch
from torch import nn

from gtrmda.data.schema import GraphPrompt


def segment_softmax(logits: torch.Tensor, groups: torch.Tensor, num_groups: int) -> torch.Tensor:
    result = torch.zeros_like(logits)
    for group in torch.unique(groups):
        mask = groups == group
        result[mask] = torch.softmax(logits[mask], dim=0)
    return result


class ReasoningLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: torch.Tensor,
        prompt: GraphPrompt,
        relation_states: torch.Tensor,
        query_relation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt.num_edges == 0:
            return states, torch.empty(0, device=states.device)
        src, dst = prompt.edge_index
        edge_relations = relation_states[prompt.edge_types]
        query = query_relation.expand(src.numel(), -1)
        messages = self.message(torch.cat([states[src], edge_relations, query], dim=-1))
        logits = self.edge_attention(
            torch.cat([states[src], states[dst], edge_relations, query], dim=-1)
        ).squeeze(-1)
        weights = segment_softmax(logits, dst, states.shape[0])
        aggregate = torch.zeros_like(states)
        aggregate.index_add_(0, dst, weights.unsqueeze(-1) * messages)
        updated = self.update(aggregate, states)
        return self.norm(updated + states), weights


class EntityGraphReasoner(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ReasoningLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        initial_states: torch.Tensor,
        prompt: GraphPrompt,
        relation_states: torch.Tensor,
        query_relation: torch.Tensor,
        max_layers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = initial_states
        edge_weights = torch.empty(0, device=states.device)
        layers = self.layers if max_layers is None else self.layers[:max_layers]
        for layer in layers:
            states, edge_weights = layer(states, prompt, relation_states, query_relation)
        query_state = torch.cat(
            [states[prompt.head_index], states[prompt.tail_index], query_relation], dim=-1
        )
        score = self.scorer(query_state).squeeze(-1)
        return score, states, edge_weights

