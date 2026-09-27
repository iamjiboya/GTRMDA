from __future__ import annotations

import math

import torch
from torch import nn


class GraphInContextConditioner(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(2, hidden_dim)
        self.demo_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_relation: torch.Tensor,
        demo_graphs: torch.Tensor | None,
        demo_relations: torch.Tensor | None,
        demo_labels: torch.Tensor | None,
    ) -> torch.Tensor:
        if demo_graphs is None or demo_graphs.numel() == 0:
            return query_relation
        demo_state = self.demo_projection(
            torch.cat([demo_graphs, self.label_embedding(demo_labels.long())], dim=-1)
        )
        logits = (self.attention(query_relation) * demo_relations).sum(-1)
        logits = logits / math.sqrt(query_relation.numel())
        weights = torch.softmax(logits, dim=0)
        context = torch.sum(weights.unsqueeze(-1) * demo_state, dim=0)
        return self.norm(query_relation + self.context_projection(context))

