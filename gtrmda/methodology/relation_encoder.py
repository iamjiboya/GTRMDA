from __future__ import annotations

import math

import torch
from torch import nn


def compatibility_codes(head_types: torch.Tensor, tail_types: torch.Tensor):
    hh = head_types[:, None] == head_types[None, :]
    ht = head_types[:, None] == tail_types[None, :]
    th = tail_types[:, None] == head_types[None, :]
    tt = tail_types[:, None] == tail_types[None, :]
    mask = hh | ht | th | tt
    codes = torch.zeros_like(head_types[:, None].expand(-1, head_types.numel()))
    codes = torch.where(ht, torch.ones_like(codes), codes)
    codes = torch.where(th, torch.full_like(codes, 2), codes)
    codes = torch.where(tt, torch.full_like(codes, 3), codes)
    return codes.long(), mask


class RelationAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.transforms = nn.Parameter(torch.empty(4, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.transforms)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.query(z) @ self.key(z).T / math.sqrt(z.shape[-1])
        logits = logits.masked_fill(~mask, -1e4)
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        transformed = torch.stack([z @ matrix for matrix in self.transforms], dim=0)
        columns = torch.arange(z.shape[0], device=z.device)[None, :].expand_as(codes)
        pair_values = transformed[codes, columns]
        update = torch.sum(attention.unsqueeze(-1) * pair_values, dim=1)
        return self.norm(z + update)


class RelationGraphEncoder(nn.Module):
    def __init__(
        self,
        relation_feature_dim: int,
        hidden_dim: int,
        num_node_types: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_projection = nn.Linear(relation_feature_dim, hidden_dim)
        self.endpoint_embedding = nn.Embedding(num_node_types, hidden_dim // 2)
        self.initial = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.layers = nn.ModuleList(
            [RelationAttentionLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        relation_features: torch.Tensor,
        head_types: torch.Tensor,
        tail_types: torch.Tensor,
    ) -> torch.Tensor:
        text = self.text_projection(relation_features)
        endpoints = torch.cat(
            [self.endpoint_embedding(head_types), self.endpoint_embedding(tail_types)], dim=-1
        )
        z = self.initial(torch.cat([text, endpoints], dim=-1))
        codes, mask = compatibility_codes(head_types, tail_types)
        mask = mask | torch.eye(mask.shape[0], dtype=torch.bool, device=mask.device)
        for layer in self.layers:
            z = layer(z, codes, mask)
        return z

