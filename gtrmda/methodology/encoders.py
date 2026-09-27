from __future__ import annotations

import torch
from torch import nn

from gtrmda.data.schema import GraphPrompt


class PromptNodeEncoder(nn.Module):
    """Projects frozen biological features and query-relative structural codes."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_node_types: int,
        max_distance: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.head_distance = nn.Embedding(max_distance + 1, hidden_dim)
        self.tail_distance = nn.Embedding(max_distance + 1, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.max_distance = max_distance

    def forward(self, prompt: GraphPrompt) -> torch.Tensor:
        head_distance = prompt.distances_to_head.clamp(max=self.max_distance)
        tail_distance = prompt.distances_to_tail.clamp(max=self.max_distance)
        pieces = [
            self.feature_projection(prompt.node_features),
            self.type_embedding(prompt.node_types),
            self.head_distance(head_distance),
            self.tail_distance(tail_distance),
        ]
        return self.fusion(torch.cat(pieces, dim=-1))


class FrozenEncoderAdapter:
    """Optional offline adapters for RNA-FM/PubMedBERT feature extraction.

    The training code consumes cached numeric features so frozen foundation
    models are never updated and can be prepared on a separate machine.
    """

    @staticmethod
    def pad_or_trim(vector: torch.Tensor, output_dim: int) -> torch.Tensor:
        vector = vector.flatten()
        if vector.numel() >= output_dim:
            return vector[:output_dim]
        return torch.nn.functional.pad(vector, (0, output_dim - vector.numel()))

