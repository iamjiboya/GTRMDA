from __future__ import annotations

import torch
from torch import nn


class EvidenceVerifier(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def verifier_features(
    structural_validity: torch.Tensor,
    provenance_consistency: torch.Tensor,
    score: torch.Tensor,
    score_without_key: torch.Tensor,
    score_without_random: torch.Tensor,
) -> torch.Tensor:
    counterfactual = (score - score_without_key) - 0.5 * torch.abs(
        score - score_without_random
    )
    return torch.stack([structural_validity, provenance_consistency, counterfactual])

