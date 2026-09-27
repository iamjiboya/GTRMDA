from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LossOutput:
    total: torch.Tensor
    relation: torch.Tensor
    ranking: torch.Tensor
    counterfactual: torch.Tensor
    calibration: torch.Tensor


class GTRMDALoss(nn.Module):
    def __init__(self, lambda_rank: float = 0.5, lambda_cf: float = 0.2, lambda_cal: float = 0.1):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.lambda_cf = lambda_cf
        self.lambda_cal = lambda_cal
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        positive_labels: torch.Tensor,
        score_without_key: torch.Tensor | None = None,
        score_without_random: torch.Tensor | None = None,
        temperature: torch.Tensor | float = 1.0,
        verifier_logits: torch.Tensor | None = None,
    ) -> LossOutput:
        relation_logits = torch.cat([positive.reshape(-1), negative.reshape(-1)])
        relation_labels = torch.cat([
            positive_labels.float().reshape(-1),
            torch.zeros_like(negative).reshape(-1),
        ])
        relation = self.bce(relation_logits, relation_labels)
        ranking = torch.relu(1.0 - positive.unsqueeze(-1) + negative).mean()
        if score_without_key is None or score_without_random is None:
            counterfactual = positive.new_zeros(())
        else:
            counterfactual = (
                torch.relu(0.5 - positive + score_without_key).mean()
                + torch.abs(positive - score_without_random).mean()
            )
        calibrated = torch.sigmoid(positive / temperature)
        calibration = torch.mean((calibrated - positive_labels.float()) ** 2)
        if verifier_logits is not None:
            calibration = calibration + self.bce(
                verifier_logits.reshape(-1), positive_labels.float().reshape(-1)
            )
        total = (
            relation + self.lambda_rank * ranking
            + self.lambda_cf * counterfactual + self.lambda_cal * calibration
        )
        return LossOutput(total, relation, ranking, counterfactual, calibration)
