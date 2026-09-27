from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(labels, probabilities, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (probabilities > low) & (probabilities <= high)
        if not np.any(mask):
            continue
        error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)


def binary_metrics(labels, probabilities) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    output = {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": expected_calibration_error(labels, probabilities),
    }
    if len(np.unique(labels)) == 2:
        output["auc"] = float(roc_auc_score(labels, probabilities))
        output["aupr"] = float(average_precision_score(labels, probabilities))
    else:
        output["auc"] = math.nan
        output["aupr"] = math.nan
    return output


def ranking_metrics(ranks: list[int], hits=(10, 50), ndcg_k: int = 50) -> dict[str, float]:
    ranks_array = np.asarray(ranks, dtype=float)
    result = {"mrr": float(np.mean(1.0 / ranks_array)) if len(ranks) else math.nan}
    for k in hits:
        result[f"hits@{k}"] = float(np.mean(ranks_array <= k)) if len(ranks) else math.nan
    result[f"recall@{ndcg_k}"] = (
        float(np.mean(ranks_array <= ndcg_k)) if len(ranks) else math.nan
    )
    gains = np.where(ranks_array <= ndcg_k, 1.0 / np.log2(ranks_array + 1.0), 0.0)
    result[f"ndcg@{ndcg_k}"] = float(np.mean(gains)) if len(ranks) else math.nan
    return result

