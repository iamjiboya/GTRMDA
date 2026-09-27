#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtrmda.experiments.metrics import binary_metrics, ranking_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hits", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--ndcg-k", type=int, default=50)
    args = parser.parse_args()
    table = pd.read_csv(args.predictions)
    required = {"mirna", "disease", "label", "score"}
    if not required.issubset(table.columns):
        raise ValueError(f"Prediction file must contain {sorted(required)}")
    metrics = binary_metrics(table["label"], table["score"])
    positives = table[table["label"] == 1]
    ranks = []
    for positive in positives.itertuples(index=False):
        other_positives = set(positives[
            (positives["mirna"] == positive.mirna)
            & (positives["disease"] != positive.disease)
        ]["disease"])
        candidates = table[
            (table["mirna"] == positive.mirna)
            & (~table["disease"].isin(other_positives))
        ].sort_values("score", ascending=False).reset_index(drop=True)
        ranks.append(int(candidates.index[candidates["disease"] == positive.disease][0]) + 1)
    metrics.update(ranking_metrics(ranks, tuple(args.hits), args.ndcg_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

