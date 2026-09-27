#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtrmda.data.io import load_dataset
from gtrmda.data.split import make_protocol_split
from gtrmda.experiments.evaluate import evaluate_query_table
from gtrmda.experiments.factory import build_model, build_prompt_components
from gtrmda.utils.config import load_config, resolve_device
from gtrmda.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GTRMDA")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", choices=["relation-zs", "unseen-mirna", "unseen-disease", "double-cs", "temporal"], default="relation-zs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    data_root = Path(config["data"]["root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    bundle = load_dataset(data_root, set(config["data"]["target_aliases"]))
    model = build_model(bundle, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trained_protocol = checkpoint.get("config", {}).get("protocol")
    if trained_protocol and trained_protocol != args.protocol:
        raise ValueError(
            f"Checkpoint was trained for {trained_protocol}, not {args.protocol}; "
            "train a protocol-specific checkpoint to avoid leakage"
        )
    model.load_state_dict(checkpoint["model"])
    builder, sampler = build_prompt_components(bundle, config)
    split = make_protocol_split(bundle.queries, args.protocol, config["seed"])
    metrics, predictions = evaluate_query_table(
        model, bundle, builder, sampler, config, device,
        query_table=split.test, budget=args.budget,
        candidate_diseases=split.heldout_diseases or None,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
