#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtrmda.data.io import load_dataset
from gtrmda.experiments.ablation import ABLATIONS, ablation_config
from gtrmda.experiments.evaluate import evaluate_query_table
from gtrmda.experiments.factory import build_model, build_prompt_components
from gtrmda.experiments.trainer import Trainer
from gtrmda.utils.config import load_config, resolve_device
from gtrmda.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="*", default=list(ABLATIONS))
    args = parser.parse_args()
    base = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in args.variants:
        config, flags = ablation_config(base, variant)
        seed_everything(config["seed"])
        device = resolve_device(config["device"])
        data_root = Path(config["data"]["root"])
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        bundle = load_dataset(data_root, set(config["data"]["target_aliases"]))
        if flags.get("zero_relation_features"):
            bundle.relation_features = np.zeros_like(bundle.relation_features)
        model = build_model(bundle, config)
        builder, sampler = build_prompt_components(bundle, config)
        run_dir = args.output / variant
        trainer = Trainer(model, bundle, builder, sampler, config, device)
        trainer.fit(run_dir)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
        metrics, _ = evaluate_query_table(
            model, bundle, builder, sampler, config, device,
            uniform_aggregation=flags.get("uniform_aggregation", False),
        )
        results.append({"variant": variant, **metrics})
    with (args.output / "ablation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

