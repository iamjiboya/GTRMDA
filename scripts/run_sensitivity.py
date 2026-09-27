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
from gtrmda.experiments.evaluate import evaluate_query_table
from gtrmda.experiments.factory import build_model, build_prompt_components
from gtrmda.experiments.sensitivity import GRID, sensitivity_config
from gtrmda.experiments.trainer import Trainer
from gtrmda.utils.config import load_config, resolve_device
from gtrmda.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parameter", choices=sorted(GRID), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for value in GRID[args.parameter]:
        config = sensitivity_config(base, args.parameter, value)
        seed_everything(config["seed"])
        device = resolve_device(config["device"])
        data_root = Path(config["data"]["root"])
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        bundle = load_dataset(data_root, set(config["data"]["target_aliases"]))
        model = build_model(bundle, config)
        builder, sampler = build_prompt_components(bundle, config)
        run_dir = args.output / f"{args.parameter}_{value}"
        trainer = Trainer(model, bundle, builder, sampler, config, device)
        trainer.fit(run_dir)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
        metrics, _ = evaluate_query_table(model, bundle, builder, sampler, config, device)
        records.append({"parameter": args.parameter, "value": value, **metrics})
    with (args.output / "sensitivity_results.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

