#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtrmda.data.io import load_dataset
from gtrmda.data.split import make_protocol_split, training_bundle_for_protocol
from gtrmda.experiments.factory import build_model, build_prompt_components
from gtrmda.experiments.trainer import Trainer
from gtrmda.utils.config import load_config, resolve_device
from gtrmda.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GTRMDA on auxiliary relations")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/default")
    parser.add_argument("--protocol", choices=["relation-zs", "unseen-mirna", "unseen-disease", "double-cs", "temporal"], default="relation-zs")
    parser.add_argument("--temporal-cutoff", type=int, default=2018)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    data_root = Path(config["data"]["root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    bundle = load_dataset(data_root, set(config["data"]["target_aliases"]))
    split = make_protocol_split(
        bundle.queries, args.protocol, config["seed"], args.temporal_cutoff
    )
    bundle = training_bundle_for_protocol(
        bundle, split, args.protocol, args.temporal_cutoff
    )
    config["protocol"] = args.protocol
    config["temporal_cutoff"] = args.temporal_cutoff
    model = build_model(bundle, config)
    builder, sampler = build_prompt_components(bundle, config)
    trainer = Trainer(model, bundle, builder, sampler, config, resolve_device(config["device"]))
    history = trainer.fit(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Finished {len(history)} epochs; checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
