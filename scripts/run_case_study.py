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
from gtrmda.experiments.case_study import explain_query
from gtrmda.experiments.factory import build_model, build_prompt_components
from gtrmda.utils.config import load_config, resolve_device
from gtrmda.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mirna", required=True)
    parser.add_argument("--disease", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    data_root = Path(config["data"]["root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    bundle = load_dataset(data_root, set(config["data"]["target_aliases"]))
    model = build_model(bundle, config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    builder, sampler = build_prompt_components(bundle, config)
    result = explain_query(
        model, bundle, builder, sampler, config, args.mirna, args.disease, device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
