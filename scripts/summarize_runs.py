#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-seed metrics JSON files")
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for path in args.metrics:
        with path.open("r", encoding="utf-8") as handle:
            runs.append(json.load(handle))
    keys = sorted(set.intersection(*(set(run) for run in runs)))
    summary = {}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

