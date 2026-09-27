#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtrmda.data.toy import make_toy_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the deterministic GTRMDA toy dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/toy")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    bundle = make_toy_dataset(args.output, args.seed)
    print(f"Wrote {len(bundle.graph.entities)} entities and {len(bundle.graph.edges)} auxiliary edges to {args.output}")


if __name__ == "__main__":
    main()

