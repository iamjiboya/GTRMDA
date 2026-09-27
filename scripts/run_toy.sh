#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/make_toy_data.py
python scripts/train.py --config configs/toy.yaml --output outputs/toy
python scripts/evaluate.py \
  --config configs/toy.yaml \
  --checkpoint outputs/toy/best.pt \
  --protocol relation-zs \
  --output outputs/toy/eval
python scripts/run_case_study.py \
  --config configs/toy.yaml \
  --checkpoint outputs/toy/best.pt \
  --mirna mir:21 \
  --disease dis:1 \
  --output outputs/toy/case_mir21_breast.json

