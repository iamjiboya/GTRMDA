# GTRMDA

Official research code for **Graph In-Context Reasoning with Test-Time Verification
for Relation-Level Zero-Shot miRNA--Disease Association Prediction**.

GTRMDA treats miRNA--disease association (MDA) prediction as relation-level
zero-shot inference. The model is trained only on auxiliary biomedical relations;
all MDA edges, inverse edges, aliases, and MDA-derived interaction profiles are
excluded from training and model selection.

> **Reproducibility status.** The implementation, toy example, evaluation protocols,
> tests, and plotting pipeline are runnable. HMDD and auxiliary biomedical databases
> are not redistributed because their licenses and download procedures differ. The
> numerical tables and case-study figure inputs currently bundled under `figures/`


## Method

The implementation follows the paper's four stages:

1. **Query-centered graph prompts** combine cached RNA-FM sequence embeddings,
   PubMedBERT disease/relation embeddings, endpoint types, and anonymous structural
   distance codes.
2. **Relation-conditioned reasoning** uses a two-layer relation graph encoder and
   Bellman--Ford-style entity message passing.
3. **Graph in-context conditioning** uses labeled demonstrations sampled only from
   auxiliary relations, without parameter updates at inference.
4. **Test-time verification** samples multiple evidence trajectories and scores
   structural validity, provenance consistency, and counterfactual stability before
   calibrated aggregation.

## Repository layout

```text
code/
├── configs/                 # Paper and toy hyperparameters
├── data/
│   ├── raw/                 # User-downloaded source databases (gitignored)
│   └── processed/           # Validated GTRMDA datasets (gitignored)
├── gtrmda/
│   ├── data/                # Section: datasets, protocols, prompts, leakage guards
│   ├── methodology/         # Section: model modules, verifier, and objectives
│   ├── experiments/         # Section: training, metrics, ablations, cases
│   └── utils/
├── scripts/                 # Reproduction command-line entry points
├── figures/                 # Figures 2--6 and source tables
├── baselines/               # Fair prediction-file interface for external baselines
├── tests/                   # Leakage, prompt, and model smoke tests
├── pyproject.toml
└── README.md
```

## Installation

Python 3.10 and PyTorch 2.3 or newer are recommended.

```bash
git clone https://github.com/iamjiboya/GTRMDA.git
cd GTRMDA
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

For offline RNA-FM and PubMedBERT feature extraction:

```bash
pip install -e ".[encoders]"
pip install "git+https://github.com/ml4bio/RNA-FM.git"
```

## Five-minute smoke test

The toy graph contains no MDA edge in its auxiliary training graph and requires no
downloads.

```bash
bash scripts/run_toy.sh
pytest
```

Outputs are written to `outputs/toy/`. 

## Data contract

Prepare one directory containing the following files:

| File | Required columns or shape |
|---|---|
| `entities.csv` | `id,type,name`; optional `description,sequence` |
| `relations.csv` | `id,head_type,tail_type,description` |
| `auxiliary_edges.csv` | `head,relation,tail,provenance`; optional `year` |
| `queries.csv` | `mirna,disease,label`; `year` for temporal evaluation |
| `entity_features.npy` | `[num_entities, feature_dim]` |
| `relation_features.npy` | `[num_relations, relation_feature_dim]` |
| `metadata.json` | source and preprocessing metadata |

Recommended sources are HMDD v3.2/v4.0 and miR2Disease for evaluation, with
miRTarBase, CTD/DisGeNET, STRING, Reactome, Disease Ontology, and phenotype
relations for the auxiliary graph. Normalize mature miRNAs to miRBase identifiers
and diseases to DOID or UMLS concepts before packaging.

### Cache frozen biological features

`entities.csv` must contain miRNA sequences and disease/entity descriptions.

```bash
python scripts/cache_features.py \
  --entities work/entities.csv \
  --relations work/relations.csv \
  --output-dir work/features
```

### Validate and package

```bash
python scripts/prepare_data.py \
  --entities work/entities.csv \
  --relations work/relations.csv \
  --auxiliary-edges work/auxiliary_edges.csv \
  --queries work/queries.csv \
  --entity-features work/features/entity_features.npy \
  --relation-features work/features/relation_features.npy \
  --target-relation MDA \
  --target-alias miRNA_disease \
  --output data/processed/hmdd_v4
```

The loader fails immediately if a target relation or configured alias is found in
`auxiliary_edges.csv`.

## Training

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --protocol relation-zs \
  --output outputs/hmdd_v4/relation-zs/seed_2026
```

The default configuration matches the manuscript: hidden dimension 256, two
relation layers, four entity layers, `K=8` demonstrations, AdamW at `2e-4`, and
loss weights `0.5/0.2/0.1` for ranking, counterfactual fidelity, and calibration.

## Evaluation protocols

```bash
for protocol in relation-zs unseen-mirna unseen-disease double-cs temporal; do
  python scripts/train.py \
    --config configs/default.yaml \
    --protocol "$protocol" \
    --output "outputs/hmdd_v4/$protocol/seed_2026"
  python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint "outputs/hmdd_v4/$protocol/seed_2026/best.pt" \
    --protocol "$protocol" \
    --output "outputs/hmdd_v4/$protocol/seed_2026/eval"
done
```

Cold-start training removes every auxiliary edge incident to held-out entities;
temporal training removes auxiliary evidence newer than the cutoff. Evaluation then
restores the permitted test-time auxiliary graph. Checkpoints are protocol-specific,
and the evaluator rejects a mismatched checkpoint.

The evaluator scores the full type-compatible disease candidate set and reports
PU-style AUC/AUPR, Brier score, ECE, filtered MRR, Hits@K, Recall@K, and NDCG@K.
For publication, repeat with seeds `2022--2026` and aggregate the resulting JSON
files as mean and standard deviation.

Aggregate per-seed metric files with:

```bash
python scripts/summarize_runs.py outputs/*/eval/metrics.json \
  --output outputs/summary.json
```

External baselines use the prediction contract documented in
[`baselines/README.md`](baselines/README.md).

## Ablation and sensitivity experiments

```bash
python scripts/run_ablation.py \
  --config configs/default.yaml \
  --output outputs/ablation

python scripts/run_sensitivity.py \
  --config configs/default.yaml \
  --parameter budget \
  --output outputs/sensitivity/budget
```

Available ablations remove relation semantics, in-context demonstrations, the
test-time verifier, the counterfactual loss, or multi-trajectory inference.
Sensitivity grids cover `K`, test-time budget `B`, prompt depth, and
`lambda_cf`.

## Case-study evidence

```bash
python scripts/run_case_study.py \
  --config configs/default.yaml \
  --checkpoint outputs/hmdd_v4/seed_2026/best.pt \
  --mirna hsa-miR-21-5p \
  --disease DOID:1612 \
  --output outputs/cases/mir21_breast.json
```

The JSON output includes the verifier-weighted score, trajectory weights,
verifier features, selected edges, and edge attention values. Evidence paths are
hypotheses for experimental prioritization and are not causal validation.

## Reproduce manuscript figures

See [`figures/README.md`](figures/README.md). Figure scripts export vector PDF/SVG
and high-resolution raster files. 

## Reproducibility checklist

- Target MDA relation and aliases excluded from auxiliary edges.
- No MDA demonstration used under Relation-ZS.
- Frozen encoders cached before graph training.
- Hyperparameters selected on auxiliary relations only.
- Entity cold-start partitions are disjoint.
- Temporal graph uses evidence no later than the configured cutoff.
- Full candidate ranking and calibration metrics retained with predictions.
- Seeds, resolved configuration, checkpoints, and per-query scores saved.

## Citation


Update the citation and repository URL after the anonymous review period.

## License

Code is released under the MIT License. Biomedical source databases and pretrained
model weights remain subject to their original licenses and terms of use.
