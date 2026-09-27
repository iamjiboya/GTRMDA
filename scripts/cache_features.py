#!/usr/bin/env python3
"""Cache frozen RNA-FM and PubMedBERT features in dataset table order."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def pad(vector: np.ndarray, width: int = 768) -> np.ndarray:
    output = np.zeros(width, dtype=np.float32)
    output[: min(width, len(vector))] = vector[:width]
    return output


@torch.no_grad()
def encode_text(texts: list[str], model_name: str, device: torch.device) -> np.ndarray:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install optional encoders: pip install -e '.[encoders]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    rows = []
    for start in range(0, len(texts), 16):
        tokens = tokenizer(
            texts[start:start + 16], padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        ).to(device)
        hidden = model(**tokens).last_hidden_state[:, 0]
        rows.extend(hidden.cpu().numpy().astype(np.float32))
    return np.asarray(rows)


@torch.no_grad()
def encode_rna(sequences: list[str], device: torch.device) -> np.ndarray:
    try:
        import fm
    except ImportError as exc:
        raise RuntimeError("RNA-FM package is required for miRNA sequence encoding") from exc
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.eval().to(device)
    converter = alphabet.get_batch_converter()
    rows = []
    for index, sequence in enumerate(sequences):
        _, _, tokens = converter([(str(index), sequence.upper().replace("T", "U"))])
        representation = model(tokens.to(device), repr_layers=[12])["representations"][12]
        rows.append(representation[0, 1: len(sequence) + 1].mean(0).cpu().numpy())
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    entities = pd.read_csv(args.entities).fillna("")
    relations = pd.read_csv(args.relations).fillna("")

    text_inputs = []
    for row in entities.itertuples(index=False):
        description = getattr(row, "description", "") or row.name
        text_inputs.append(str(description))
    entity_text = encode_text(text_inputs, args.text_model, device)
    mirna_rows = [i for i, row in entities.iterrows() if row["type"] == "miRNA"]
    if mirna_rows:
        if "sequence" not in entities.columns or any(not entities.loc[i, "sequence"] for i in mirna_rows):
            raise ValueError("Every miRNA entity requires a sequence column for RNA-FM")
        rna = encode_rna([entities.loc[i, "sequence"] for i in mirna_rows], device)
        for row_index, vector in zip(mirna_rows, rna):
            entity_text[row_index] = pad(vector, entity_text.shape[1])
    relation_text = encode_text(relations["description"].astype(str).tolist(), args.text_model, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "entity_features.npy", entity_text.astype(np.float32))
    np.save(args.output_dir / "relation_features.npy", relation_text.astype(np.float32))
    print(f"Cached entity and relation features under {args.output_dir}")


if __name__ == "__main__":
    main()

