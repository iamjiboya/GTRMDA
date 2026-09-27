from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Edge, Entity, KnowledgeGraph, Relation


@dataclass
class DatasetBundle:
    graph: KnowledgeGraph
    entity_features: np.ndarray
    relation_features: np.ndarray
    queries: pd.DataFrame
    metadata: dict


REQUIRED_FILES = {
    "entities.csv",
    "relations.csv",
    "auxiliary_edges.csv",
    "queries.csv",
    "entity_features.npy",
    "relation_features.npy",
    "metadata.json",
}


def _optional_int(value):
    if pd.isna(value) or value == "":
        return None
    return int(value)


def load_dataset(root: str | Path, forbidden_relations: set[str] | None = None) -> DatasetBundle:
    root = Path(root)
    missing = REQUIRED_FILES - {path.name for path in root.glob("*")}
    if missing:
        raise FileNotFoundError(f"Missing dataset files under {root}: {sorted(missing)}")

    entity_table = pd.read_csv(root / "entities.csv").fillna("")
    relation_table = pd.read_csv(root / "relations.csv").fillna("")
    edge_table = pd.read_csv(root / "auxiliary_edges.csv")
    queries = pd.read_csv(root / "queries.csv")
    entities = [Entity(str(row.id), str(row.type), str(row.name))
                for row in entity_table.itertuples(index=False)]
    relations = [Relation(str(row.id), str(row.head_type), str(row.tail_type),
                          str(row.description))
                 for row in relation_table.itertuples(index=False)]
    edges = [Edge(str(row.head), str(row.relation), str(row.tail),
                  str(getattr(row, "provenance", "") or ""),
                  _optional_int(getattr(row, "year", None)))
             for row in edge_table.itertuples(index=False)]

    forbidden = {item.lower() for item in (forbidden_relations or set())}
    leaked = sorted({edge.relation for edge in edges if edge.relation.lower() in forbidden})
    if leaked:
        raise ValueError(f"Target-relation leakage detected in auxiliary_edges.csv: {leaked}")

    graph = KnowledgeGraph(entities, relations, edges)
    entity_features = np.load(root / "entity_features.npy").astype(np.float32)
    relation_features = np.load(root / "relation_features.npy").astype(np.float32)
    if entity_features.shape[0] != len(entities):
        raise ValueError("entity_features.npy row count does not match entities.csv")
    if relation_features.shape[0] != len(relations):
        raise ValueError("relation_features.npy row count does not match relations.csv")
    with (root / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return DatasetBundle(graph, entity_features, relation_features, queries, metadata)


def save_dataset(bundle: DatasetBundle, root: str | Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(item) for item in bundle.graph.entities]).to_csv(
        root / "entities.csv", index=False
    )
    pd.DataFrame([vars(item) for item in bundle.graph.relations]).to_csv(
        root / "relations.csv", index=False
    )
    pd.DataFrame([vars(item) for item in bundle.graph.edges]).to_csv(
        root / "auxiliary_edges.csv", index=False
    )
    bundle.queries.to_csv(root / "queries.csv", index=False)
    np.save(root / "entity_features.npy", bundle.entity_features.astype(np.float32))
    np.save(root / "relation_features.npy", bundle.relation_features.astype(np.float32))
    with (root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(bundle.metadata, handle, indent=2, sort_keys=True)

