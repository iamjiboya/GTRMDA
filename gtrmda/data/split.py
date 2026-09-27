from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import DatasetBundle
from .schema import KnowledgeGraph


@dataclass
class ProtocolSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    heldout_mirnas: set[str]
    heldout_diseases: set[str]


def _partition(values: pd.Series, seed: int):
    counts = values.value_counts()
    entities = list(counts.index)
    rng = np.random.default_rng(seed)
    jitter = {entity: float(rng.random()) for entity in entities}
    entities.sort(key=lambda entity: (-int(counts[entity]), jitter[entity]))
    if len(entities) < 3:
        return set(entities), set(), set()
    train, validation, test = set(), set(), set()
    for start in range(0, len(entities), 10):
        block = entities[start:start + 10]
        n_test = max(1, round(0.1 * len(block)))
        n_validation = max(1, round(0.1 * len(block)))
        test.update(block[-n_test:])
        validation.update(block[-(n_test + n_validation):-n_test])
        train.update(block[:-(n_test + n_validation)])
    return train, validation, test


def make_protocol_split(
    queries: pd.DataFrame,
    protocol: str,
    seed: int = 2026,
    temporal_cutoff: int = 2018,
) -> ProtocolSplit:
    required = {"mirna", "disease", "label"}
    if not required.issubset(queries.columns):
        raise ValueError(f"queries must contain {sorted(required)}")
    protocol = protocol.lower()
    empty = queries.iloc[0:0].copy()
    if protocol == "relation-zs":
        return ProtocolSplit(empty, empty, queries.copy(), set(), set())
    if protocol == "temporal":
        if "year" not in queries.columns:
            raise ValueError("Temporal evaluation requires a year column")
        test = queries[queries["year"] > temporal_cutoff].copy()
        return ProtocolSplit(empty, empty, test, set(), set())

    mirna_parts = _partition(queries["mirna"], seed)
    disease_parts = _partition(queries["disease"], seed + 1)
    if protocol == "unseen-mirna":
        valid = queries[queries["mirna"].isin(mirna_parts[1])]
        test = queries[queries["mirna"].isin(mirna_parts[2])]
        return ProtocolSplit(empty, valid, test, mirna_parts[2], set())
    if protocol == "unseen-disease":
        valid = queries[queries["disease"].isin(disease_parts[1])]
        test = queries[queries["disease"].isin(disease_parts[2])]
        return ProtocolSplit(empty, valid, test, set(), disease_parts[2])
    if protocol in {"double-cs", "double-cold-start"}:
        valid = queries[
            queries["mirna"].isin(mirna_parts[1])
            & queries["disease"].isin(disease_parts[1])
        ]
        test = queries[
            queries["mirna"].isin(mirna_parts[2])
            & queries["disease"].isin(disease_parts[2])
        ]
        return ProtocolSplit(empty, valid, test, mirna_parts[2], disease_parts[2])
    raise ValueError(f"Unknown protocol: {protocol}")


def training_bundle_for_protocol(
    bundle: DatasetBundle,
    split: ProtocolSplit,
    protocol: str,
    temporal_cutoff: int = 2018,
) -> DatasetBundle:
    """Return the auxiliary graph visible during protocol-specific pretraining."""
    excluded = split.heldout_mirnas | split.heldout_diseases
    protocol = protocol.lower()
    edges = []
    for edge in bundle.graph.edges:
        if edge.head in excluded or edge.tail in excluded:
            continue
        if protocol == "temporal" and (edge.year is None or edge.year > temporal_cutoff):
            continue
        edges.append(edge)
    graph = KnowledgeGraph(bundle.graph.entities, bundle.graph.relations, edges)
    metadata = dict(bundle.metadata)
    metadata.update({
        "training_protocol": protocol,
        "training_excluded_entities": sorted(excluded),
        "temporal_cutoff": temporal_cutoff if protocol == "temporal" else None,
    })
    return DatasetBundle(
        graph, bundle.entity_features, bundle.relation_features,
        bundle.queries, metadata,
    )
