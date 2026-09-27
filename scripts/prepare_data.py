#!/usr/bin/env python3
"""Validate and package normalized biomedical tables into the GTRMDA contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtrmda.data.io import DatasetBundle, save_dataset
from gtrmda.data.schema import Edge, Entity, KnowledgeGraph, Relation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--auxiliary-edges", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--entity-features", type=Path, required=True)
    parser.add_argument("--relation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-relation", default="MDA")
    parser.add_argument("--target-alias", action="append", default=[])
    args = parser.parse_args()

    entity_table = pd.read_csv(args.entities).fillna("")
    relation_table = pd.read_csv(args.relations).fillna("")
    edge_table = pd.read_csv(args.auxiliary_edges).fillna("")
    queries = pd.read_csv(args.queries)
    required = {
        "entities": ({"id", "type", "name"}, entity_table),
        "relations": ({"id", "head_type", "tail_type", "description"}, relation_table),
        "edges": ({"head", "relation", "tail", "provenance"}, edge_table),
        "queries": ({"mirna", "disease", "label"}, queries),
    }
    for name, (columns, table) in required.items():
        missing = columns - set(table.columns)
        if missing:
            raise ValueError(f"{name} table is missing columns: {sorted(missing)}")

    forbidden = {args.target_relation.lower(), *(item.lower() for item in args.target_alias)}
    leaked = edge_table[edge_table["relation"].str.lower().isin(forbidden)]
    if not leaked.empty:
        raise ValueError(
            f"Found {len(leaked)} target-relation edges in auxiliary input; remove them first"
        )
    entities = [Entity(str(row.id), str(row.type), str(row.name))
                for row in entity_table.itertuples(index=False)]
    relations = [Relation(str(row.id), str(row.head_type), str(row.tail_type), str(row.description))
                 for row in relation_table.itertuples(index=False)]
    edges = [Edge(str(row.head), str(row.relation), str(row.tail), str(row.provenance),
                  int(row.year) if hasattr(row, "year") and str(row.year) else None)
             for row in edge_table.itertuples(index=False)]
    bundle = DatasetBundle(
        KnowledgeGraph(entities, relations, edges),
        np.load(args.entity_features).astype(np.float32),
        np.load(args.relation_features).astype(np.float32),
        queries,
        {"target_relation": args.target_relation, "synthetic": False,
         "source_manifest": json.dumps({"entities": str(args.entities), "relations": str(args.relations),
                                          "edges": str(args.auxiliary_edges), "queries": str(args.queries)})},
    )
    save_dataset(bundle, args.output)
    print(f"Validated and wrote dataset to {args.output}")


if __name__ == "__main__":
    main()

