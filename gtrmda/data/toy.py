from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import DatasetBundle, save_dataset
from .schema import Edge, Entity, KnowledgeGraph, Relation


def make_toy_dataset(root: str | Path, seed: int = 7) -> DatasetBundle:
    entities = [
        *[Entity(f"mir:{i}", "miRNA", f"hsa-miR-{i}") for i in [21, 146, 155, 34]],
        *[Entity(f"dis:{i}", "disease", name) for i, name in [
            (1, "Breast Neoplasms"), (2, "Alzheimer's Disease"),
            (3, "Lung Neoplasms"), (4, "Parkinson's Disease")]],
        *[Entity(f"gene:{name}", "gene", name) for name in ["PTEN", "PDCD4", "IRAK1", "TRAF6", "TP53", "AKT1"]],
        *[Entity(f"path:{i}", "pathway", name) for i, name in [
            (1, "PI3K-Akt"), (2, "Apoptosis"), (3, "NF-kB"), (4, "Neuroinflammation")]],
    ]
    relations = [
        Relation("miRNA_gene", "miRNA", "gene", "a microRNA targets a gene"),
        Relation("gene_pathway", "gene", "pathway", "a gene participates in a pathway"),
        Relation("gene_disease", "gene", "disease", "a gene is implicated in a disease"),
        Relation("pathway_disease", "pathway", "disease", "a pathway participates in a disease"),
        Relation("MDA", "miRNA", "disease", "a microRNA is experimentally associated with a disease"),
    ]
    edges = [
        Edge("mir:21", "miRNA_gene", "gene:PTEN", "miRTarBase", 2015),
        Edge("mir:21", "miRNA_gene", "gene:PDCD4", "miRTarBase", 2016),
        Edge("mir:146", "miRNA_gene", "gene:IRAK1", "miRTarBase", 2015),
        Edge("mir:146", "miRNA_gene", "gene:TRAF6", "miRTarBase", 2017),
        Edge("mir:155", "miRNA_gene", "gene:TP53", "miRTarBase", 2016),
        Edge("mir:34", "miRNA_gene", "gene:AKT1", "miRTarBase", 2018),
        Edge("gene:PTEN", "gene_pathway", "path:1", "Reactome", 2014),
        Edge("gene:PDCD4", "gene_pathway", "path:2", "Reactome", 2014),
        Edge("gene:IRAK1", "gene_pathway", "path:3", "Reactome", 2014),
        Edge("gene:TRAF6", "gene_pathway", "path:4", "Reactome", 2014),
        Edge("gene:TP53", "gene_pathway", "path:2", "Reactome", 2014),
        Edge("gene:AKT1", "gene_pathway", "path:1", "Reactome", 2014),
        Edge("gene:PTEN", "gene_disease", "dis:1", "CTD", 2017),
        Edge("gene:PDCD4", "gene_disease", "dis:1", "DisGeNET", 2017),
        Edge("gene:IRAK1", "gene_disease", "dis:2", "CTD", 2017),
        Edge("gene:TRAF6", "gene_disease", "dis:2", "DisGeNET", 2017),
        Edge("gene:TP53", "gene_disease", "dis:3", "CTD", 2018),
        Edge("gene:AKT1", "gene_disease", "dis:4", "DisGeNET", 2018),
        Edge("path:1", "pathway_disease", "dis:1", "Reactome", 2017),
        Edge("path:2", "pathway_disease", "dis:1", "Reactome", 2017),
        Edge("path:3", "pathway_disease", "dis:2", "Reactome", 2017),
        Edge("path:4", "pathway_disease", "dis:2", "Reactome", 2017),
    ]
    graph = KnowledgeGraph(entities, relations, edges)
    rng = np.random.default_rng(seed)
    entity_features = rng.normal(size=(len(entities), 32)).astype(np.float32)
    relation_features = rng.normal(size=(len(relations), 24)).astype(np.float32)
    queries = pd.DataFrame([
        {"mirna": "mir:21", "disease": "dis:1", "label": 1, "year": 2019},
        {"mirna": "mir:146", "disease": "dis:2", "label": 1, "year": 2020},
        {"mirna": "mir:155", "disease": "dis:3", "label": 1, "year": 2020},
        {"mirna": "mir:34", "disease": "dis:4", "label": 1, "year": 2021},
        {"mirna": "mir:21", "disease": "dis:4", "label": 0, "year": 2021},
        {"mirna": "mir:146", "disease": "dis:3", "label": 0, "year": 2021},
    ])
    metadata = {
        "name": "GTRMDA toy graph",
        "synthetic": True,
        "target_relation": "MDA",
        "entity_feature_source": "seeded random smoke-test vectors",
        "relation_feature_source": "seeded random smoke-test vectors",
    }
    bundle = DatasetBundle(graph, entity_features, relation_features, queries, metadata)
    save_dataset(bundle, root)
    return bundle

