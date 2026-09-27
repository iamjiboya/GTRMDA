from pathlib import Path

import pandas as pd
import pytest

from gtrmda.data.io import load_dataset
from gtrmda.data.prompt import QueryPromptBuilder
from gtrmda.data.split import make_protocol_split, training_bundle_for_protocol
from gtrmda.data.toy import make_toy_dataset


def test_target_relation_is_absent_from_auxiliary_edges(tmp_path: Path):
    make_toy_dataset(tmp_path)
    bundle = load_dataset(tmp_path, {"MDA", "miRNA_disease"})
    assert all(edge.relation != "MDA" for edge in bundle.graph.edges)


def test_leakage_guard_rejects_target_edge(tmp_path: Path):
    make_toy_dataset(tmp_path)
    edges = pd.read_csv(tmp_path / "auxiliary_edges.csv")
    edges.loc[len(edges)] = ["mir:21", "MDA", "dis:1", "HMDD", 2020]
    edges.to_csv(tmp_path / "auxiliary_edges.csv", index=False)
    with pytest.raises(ValueError, match="leakage"):
        load_dataset(tmp_path, {"MDA"})


def test_prompt_contains_no_mda_edge(tmp_path: Path):
    bundle = make_toy_dataset(tmp_path)
    builder = QueryPromptBuilder(
        bundle.graph, bundle.entity_features, always_excluded_relations={"MDA"}
    )
    prompt = builder.build("mir:21", "MDA", "dis:1")
    relation_names = {
        bundle.graph.relations[index].id for index in prompt.edge_types.tolist()
    }
    assert "MDA" not in relation_names


def test_cold_start_training_removes_incident_auxiliary_edges(tmp_path: Path):
    bundle = make_toy_dataset(tmp_path)
    split = make_protocol_split(bundle.queries, "unseen-mirna", seed=7)
    training = training_bundle_for_protocol(bundle, split, "unseen-mirna")
    assert split.heldout_mirnas
    assert all(
        edge.head not in split.heldout_mirnas and edge.tail not in split.heldout_mirnas
        for edge in training.graph.edges
    )
