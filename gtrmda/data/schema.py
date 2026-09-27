from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    name: str


@dataclass(frozen=True)
class Relation:
    id: str
    head_type: str
    tail_type: str
    description: str


@dataclass(frozen=True)
class Edge:
    head: str
    relation: str
    tail: str
    provenance: str = ""
    year: int | None = None


class KnowledgeGraph:
    def __init__(
        self,
        entities: Iterable[Entity],
        relations: Iterable[Relation],
        edges: Iterable[Edge],
    ) -> None:
        self.entities = list(entities)
        self.relations = list(relations)
        self.edges = list(edges)
        self.entity_to_idx = {entity.id: i for i, entity in enumerate(self.entities)}
        self.relation_to_idx = {relation.id: i for i, relation in enumerate(self.relations)}
        self.type_names = sorted({entity.type for entity in self.entities})
        self.type_to_idx = {name: i for i, name in enumerate(self.type_names)}
        self._validate()

    def _validate(self) -> None:
        if len(self.entity_to_idx) != len(self.entities):
            raise ValueError("Entity identifiers must be unique")
        if len(self.relation_to_idx) != len(self.relations):
            raise ValueError("Relation identifiers must be unique")
        for edge in self.edges:
            if edge.head not in self.entity_to_idx or edge.tail not in self.entity_to_idx:
                raise ValueError(f"Unknown endpoint in edge: {edge}")
            if edge.relation not in self.relation_to_idx:
                raise ValueError(f"Unknown relation in edge: {edge}")
            relation = self.relations[self.relation_to_idx[edge.relation]]
            head_type = self.entities[self.entity_to_idx[edge.head]].type
            tail_type = self.entities[self.entity_to_idx[edge.tail]].type
            if (head_type, tail_type) != (relation.head_type, relation.tail_type):
                raise ValueError(
                    f"Typed edge mismatch for {edge.relation}: "
                    f"expected {relation.head_type}->{relation.tail_type}, "
                    f"received {head_type}->{tail_type}"
                )


@dataclass
class GraphPrompt:
    node_ids: list[str]
    node_features: torch.Tensor
    node_types: torch.Tensor
    distances_to_head: torch.Tensor
    distances_to_tail: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    edge_provenance: torch.Tensor
    head_index: int
    tail_index: int
    query_relation: int

    def to(self, device: torch.device | str) -> "GraphPrompt":
        return replace(
            self,
            node_features=self.node_features.to(device),
            node_types=self.node_types.to(device),
            distances_to_head=self.distances_to_head.to(device),
            distances_to_tail=self.distances_to_tail.to(device),
            edge_index=self.edge_index.to(device),
            edge_types=self.edge_types.to(device),
            edge_provenance=self.edge_provenance.to(device),
        )

    def select_edges(self, indices: torch.Tensor) -> "GraphPrompt":
        return replace(
            self,
            edge_index=self.edge_index[:, indices],
            edge_types=self.edge_types[indices],
            edge_provenance=self.edge_provenance[indices],
        )

    def without_edges(self, indices: torch.Tensor) -> "GraphPrompt":
        keep = torch.ones(self.edge_types.numel(), dtype=torch.bool, device=self.edge_types.device)
        keep[indices] = False
        return self.select_edges(torch.where(keep)[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_types.numel())


@dataclass
class LabeledPrompt:
    prompt: GraphPrompt
    label: float

