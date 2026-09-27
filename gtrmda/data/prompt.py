from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np
import torch

from .schema import GraphPrompt, KnowledgeGraph, LabeledPrompt


class QueryPromptBuilder:
    def __init__(
        self,
        graph: KnowledgeGraph,
        entity_features: np.ndarray,
        hops: int = 3,
        max_neighbors: int = 32,
        always_excluded_relations: set[str] | None = None,
    ) -> None:
        self.graph = graph
        self.features = entity_features
        self.hops = hops
        self.max_neighbors = max_neighbors
        self.always_excluded = {name.lower() for name in (always_excluded_relations or set())}
        self.adjacency = defaultdict(list)
        for edge_idx, edge in enumerate(graph.edges):
            self.adjacency[edge.head].append((edge_idx, edge.tail))
            self.adjacency[edge.tail].append((edge_idx, edge.head))

    def _distances(self, source: str, excluded: set[str]) -> dict[str, int]:
        distances = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if distances[node] >= self.hops:
                continue
            for edge_idx, neighbor in self.adjacency[node]:
                edge = self.graph.edges[edge_idx]
                if edge.relation.lower() in excluded or neighbor in distances:
                    continue
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
        return distances

    def build(
        self,
        head: str,
        relation: str,
        tail: str,
        seed: int = 0,
        masked_relations: set[str] | None = None,
    ) -> GraphPrompt:
        if head not in self.graph.entity_to_idx or tail not in self.graph.entity_to_idx:
            raise KeyError(f"Unknown query endpoint: {head}, {tail}")
        if relation not in self.graph.relation_to_idx:
            raise KeyError(f"Unknown query relation: {relation}")
        excluded = self.always_excluded | {
            item.lower() for item in (masked_relations or set())
        }
        head_dist = self._distances(head, excluded)
        tail_dist = self._distances(tail, excluded)
        selected_nodes = set(head_dist) | set(tail_dist) | {head, tail}

        rng = random.Random(seed)
        grouped = defaultdict(list)
        for edge_idx, edge in enumerate(self.graph.edges):
            if edge.relation.lower() in excluded:
                continue
            if edge.head in selected_nodes and edge.tail in selected_nodes:
                grouped[(edge.head, edge.relation)].append(edge_idx)
        selected_edges = []
        for candidates in grouped.values():
            rng.shuffle(candidates)
            selected_edges.extend(candidates[: self.max_neighbors])

        node_ids = sorted(selected_nodes)
        local = {node_id: i for i, node_id in enumerate(node_ids)}
        global_indices = [self.graph.entity_to_idx[node_id] for node_id in node_ids]
        node_types = [
            self.graph.type_to_idx[self.graph.entities[index].type] for index in global_indices
        ]
        max_distance = self.hops + 1
        distances_head = [head_dist.get(node_id, max_distance) for node_id in node_ids]
        distances_tail = [tail_dist.get(node_id, max_distance) for node_id in node_ids]

        src, dst, rel, provenance = [], [], [], []
        provenance_names = sorted({edge.provenance for edge in self.graph.edges if edge.provenance})
        provenance_to_idx = {name: i + 1 for i, name in enumerate(provenance_names)}
        for edge_idx in selected_edges:
            edge = self.graph.edges[edge_idx]
            src.append(local[edge.head])
            dst.append(local[edge.tail])
            rel.append(self.graph.relation_to_idx[edge.relation])
            provenance.append(provenance_to_idx.get(edge.provenance, 0))
            # Add inverse flow while keeping the original relation identity.
            src.append(local[edge.tail])
            dst.append(local[edge.head])
            rel.append(self.graph.relation_to_idx[edge.relation])
            provenance.append(provenance_to_idx.get(edge.provenance, 0))

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return GraphPrompt(
            node_ids=node_ids,
            node_features=torch.from_numpy(self.features[global_indices]).float(),
            node_types=torch.tensor(node_types, dtype=torch.long),
            distances_to_head=torch.tensor(distances_head, dtype=torch.long),
            distances_to_tail=torch.tensor(distances_tail, dtype=torch.long),
            edge_index=edge_index,
            edge_types=torch.tensor(rel, dtype=torch.long),
            edge_provenance=torch.tensor(provenance, dtype=torch.long),
            head_index=local[head],
            tail_index=local[tail],
            query_relation=self.graph.relation_to_idx[relation],
        )


class DemonstrationSampler:
    def __init__(self, builder: QueryPromptBuilder, seed: int = 0) -> None:
        self.builder = builder
        self.rng = random.Random(seed)

    def sample(self, k: int, forbidden_relation: str) -> list[LabeledPrompt]:
        candidates = [
            edge for edge in self.builder.graph.edges
            if edge.relation != forbidden_relation
            and edge.relation.lower() not in self.builder.always_excluded
        ]
        if not candidates or k == 0:
            return []
        demonstrations = []
        for index in range(k):
            edge = self.rng.choice(candidates)
            label = float(index % 2 == 0)
            tail = edge.tail
            if label == 0.0:
                relation = self.builder.graph.relations[
                    self.builder.graph.relation_to_idx[edge.relation]
                ]
                pool = [entity.id for entity in self.builder.graph.entities
                        if entity.type == relation.tail_type and entity.id != edge.tail]
                if pool:
                    tail = self.rng.choice(pool)
            prompt = self.builder.build(
                edge.head, edge.relation, tail, seed=self.rng.randrange(1_000_000),
                masked_relations={edge.relation},
            )
            demonstrations.append(LabeledPrompt(prompt, label))
        return demonstrations

