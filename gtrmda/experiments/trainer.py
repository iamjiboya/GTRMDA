from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from gtrmda.data.io import DatasetBundle
from gtrmda.data.schema import Edge
from gtrmda.methodology.losses import GTRMDALoss
from gtrmda.methodology.verifier import verifier_features

from .factory import move_demonstrations


class Trainer:
    def __init__(
        self,
        model,
        bundle: DatasetBundle,
        builder,
        demonstration_sampler,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.bundle = bundle
        self.builder = builder
        self.demonstration_sampler = demonstration_sampler
        self.config = config
        self.device = device
        training = config["training"]
        self.optimizer = AdamW(
            model.parameters(), lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        self.criterion = GTRMDALoss(
            training["lambda_rank"], training["lambda_cf"], training["lambda_cal"]
        )
        self.relation_features = torch.from_numpy(bundle.relation_features).float().to(device)
        self.rng = random.Random(config["seed"])

    def _tail_pool(self, edge: Edge) -> list[str]:
        relation = self.bundle.graph.relations[
            self.bundle.graph.relation_to_idx[edge.relation]
        ]
        excluded = set(self.bundle.metadata.get("training_excluded_entities", []))
        return [entity.id for entity in self.bundle.graph.entities
                if entity.type == relation.tail_type and entity.id != edge.tail
                and entity.id not in excluded]

    def _prompt(self, edge: Edge, tail: str | None = None, seed: int = 0):
        return self.builder.build(
            edge.head, edge.relation, tail or edge.tail, seed=seed,
            masked_relations={edge.relation},
        ).to(self.device)

    def _counterfactual_scores(self, prompt, demonstrations, trajectory):
        if prompt.num_edges == 0:
            return trajectory.score, trajectory.score
        key = trajectory.selected_edges[:1]
        random_edge = torch.tensor(
            [self.rng.randrange(prompt.num_edges)], dtype=torch.long, device=self.device
        )
        score_key = self.model(
            prompt.without_edges(key), self.relation_features, demonstrations
        ).score
        score_random = self.model(
            prompt.without_edges(random_edge), self.relation_features, demonstrations
        ).score
        return score_key, score_random

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        edges = list(self.bundle.graph.edges)
        self.rng.shuffle(edges)
        limit = self.config["training"].get("max_edges_per_epoch")
        if limit:
            edges = edges[: int(limit)]
        sums = {"total": 0.0, "relation": 0.0, "ranking": 0.0,
                "counterfactual": 0.0, "calibration": 0.0}
        batch_size = self.config["training"]["batch_size"]
        negative_ratio = self.config["training"]["negative_ratio"]
        self.optimizer.zero_grad(set_to_none=True)
        for step, edge in enumerate(edges, start=1):
            demonstrations = self.demonstration_sampler.sample(
                self.config["data"]["demonstrations"], edge.relation
            )
            demonstrations = move_demonstrations(demonstrations, self.device)
            prompt = self._prompt(edge, seed=epoch * 100_000 + step)
            positive = self.model(prompt, self.relation_features, demonstrations)
            pool = self._tail_pool(edge)
            negative_scores = []
            for _ in range(negative_ratio):
                corrupted_tail = self.rng.choice(pool) if pool else edge.tail
                negative_prompt = self._prompt(
                    edge, tail=corrupted_tail, seed=epoch * 200_000 + step
                )
                negative_scores.append(
                    self.model(negative_prompt, self.relation_features, demonstrations).score
                )
            negative_tensor = torch.stack(negative_scores).unsqueeze(0)
            if self.config["training"]["lambda_cf"] > 0:
                score_key, score_random = self._counterfactual_scores(
                    prompt, demonstrations, positive
                )
                structural = self.model._structural_validity(prompt)
                provenance = (
                    (prompt.edge_provenance > 0).float().mean()
                    if prompt.num_edges else structural.new_zeros(())
                )
                verification = verifier_features(
                    structural, provenance, positive.score, score_key, score_random
                )
                verifier_logits = self.model.verifier(verification).unsqueeze(0)
            else:
                score_key = score_random = None
                verifier_logits = None
            losses = self.criterion(
                positive.score.unsqueeze(0), negative_tensor,
                torch.ones(1, device=self.device),
                score_key.unsqueeze(0) if score_key is not None else None,
                score_random.unsqueeze(0) if score_random is not None else None,
                self.model.temperature.clamp_min(0.05),
                verifier_logits,
            )
            (losses.total / batch_size).backward()
            if step % batch_size == 0 or step == len(edges):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["gradient_clip"]
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            for name in sums:
                sums[name] += float(getattr(losses, name).detach().cpu())
        return {name: value / max(1, len(edges)) for name, value in sums.items()}

    def fit(self, output_dir: str | Path) -> list[dict]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        history = []
        best = float("inf")
        stale = 0
        for epoch in range(1, self.config["training"]["epochs"] + 1):
            metrics = self.train_epoch(epoch)
            metrics["epoch"] = epoch
            history.append(metrics)
            if metrics["total"] < best:
                best = metrics["total"]
                stale = 0
                self.save(output_dir / "best.pt")
            else:
                stale += 1
            if stale >= self.config["training"]["patience"]:
                break
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        return history

    def save(self, path: str | Path) -> None:
        torch.save(
            {"model": self.model.state_dict(), "config": self.config,
             "metadata": self.bundle.metadata}, path
        )
