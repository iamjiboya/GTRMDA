from pathlib import Path

import torch

from gtrmda.data.toy import make_toy_dataset
from gtrmda.experiments.factory import build_model, build_prompt_components, move_demonstrations
from gtrmda.utils.config import load_config
from gtrmda.utils.seed import seed_everything


ROOT = Path(__file__).resolve().parents[1]


def test_forward_and_verified_inference(tmp_path: Path):
    seed_everything(7)
    config = load_config(ROOT / "configs/toy.yaml")
    bundle = make_toy_dataset(tmp_path)
    model = build_model(bundle, config).eval()
    builder, sampler = build_prompt_components(bundle, config)
    prompt = builder.build("mir:21", "MDA", "dis:1")
    demonstrations = move_demonstrations(sampler.sample(1, "MDA"), "cpu")
    relation_features = torch.from_numpy(bundle.relation_features)
    trajectory = model(prompt, relation_features, demonstrations)
    assert trajectory.score.ndim == 0
    result = model.infer(prompt, relation_features, demonstrations, budget=1, seed=7)
    assert 0.0 <= float(result.score) <= 1.0
    assert len(result.trajectories) == 1
    assert torch.isclose(result.aggregation_weights.sum(), torch.tensor(1.0))

