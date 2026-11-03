import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from rl_policy import digest


def load_trainable_checkpoint(model, checkpoint):
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "adapter_config.json").read_text())
    weights_path = checkpoint / "adapters.safetensors"
    if digest(weights_path) != metadata.get("adapter_sha256"):
        raise ValueError("Checkpoint adapter checksum does not match its metadata")
    weights = mx.load(str(weights_path))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(weights) != set(expected) or any(
        weights[key].shape != expected[key].shape
        or weights[key].dtype != expected[key].dtype
        for key in weights
    ):
        raise ValueError(
            "Checkpoint must contain exactly the compatible trainable parameters"
        )
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.trainable_parameters())
    return metadata


def initialize_rl_stage(model, checkpoint, config):
    checkpoint = Path(checkpoint)
    metadata_path = checkpoint / "adapter_config.json"
    previous = json.loads(metadata_path.read_text())
    if previous.get("training_kind") != "match_priority_policy_gradient":
        raise ValueError(
            "Stage initialization requires an actual RL checkpoint, not supervised or system-diagnostic weights"
        )
    if previous.get("completed_groups", 0) < 1:
        raise ValueError(
            "Stage initialization requires a completed training checkpoint"
        )
    for key in [
        "model_hashes",
        "num_layers",
        "lora_parameters",
        "board_encoding",
        "schema",
        "input_representation",
    ]:
        if previous.get(key) != config.get(key):
            raise ValueError(f"RL stage initializer has incompatible {key}")
    load_trainable_checkpoint(model, checkpoint)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "metadata_sha256": digest(metadata_path),
        "adapter_sha256": previous["adapter_sha256"],
        "training_kind": previous["training_kind"],
        "source_hashes": previous["source_hashes"],
        "data_sha256": previous["data_sha256"],
        "completed_groups": previous["completed_groups"],
        "optimizer_updates": previous["optimizer_updates"],
        "optimizer_state": "reset for the new RL stage; weights retained",
    }
