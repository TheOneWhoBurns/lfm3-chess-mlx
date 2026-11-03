import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
import torch.nn.functional as F
from mlx.utils import tree_flatten
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mlx_lm.models.cache import ArraysCache, BatchKVCache, BatchRotatingKVCache
from mlx_lm.models.lfm3 import Model, ModelArgs
from mlx_lm.tuner.utils import linear_to_lora_layers

REFERENCE = "LiquidAI/LFM3-1.7B-Base"
REVISION = "545fc8f4c7b5fa9f405105274cd29c4f8f44b526"


def compare(actual, expected, label, atol=3e-5, rtol=3e-4):
    actual = np.array(actual)
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)
    return float(np.max(np.abs(actual - expected)))


def validate_variant(config_class, reference_class, variant):
    torch.manual_seed(42)
    mx.random.seed(42)
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=8,
        layer_types=[
            "conv",
            "full_attention",
            "sliding_attention",
            "conv",
            "full_attention",
        ],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **variant,
    )
    config._attn_implementation = "eager"
    reference = reference_class(config).eval()
    model = Model(ModelArgs.from_dict(config.to_dict()))
    state = {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
    model.load_weights(list(model.sanitize(state).items()), strict=True)
    model.eval()
    tokens = np.random.default_rng(42).integers(3, 128, size=(2, 41), dtype=np.int64)
    with torch.no_grad():
        expected = reference(torch.tensor(tokens), use_cache=False).logits.numpy()
    actual = model(mx.array(tokens))
    report = {"forward_max_abs_error": compare(actual, expected, "forward")}
    cache = model.make_cache()
    chunks = []
    start = 0
    for end in [3, 5, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 41]:
        chunks.append(model(mx.array(tokens[:, start:end]), cache=cache))
        mx.eval(chunks[-1], [c.state for c in cache])
        start = end
    report["cached_max_abs_error"] = compare(
        mx.concatenate(chunks, axis=1), expected, "cached"
    )
    padding = [4, 0]
    padded = tokens.copy()
    padded[0, :4] = 0
    cache = [
        ArraysCache(1, left_padding=padding)
        if kind == "conv"
        else BatchRotatingKVCache(config.sliding_window, padding)
        if kind == "sliding_attention"
        else BatchKVCache(padding)
        for kind in config.layer_types
    ]
    padded_output = model(mx.array(padded[:, :24]), cache=cache)
    decoded_output = model(mx.array(padded[:, 24:]), cache=cache)
    report["padded_max_abs_error"] = 0.0
    for row, pad in enumerate(padding):
        with torch.no_grad():
            unpadded = reference(
                torch.tensor(padded[row : row + 1, pad:]), use_cache=False
            ).logits.numpy()
        combined = mx.concatenate(
            [padded_output[row : row + 1, pad:], decoded_output[row : row + 1]], axis=1
        )
        error = compare(combined, unpadded, "padded batch")
        report["padded_max_abs_error"] = max(report["padded_max_abs_error"], error)
    short = tokens[:, :17]
    ref_logits = reference(torch.tensor(short[:, :-1]), use_cache=False).logits
    ref_loss = F.cross_entropy(
        ref_logits.reshape(-1, 128), torch.tensor(short[:, 1:]).reshape(-1)
    )
    ref_loss.backward()

    def loss_fn(m):
        logits = m(mx.array(short[:, :-1]))
        return nn.losses.cross_entropy(logits, mx.array(short[:, 1:]), reduction="mean")

    model.train()
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    report["loss_abs_error"] = compare(loss, ref_loss.detach().numpy(), "loss")
    ref_grads = {
        name: param.grad.detach().numpy()
        for name, param in reference.named_parameters()
    }
    grad_error = 0.0
    for name, grad in tree_flatten(grads):
        expected_grad = ref_grads[name]
        if name.endswith("conv.conv.weight"):
            expected_grad = expected_grad.transpose(0, 2, 1)
        grad_error = max(
            grad_error, compare(grad, expected_grad, name, atol=8e-5, rtol=2e-3)
        )
    report["gradient_max_abs_error"] = grad_error
    model.freeze()
    linear_to_lora_layers(
        model, len(model.layers), {"rank": 2, "scale": 2.0, "dropout": 0.0}
    )
    optimizer = optim.Adam(learning_rate=1e-3)
    initial = float(loss_fn(model).item())
    for _ in range(3):
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    final = float(loss_fn(model).item())
    if not np.isfinite(final) or final >= initial:
        raise AssertionError(f"LoRA optimization failed: {initial} -> {final}")
    report["lora_initial_loss"] = initial
    report["lora_final_loss"] = final
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config_class = type(
        AutoConfig.from_pretrained(REFERENCE, revision=REVISION, trust_remote_code=True)
    )
    reference_class = get_class_from_dynamic_module(
        "modeling_lfm3.Lfm3ForCausalLM", REFERENCE, revision=REVISION
    )
    report = {
        "reference": REFERENCE,
        "revision": REVISION,
        "device": str(mx.default_device()),
        "variants": {},
    }
    variants = {
        "released_architecture": {},
        "biased_untied_without_sandwich": {
            "conv_bias": True,
            "tie_word_embeddings": False,
            "use_sandwich_norm": False,
        },
        "width_one_convolution": {"conv_L_cache": 1},
    }
    for name, variant in variants.items():
        report["variants"][name] = validate_variant(
            config_class, reference_class, variant
        )
        print(name, json.dumps(report["variants"][name]), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
