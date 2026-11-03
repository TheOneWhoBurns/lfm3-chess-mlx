import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm import generate, load
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".artifacts/checkpoint")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(42)
    started = time.perf_counter()
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": False})
    model.set_dtype(mx.float32)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "What is the capital of Australia? Answer with just the city name.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    report = {
        "checkpoint": args.model,
        "device": str(mx.default_device()),
        "floating_dtype": "float32",
        "quantized_weights_preserved": True,
    }
    report["response"] = generate(
        model, tokenizer, prompt=prompt, max_tokens=20, sampler=make_sampler(temp=0.0)
    )
    print("Generation:", report["response"], flush=True)
    tokens = tokenizer.encode("The chess board contains pieces and squares. " * 48)[
        :320
    ]
    if len(tokens) <= model.args.sliding_window:
        raise ValueError("Validation prompt must cross the sliding-window boundary")
    x = mx.array([tokens])
    whole = model(x)
    mx.eval(whole)
    cache = model.make_cache()
    parts = []
    start = 0
    for end in [128, 257, 258, 259, len(tokens)]:
        parts.append(model(x[:, start:end], cache=cache))
        mx.eval(parts[-1], [c.state for c in cache])
        start = end
    cached = mx.concatenate(parts, axis=1)
    difference = mx.abs(whole.astype(mx.float32) - cached.astype(mx.float32))
    report["cache_max_abs_error"] = float(mx.max(difference).item())
    report["cache_mean_abs_error"] = float(mx.mean(difference).item())
    report["cache_top1_agreement"] = float(
        mx.mean(mx.argmax(whole, axis=-1) == mx.argmax(cached, axis=-1)).item()
    )
    if (
        not bool(mx.all(mx.isfinite(cached)).item())
        or report["cache_max_abs_error"] > 0.002
        or report["cache_top1_agreement"] < 0.99
    ):
        raise AssertionError(f"Cached forward diverged: {report}")
    cache_path = args.output_dir / "prompt-cache.safetensors"
    save_prompt_cache(str(cache_path), cache)
    restored = load_prompt_cache(str(cache_path))
    next_token = mx.array([[tokens[-1]]])
    expected = model(next_token, cache=cache)
    actual = model(next_token, cache=restored)
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
    )
    report["cache_roundtrip_exact"] = True
    del whole, cached, parts, cache, restored, expected, actual, difference
    gc.collect()
    mx.clear_cache()
    model.freeze()
    lora_config = {"rank": 4, "scale": 4.0, "dropout": 0.0}
    linear_to_lora_layers(model, 4, lora_config)
    model.train()
    optimizer = optim.Adam(learning_rate=1e-4)
    sample = mx.array(
        [
            tokenizer.encode(
                "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nMove: e2e4\n"
            )
        ]
    )

    def loss_fn(m):
        return nn.losses.cross_entropy(
            m(sample[:, :-1]).astype(mx.float32), sample[:, 1:], reduction="mean"
        )

    report["training_initial_loss"] = float(loss_fn(model).item())
    for step in range(3):
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)
        if not all(bool(mx.all(mx.isfinite(g)).item()) for _, g in tree_flatten(grads)):
            raise AssertionError("Non-finite adapter gradients")
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        print("Training step", step + 1, "loss", loss.item(), flush=True)
    report["training_final_loss"] = float(loss_fn(model).item())
    if report["training_final_loss"] >= report["training_initial_loss"]:
        raise AssertionError("Adapter training did not reduce loss")
    report["trainable_parameters"] = sum(
        p.size for _, p in tree_flatten(model.trainable_parameters())
    )
    adapter_dir = args.output_dir / "smoke-adapter"
    adapter_dir.mkdir(exist_ok=True)
    mx.save_safetensors(
        str(adapter_dir / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": 4,
                "lora_parameters": lora_config,
            },
            indent=2,
        )
        + "\n"
    )
    model.eval()
    trained_logits = np.array(model(sample[:, :-1]).astype(mx.float32))
    report["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    del model, optimizer, grads
    gc.collect()
    mx.clear_cache()
    restored_model, _ = load(
        args.model,
        adapter_path=str(adapter_dir),
        tokenizer_config={"trust_remote_code": False},
    )
    restored_model.set_dtype(mx.float32)
    restored_model.eval()
    restored_logits = np.array(restored_model(sample[:, :-1]).astype(mx.float32))
    np.testing.assert_array_equal(restored_logits, trained_logits)
    report["adapter_roundtrip_exact"] = True
    report["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
