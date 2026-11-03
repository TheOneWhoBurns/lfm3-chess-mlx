import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mlx_lm import load

REFERENCE = "LiquidAI/LFM3-1.7B-Base"
REVISION = "545fc8f4c7b5fa9f405105274cd29c4f8f44b526"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config_dict = json.loads((args.model / "config.json").read_text())
    quantization = config_dict["quantization"]
    if quantization.get("mode", "affine") != "affine":
        raise ValueError("This reference comparison supports affine exports only")
    model, tokenizer = load(
        str(args.model), tokenizer_config={"trust_remote_code": False}
    )
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
    tokens = tokenizer.encode(prompt)
    expected = np.array(model(mx.array([tokens])))
    del model
    gc.collect()
    mx.clear_cache()
    config = AutoConfig.from_pretrained(
        REFERENCE, revision=REVISION, trust_remote_code=True
    )
    config._attn_implementation = "eager"
    reference_class = get_class_from_dynamic_module(
        "modeling_lfm3.Lfm3ForCausalLM", REFERENCE, revision=REVISION
    )
    with init_empty_weights():
        reference = reference_class(config)
    raw = {}
    for file in args.model.glob("*.safetensors"):
        raw.update(mx.load(str(file)))
    state = {}
    for name, weight in raw.items():
        if name.endswith((".scales", ".biases")):
            continue
        stem = name.removesuffix(".weight")
        if stem + ".scales" in raw:
            weight = mx.dequantize(
                weight,
                raw[stem + ".scales"].astype(mx.float32),
                raw[stem + ".biases"].astype(mx.float32),
                group_size=quantization["group_size"],
                bits=quantization["bits"],
            )
        if name.endswith("conv.conv.weight") and weight.shape[1:] == (
            config.conv_L_cache,
            1,
        ):
            weight = weight.transpose(0, 2, 1)
        state[name.removeprefix("language_model.")] = torch.from_numpy(
            np.array(weight.astype(mx.float32))
        )
    missing, unexpected = reference.load_state_dict(state, strict=False, assign=True)
    if missing != ["lm_head.weight"] or unexpected:
        raise AssertionError((missing, unexpected))
    reference.tie_weights()
    reference.eval()
    del raw, state, weight
    gc.collect()
    mx.clear_cache()
    with torch.no_grad():
        actual = reference(torch.tensor([tokens]), use_cache=False).logits.numpy()
    difference = np.abs(actual - expected)
    report = {
        "checkpoint": str(args.model),
        "reference_revision": REVISION,
        "floating_dtype": "float32",
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "top1_agreement": float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
        "tokens": len(tokens),
    }
    np.testing.assert_allclose(expected, actual, atol=0.002, rtol=0.002)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
