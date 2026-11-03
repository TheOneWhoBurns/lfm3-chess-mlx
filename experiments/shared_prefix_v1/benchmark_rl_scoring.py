import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np

from mlx_lm import JSONSchemaConstraint
from board_encoder import board_prompt, load_board_policy
from oracle import read_rows
from rl_policy import MOVE_SCHEMA, digest, rollout
from rl_scoring import shared_prefix_log_probabilities, shared_prefix_policy_loss
from train_rl import action_support, group_log_probabilities, policy_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--encoder", choices=["compact", "semantic"], default="semantic"
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--gradients", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model, tokenizer = load_board_policy(args.model, encoder=args.encoder)
    constraint = JSONSchemaConstraint(tokenizer, MOVE_SCHEMA)
    rows = read_rows(args.data / "train.jsonl")
    case = rows[0]
    prompt = board_prompt(tokenizer, case["fen"], model.board_encoding)
    group = []
    for index in range(args.samples):
        mx.random.seed(2026103116 + index)
        row = rollout(model, tokenizer, prompt, constraint=constraint)
        group.append(
            (
                row,
                *action_support(constraint, row["tokens"]),
                0.3 if index % 2 else -0.2,
            )
        )
    results = {}
    for name, function in [
        ("full_forward", group_log_probabilities),
        ("shared_prefix", shared_prefix_log_probabilities),
        ("shared_prefix_repeat", shared_prefix_log_probabilities),
    ]:
        mx.synchronize()
        started = time.perf_counter()
        values = function(model, prompt, group)
        mx.synchronize()
        results[name] = {
            "seconds": time.perf_counter() - started,
            "logprobs": values.tolist(),
        }
    expected = np.array(results["full_forward"]["logprobs"])
    actual = np.array(results["shared_prefix"]["logprobs"])
    repeated = np.array(results["shared_prefix_repeat"]["logprobs"])
    error = float(np.max(np.abs(expected - actual)))
    repeat_error = float(np.max(np.abs(actual - repeated)))
    if error > 0.002 or repeat_error > 1e-6:
        raise AssertionError(
            f"Shared-prefix scoring changed probabilities: {error}, repeat {repeat_error}"
        )
    gradient_report = None
    if args.gradients:
        value_and_grad = nn.value_and_grad(model, policy_loss)
        started = time.perf_counter()
        expected_gradient, expected_loss = None, 0.0
        for row, support, targets, advantage in group:
            loss, gradient = value_and_grad(
                model,
                prompt,
                row["tokens"],
                support,
                targets,
                mx.array(row["old_logprobs"]),
                advantage,
                0.2,
                0.0,
            )
            mx.eval(loss, gradient)
            expected_loss += float(loss.item())
            expected_gradient = (
                gradient
                if expected_gradient is None
                else tree_map(lambda a, b: a + b, expected_gradient, gradient)
            )
            mx.eval(expected_gradient)
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        loss, actual_gradient = nn.value_and_grad(model, shared_prefix_policy_loss)(
            model, prompt, group
        )
        mx.eval(loss, actual_gradient)
        shared_seconds = time.perf_counter() - started
        reference = dict(tree_flatten(expected_gradient))
        actual_gradients = dict(tree_flatten(actual_gradient))
        squared_error = sum(
            float(mx.sum((value - reference[name]) ** 2).item())
            for name, value in actual_gradients.items()
        )
        squared_norm = sum(
            float(mx.sum(value**2).item()) for value in reference.values()
        )
        relative_error = (squared_error / max(squared_norm, 1e-30)) ** 0.5
        if relative_error > 0.002 or abs(float(loss.item()) - expected_loss) > 0.002:
            raise AssertionError(
                f"Shared-prefix loss or gradient differs: {float(loss.item()) - expected_loss}, relative gradient {relative_error}"
            )
        gradient_report = {
            "relative_l2_error": relative_error,
            "loss_error": abs(float(loss.item()) - expected_loss),
            "full_forward_seconds": baseline_seconds,
            "shared_prefix_seconds": shared_seconds,
            "speedup": baseline_seconds / shared_seconds,
        }
    report = {
        "passed": True,
        "gradient_parity": gradient_report,
        "data_sha256": digest(args.data / "train.jsonl"),
        "runtime_sha256": digest(
            Path(__file__).resolve().parents[2] / "mlx_lm/models/lfm3.py"
        ),
        "model_hashes": {
            path.name: digest(path) for path in args.model.glob("*.safetensors")
        },
        "fen": case["fen"],
        "samples": args.samples,
        "board_encoding": model.board_encoding,
        "results": results,
        "max_logprob_error": error,
        "repeat_error": repeat_error,
        "speedup": results["full_forward"]["seconds"]
        / results["shared_prefix"]["seconds"],
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "source_hashes": {
            name: digest(Path(__file__).with_name(name))
            for name in [
                "benchmark_rl_scoring.py",
                "rl_scoring.py",
                "train_rl.py",
                "board_encoder.py",
                "semantic_board_encoder.py",
                "rl_policy.py",
            ]
        },
        "scope": "Real-model system parity and timing, no parameter updates. Timing under concurrent workloads is only indicative.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
