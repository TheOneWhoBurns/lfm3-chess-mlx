import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_lm import JSONSchemaConstraint, load
from mlx_lm.tuner.utils import linear_to_lora_layers
from oracle import Oracle, read_rows
from rl_policy import MOVE_SCHEMA, RewardOracle, digest, prompt_tokens, rollout
from board_encoder import BoardPrompt, board_prompt, load_board_policy
from rewards import REWARD_CONFIG, reward_advantages
from rl_scoring import shared_prefix_log_probabilities, shared_prefix_policy_loss


def action_support(constraint, tokens):
    support, targets = [], []
    for index, token in enumerate(tokens):
        allowed = constraint.allowed_tokens(tokens[:index])
        if token not in allowed:
            raise ValueError("A sampled token is outside the recorded schema support")
        support.append(mx.array(allowed))
        targets.append(allowed.index(token))
    return support, targets


def conditional_logprobs(model, prompt, tokens, support, targets):
    prompt_ids = prompt.tokens if isinstance(prompt, BoardPrompt) else prompt
    inputs = mx.array([prompt_ids + tokens[:-1]])
    embeddings = (
        model.embed_prompt(inputs, prompt) if isinstance(prompt, BoardPrompt) else None
    )
    logits = model(inputs, input_embeddings=embeddings)[
        0, len(prompt_ids) - 1 :
    ].astype(mx.float32)
    selected, entropies = [], []
    for index, (allowed, target) in enumerate(zip(support, targets)):
        values = logits[index, allowed]
        logprobs = values - mx.logsumexp(values)
        selected.append(logprobs[target])
        entropies.append(-(mx.exp(logprobs) * logprobs).sum())
    return mx.stack(selected), mx.stack(entropies)


def policy_loss(model, prompt, tokens, support, targets, old, advantage, clip, entropy):
    logprobs, entropies = conditional_logprobs(model, prompt, tokens, support, targets)
    ratio = mx.exp(logprobs - old)
    objective = mx.minimum(
        ratio * advantage, mx.clip(ratio, 1 - clip, 1 + clip) * advantage
    )
    return -objective.sum() - entropy * entropies.sum()


def prepare_group(constraint, responses):
    advantages = reward_advantages(responses)
    if float(np.max(np.abs(advantages))) < 1e-7:
        return []
    return [
        (row, *action_support(constraint, row["tokens"]), float(advantage))
        for row, advantage in zip(responses, advantages)
    ]


def gradient_norm(tree):
    return mx.sqrt(
        sum(mx.sum(value.astype(mx.float32) ** 2) for _, value in tree_flatten(tree))
    )


def group_log_probabilities(model, prompt, group):
    return np.array(
        [
            float(
                conditional_logprobs(model, prompt, row["tokens"], support, targets)[0]
                .sum()
                .item()
            )
            for row, support, targets, _ in group
        ]
    )


def update_group(
    model,
    optimizer,
    prompt,
    group,
    clip=0.2,
    entropy=0.0,
    max_update_rms=1e-5,
    max_policy_logprob_change=0.5,
    shared_prefix=False,
):
    if not group:
        return {"updated": False}
    matched, ordinary, total_loss = None, None, 0.0
    if shared_prefix:
        value_and_grad = nn.value_and_grad(model, shared_prefix_policy_loss)
        for exact in [True, False]:
            subset = [item for item in group if bool(item[0]["exact_match"]) == exact]
            if not subset:
                continue
            loss, gradient = value_and_grad(model, prompt, subset, clip, entropy)
            mx.eval(loss, gradient)
            if exact:
                matched = gradient
            else:
                ordinary = gradient
            total_loss += float(loss.item())
    else:
        value_and_grad = nn.value_and_grad(model, policy_loss)
        for row, support, targets, advantage in group:
            loss, gradient = value_and_grad(
                model,
                prompt,
                row["tokens"],
                support,
                targets,
                mx.array(row["old_logprobs"]),
                advantage,
                clip,
                entropy,
            )
            mx.eval(loss, gradient)
            accumulated = matched if row["exact_match"] else ordinary
            accumulated = (
                gradient
                if accumulated is None
                else tree_map(lambda x, y: x + y, accumulated, gradient)
            )
            mx.eval(accumulated)
            if row["exact_match"]:
                matched = accumulated
            else:
                ordinary = accumulated
            total_loss += float(loss.item())
    match_norm = float(gradient_norm(matched).item()) if matched is not None else 0.0
    ordinary_norm = (
        float(gradient_norm(ordinary).item()) if ordinary is not None else 0.0
    )
    other_scale = 1.0
    if match_norm > 0 and ordinary_norm > 0:
        target_share = REWARD_CONFIG["match_gradient_share"]
        other_scale = min(
            1.0, (1.0 - target_share) / target_share * match_norm / ordinary_norm
        )
    if matched is None:
        accumulated = ordinary
    elif ordinary is None:
        accumulated = matched
    else:
        accumulated = tree_map(
            lambda exact, other: exact + other_scale * other, matched, ordinary
        )
    accumulated = tree_map(lambda value: value / len(group), accumulated)
    gradient, norm = optim.clip_grad_norm(accumulated, 1.0)
    mx.eval(norm, gradient)
    if not math.isfinite(float(norm.item())) or not math.isfinite(total_loss):
        raise FloatingPointError("Non-finite RL gradient or loss")
    previous = model.trainable_parameters()
    previous_optimizer = tree_map(lambda value: value, optimizer.state)
    score_group = (
        shared_prefix_log_probabilities if shared_prefix else group_log_probabilities
    )
    logprobs_before = score_group(model, prompt, group)
    match_mask = np.array([row["exact_match"] for row, _, _, _ in group], dtype=bool)
    match_before = (
        float(logprobs_before[match_mask].mean()) if match_mask.any() else None
    )
    optimizer.update(model, gradient)
    mx.eval(model.trainable_parameters(), optimizer.state)
    proposed = model.trainable_parameters()
    delta = tree_map(lambda after, before: after - before, proposed, previous)
    count = sum(value.size for _, value in tree_flatten(delta))
    rms = float((gradient_norm(delta) / math.sqrt(count)).item())
    step_scale = min(1.0, max_update_rms / max(rms, 1e-30))
    if step_scale < 1:
        model.update(
            tree_map(
                lambda before, after: before + step_scale * (after - before),
                previous,
                proposed,
            )
        )
        mx.eval(model.trainable_parameters())
    accepted = False
    match_after = None
    backtracks = 0
    while True:
        logprobs_after = score_group(model, prompt, group)
        policy_change = float(np.max(np.abs(logprobs_after - logprobs_before)))
        match_after = (
            float(logprobs_after[match_mask].mean()) if match_mask.any() else None
        )
        if (
            np.all(np.isfinite(logprobs_after))
            and policy_change <= max_policy_logprob_change
            and (match_before is None or match_after > match_before)
        ):
            accepted = True
            break
        if backtracks >= 12:
            break
        step_scale *= 0.5
        model.update(
            tree_map(
                lambda before, after: before + step_scale * (after - before),
                previous,
                proposed,
            )
        )
        mx.eval(model.trainable_parameters())
        backtracks += 1
    if not accepted:
        model.update(previous)
        optimizer.state = previous_optimizer
        mx.eval(model.trainable_parameters(), optimizer.state)
        step_scale = 0.0
        match_after = match_before
        policy_change = 0.0
    mx.clear_cache()
    return {
        "updated": accepted,
        "loss": total_loss / len(group),
        "gradient_norm": float(norm.item()),
        "match_gradient_norm": match_norm,
        "ordinary_gradient_norm": ordinary_norm,
        "ordinary_gradient_scale": other_scale,
        "match_gradient_share": match_norm
        / max(match_norm + other_scale * ordinary_norm, 1e-30),
        "parameter_update_rms": rms * step_scale,
        "parameter_step_scale": step_scale,
        "match_logprob_before": match_before,
        "match_logprob_after": match_after,
        "match_guard_backtracks": backtracks,
        "match_guard_rejected": not accepted,
        "max_sampled_logprob_change": policy_change,
    }


def save_checkpoint(directory, model, optimizer, config, group_index, updates):
    destination = directory / "checkpoints" / str(group_index)
    destination.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(
        str(destination / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    mx.save_safetensors(
        str(destination / "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )
    metadata = {
        **config,
        "completed_groups": group_index,
        "optimizer_updates": updates,
        "adapter_sha256": digest(destination / "adapters.safetensors"),
    }
    (destination / "adapter_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("saved", destination, "updates", updates, flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=200)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026103110)
    parser.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--nodes", type=int, default=40000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--input", choices=["board", "text"], default="board")
    parser.add_argument("--encoder", choices=["compact", "semantic"], default="compact")
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--shaping-weight", type=float, default=1.0)
    parser.add_argument("--max-update-rms", type=float, default=1e-5)
    parser.add_argument("--max-policy-logprob-change", type=float, default=0.5)
    args = parser.parse_args()
    if min(args.groups, args.save_every, args.max_tokens) < 1 or args.samples < 2:
        parser.error("Positive budgets and at least two samples per group are required")
    if (
        args.learning_rate <= 0
        or args.max_update_rms <= 0
        or args.max_policy_logprob_change <= 0
    ):
        parser.error("Learning rate and update RMS bound must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = read_rows(args.data / "train.jsonl")
    if not rows:
        raise ValueError("No training positions")
    random.Random(args.seed).shuffle(rows)
    mx.random.seed(args.seed)
    lora = {"rank": 8, "scale": 16.0, "dropout": 0.0}
    if args.input == "board":
        model, tokenizer = load_board_policy(
            args.model, layers=args.layers, seed=args.seed, encoder=args.encoder
        )
    else:
        model, tokenizer = load(
            str(args.model), tokenizer_config={"trust_remote_code": False}
        )
        model.set_dtype(mx.float32)
        model.freeze()
        linear_to_lora_layers(model, args.layers, lora)
    model.eval()
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    optimizer.init(model.trainable_parameters())
    constraint = JSONSchemaConstraint(tokenizer, MOVE_SCHEMA)
    engine = Oracle(args.engine, args.nodes)
    oracle = RewardOracle(engine, args.output, args.shaping_weight)
    sources = [
        Path(__file__),
        Path(__file__).with_name("rl_policy.py"),
        Path(__file__).with_name("rl_scoring.py"),
        Path(__file__).with_name("oracle.py"),
        Path(__file__).with_name("board_encoder.py"),
        Path(__file__).with_name("semantic_board_encoder.py"),
        Path(__file__).with_name("rewards.py"),
        Path(__file__).resolve().parents[2] / "mlx_lm" / "structured.py",
        Path(__file__).resolve().parents[2] / "mlx_lm" / "models" / "lfm3.py",
    ]
    config = {
        "training_kind": "match_priority_policy_gradient",
        "model": str(args.model.resolve()),
        "fine_tune_type": "lora",
        "num_layers": args.layers,
        "lora_parameters": lora,
        "data_sha256": digest(args.data / "train.jsonl"),
        "model_hashes": {
            path.name: digest(path) for path in args.model.glob("*.safetensors")
        },
        "source_hashes": {path.name: digest(path) for path in sources},
        "engine": engine.identity,
        "schema": MOVE_SCHEMA,
        "algorithm": "fixed-scale leave-one-out base advantages plus uncentered exact-match bonus; match-prioritized gradient; summed PPO surrogate; clip=0.2; no reference KL",
        "input_representation": args.input,
        "board_encoding": model.board_encoding if args.input == "board" else None,
        "encoder": args.encoder if args.input == "board" else None,
        "reward_config": REWARD_CONFIG,
        "shaping_weight": args.shaping_weight,
        "shared_prefix": args.shared_prefix,
        "max_update_rms": args.max_update_rms,
        "max_policy_logprob_change": args.max_policy_logprob_change,
        "randomness": "deterministic per-rollout seed: seed + group_index * samples + sample_index",
        "policy_temperature": 1.0,
        "probabilities": "normalized over JSON-schema support",
        "initialization": "original checkpoint with zero-delta LoRA and optional native-embedding-initialized board encoder; no supervised adapter",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "entropy": args.entropy,
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "planned_groups": args.groups,
        "history": "FEN only; repetition history omitted",
        "resume": str(args.resume) if args.resume else None,
    }
    start, updates = 0, 0
    if args.resume:
        previous = json.loads((args.resume / "adapter_config.json").read_text())
        for key in [
            "training_kind",
            "model_hashes",
            "data_sha256",
            "source_hashes",
            "num_layers",
            "lora_parameters",
            "engine",
            "schema",
            "seed",
            "samples",
            "learning_rate",
            "entropy",
            "max_tokens",
            "board_encoding",
            "reward_config",
            "shaping_weight",
            "shared_prefix",
            "max_update_rms",
            "max_policy_logprob_change",
        ]:
            if previous.get(key) != config[key]:
                raise ValueError(f"Cannot resume with changed {key}")
        model.load_weights(str(args.resume / "adapters.safetensors"), strict=False)
        optimizer.state = tree_unflatten(
            list(mx.load(str(args.resume / "optimizer.safetensors")).items())
        )
        start, updates = previous["completed_groups"], previous["optimizer_updates"]
    if start >= args.groups:
        raise ValueError("Requested group budget must exceed resumed progress")
    (args.output / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    (args.output / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, args.output / "source" / path.name)
    started = time.perf_counter()
    last_saved = start
    try:
        for group_index in range(start + 1, args.groups + 1):
            case = rows[(group_index - 1) % len(rows)]
            prompt = (
                board_prompt(tokenizer, case["fen"], model.board_encoding)
                if args.input == "board"
                else prompt_tokens(tokenizer, case["fen"])
            )
            responses = []
            for sample in range(args.samples):
                mx.random.seed(args.seed + group_index * args.samples + sample)
                response = rollout(
                    model, tokenizer, prompt, args.max_tokens, 1.0, constraint
                )
                response.update(oracle.score(case["fen"], response))
                responses.append(response)
                with (args.output / "rollouts.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "group": group_index,
                                "id": case["id"],
                                "fen": case["fen"],
                                "sample": sample,
                                **response,
                            }
                        )
                        + "\n"
                    )
            prepared = prepare_group(constraint, responses)
            metrics = update_group(
                model,
                optimizer,
                prompt,
                prepared,
                entropy=args.entropy,
                max_update_rms=args.max_update_rms,
                max_policy_logprob_change=args.max_policy_logprob_change,
                shared_prefix=args.shared_prefix,
            )
            updates += int(metrics["updated"])
            metrics.update(
                {
                    "group": group_index,
                    "optimizer_updates": updates,
                    "id": case["id"],
                    "samples": args.samples,
                    "legal": sum(row["accepted"] for row in responses),
                    "exact_match": sum(row["exact_match"] for row in responses),
                    "mean_reward": float(np.mean([row["reward"] for row in responses])),
                    "seconds": time.perf_counter() - started,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
            )
            with (args.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if group_index % args.save_every == 0 or group_index == args.groups:
                save_checkpoint(
                    args.output, model, optimizer, config, group_index, updates
                )
                last_saved = group_index
    except KeyboardInterrupt:
        (args.output / "interrupted.json").write_text(
            json.dumps(
                {
                    "last_complete_checkpoint": last_saved,
                    "unsaved_updates_discarded": True,
                }
            )
            + "\n"
        )
        raise
    finally:
        engine.close()
    (args.output / "completed.json").write_text(
        json.dumps(
            {
                "groups": args.groups,
                "updates": updates,
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
