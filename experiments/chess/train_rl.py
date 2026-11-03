import argparse
import json
import math
import shutil
import os
import signal
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_lm import JSONSchemaConstraint, load
from mlx_lm.tuner.utils import linear_to_lora_layers
from oracle import Oracle
from corpus_access import corpus_signature, load_positions
from rl_policy import MOVE_SCHEMA, RewardOracle, digest, prompt_tokens, rollout
from batched_policy import batch_rollout
from board_encoder import BoardPrompt, board_prompt, load_board_policy
from rewards import REWARD_CONFIG, reward_advantages
from exploration import behavior_logprobs, response_loss, sequence_terms
from rl_checkpoint import initialize_rl_stage, load_trainable_checkpoint
from rl_scoring import shared_prefix_log_probabilities, shared_prefix_policy_loss
from rl_batched_scoring import (
    batched_prefix_log_probabilities,
    batched_prefix_policy_loss,
)


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
        logprobs = behavior_logprobs(values)
        selected.append(logprobs[target])
        entropies.append(-(mx.exp(logprobs) * logprobs).sum())
    return mx.stack(selected), mx.stack(entropies)


def policy_loss(
    model, prompt, row, support, targets, advantage, clip, entropy, uniform_kl
):
    prompt_ids = prompt.tokens if isinstance(prompt, BoardPrompt) else prompt
    inputs = mx.array([prompt_ids + row["tokens"][:-1]])
    embeddings = (
        model.embed_prompt(inputs, prompt) if isinstance(prompt, BoardPrompt) else None
    )
    logits = model(inputs, input_embeddings=embeddings)[0, len(prompt_ids) - 1 :]
    return response_loss(
        logits, support, targets, row, advantage, clip, entropy, uniform_kl
    )


def prepare_group(constraint, responses, exploration=False):
    advantages = reward_advantages(responses)
    if not exploration and float(np.max(np.abs(advantages))) < 1e-7:
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
    result = []
    for row, support, targets, _ in group:
        prompt_ids = prompt.tokens if isinstance(prompt, BoardPrompt) else prompt
        inputs = mx.array([prompt_ids + row["tokens"][:-1]])
        embeddings = (
            model.embed_prompt(inputs, prompt)
            if isinstance(prompt, BoardPrompt)
            else None
        )
        logits = model(inputs, input_embeddings=embeddings)[0, len(prompt_ids) - 1 :]
        selected, _, _ = sequence_terms(logits, support, targets, row)
        result.append(float(selected.sum().item()))
    return np.array(result)


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
    batched_scoring=False,
    uniform_kl=0.0,
):
    if batched_scoring and not shared_prefix:
        raise ValueError("Batched scoring requires shared-prefix scoring")
    if not group:
        return {"updated": False, "skip_reason": "zero_advantage_without_exploration"}
    matched, ordinary, total_loss = None, None, 0.0
    if shared_prefix:
        value_and_grad = nn.value_and_grad(
            model,
            batched_prefix_policy_loss
            if batched_scoring
            else shared_prefix_policy_loss,
        )
        for exact in [True, False]:
            subset = [item for item in group if bool(item[0]["exact_match"]) == exact]
            if not subset:
                continue
            loss, gradient = value_and_grad(model, prompt, subset, clip, 0.0, 0.0)
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
                row,
                support,
                targets,
                advantage,
                clip,
                0.0,
                0.0,
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
    exploration_norm = 0.0
    if entropy or uniform_kl:
        neutral = [(row, support, targets, 0.0) for row, support, targets, _ in group]
        exploration_grad = nn.value_and_grad(
            model,
            batched_prefix_policy_loss
            if batched_scoring
            else shared_prefix_policy_loss,
        )
        loss, gradient = exploration_grad(
            model, prompt, neutral, clip, entropy, uniform_kl
        )
        mx.eval(loss, gradient)
        exploration_norm = float(gradient_norm(gradient).item())
        ordinary = (
            gradient
            if ordinary is None
            else tree_map(lambda x, y: x + y, ordinary, gradient)
        )
        mx.eval(ordinary)
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
        batched_prefix_log_probabilities
        if batched_scoring
        else shared_prefix_log_probabilities
        if shared_prefix
        else group_log_probabilities
    )
    logprobs_before = score_group(model, prompt, group)
    base_group = [
        (
            dict(row, sampling_temperature=1.0, exploration_epsilon=0.0),
            support,
            targets,
            advantage,
        )
        for row, support, targets, advantage in group
    ]
    base_before = score_group(model, prompt, base_group)
    match_mask = np.array([row["exact_match"] for row, _, _, _ in group], dtype=bool)
    all_illegal = all(
        not row.get("accepted", row.get("legal", False)) for row, _, _, _ in group
    )
    match_before = float(base_before[match_mask].mean()) if match_mask.any() else None
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
        base_after = score_group(model, prompt, base_group)
        policy_change = float(
            max(
                np.max(np.abs(logprobs_after - logprobs_before)),
                np.max(np.abs(base_after - base_before)),
            )
        )
        match_after = float(base_after[match_mask].mean()) if match_mask.any() else None
        if (
            np.all(np.isfinite(logprobs_after))
            and np.all(np.isfinite(base_after))
            and policy_change <= max_policy_logprob_change
            and (match_before is None or match_after > match_before)
            and (
                not all_illegal or float(base_after.mean()) < float(base_before.mean())
            )
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
        base_after = base_before
    mx.clear_cache()
    return {
        "updated": accepted,
        "skip_reason": None if accepted else "step_guard_rejected",
        "exploration_gradient_norm": exploration_norm,
        "all_illegal": all_illegal,
        "illegal_logprob_before": float(base_before.mean()) if all_illegal else None,
        "illegal_logprob_after": float(base_after.mean()) if all_illegal else None,
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".pending-{group_index}-{os.getpid()}")
    temporary.mkdir(exist_ok=False)
    mx.save_safetensors(
        str(temporary / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    mx.save_safetensors(
        str(temporary / "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )
    metadata = {
        **config,
        "completed_groups": group_index,
        "optimizer_updates": updates,
        "adapter_sha256": digest(temporary / "adapters.safetensors"),
        "optimizer_sha256": digest(temporary / "optimizer.safetensors"),
    }
    (temporary / "adapter_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    for path in temporary.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    temporary.rename(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
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
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--uniform-kl", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--exploration-epsilon", type=float, default=0.08)
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026103110)
    parser.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--nodes", type=int, default=40000)
    parser.add_argument("--oracle-cache", type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--initialize-from", type=Path)
    parser.add_argument("--input", choices=["board", "text"], default="board")
    parser.add_argument("--encoder", choices=["compact", "semantic"], default="compact")
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--batched-rollouts", action="store_true")
    parser.add_argument("--batched-scoring", action="store_true")
    parser.add_argument("--shaping-weight", type=float, default=1.0)
    parser.add_argument("--max-update-rms", type=float, default=1e-5)
    parser.add_argument("--max-policy-logprob-change", type=float, default=0.5)
    args = parser.parse_args()
    if not (
        math.isfinite(args.temperature)
        and args.temperature > 0
        and math.isfinite(args.exploration_epsilon)
        and 0 <= args.exploration_epsilon < 1
        and math.isfinite(args.entropy)
        and args.entropy >= 0
        and math.isfinite(args.uniform_kl)
        and args.uniform_kl >= 0
        and args.position_offset >= 0
    ):
        parser.error(
            "Exploration parameters must be finite and within their valid ranges"
        )
    stop_requested = []

    def request_stop(signum, frame):
        stop_requested.append(signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if args.batched_rollouts and args.input != "board":
        parser.error("Batched rollouts require numeric board input")
    if args.batched_scoring and not args.shared_prefix:
        parser.error("Batched scoring requires --shared-prefix")
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
    rows, data_format = load_positions(args.data, "train", args.seed)
    if not rows:
        raise ValueError("No training positions")
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
    try:
        oracle = RewardOracle(
            engine, args.output, args.shaping_weight, args.oracle_cache
        )
        sources = [
            Path(__file__),
            Path(__file__).with_name("rl_policy.py"),
            Path(__file__).with_name("batched_policy.py"),
            Path(__file__).with_name("rl_scoring.py"),
            Path(__file__).with_name("rl_batched_scoring.py"),
            Path(__file__).with_name("rl_checkpoint.py"),
            Path(__file__).with_name("corpus_access.py"),
            Path(__file__).with_name("indexed_corpus.py"),
            Path(__file__).with_name("oracle.py"),
            Path(__file__).with_name("oracle_cache.py"),
            Path(__file__).with_name("board_encoder.py"),
            Path(__file__).with_name("semantic_board_encoder.py"),
            Path(__file__).with_name("rewards.py"),
            Path(__file__).with_name("exploration.py"),
            Path(__file__).resolve().parents[2] / "mlx_lm" / "structured.py",
            Path(__file__).resolve().parents[2] / "mlx_lm" / "models" / "lfm3.py",
        ]
        config = {
            "training_kind": "match_priority_policy_gradient",
            "model": str(args.model.resolve()),
            "fine_tune_type": "lora",
            "num_layers": args.layers,
            "lora_parameters": lora,
            "data": str(args.data.resolve()),
            "data_sha256": corpus_signature(args.data, "train"),
            "data_format": data_format,
            "training_positions": len(rows),
            "model_hashes": {
                path.name: digest(path) for path in args.model.glob("*.safetensors")
            },
            "source_hashes": {path.name: digest(path) for path in sources},
            "engine": engine.identity,
            "oracle_cache": str(args.oracle_cache.resolve())
            if args.oracle_cache
            else None,
            "oracle_namespace": oracle.namespace,
            "schema": MOVE_SCHEMA,
            "algorithm": "leave-one-out base advantages with negative illegal floor plus uncentered exact bonus; syntax-only mixture exploration; entropy and uniform-to-policy KL on branching tokens; match-prioritized gradient; summed PPO surrogate; clip=0.2",
            "input_representation": args.input,
            "board_encoding": model.board_encoding if args.input == "board" else None,
            "encoder": args.encoder if args.input == "board" else None,
            "reward_config": REWARD_CONFIG,
            "shaping_weight": args.shaping_weight,
            "shared_prefix": args.shared_prefix,
            "batched_rollouts": args.batched_rollouts,
            "batched_scoring": args.batched_scoring,
            "max_update_rms": args.max_update_rms,
            "max_policy_logprob_change": args.max_policy_logprob_change,
            "randomness": (
                "deterministic per-group batch seed: seed + (position_offset + group_index) * samples"
                if args.batched_rollouts
                else "deterministic per-rollout seed: seed + (position_offset + group_index) * samples + sample_index"
            ),
            "policy_temperature": args.temperature,
            "exploration_epsilon": args.exploration_epsilon,
            "position_offset": args.position_offset,
            "probabilities": "recorded behavior probabilities include temperature and uniform mixture over JSON-schema support; greedy inference remains unmodified",
            "initialization": "original checkpoint with zero-delta LoRA and optional native-embedding-initialized board encoder; no supervised adapter",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "entropy": args.entropy,
            "uniform_kl": args.uniform_kl,
            "save_every": args.save_every,
            "samples": args.samples,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "planned_groups": args.groups,
            "history": "FEN only; repetition history omitted",
            "resume": str(args.resume) if args.resume else None,
            "parent_rl_checkpoint": None,
        }
        if args.initialize_from:
            config["parent_rl_checkpoint"] = initialize_rl_stage(
                model, args.initialize_from, config
            )
            config["initialization"] = (
                "encoder and LoRA weights from a verified RL checkpoint; fresh Adam state for a new stage; no supervised adapter"
            )
        start, updates = 0, 0
        if args.resume:
            previous = json.loads((args.resume / "adapter_config.json").read_text())
            for key in [
                "training_kind",
                "model_hashes",
                "data_sha256",
                "data_format",
                "training_positions",
                "source_hashes",
                "num_layers",
                "lora_parameters",
                "engine",
                "oracle_namespace",
                "schema",
                "seed",
                "samples",
                "learning_rate",
                "entropy",
                "uniform_kl",
                "policy_temperature",
                "exploration_epsilon",
                "position_offset",
                "max_tokens",
                "board_encoding",
                "reward_config",
                "shaping_weight",
                "shared_prefix",
                "batched_rollouts",
                "batched_scoring",
                "randomness",
                "max_update_rms",
                "max_policy_logprob_change",
            ]:
                if previous.get(key) != config[key]:
                    raise ValueError(f"Cannot resume with changed {key}")
            load_trainable_checkpoint(model, args.resume)
            if digest(args.resume / "optimizer.safetensors") != previous.get(
                "optimizer_sha256"
            ):
                raise ValueError("Checkpoint optimizer checksum does not match")
            config["parent_rl_checkpoint"] = previous.get("parent_rl_checkpoint")
            config["initialization"] = previous["initialization"]
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
        if args.initialize_from:
            shutil.copyfile(
                args.initialize_from / "adapter_config.json",
                args.output / "source" / "parent_adapter_config.json",
            )
        started = time.perf_counter()
        last_saved = start
        try:
            for group_index in range(start + 1, args.groups + 1):
                if stop_requested:
                    if last_saved != group_index - 1 or not last_saved:
                        save_checkpoint(
                            args.output,
                            model,
                            optimizer,
                            config,
                            group_index - 1,
                            updates,
                        )
                    (args.output / "interrupted.json").write_text(
                        json.dumps(
                            {
                                "last_complete_checkpoint": group_index - 1,
                                "unsaved_updates_discarded": False,
                                "signals": stop_requested,
                            }
                        )
                        + "\n"
                    )
                    return
                case = rows[(args.position_offset + group_index - 1) % len(rows)]
                prompt = (
                    board_prompt(tokenizer, case["fen"], model.board_encoding)
                    if args.input == "board"
                    else prompt_tokens(tokenizer, case["fen"])
                )
                responses = []
                sampling_started = time.perf_counter()
                if args.batched_rollouts:
                    mx.random.seed(
                        args.seed + (args.position_offset + group_index) * args.samples
                    )
                    sampled = batch_rollout(
                        model,
                        tokenizer,
                        [prompt] * args.samples,
                        constraint,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        exploration_epsilon=args.exploration_epsilon,
                    )
                else:
                    sampled = []
                    for sample in range(args.samples):
                        mx.random.seed(
                            args.seed
                            + (args.position_offset + group_index) * args.samples
                            + sample
                        )
                        sampled.append(
                            rollout(
                                model,
                                tokenizer,
                                prompt,
                                args.max_tokens,
                                args.temperature,
                                constraint,
                                exploration_epsilon=args.exploration_epsilon,
                            )
                        )
                sampling_seconds = time.perf_counter() - sampling_started
                for sample in range(args.samples):
                    response = sampled[sample]
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
                prepared = prepare_group(
                    constraint,
                    responses,
                    exploration=bool(args.entropy or args.uniform_kl),
                )
                metrics = update_group(
                    model,
                    optimizer,
                    prompt,
                    prepared,
                    entropy=args.entropy,
                    max_update_rms=args.max_update_rms,
                    max_policy_logprob_change=args.max_policy_logprob_change,
                    shared_prefix=args.shared_prefix,
                    batched_scoring=args.batched_scoring,
                    uniform_kl=args.uniform_kl,
                )
                updates += int(metrics["updated"])
                metrics.update(
                    {
                        "group": group_index,
                        "position_index": args.position_offset + group_index,
                        "unique_responses": len({row["raw"] for row in responses}),
                        "reward_std": float(
                            np.std([row["reward"] for row in responses])
                        ),
                        "advantage_min": float(min(item[3] for item in prepared))
                        if prepared
                        else 0.0,
                        "advantage_max": float(max(item[3] for item in prepared))
                        if prepared
                        else 0.0,
                        "optimizer_updates": updates,
                        "id": case["id"],
                        "samples": args.samples,
                        "sampling_seconds": sampling_seconds,
                        "legal": sum(row["accepted"] for row in responses),
                        "exact_match": sum(row["exact_match"] for row in responses),
                        "mean_reward": float(
                            np.mean([row["reward"] for row in responses])
                        ),
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
            (args.output / "oracle-statistics.json").write_text(
                json.dumps(oracle.statistics, indent=2) + "\n"
            )
            if hasattr(rows, "close"):
                rows.close()
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
    finally:
        if "oracle" in locals():
            oracle.close()
        engine.close()
        if hasattr(rows, "close"):
            rows.close()


if __name__ == "__main__":
    main()
