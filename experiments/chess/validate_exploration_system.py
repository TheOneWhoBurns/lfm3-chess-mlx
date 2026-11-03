import argparse
import hashlib
import json
from pathlib import Path

import chess
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from mlx_lm import JSONSchemaConstraint
from batched_policy import batch_rollout
from board_encoder import board_prompt, load_board_policy
from exploration import sequence_terms
from oracle import Oracle
from rewards import reward_advantages
from rl_batched_scoring import response_logits, batched_prefix_log_probabilities
from rl_policy import MOVE_SCHEMA, RewardOracle, rollout
from train_rl import action_support, prepare_group, update_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {}
    for line in args.rollouts.open():
        row = json.loads(line)
        if row["group"] >= 2500 and not row["accepted"]:
            cases[row["fen"]] = row
    model, tokenizer = load_board_policy(
        args.model, args.checkpoint, encoder="semantic"
    )
    model.eval()
    initial = model.trainable_parameters()
    constraint = JSONSchemaConstraint(tokenizer, MOVE_SCHEMA)
    engine = Oracle("/opt/homebrew/bin/stockfish", 40000)
    oracle = RewardOracle(engine, args.output)
    report = {
        "scope": "Real-model system diagnostics only; constructed responses and diagnostic updates never initialize training",
        "colors": {},
    }

    def fresh_optimizer():
        model.update(initial)
        mx.eval(model.trainable_parameters())
        optimizer = optim.Adam(learning_rate=1e-5)
        optimizer.init(model.trainable_parameters())
        return optimizer

    def set_old(prompt, responses):
        group = [(r, *action_support(constraint, r["tokens"]), 0.0) for r in responses]
        for row, support, targets, logits in response_logits(
            model, prompt, group, True
        ):
            selected, _, _ = sequence_terms(logits, support, targets, row)
            row["old_logprobs"] = np.array(selected).tolist()

    def bounded(metrics):
        assert metrics["updated"], metrics
        assert metrics["max_sampled_logprob_change"] <= 0.500001, metrics
        assert metrics["parameter_update_rms"] <= 1.00001e-5, metrics

    try:
        for color in (chess.WHITE, chess.BLACK):
            fresh_optimizer()
            for fen in reversed(list(cases)):
                if chess.Board(fen).turn != color:
                    continue
                prompt = board_prompt(tokenizer, fen, model.board_encoding)
                greedy = batch_rollout(model, tokenizer, [prompt], constraint)[0]
                greedy.update(oracle.score(fen, greedy))
                if not greedy["accepted"]:
                    break
            else:
                raise AssertionError("No naturally generated illegal diagnostic found")
            sampled = []
            for seed in (71, 72, 73, 74):
                mx.random.seed(seed)
                sampled.extend(
                    batch_rollout(
                        model,
                        tokenizer,
                        [prompt] * 8,
                        constraint,
                        temperature=1.2,
                        exploration_epsilon=0.08,
                    )
                )
            assert all(r["schema_valid"] and r["ended"] for r in sampled)
            unique = len({r["raw"] for r in sampled})
            assert unique > 1, "Exploration failed to diversify a collapsed policy"
            mx.random.seed(71)
            replay = batch_rollout(
                model,
                tokenizer,
                [prompt] * 8,
                constraint,
                temperature=1.2,
                exploration_epsilon=0.08,
            )
            assert replay == sampled[:8], "Seeded batch replay changed"
            group = [
                (r, *action_support(constraint, r["tokens"]), 0.0) for r in sampled[:8]
            ]
            computed = batched_prefix_log_probabilities(model, prompt, group)
            parity = float(
                np.max(np.abs(computed - [sum(r["old_logprobs"]) for r in sampled[:8]]))
            )
            assert parity < 0.002, parity
            serial = rollout(
                model,
                tokenizer,
                prompt,
                temperature=1.2,
                constraint=constraint,
                exploration_epsilon=0.08,
            )
            sg = [(serial, *action_support(constraint, serial["tokens"]), 0.0)]
            serial_error = abs(
                float(batched_prefix_log_probabilities(model, prompt, sg)[0])
                - sum(serial["old_logprobs"])
            )
            assert serial_error < 0.002, serial_error
            duplicates = [
                dict(greedy, sampling_temperature=1.2, exploration_epsilon=0.08)
                for _ in range(8)
            ]
            set_old(prompt, duplicates)
            advantages = reward_advantages(duplicates)
            assert np.all(advantages <= -1.0), advantages
            optimizer = fresh_optimizer()
            bad_update = update_group(
                model,
                optimizer,
                prompt,
                prepare_group(constraint, duplicates, exploration=True),
                shared_prefix=True,
                batched_scoring=True,
                entropy=0.01,
                uniform_kl=0.02,
            )
            bounded(bad_update)
            assert (
                bad_update["illegal_logprob_after"]
                < bad_update["illegal_logprob_before"]
            ), bad_update
            print(
                json.dumps({"color": color, "illegal_update": bad_update}), flush=True
            )
            optimizer = fresh_optimizer()
            neutral = [
                dict(
                    r,
                    accepted=True,
                    legal=True,
                    exact_match=False,
                    base_reward=2.0,
                    match_bonus=0.0,
                )
                for r in duplicates
            ]
            assert not np.any(reward_advantages(neutral))
            neutral_group = prepare_group(constraint, neutral, exploration=True)
            entropy_before = []
            for row, support, targets, logits in response_logits(
                model, prompt, neutral_group, True
            ):
                entropy_before.append(
                    float(sequence_terms(logits, support, targets, row)[2].item())
                )
            neutral_update = update_group(
                model,
                optimizer,
                prompt,
                neutral_group,
                shared_prefix=True,
                batched_scoring=True,
                entropy=0.01,
                uniform_kl=0.02,
            )
            bounded(neutral_update)
            entropy_after = []
            for row, support, targets, logits in response_logits(
                model, prompt, neutral_group, True
            ):
                entropy_after.append(
                    float(sequence_terms(logits, support, targets, row)[2].item())
                )
            assert np.mean(entropy_after) < np.mean(entropy_before), (
                entropy_before,
                entropy_after,
            )
            optimizer = fresh_optimizer()
            best = oracle.analyse(fen)["move"]
            raw = json.dumps({"move": best})
            exact = {
                "raw": raw,
                "tokens": tokenizer.encode(raw, add_special_tokens=False)
                + [tokenizer.eos_token_id],
                "ended": True,
                "schema_valid": True,
                "response_format": "json_schema",
                "sampling_temperature": 1.2,
                "exploration_epsilon": 0.08,
            }
            exact.update(oracle.score(fen, exact))
            mixed = [exact] + [dict(r) for r in duplicates[:7]]
            set_old(prompt, mixed)
            assert (
                reward_advantages([dict(exact) for _ in range(8)]).tolist() == [4.0] * 8
            )
            match_update = update_group(
                model,
                optimizer,
                prompt,
                prepare_group(constraint, mixed, exploration=True),
                shared_prefix=True,
                batched_scoring=True,
                entropy=0.01,
                uniform_kl=0.02,
            )
            bounded(match_update)
            assert match_update["match_gradient_share"] >= 0.8 - 1e-6, match_update
            assert (
                match_update["match_logprob_after"]
                > match_update["match_logprob_before"]
            ), match_update
            report["colors"]["white" if color else "black"] = {
                "fen": fen,
                "collapsed_illegal_move": greedy["answer"],
                "unique_exploration_responses_of_32": unique,
                "batch_probability_error": parity,
                "serial_probability_error": serial_error,
                "seeded_replay_exact": True,
                "identical_illegal_advantages": advantages.tolist(),
                "identical_illegal_update": bad_update,
                "zero_advantage_exploration_update": neutral_update,
                "uniform_kl_before": float(np.mean(entropy_before)),
                "uniform_kl_after": float(np.mean(entropy_after)),
                "match_bonus_advantage": 4.0,
                "match_update": match_update,
            }
            (args.output / "partial.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        report["passed"] = True
        report["source_hashes"] = {
            str(path.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                Path(__file__).resolve().with_name(name)
                for name in (
                    "train_rl.py",
                    "exploration.py",
                    "rewards.py",
                    "rl_policy.py",
                    "batched_policy.py",
                    "rl_scoring.py",
                    "rl_batched_scoring.py",
                    "validate_exploration_system.py",
                )
            ]
        }
        report["peak_memory_gb"] = mx.get_peak_memory() / 1e9
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        oracle.close()
        engine.close()


if __name__ == "__main__":
    main()
