import argparse
import fcntl
import json
import time
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from pathlib import Path

import chess
import mlx.core as mx

from mlx_lm import JSONSchemaConstraint
from batched_policy import batch_rollout
from board_encoder import board_prompt, load_board_policy
from corpus_access import corpus_signature, load_positions
from oracle import Oracle
from rl_policy import MOVE_SCHEMA, RewardOracle, digest, rollout
from prefix_policy import FixedPrefixPolicy
from rewards import REWARD_CONFIG


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_committed(path):
    if not path.exists():
        return
    with path.open("rb+") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                backup = path.with_name(f"{path.name}.uncommitted-{time.time_ns()}")
                backup.write_bytes(line)
                handle.truncate(offset)
                break
            yield json.loads(line)


def record_phase(row):
    if row.get("evaluation_phase"):
        return row["evaluation_phase"]
    board = chess.Board(row["fen"])
    return (
        "endgame"
        if chess.popcount(board.occupied) <= 10
        else "opening"
        if row["ply"] <= 20
        else "middlegame"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Resumable single-attempt greedy evaluation over a complete board-position split."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--positions", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-fixed-prefix", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026103108)
    parser.add_argument("--encoder-seed", type=int, default=2026103110)
    parser.add_argument(
        "--encoder", choices=["auto", "compact", "semantic"], default="auto"
    )
    parser.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--nodes", type=int, default=40000)
    parser.add_argument("--oracle-cache", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-batches", type=int)
    args = parser.parse_args()
    if args.cache_fixed_prefix and args.batch_size != 1:
        parser.error("Fixed instruction caching currently requires batch size 1")
    if min(args.batch_size, args.max_tokens, args.nodes) < 1 or (
        args.max_new_batches is not None and args.max_new_batches < 1
    ):
        parser.error(
            "Positive batch, token, engine, and optional invocation budgets required"
        )
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if args.resume and not (args.output / "manifest.json").exists():
        raise ValueError("Resume requires an existing evaluation manifest")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (
        (args.output / ".evaluation.lock").open("a+") as lock,
        ExitStack() as resources,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, data_format = load_positions(args.data, args.split, args.seed)
        if hasattr(rows, "close"):
            resources.callback(rows.close)
        if args.positions is None:
            args.positions = len(rows)
        if not 1 <= args.positions <= len(rows):
            raise ValueError("Requested evaluation size is outside the frozen split")
        encoder = (
            (None if args.adapter else "semantic")
            if args.encoder == "auto"
            else args.encoder
        )
        model, tokenizer = load_board_policy(
            args.model, args.adapter, seed=args.encoder_seed, encoder=encoder
        )
        constraint = JSONSchemaConstraint(tokenizer, MOVE_SCHEMA)
        engine = Oracle(args.engine, args.nodes)
        resources.callback(engine.close)
        oracle = RewardOracle(engine, args.output, shared_cache=args.oracle_cache)
        resources.callback(oracle.close)
        source_names = [
            "evaluate_policy.py",
            "batched_policy.py",
            "prefix_policy.py",
            "rl_scoring.py",
            "rl_policy.py",
            "oracle.py",
            "oracle_cache.py",
            "board_encoder.py",
            "semantic_board_encoder.py",
            "corpus_access.py",
            "indexed_corpus.py",
            "rewards.py",
        ]
        arguments = {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in ["resume", "max_new_batches", "oracle_cache"]
        }
        arguments.update(samples=1, temperature=0.0, style="json", input="board")
        manifest = {
            "arguments": arguments,
            "source_hashes": {
                name: digest(Path(__file__).with_name(name)) for name in source_names
            },
            "model_hashes": {
                path.name: digest(path)
                for path in sorted(args.model.iterdir())
                if path.suffix in [".safetensors", ".json"]
            },
            "adapter_hash": digest(args.adapter / "adapters.safetensors")
            if args.adapter
            else None,
            "data_hash": corpus_signature(args.data, args.split),
            "data_format": data_format,
            "board_encoding": model.board_encoding,
            "engine": engine.identity,
            "oracle_namespace": oracle.namespace,
            "response_schema": MOVE_SCHEMA,
            "reward_config": REWARD_CONFIG,
            "model_runtime_source_hash": digest(
                Path(__file__).resolve().parents[2] / "mlx_lm/models/lfm3.py"
            ),
            "runtime_source_hash": digest(
                Path(__file__).resolve().parents[2] / "mlx_lm/structured.py"
            ),
            "decoding": "greedy; one committed response per position; no legality masks or repairs",
            "history": "FEN only; no repetition history",
            "scope": "Full requested split. Resumption replays only unfinished batches deterministically, checks their committed prefix, and never resamples based on move quality.",
        }
        if args.resume:
            previous = json.loads((args.output / "manifest.json").read_text())
            if previous != manifest:
                changes = [
                    key
                    for key in set(previous) | set(manifest)
                    if previous.get(key) != manifest.get(key)
                ]
                raise ValueError(
                    f"Evaluation protocol changed since its start: {changes}"
                )
        else:
            write_json(args.output / "manifest.json", manifest)
            sources = args.output / "source"
            sources.mkdir()
            for name in source_names:
                (sources / name).write_bytes(
                    Path(__file__).with_name(name).read_bytes()
                )
            for name in ["models/lfm3.py", "structured.py"]:
                (sources / Path(name).name).write_bytes(
                    (Path(__file__).resolve().parents[2] / "mlx_lm" / name).read_bytes()
                )
        counters = defaultdict(Counter)
        rewards = Counter()
        prefix = deque(maxlen=args.batch_size)
        completed = 0

        def accumulate(row):
            color = "white" if chess.Board(row["fen"]).turn else "black"
            for category in ["all", color, row["evaluation_phase"]]:
                for key in [
                    "ended",
                    "format_valid",
                    "legal",
                    "accepted",
                    "exact_match",
                ]:
                    counters[category][key] += int(row[key])
                counters[category]["samples"] += 1
            rewards[row["reward"]] += 1

        for result in read_committed(args.output / "responses.jsonl"):
            if completed >= args.positions:
                raise ValueError("Evaluation contains more responses than requested")
            expected = rows[completed]
            if (
                result["id"] != expected["id"]
                or result["fen"] != expected["fen"]
                or result["sample"] != 0
                or result["game_id"] != expected["game_id"]
                or result["evaluation_phase"] != record_phase(expected)
            ):
                raise ValueError(
                    "Committed evaluation records do not match the frozen position order"
                )
            accumulate(result)
            prefix.append((completed, result))
            completed += 1
        resumed_at = completed
        started = time.perf_counter()
        batches = 0
        fixed_policy = None

        def progress(status):
            write_json(
                args.output / "progress.json",
                {
                    "status": status,
                    "completed_positions": completed,
                    "total_positions": args.positions,
                    "resumed_at": resumed_at,
                    "seconds_this_invocation": time.perf_counter() - started,
                    "counts": dict(counters["all"]),
                    "oracle_statistics": oracle.statistics,
                    "shared_oracle_cache": str(args.oracle_cache.resolve())
                    if args.oracle_cache
                    else None,
                },
            )

        try:
            progress("running")
            with (args.output / "responses.jsonl").open("a", buffering=1) as output:
                first_batch = (completed // args.batch_size) * args.batch_size
                if completed == args.positions:
                    first_batch = args.positions
                for offset in range(first_batch, args.positions, args.batch_size):
                    cases = rows[offset : min(offset + args.batch_size, args.positions)]
                    prompts = [
                        board_prompt(tokenizer, row["fen"], model.board_encoding)
                        for row in cases
                    ]
                    if args.cache_fixed_prefix:
                        if fixed_policy is None:
                            fixed_policy = FixedPrefixPolicy(
                                model, tokenizer, prompts[0], constraint
                            )
                        responses = [fixed_policy.rollout(prompts[0], args.max_tokens)]
                    elif args.batch_size == 1:
                        responses = [
                            rollout(
                                model,
                                tokenizer,
                                prompts[0],
                                args.max_tokens,
                                0.0,
                                constraint,
                            )
                        ]
                    else:
                        responses = batch_rollout(
                            model,
                            tokenizer,
                            prompts,
                            constraint,
                            max_tokens=args.max_tokens,
                            temperature=0.0,
                        )
                    saved = dict(prefix)
                    for local, (case, response) in enumerate(zip(cases, responses)):
                        index = offset + local
                        if index < completed:
                            prior = saved[index]
                            if any(
                                prior[key] != response[key]
                                for key in ["tokens", "raw", "ended", "schema_valid"]
                            ):
                                raise ValueError(
                                    "Replayed batch changed a committed response"
                                )
                            continue
                        result = {
                            "id": case["id"],
                            "game_id": case["game_id"],
                            "fen": case["fen"],
                            "evaluation_phase": record_phase(case),
                            "sample": 0,
                            **response,
                            **oracle.score(case["fen"], response),
                        }
                        output.write(json.dumps(result) + "\n")
                        accumulate(result)
                        prefix.append((index, result))
                        completed += 1
                    batches += 1
                    progress("running")
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": args.positions,
                                "legal": counters["all"]["accepted"],
                                "exact": counters["all"]["exact_match"],
                                "seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
                    if (
                        args.max_new_batches is not None
                        and batches >= args.max_new_batches
                        and completed < args.positions
                    ):
                        progress("partial")
                        return
        except BaseException:
            progress("interrupted")
            raise
        finally:
            write_json(args.output / "oracle-statistics.json", oracle.statistics)
        report = {
            "positions": args.positions,
            "samples": completed,
            **{
                key: counters["all"][key]
                for key in ["ended", "format_valid", "legal", "accepted", "exact_match"]
            },
            "by_color": {
                color: {
                    "samples": counters[color]["samples"],
                    "legal": counters[color]["accepted"],
                    "exact_match": counters[color]["exact_match"],
                }
                for color in ["white", "black"]
            },
            "by_phase": {
                name: dict(counters[name])
                for name in ["opening", "middlegame", "endgame"]
            },
            "rewards": dict(rewards),
            "groups_with_reward_variation": 0,
            "complete": completed == args.positions,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
        }
        if not report["complete"]:
            raise AssertionError("Evaluation cannot finish with missing positions")
        write_json(args.output / "report.json", report)
        progress("complete")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
