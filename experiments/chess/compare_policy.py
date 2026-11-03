import argparse
import json
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np

from rl_policy import digest
from corpus_access import corpus_signature, load_positions


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(rows):
    return {
        "positions": len(rows),
        "legal": sum(row["accepted"] for row in rows),
        "exact_match": sum(row["exact_match"] for row in rows),
        "format_valid": sum(row["format_valid"] for row in rows),
        "mean_oracle_loss_invalid_penalty_1": float(
            np.mean([1.0 - row["quality"] if row["accepted"] else 1.0 for row in rows])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifests = [
        json.loads((path / "manifest.json").read_text())
        for path in [args.before, args.after]
    ]
    for key in [
        "data_hash",
        "data_format",
        "model_hashes",
        "engine",
        "oracle_namespace",
        "response_schema",
        "board_encoding",
        "runtime_source_hash",
        "model_runtime_source_hash",
        "source_hashes",
        "reward_config",
    ]:
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"Comparisons require matching {key}")
    for key in [
        "input",
        "style",
        "positions",
        "samples",
        "temperature",
        "max_tokens",
        "seed",
        "encoder_seed",
        "batch_size",
        "cache_fixed_prefix",
        "split",
    ]:
        if manifests[0]["arguments"].get(key) != manifests[1]["arguments"].get(key):
            raise ValueError(f"Comparisons require matching decoding argument {key}")
    if manifests[0]["arguments"]["samples"] != 1:
        raise ValueError("This comparator requires one attempt per position")
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 cluster-bootstrap draws")
    results = []
    for path in [args.before, args.after]:
        if not (path / "report.json").exists():
            raise ValueError(f"Evaluation has not completed: {path}")
        rows = read_rows(path / "responses.jsonl")
        if len(rows) != manifests[0]["arguments"]["positions"] or len(
            {row["id"] for row in rows}
        ) != len(rows):
            raise ValueError("Evaluation contains missing or duplicated positions")
        results.append({row["id"]: row for row in rows})
    if set(results[0]) != set(results[1]):
        raise ValueError("The two evaluations cover different positions")
    data_args = manifests[0]["arguments"]
    source = Path(data_args["data"])
    split = data_args.get("split", "valid")
    if corpus_signature(source, split) != manifests[0]["data_hash"]:
        raise ValueError("Source data changed after evaluation")
    rows, _ = load_positions(source, split, data_args["seed"])
    cases = {row["id"]: row for row in rows}
    if hasattr(rows, "close"):
        rows.close()
    grouped, games = defaultdict(list), defaultdict(list)
    paired = []
    for identifier in results[0]:
        first, second = results[0][identifier], results[1][identifier]
        if first["fen"] != second["fen"] or first["fen"] != cases[identifier]["fen"]:
            raise ValueError("Position state changed between evaluations")
        board = chess.Board(first["fen"])
        phase = cases[identifier].get("evaluation_phase") or (
            "endgame"
            if len(board.piece_map()) <= 10
            else "opening"
            if cases[identifier]["ply"] <= 20
            else "middlegame"
        )
        color = "white" if board.turn else "black"
        pair = {
            "id": identifier,
            "game_id": cases[identifier]["game_id"],
            "phase": phase,
            "color": color,
            "before": first,
            "after": second,
        }
        paired.append(pair)
        for category in ["all", color, phase]:
            grouped[category].append(pair)
        games[pair["game_id"]].append(pair)
    breakdown = {
        key: {
            side: summarize([pair[side] for pair in pairs])
            for side in ["before", "after"]
        }
        for key, pairs in grouped.items()
    }
    game_groups = list(games.values())
    counts = np.array([len(group) for group in game_groups])
    differences = {}
    for label, field in [("legality", "accepted"), ("exact_agreement", "exact_match")]:
        changes = np.array(
            [
                sum(
                    int(pair["after"][field]) - int(pair["before"][field])
                    for pair in group
                )
                for group in game_groups
            ]
        )
        rng = np.random.default_rng(2026103113)
        estimates = np.empty(args.bootstrap)
        for start in range(0, args.bootstrap, 64):
            end = min(start + 64, args.bootstrap)
            draws = rng.integers(
                0, len(game_groups), size=(end - start, len(game_groups))
            )
            estimates[start:end] = changes[draws].sum(axis=1) / counts[draws].sum(
                axis=1
            )
        differences[label] = {
            "rate_change": float(changes.sum() / counts.sum()),
            "game_cluster_bootstrap_95_percent_interval": np.quantile(
                estimates, [0.025, 0.975]
            ).tolist(),
            "improved_positions": sum(
                pair["after"][field] and not pair["before"][field] for pair in paired
            ),
            "regressed_positions": sum(
                pair["before"][field] and not pair["after"][field] for pair in paired
            ),
        }
    report = {
        "before": str(args.before.resolve()),
        "after": str(args.after.resolve()),
        "before_adapter_hash": manifests[0]["adapter_hash"],
        "after_adapter_hash": manifests[1]["adapter_hash"],
        "source_hash": digest(Path(__file__)),
        "data_hash": manifests[0]["data_hash"],
        "source_games": len(games),
        "bootstrap_draws": args.bootstrap,
        "breakdown": breakdown,
        "paired_differences": differences,
        "scope": "Paired single-attempt comparison; bootstrap resamples source games. Development results are not the reserved final evaluation.",
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
