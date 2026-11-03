import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import chess

from build_position_corpus import digest, phase, state_key
from indexed_corpus import IndexedCorpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Cannot audit an incomplete corpus export")
    connection = sqlite3.connect(args.output / "audit.sqlite")
    connection.executescript(
        "CREATE TABLE states (key BLOB PRIMARY KEY, split TEXT); CREATE TABLE games (id TEXT PRIMARY KEY, split TEXT, n INTEGER);"
    )
    result = {}
    allowed = {
        "id",
        "game_id",
        "ply",
        "fen",
        "evaluation_phase",
        "color",
        "coverage_tags",
    }
    for split in ["train", "valid", "test"]:
        info = manifest["splits"][split]
        positions, categories, tags = 0, Counter(), Counter()
        sample_indices = set(
            random.Random(2026103102).sample(
                range(info["positions"]), min(1000, info["positions"])
            )
        )
        reference = {}
        for shard in info["shards"]:
            path = args.corpus / split / shard["file"]
            index = args.corpus / split / shard["index"]
            if (
                digest(path) != shard["sha256"]
                or digest(index) != shard["index_sha256"]
            ):
                raise ValueError(f"Shard checksum failed: {path}")
            offsets = index.read_bytes()
            if len(offsets) != shard["positions"] * 8:
                raise ValueError("Index length does not match the shard")
            count = 0
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if offset != int.from_bytes(
                        offsets[count * 8 : (count + 1) * 8], "little"
                    ):
                        raise ValueError(
                            "Random-access index points to the wrong record"
                        )
                    row = json.loads(line)
                    if set(row) != allowed:
                        raise ValueError(
                            "Unexpected input fields or target responses in exported data"
                        )
                    board = chess.Board(row["fen"])
                    if not board.is_valid() or board.is_game_over():
                        raise ValueError(
                            "Export contains invalid or terminal FEN state"
                        )
                    color = "white" if board.turn else "black"
                    if (
                        color != row["color"]
                        or phase(board, row["ply"]) != row["evaluation_phase"]
                    ):
                        raise ValueError("Coverage labels contradict board state")
                    expected_tags = []
                    if board.is_check():
                        expected_tags.append("in_check")
                    if any(board.generate_castling_moves()):
                        expected_tags.append("castling_available")
                    if board.has_legal_en_passant():
                        expected_tags.append("en_passant_available")
                    if any(
                        move.promotion
                        for move in board.generate_legal_moves(
                            from_mask=board.pawns
                            & (chess.BB_RANK_7 if board.turn else chess.BB_RANK_2)
                        )
                    ):
                        expected_tags.append("promotion_available")
                    if row["coverage_tags"] != expected_tags:
                        raise ValueError("Rule-coverage tags contradict the board")
                    try:
                        connection.execute(
                            "INSERT INTO states VALUES (?,?)",
                            (state_key(row["fen"]), split),
                        )
                    except sqlite3.IntegrityError:
                        raise ValueError(
                            "Duplicate or cross-split board state"
                        ) from None
                    connection.execute(
                        "INSERT INTO games VALUES (?,?,1) ON CONFLICT(id) DO UPDATE SET split=CASE WHEN games.split=excluded.split THEN games.split ELSE 'LEAK' END,n=games.n+1",
                        (row["game_id"], split),
                    )
                    categories[f"{color}/{row['evaluation_phase']}"] += 1
                    tags.update(row["coverage_tags"])
                    if positions in sample_indices:
                        reference[positions] = row
                    count += 1
                    positions += 1
            if count != shard["positions"]:
                raise ValueError("Wrong shard position count")
            connection.commit()
        with IndexedCorpus(args.corpus, split) as indexed:
            indices = list(reference)
            random.Random(2026103103).shuffle(indices)
            for index in indices:
                if indexed[index] != reference[index]:
                    raise ValueError(
                        "Indexed random access does not reproduce sequential records"
                    )
        game_count, maximum = connection.execute(
            "SELECT count(*),max(n) FROM games WHERE split=?", (split,)
        ).fetchone()
        if (
            positions != info["positions"]
            or positions != manifest["targets"][split]
            or game_count != info["source_games"]
            or maximum != info["maximum_positions_per_game"]
        ):
            raise ValueError("Manifest counts disagree with the exported records")
        if (
            dict(categories) != info["categories"]
            or dict(tags) != info["coverage_tags"]
        ):
            raise ValueError("Coverage totals disagree with the exported records")
        if split != "train" and maximum > 2:
            raise ValueError(
                "Evaluation exceeds two positions per independent game trajectory"
            )
        result[split] = {
            "positions": positions,
            "game_trajectories": game_count,
            "maximum_positions_per_game": maximum,
            "categories": dict(categories),
            "coverage_tags": dict(tags),
        }
        print(split, positions, "validated", flush=True)
    if connection.execute("SELECT count(*) FROM games WHERE split='LEAK'").fetchone()[
        0
    ]:
        raise ValueError("A game trajectory crosses dataset splits")
    for path, expected in manifest["config"]["exclusions"].items():
        path = Path(path)
        if digest(path) != expected:
            raise ValueError("Prior-data exclusion file changed")
        with path.open() as handle:
            for line in handle:
                key = state_key(chess.Board(json.loads(line)["fen"]).fen())
                if connection.execute(
                    "SELECT 1 FROM states WHERE key=?", (key,)
                ).fetchone():
                    raise ValueError("Previously used board leaked into the new corpus")
    report = {
        "passed": True,
        "corpus_manifest_sha256": digest(args.corpus / "manifest.json"),
        "splits": result,
        "position_overlap": 0,
        "trajectory_overlap": 0,
        "previous_position_overlap": 0,
        "targets_in_input": False,
        "source_sha256": digest(Path(__file__)),
        "index_reader_sha256": digest(Path(__file__).with_name("indexed_corpus.py")),
        "indexed_random_access_verified": True,
        "scope": "Dataset system audit only; includes sealed-split hygiene but never queries a model or reveals test labels.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    connection.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
