import argparse
import functools
import hashlib
import io
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.pgn
import zstandard


SPLITS = {"train": 1, "valid": 2, "test": 4}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def hashed(value):
    return hashlib.sha256(value.encode()).digest()


def state_key(fen):
    return hashed(" ".join(fen.split()[:4]))


def game_split(identifier, seed):
    bucket = int.from_bytes(hashed(f"{seed}/split/{identifier}")[:8], "big") % 1000
    return "train" if bucket < 850 else "valid" if bucket < 900 else "test"


def phase(board, ply):
    return (
        "endgame"
        if chess.popcount(board.occupied) <= 10
        else "opening"
        if ply <= 20
        else "middlegame"
    )


class PositionVisitor(chess.pgn.BaseVisitor):
    def __init__(self, seed, per_bucket):
        self.seed = seed
        self.per_bucket = per_bucket
        self.headers = {}
        self.buckets = defaultdict(list)
        self.special = {}
        self.errors = []
        self.ply = -1
        self.identifier = None
        self.trajectory = hashlib.sha256()

    def visit_header(self, name, value):
        self.headers[name] = value

    def end_headers(self):
        match = re.fullmatch(
            r"https?://lichess.org/([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?/?",
            self.headers.get("Site", ""),
        )
        if (
            self.headers.get("Variant", "Standard") != "Standard"
            or "FEN" in self.headers
            or not match
        ):
            self.errors.append("unsupported_headers")
            return chess.pgn.SKIP
        self.identifier = "lichess:" + match.group(1)

    def begin_variation(self):
        return chess.pgn.SKIP

    def handle_error(self, error):
        self.errors.append(type(error).__name__)

    def rank(self):
        return hashed(f"{self.seed}/within-game/{self.identifier}/{self.ply}")[:8]

    def visit_board(self, board):
        self.ply += 1
        if self.ply < 4 or not self.identifier:
            return
        bucket = self.buckets[(phase(board, self.ply), bool(board.turn))]
        rank = self.rank()
        if len(bucket) < self.per_bucket or rank < bucket[-1][0]:
            bucket.append((rank, self.ply, board.fen()))
            bucket.sort(key=lambda item: item[0])
            del bucket[self.per_bucket :]

    def visit_move(self, board, move):
        if not move:
            self.errors.append("null_move")
        self.trajectory.update((move.uci() + " ").encode())
        if self.ply < 4:
            return
        name = (
            "promotion_available"
            if move.promotion
            else "en_passant_available"
            if board.is_en_passant(move)
            else "castling_available"
            if board.is_castling(move)
            else None
        )
        if name and (name not in self.special or self.rank() < self.special[name][0]):
            self.special[name] = (self.rank(), self.ply, board.fen())

    def result(self):
        if self.errors or not self.identifier:
            return {"rejected": True, "errors": self.errors}
        chosen = {
            ply: fen for bucket in self.buckets.values() for _, ply, fen in bucket
        }
        chosen.update({ply: fen for _, ply, fen in self.special.values()})
        trajectory_id = "trajectory:" + self.trajectory.hexdigest()
        rows = []
        for ply, fen in sorted(chosen.items()):
            board = chess.Board(fen)
            if not board.is_valid() or board.is_game_over():
                continue
            tags = []
            if board.is_check():
                tags.append("in_check")
            if any(board.generate_castling_moves()):
                tags.append("castling_available")
            if board.has_legal_en_passant():
                tags.append("en_passant_available")
            if any(
                move.promotion
                for move in board.generate_legal_moves(
                    from_mask=board.pawns
                    & (chess.BB_RANK_7 if board.turn else chess.BB_RANK_2)
                )
            ):
                tags.append("promotion_available")
            rows.append(
                {
                    "id": f"{self.identifier}-p{ply:04d}",
                    "game_id": trajectory_id,
                    "ply": ply,
                    "fen": fen,
                    "phase": phase(board, ply),
                    "color": "white" if board.turn else "black",
                    "tags": tags,
                }
            )
        return {
            "rejected": False,
            "game_id": trajectory_id,
            "rows": rows,
            "metadata": {
                "source_game_id": self.identifier,
                **{
                    key: self.headers.get(key)
                    for key in [
                        "UTCDate",
                        "Date",
                        "WhiteElo",
                        "BlackElo",
                        "TimeControl",
                        "ECO",
                    ]
                },
            },
        }


def database(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sources (sha TEXT PRIMARY KEY, path TEXT NOT NULL, cursor INTEGER NOT NULL, accepted INTEGER NOT NULL, rejected INTEGER NOT NULL, eof INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, split TEXT NOT NULL, source TEXT NOT NULL, source_index INTEGER NOT NULL, metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS positions (key BLOB PRIMARY KEY, mask INTEGER NOT NULL, split TEXT NOT NULL, game_id TEXT, ply INTEGER, fen TEXT, phase TEXT, color TEXT, tags TEXT, rank BLOB);
    """)
    return connection


def configure(connection, args):
    exclusions = {str(path.resolve()): digest(path) for path in args.exclude}
    config = {
        "version": "game_disjoint_position_corpus_v3",
        "game_sampling_stride": args.game_stride,
        "game_sampling": "deterministic hash sampling across source record indices; production scans complete archives",
        "game_grouping": "SHA256 of complete legal mainline UCI sequence; identical games share split and group",
        "seed": args.seed,
        "per_color_phase_per_game": args.per_bucket,
        "extra_special_positions_per_game": 3,
        "split_hash_buckets": {"train": 850, "valid": 50, "test": 100},
        "deduplication": "first four canonical FEN fields; every cross-split collision excluded",
        "history": "FEN only; repetition history omitted",
        "exclusions": exclusions,
        "builder_sha256": digest(Path(__file__)),
        "python": sys.version.split()[0],
        "python_chess": chess.__version__,
        "zstandard": zstandard.__version__,
    }
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='config'"
    ).fetchone()
    if existing:
        if json.loads(existing[0]) != config:
            raise ValueError(
                "Ingestion configuration or source changed; use a separate corpus registry"
            )
        return config
    with connection:
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("config", json.dumps(config, sort_keys=True)),
        )
        for path in args.exclude:
            with path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    connection.execute(
                        "INSERT OR IGNORE INTO positions (key,mask,split) VALUES (?,8,?)",
                        (state_key(chess.Board(row["fen"]).fen()), "excluded"),
                    )
    return config


def ingest(args):
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    connection = database(args.registry)
    configure(connection, args)
    started = time.monotonic()
    for source in args.sources:
        sha = digest(source)
        metadata_path = Path(str(source) + ".json")
        if source.suffix == ".zst":
            metadata = json.loads(metadata_path.read_text())
            if not metadata["complete"] or sha != metadata["expected_sha256"]:
                raise ValueError(
                    "Compressed source must pass its upstream checksum before ingestion"
                )
        prior = connection.execute(
            "SELECT cursor,accepted,rejected,eof FROM sources WHERE sha=?", (sha,)
        ).fetchone()
        cursor, accepted, rejected, eof = prior if prior else (0, 0, 0, 0)
        if eof or cursor >= args.games_per_source:
            print(
                "source already processed to requested boundary",
                source.name,
                cursor,
                flush=True,
            )
            continue
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO sources VALUES (?,?,0,0,0,0)",
                (sha, str(source.resolve())),
            )
        raw = source.open("rb")
        reader = (
            zstandard.ZstdDecompressor().stream_reader(raw)
            if source.suffix == ".zst"
            else raw
        )
        handle = io.TextIOWrapper(reader, encoding="utf-8")
        try:
            for _ in range(cursor):
                if not chess.pgn.skip_game(handle):
                    raise ValueError("Source ended before its committed cursor")
            factory = functools.partial(PositionVisitor, args.seed, args.per_bucket)
            while cursor < args.games_per_source:
                selected = (
                    int.from_bytes(
                        hashed(f"{args.seed}/source-sample/{sha}/{cursor}")[:8], "big"
                    )
                    % args.game_stride
                    == 0
                )
                game = (
                    chess.pgn.read_game(handle, Visitor=factory)
                    if selected
                    else chess.pgn.skip_game(handle)
                )
                if game is None or game is False:
                    eof = 1
                    break
                cursor += 1
                if not selected:
                    pass
                elif game["rejected"]:
                    rejected += 1
                else:
                    split = game_split(game["game_id"], args.seed)
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO games VALUES (?,?,?,?,?)",
                        (
                            game["game_id"],
                            split,
                            sha,
                            cursor,
                            json.dumps(game["metadata"]),
                        ),
                    ).rowcount
                    if inserted:
                        accepted += 1
                        for row in game["rows"]:
                            key = state_key(row["fen"])
                            connection.execute(
                                """INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?)
                                ON CONFLICT(key) DO UPDATE SET mask=positions.mask | excluded.mask,
                                split=CASE WHEN positions.mask=excluded.mask THEN positions.split ELSE 'excluded' END""",
                                (
                                    key,
                                    SPLITS[split],
                                    split,
                                    row["game_id"],
                                    row["ply"],
                                    row["fen"],
                                    row["phase"],
                                    row["color"],
                                    json.dumps(row["tags"]),
                                    hashed(f"{args.seed}/export/{key.hex()}")[:8],
                                ),
                            )
                if cursor % args.commit_every == 0:
                    connection.execute(
                        "UPDATE sources SET cursor=?,accepted=?,rejected=?,eof=? WHERE sha=?",
                        (cursor, accepted, rejected, eof, sha),
                    )
                    connection.commit()
                    print(
                        json.dumps(
                            {
                                "source": source.name,
                                "games_scanned": cursor,
                                "accepted_games": accepted,
                                "rejected_games": rejected,
                                "seconds_this_invocation": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
            connection.execute(
                "UPDATE sources SET cursor=?,accepted=?,rejected=?,eof=? WHERE sha=?",
                (cursor, accepted, rejected, eof, sha),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            handle.close()
            raw.close()
    connection.execute(
        "CREATE INDEX IF NOT EXISTS positions_split_category_rank ON positions(split,phase,color,rank)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS positions_split_rank ON positions(split,rank)"
    )
    connection.commit()
    report = {
        "sources": [
            dict(
                zip(
                    [
                        "sha256",
                        "path",
                        "games_scanned",
                        "accepted_games",
                        "rejected_games",
                        "eof",
                    ],
                    row,
                )
            )
            for row in connection.execute("SELECT * FROM sources")
        ],
        "available_positions": {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT split,count(*) FROM positions GROUP BY split"
            )
        },
        "available_games": {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT split,count(*) FROM games GROUP BY split"
            )
        },
        "seconds": time.monotonic() - started,
    }
    Path(str(args.registry) + ".ingest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    connection.close()


def export(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    connection = sqlite3.connect(f"file:{args.registry.resolve()}?mode=ro", uri=True)
    connection.execute("BEGIN")
    config = json.loads(
        connection.execute("SELECT value FROM metadata WHERE key='config'").fetchone()[
            0
        ]
    )
    targets = {"train": args.train, "valid": args.valid, "test": args.test}
    args.output.mkdir(parents=True)
    report = {
        "config": config,
        "exporter_sha256": digest(Path(__file__)),
        "evaluation_selection_order": ["endgame", "middlegame", "opening"],
        "registry": str(args.registry.resolve()),
        "sources": [
            dict(
                zip(
                    [
                        "sha256",
                        "path",
                        "games_scanned",
                        "accepted_games",
                        "rejected_games",
                        "eof",
                    ],
                    row,
                )
            )
            for row in connection.execute("SELECT * FROM sources")
        ],
        "targets": targets,
        "splits": {},
        "complete": False,
        "target_responses_present": False,
        "scope": "Initial corpus increment, not a claim of sufficient chess coverage. Test files are sealed for recipe selection; no model evaluation performed by the builder.",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    for split, target in targets.items():
        directory = args.output / split
        directory.mkdir()
        categories, tags, games = Counter(), Counter(), Counter()
        selected, shard, handle, index_handle, shard_count = 0, 0, None, None, 0
        shards = []
        fractions = (
            {"endgame": 0.3, "middlegame": 0.5, "opening": 0.2}
            if split != "train"
            else None
        )
        queries = (
            [(None, None, target)]
            if fractions is None
            else [
                (phase_name, color, int(target * fraction / 2))
                for phase_name, fraction in fractions.items()
                for color in ["white", "black"]
            ]
        )
        if sum(q[2] for q in queries) != target:
            raise ValueError(
                "Evaluation sizes must support exact 20/50/30 phase and equal-color strata"
            )
        try:
            for phase_name, color, quota in queries:
                parameters = [split]
                where = "split=?"
                if phase_name:
                    where += " AND phase=? AND color=?"
                    parameters += [phase_name, color]
                count = 0
                for (
                    game_id,
                    ply,
                    fen,
                    position_phase,
                    position_color,
                    position_tags,
                    rank,
                ) in connection.execute(
                    f"SELECT game_id,ply,fen,phase,color,tags,rank FROM positions WHERE {where} ORDER BY rank",
                    parameters,
                ):
                    if split != "train" and games[game_id] >= args.evaluation_per_game:
                        continue
                    if handle is None or shard_count == args.shard_size:
                        if handle:
                            handle.close()
                            index_handle.close()
                            shards[-1].update(
                                sha256=digest(directory / shards[-1]["file"]),
                                index_sha256=digest(directory / shards[-1]["index"]),
                                positions=shard_count,
                            )
                        name = f"{shard:06d}.jsonl"
                        handle = (directory / name).open("wb")
                        index_handle = (directory / (name + ".idx")).open("wb")
                        shards.append({"file": name, "index": name + ".idx"})
                        shard += 1
                        shard_count = 0
                    row = {
                        "id": f"{game_id}-p{ply:04d}",
                        "game_id": game_id,
                        "ply": ply,
                        "fen": fen,
                        "evaluation_phase": position_phase,
                        "color": position_color,
                        "coverage_tags": json.loads(position_tags),
                    }
                    index_handle.write(handle.tell().to_bytes(8, "little"))
                    handle.write(
                        (json.dumps(row, separators=(",", ":")) + "\n").encode()
                    )
                    selected += 1
                    count += 1
                    shard_count += 1
                    games[game_id] += 1
                    categories[f"{position_color}/{position_phase}"] += 1
                    tags.update(row["coverage_tags"])
                    if count == quota:
                        break
                if count != quota:
                    raise ValueError(
                        f"Insufficient {split}/{phase_name}/{color}: selected {count} of {quota}; ingest more games"
                    )
        finally:
            if handle:
                handle.close()
                index_handle.close()
                shards[-1].update(
                    sha256=digest(directory / shards[-1]["file"]),
                    index_sha256=digest(directory / shards[-1]["index"]),
                    positions=shard_count,
                )
        split_report = {
            "positions": selected,
            "source_games": len(games),
            "maximum_positions_per_game": max(games.values(), default=0),
            "categories": dict(categories),
            "coverage_tags": dict(tags),
            "shards": shards,
        }
        (directory / "manifest.json").write_text(
            json.dumps(split_report, indent=2) + "\n"
        )
        report["splits"][split] = split_report
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(split, selected, "positions", len(games), "games", flush=True)
    report["complete"] = True
    report["exporter_sha256"] = digest(Path(__file__))
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    connection.close()


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("ingest")
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--sources", type=Path, nargs="+", required=True)
    build.add_argument("--exclude", type=Path, nargs="*", default=[])
    build.add_argument("--seed", type=int, default=2026103101)
    build.add_argument("--per-bucket", type=int, default=2)
    build.add_argument("--games-per-source", type=int, default=9223372036854775807)
    build.add_argument("--game-stride", type=int, default=16)
    build.add_argument("--commit-every", type=int, default=1000)
    freeze = actions.add_parser("export")
    freeze.add_argument("--registry", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--train", type=int, default=2000000)
    freeze.add_argument("--valid", type=int, default=50000)
    freeze.add_argument("--test", type=int, default=100000)
    freeze.add_argument("--evaluation-per-game", type=int, default=2)
    freeze.add_argument("--shard-size", type=int, default=100000)
    args = parser.parse_args()
    if args.action == "ingest":
        if (
            min(
                args.per_bucket,
                args.games_per_source,
                args.commit_every,
                args.game_stride,
            )
            < 1
        ):
            parser.error("All ingestion budgets must be positive")
        ingest(args)
    else:
        if (
            min(
                args.train,
                args.valid,
                args.test,
                args.evaluation_per_game,
                args.shard_size,
            )
            < 1
        ):
            parser.error("All export budgets must be positive")
        export(args)


if __name__ == "__main__":
    main()
