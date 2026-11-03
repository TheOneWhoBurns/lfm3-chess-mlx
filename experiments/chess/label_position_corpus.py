import argparse
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import chess

from indexed_corpus import IndexedCorpus
from oracle import Oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--nodes", type=int, default=40000)
    args = parser.parse_args()
    if args.workers < 1 or args.nodes < 1:
        parser.error("Positive worker count and node budget required")
    if args.report.exists():
        raise FileExistsError(args.report)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    oracle = Oracle(args.engine, args.nodes)
    identity = oracle.identity
    oracle.close()
    identity["oracle_source_sha256"] = hashlib.sha256(
        Path(__file__).with_name("oracle.py").read_bytes()
    ).hexdigest()
    engine_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    corpus_key = hashlib.sha256(
        (args.corpus / "manifest.json").read_bytes()
    ).hexdigest()
    database = sqlite3.connect(args.database)
    database.execute("PRAGMA journal_mode=WAL")
    database.executescript(
        "CREATE TABLE IF NOT EXISTS engines (id TEXT PRIMARY KEY,metadata TEXT); CREATE TABLE IF NOT EXISTS labels (engine TEXT,fen TEXT,analysis TEXT,PRIMARY KEY(engine,fen)); CREATE TABLE IF NOT EXISTS provenance (corpus TEXT,split TEXT,row_id TEXT,engine TEXT,fen TEXT,PRIMARY KEY(corpus,split,row_id,engine));"
    )
    database.execute(
        "INSERT OR IGNORE INTO engines VALUES (?,?)",
        (engine_key, json.dumps(identity, sort_keys=True)),
    )
    database.commit()
    engines, lock, local = [], threading.Lock(), threading.local()

    def initialize():
        local.oracle = Oracle(args.engine, args.nodes)
        with lock:
            engines.append(local.oracle)

    def label(row):
        board = chess.Board(row["fen"])
        result = local.oracle.analyse(board, multipv=1)[0]
        if chess.Move.from_uci(result["move"]) not in board.legal_moves:
            raise ValueError("Oracle returned an illegal target")
        return row, result

    count, cached, produced = 0, 0, 0
    started = time.monotonic()
    try:
        with (
            IndexedCorpus(args.corpus, args.split) as corpus,
            ThreadPoolExecutor(
                max_workers=args.workers, initializer=initialize
            ) as executor,
        ):
            pending = set()
            iterator = iter(range(len(corpus)))
            exhausted = False
            while pending or not exhausted:
                while len(pending) < 2 * args.workers and not exhausted:
                    index = next(iterator, None)
                    if index is None:
                        exhausted = True
                        break
                    row = corpus[index]
                    known = database.execute(
                        "SELECT 1 FROM labels WHERE engine=? AND fen=?",
                        (engine_key, row["fen"]),
                    ).fetchone()
                    if known:
                        database.execute(
                            "INSERT OR IGNORE INTO provenance VALUES (?,?,?,?,?)",
                            (corpus_key, args.split, row["id"], engine_key, row["fen"]),
                        )
                        cached += 1
                        count += 1
                        if count % 100 == 0:
                            database.commit()
                    else:
                        pending.add(executor.submit(label, row))
                if pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        row, result = future.result()
                        database.execute(
                            "INSERT OR IGNORE INTO labels VALUES (?,?,?)",
                            (engine_key, row["fen"], json.dumps(result)),
                        )
                        database.execute(
                            "INSERT OR IGNORE INTO provenance VALUES (?,?,?,?,?)",
                            (corpus_key, args.split, row["id"], engine_key, row["fen"]),
                        )
                        produced += 1
                        count += 1
                    database.commit()
                    if count % 100 < len(completed) or exhausted:
                        print(
                            json.dumps(
                                {
                                    "processed": count,
                                    "total": len(corpus),
                                    "cached": cached,
                                    "new_labels": produced,
                                    "seconds": time.monotonic() - started,
                                }
                            ),
                            flush=True,
                        )
            database.commit()
            coverage = database.execute(
                "SELECT count(*) FROM provenance WHERE corpus=? AND split=? AND engine=?",
                (corpus_key, args.split, engine_key),
            ).fetchone()[0]
            if coverage != len(corpus):
                raise ValueError(
                    "Oracle cache does not cover the entire requested split"
                )
    finally:
        for engine in engines:
            engine.close()
        database.close()
    report = {
        "complete": True,
        "corpus_sha256": corpus_key,
        "split": args.split,
        "positions": count,
        "cached": cached,
        "new_labels": produced,
        "engine_key": engine_key,
        "engine": identity,
        "workers": args.workers,
        "seconds": time.monotonic() - started,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Offline reward/evaluation references only. No model training, no inference assistance, and no target responses added to model-input records.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
