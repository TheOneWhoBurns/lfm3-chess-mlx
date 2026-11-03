import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import chess
import chess.engine


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def position_key(board):
    return " ".join(board.fen().split()[:4])


def move_prompt(fen, representation="fen"):
    instruction = (
        "Choose the best chess move for the side to move in this FEN. "
        "Return only one move in UCI notation, such as e2e4 or a7a8q.\nFEN: " + fen
    )
    if representation == "fen":
        return instruction
    if representation != "pieces":
        raise ValueError(f"Unknown board representation: {representation}")
    return (
        "Choose the best chess move for the side to move. "
        "Return only one move in UCI notation, such as e2e4 or a7a8q.\n"
        + board_text(chess.Board(fen))
    )


def board_text(board):
    lines = ["All pieces are listed below. Any unlisted square is empty."]
    for color, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
        pieces = [
            f"{chess.piece_name(piece.piece_type)} {chess.square_name(square)}"
            for square, piece in sorted(board.piece_map().items())
            if piece.color == color
        ]
        lines.append(name + ": " + ", ".join(pieces))
    lines.append("Side to move: " + ("White" if board.turn else "Black"))
    rights = [
        f"{name} {side}"
        for color, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]
        for side, allowed in [
            ("kingside", board.has_kingside_castling_rights(color)),
            ("queenside", board.has_queenside_castling_rights(color)),
        ]
        if allowed
    ]
    lines.append("Castling rights: " + (", ".join(rights) if rights else "none"))
    lines.append(
        "En-passant target: "
        + (
            chess.square_name(board.ep_square)
            if board.ep_square is not None
            else "none"
        )
    )
    lines.append(
        f"Halfmove clock: {board.halfmove_clock}; fullmove number: {board.fullmove_number}"
    )
    return "\n".join(lines)


class Oracle:
    def __init__(self, path, nodes):
        self.path = str(Path(path).resolve())
        self.nodes = nodes
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self.engine.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})
        self.identity = {
            "id": self.engine.id,
            "path": self.path,
            "sha256": hashlib.sha256(Path(self.path).read_bytes()).hexdigest(),
            "nodes": nodes,
            "threads": 1,
            "hash_mb": 64,
        }

    def analyse(self, board, multipv=1, root_moves=None):
        if not board.is_valid() or board.is_game_over():
            raise ValueError("Oracle requires a valid nonterminal position")
        board = chess.Board(board.fen())
        self.engine.configure({"Clear Hash": None})
        infos = self.engine.analyse(
            board,
            chess.engine.Limit(nodes=self.nodes),
            multipv=multipv,
            root_moves=root_moves,
            game=object(),
        )
        result = []
        for info in infos:
            score = info["score"].pov(board.turn)
            wdl = info["wdl"].pov(board.turn)
            result.append(
                {
                    "move": info["pv"][0].uci(),
                    "cp": score.score(),
                    "mate": score.mate(),
                    "wdl": list(wdl),
                    "expected_score": wdl.expectation(),
                    "pv": [move.uci() for move in info["pv"]],
                    "depth": info.get("depth"),
                    "nodes": info.get("nodes"),
                }
            )
        return result

    def close(self):
        self.engine.quit()


def prepare(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to replace frozen data: {out}")
    out.mkdir(parents=True)
    rng = random.Random(args.seed)
    oracle = Oracle(args.engine, args.nodes)
    seen = set()
    rows = {"train": [], "valid": [], "test": []}
    try:
        for game_id in range(args.games):
            split = (
                "test" if game_id % 8 == 0 else "valid" if game_id % 8 == 1 else "train"
            )
            board = chess.Board()
            for ply in range(args.plies):
                if board.is_game_over():
                    break
                candidates = oracle.analyse(board, multipv=3)
                key = position_key(board)
                if ply >= 8 and (ply - 8) % 3 == 0 and key not in seen:
                    seen.add(key)
                    rows[split].append(
                        {
                            "id": f"g{game_id:03d}-p{ply:03d}",
                            "game_id": game_id,
                            "ply": ply,
                            "fen": board.fen(),
                            "position_key": key,
                            "oracle": candidates,
                            "prompt": move_prompt(board.fen()),
                            "completion": candidates[0]["move"],
                        }
                    )
                if rng.random() < 0.15:
                    chosen = rng.choice(
                        sorted(board.legal_moves, key=lambda m: m.uci())
                    )
                else:
                    chosen = chess.Move.from_uci(rng.choice(candidates)["move"])
                board.push(chosen)
            print(
                f"game {game_id + 1}/{args.games}: "
                + str({k: len(v) for k, v in rows.items()}),
                flush=True,
            )
        for split, items in rows.items():
            if not items:
                raise ValueError(f"Empty split: {split}")
            rng.shuffle(items)
            write_rows(out / f"{split}.jsonl", items)
        metadata = {
            "engine": oracle.identity,
            "seed": args.seed,
            "games": args.games,
            "plies": args.plies,
            "counts": {k: len(v) for k, v in rows.items()},
            "position_only": True,
            "split_rule": "source game; all duplicate board/turn/castling/en-passant keys removed globally",
            "sha256": {
                k: hashlib.sha256((out / f"{k}.jsonl").read_bytes()).hexdigest()
                for k in rows
            },
        }
        (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        oracle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=shutil.which("stockfish"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--plies", type=int, default=48)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if not args.engine:
        parser.error("Stockfish was not found; pass --engine")
    if args.games < 8 or args.plies < 12 or args.nodes < 1:
        parser.error("Need at least 8 games, 12 plies, and a positive node budget")
    prepare(args)


if __name__ == "__main__":
    main()
