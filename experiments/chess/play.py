import argparse
import json
from pathlib import Path

import chess
import chess.pgn
import mlx.core as mx

from mlx_lm import JSONSchemaConstraint, StructuredOutputError, generate, load
from mlx_lm.sample_utils import make_sampler
from oracle import board_text
from rl_policy import MOVE_RESPONSE_FORMAT, MOVE_SCHEMA, prompt_tokens, rollout
from board_encoder import board_prompt, load_board_policy


def respond(model, tokenizer, prompt, max_tokens, thinking=False):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    raw = generate(
        model,
        tokenizer,
        prompt=rendered,
        sampler=make_sampler(temp=0),
        max_tokens=max_tokens,
    )
    final = (
        raw.rsplit("</think>", 1)[-1].strip()
        if "</think>" in raw
        else ""
        if thinking
        else raw.strip()
    )
    return raw, final


def explain(model, tokenizer, board, move, max_tokens):
    prompt = (
        board_text(board)
        + f"\nExplain the move {move.uci()}. Return only JSON with reason, capture (piece name or none), gives_check (boolean), and line (1 to 4 UCI moves starting with this move)."
    )
    raw, final = respond(model, tokenizer, prompt, max_tokens)
    try:
        parsed = json.loads(final)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("reason"), str):
        print("The model did not return a usable explanation.")
        return {"raw": raw, "final": final, "structured": False}
    print("Model explanation:", parsed["reason"])
    line = parsed.get("line")
    legal_line = (
        isinstance(line, list) and 1 <= len(line) <= 4 and line[0] == move.uci()
    )
    if legal_line:
        replay = board.copy()
        try:
            for text in line:
                if not isinstance(text, str):
                    legal_line = False
                    break
                replay.push_uci(text)
        except ValueError:
            legal_line = False
    if legal_line:
        print("Proposed continuation:", " ".join(line))
    elif line:
        print(
            "Its proposed continuation contains an invalid move or does not begin with its chosen move."
        )
    return {"raw": raw, "final": final, "structured": True, "legal_line": legal_line}


def main():
    parser = argparse.ArgumentParser(
        description="Play against the local LFM3 policy without engine assistance or move repair."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--human", choices=["white", "black", "none"], default="white")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--one-turn", action="store_true")
    parser.add_argument(
        "--explanation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--record", type=Path)
    parser.add_argument("--input", choices=["board", "text"], default="board")
    parser.add_argument("--encoder", choices=["compact", "semantic"])
    args = parser.parse_args()
    board = chess.Board(args.fen)
    if not board.is_valid():
        parser.error("The starting position is not a valid standard chess position")
    if args.record and args.record.exists():
        raise FileExistsError(args.record)
    if args.record:
        args.record.mkdir(parents=True)
        (args.record / "run.json").write_text(
            json.dumps(
                {
                    "model": str(args.model.resolve()),
                    "adapter": str(args.adapter.resolve()) if args.adapter else None,
                    "fen": args.fen,
                    "max_tokens": args.max_tokens,
                    "engine_assistance": False,
                    "move_repair": False,
                    "response_format": MOVE_RESPONSE_FORMAT,
                    "board_encoding": None,
                    "requested_encoder": args.encoder,
                },
                indent=2,
            )
            + "\n"
        )
    if board.is_game_over():
        print(board)
        print("Game over:", board.outcome().result())
        if args.record:
            (args.record / "game.pgn").write_text(
                str(chess.pgn.Game.from_board(board)) + "\n"
            )
        return
    if args.input == "board":
        encoder = args.encoder or (None if args.adapter else "semantic")
        model, tokenizer = load_board_policy(args.model, args.adapter, encoder=encoder)
    else:
        model, tokenizer = load(
            str(args.model),
            adapter_path=str(args.adapter) if args.adapter else None,
            tokenizer_config={"trust_remote_code": False},
        )
        model.set_dtype(mx.float32)
    model.eval()
    constraint = JSONSchemaConstraint(tokenizer, MOVE_SCHEMA)
    if args.record and args.input == "board":
        record_path = args.record / "run.json"
        metadata = json.loads(record_path.read_text())
        metadata["board_encoding"] = model.board_encoding
        record_path.write_text(json.dumps(metadata, indent=2) + "\n")
    human_color = {"white": chess.WHITE, "black": chess.BLACK, "none": None}[args.human]
    try:
        while not board.is_game_over():
            print("\n", board, sep="")
            print("White to move" if board.turn else "Black to move")
            if board.turn == human_color:
                text = input("Your move (SAN or UCI; quit to stop): ").strip()
                if text.lower() in ["quit", "exit"]:
                    break
                try:
                    move = board.parse_san(text)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("That move is not legal in this position.")
                    continue
                board.push(move)
            else:
                before = board.fen()
                raw, final, decoding_error = "", "", None
                response = {"ended": False, "schema_valid": False}
                try:
                    encoded = (
                        board_prompt(tokenizer, before, model.board_encoding)
                        if args.input == "board"
                        else None
                    )
                    tokens = (
                        encoded.tokens if encoded else prompt_tokens(tokenizer, before)
                    )
                    response = rollout(
                        model,
                        tokenizer,
                        encoded if encoded else tokens,
                        max_tokens=args.max_tokens,
                        temperature=0.0,
                        constraint=constraint,
                    )
                    raw = response["raw"]
                    final = json.loads(raw)["move"]
                    move = chess.Move.from_uci(final)
                except (StructuredOutputError, ValueError) as error:
                    move = None
                    decoding_error = str(error)
                legal = move is not None and move in board.legal_moves
                accepted = legal and response["ended"] and response["schema_valid"]
                event = {
                    "fen": before,
                    **response,
                    "raw": raw,
                    "answer": final,
                    "legal": legal,
                    "accepted": accepted,
                }
                if decoding_error:
                    event["decoding_error"] = decoding_error
                if accepted:
                    print("Model move:", board.san(move), f"({move.uci()})")
                    if args.explanation:
                        event["explanation"] = explain(
                            model, tokenizer, board, move, args.max_tokens
                        )
                    board.push(move)
                else:
                    print(
                        "The model failed to finish a legal move response:",
                        repr(final)
                        if final
                        else "no final answer within the token budget",
                    )
                    print("Stopping without substituting a move.")
                if args.record:
                    with (args.record / "events.jsonl").open("a") as file:
                        file.write(json.dumps(event) + "\n")
                if args.one_turn or not accepted:
                    break
        if board.is_game_over():
            print(board)
            print("Game over:", board.outcome().result())
    except (EOFError, KeyboardInterrupt):
        print("\nStopped.")
    finally:
        if args.record:
            game = chess.pgn.Game.from_board(board)
            (args.record / "game.pgn").write_text(str(game) + "\n")


if __name__ == "__main__":
    main()
