import hashlib
import json
import re
from pathlib import Path

import chess
import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.structured import StructuredOutputError
from oracle import board_text
from board_encoder import BoardPrompt
from rewards import REWARD_CONFIG, legality_progress

MOVE_SCHEMA = {
    "type": "object",
    "properties": {
        "move": {"type": "string", "pattern": "^[a-h][1-8][a-h][1-8][qrbn]?$"}
    },
    "required": ["move"],
    "additionalProperties": False,
}
MOVE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "chess_move", "strict": True, "schema": MOVE_SCHEMA},
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prompt_tokens(tokenizer, fen, style="json"):
    if style == "fen_completion":
        return tokenizer.encode(
            tokenizer.bos_token + "Chess position (FEN): " + fen + "\nBest move (UCI):",
            add_special_tokens=False,
        )
    question = (
        "Choose the best next chess move for the side to move. "
        "Respond with exactly one UCI move and nothing else.\n"
        + board_text(chess.Board(fen))
    )
    if style == "json":
        question = (
            "Choose the best next chess move for the side to move. "
            'Return only a JSON object with one key, "move", whose value is one UCI move.\n'
            + board_text(chess.Board(fen))
        )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if style == "move_prefix":
        prompt += "Move: "
    elif style == "uci_prefix":
        prompt += "UCI move:"
    elif style not in ["chat", "json"]:
        raise ValueError(style)
    return tokenizer.encode(prompt, add_special_tokens=False)


def rollout(model, tokenizer, prompt, max_tokens=48, temperature=1.0, constraint=None):
    prompt_ids = prompt.tokens if isinstance(prompt, BoardPrompt) else prompt
    embeddings = (
        model.embed_prompt(mx.array([prompt_ids]), prompt)[0]
        if isinstance(prompt, BoardPrompt)
        else None
    )
    tokens, logprobs = [], []
    ended = False
    generator = generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
        logits_processors=[constraint.processor(len(prompt_ids))]
        if constraint
        else None,
        input_embeddings=embeddings,
    )
    try:
        for token, probabilities in generator:
            tokens.append(token)
            logprobs.append(float(probabilities[token].item()))
            if token in tokenizer.eos_token_ids:
                ended = True
                break
    finally:
        generator.close()
    text_tokens = tokens[:-1] if ended else tokens
    raw = tokenizer.decode(text_tokens)
    schema_valid = None
    if constraint is not None:
        try:
            constraint.validate(raw)
            schema_valid = True
        except StructuredOutputError:
            schema_valid = False
    return {
        "tokens": tokens,
        "old_logprobs": logprobs,
        "raw": raw,
        "ended": ended,
        "schema_valid": schema_valid,
        "response_format": "json_schema" if constraint else "text",
    }


def extract_answer(response):
    answer = response["raw"].strip()
    if response.get("response_format") == "json_schema":
        try:
            value = json.loads(answer)
            answer = (
                value["move"]
                if isinstance(value, dict) and set(value) == {"move"}
                else ""
            )
            if not isinstance(answer, str):
                answer = ""
        except (ValueError, KeyError):
            answer = ""
    format_valid = bool(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", answer))
    return answer, format_valid


def grade(fen, response):
    answer, format_valid = extract_answer(response)
    legal = False
    if format_valid:
        try:
            legal = chess.Move.from_uci(answer) in chess.Board(fen).legal_moves
        except ValueError:
            pass
    return {
        "answer": answer,
        "format_valid": format_valid,
        "legal": bool(legal),
        "accepted": bool(legal and response["ended"]),
    }


class RewardOracle:
    def __init__(self, oracle, directory, shaping_weight=1.0):
        self.oracle = oracle
        self.directory = directory
        self.cache = {}
        if not 0 <= shaping_weight <= 1:
            raise ValueError("Legality shaping weight must be between zero and one")
        self.shaping_weight = shaping_weight
        self.path = directory / "oracle-cache.jsonl"
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["analysis"]

    def analyse(self, fen, move=None):
        key = fen + "|" + (move or "best")
        if key not in self.cache:
            result = self.oracle.analyse(
                chess.Board(fen),
                root_moves=[chess.Move.from_uci(move)] if move else None,
            )[0]
            self.cache[key] = result
            with self.path.open("a") as handle:
                handle.write(json.dumps({"key": key, "analysis": result}) + "\n")
        return self.cache[key]

    def score(self, fen, response):
        answer, format_valid = extract_answer(response)
        result = {
            "answer": answer,
            "format_valid": format_valid,
            "legal": False,
            "accepted": False,
            "reward": REWARD_CONFIG["malformed"],
            "base_reward": REWARD_CONFIG["malformed"],
            "match_bonus": 0.0,
            "exact_match": False,
            "quality": None,
        }
        if not response["ended"] or not result["format_valid"]:
            return result
        best = self.analyse(fen)
        exact = result["answer"] == best["move"]
        result["teacher_move"] = best["move"]
        if exact:
            result.update(
                {
                    "reward": REWARD_CONFIG["exact_base"]
                    + REWARD_CONFIG["exact_bonus"],
                    "base_reward": REWARD_CONFIG["exact_base"],
                    "match_bonus": REWARD_CONFIG["exact_bonus"],
                    "exact_match": True,
                    "legal": True,
                    "accepted": True,
                    "quality": 1.0,
                    "legality_verified_by": "exact oracle match",
                }
            )
            return result
        result.update(grade(fen, response))
        if not result["legal"]:
            try:
                progress, components = legality_progress(
                    chess.Board(fen), chess.Move.from_uci(answer)
                )
            except ValueError:
                progress, components = 0.0, {}
            result.update(
                {
                    "reward": -1.0 + self.shaping_weight * progress,
                    "base_reward": -1.0 + self.shaping_weight * progress,
                    "legality_progress": components,
                }
            )
            return result
        chosen = best if exact else self.analyse(fen, result["answer"])
        drop = max(0.0, min(1.0, best["expected_score"] - chosen["expected_score"]))
        result.update(
            {
                "reward": 2.0 - drop,
                "base_reward": 2.0 - drop,
                "exact_match": exact,
                "quality": 1.0 - drop,
                "teacher_move": best["move"],
                "best_expected_score": best["expected_score"],
                "chosen_expected_score": chosen["expected_score"],
            }
        )
        return result
