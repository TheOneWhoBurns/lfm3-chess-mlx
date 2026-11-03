import json
from dataclasses import dataclass
from pathlib import Path

import chess
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

SYMBOLS = ".PNBRQKpnbrqk"
ENCODING = {
    "version": "board_tensor_v1",
    "squares": "a8 through h1, rank-major",
    "piece_channels": list(SYMBOLS),
    "board_shape": [8, 8, 13],
    "en_passant_shape": [8, 8, 1],
    "state_features": [
        "white_to_move_signed",
        "white_kingside",
        "white_queenside",
        "black_kingside",
        "black_queenside",
        "halfmove_clock_div_150",
        "log1p_fullmove_div_10",
    ],
    "embedding_slots": 65,
    "history": "FEN state only; repetition history not represented",
}


@dataclass
class BoardPrompt:
    tokens: list[int]
    offset: int
    pieces: mx.array
    en_passant: mx.array
    state: mx.array
    slot_count: int = 65


def encode_position(fen):
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("Cannot encode an invalid chess position")
    pieces = np.zeros((8, 8, 13), dtype=np.float32)
    for row in range(8):
        for column in range(8):
            piece = board.piece_at(chess.square(column, 7 - row))
            pieces[row, column, SYMBOLS.index(piece.symbol() if piece else ".")] = 1
    en_passant = np.zeros((8, 8, 1), dtype=np.float32)
    if board.ep_square is not None:
        en_passant[
            7 - chess.square_rank(board.ep_square),
            chess.square_file(board.ep_square),
            0,
        ] = 1
    state = np.array(
        [
            1.0 if board.turn else -1.0,
            float(board.has_kingside_castling_rights(chess.WHITE)),
            float(board.has_queenside_castling_rights(chess.WHITE)),
            float(board.has_kingside_castling_rights(chess.BLACK)),
            float(board.has_queenside_castling_rights(chess.BLACK)),
            board.halfmove_clock / 150.0,
            np.log1p(board.fullmove_number) / 10.0,
        ],
        dtype=np.float32,
    )
    return pieces, en_passant, state


def board_prompt(tokenizer, fen, encoding=None):
    encoding = ENCODING if encoding is None else encoding
    marker = "[[BOARD_TENSOR]]"
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Choose the best next chess move for the side to move. Return only a JSON object with one key, move, whose value is a UCI move. The board embeddings list squares a8 through h1, followed by game state.\n"
                + marker,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count(marker) != 1:
        raise ValueError("The chat template changed the board insertion marker")
    prefix, suffix = rendered.split(marker)
    before = tokenizer.encode(prefix, add_special_tokens=False)
    after = tokenizer.encode(suffix, add_special_tokens=False)
    pieces, en_passant, state = encode_position(fen)
    return BoardPrompt(
        before + [tokenizer.pad_token_id] * encoding["embedding_slots"] + after,
        len(before),
        mx.array(pieces)[None],
        mx.array(en_passant)[None],
        mx.array(state)[None],
        encoding["embedding_slots"],
    )


class BoardEncoder(nn.Module):
    def __init__(self, backbone, tokenizer):
        super().__init__()
        width = backbone.args.hidden_size
        self.piece_embedding = nn.Embedding(13, width)
        ids = []
        for symbol in SYMBOLS:
            encoded = tokenizer.encode(symbol, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    "Piece-symbol initialization requires one token per symbol"
                )
            ids.append(encoded[0])
        self.piece_embedding.weight = mx.stop_gradient(
            backbone.model.embed_tokens(mx.array(ids)).astype(mx.float32)
        )
        self.rank_embedding = nn.Embedding(8, width)
        self.file_embedding = nn.Embedding(8, width)
        rank_ids = [
            tokenizer.encode(str(rank), add_special_tokens=False)[0]
            for rank in range(8, 0, -1)
        ]
        file_ids = [
            tokenizer.encode(file, add_special_tokens=False)[0] for file in "abcdefgh"
        ]
        self.rank_embedding.weight = mx.stop_gradient(
            backbone.model.embed_tokens(mx.array(rank_ids)).astype(mx.float32) * 0.25
        )
        self.file_embedding.weight = mx.stop_gradient(
            backbone.model.embed_tokens(mx.array(file_ids)).astype(mx.float32) * 0.25
        )
        self.en_passant_projection = nn.Linear(1, width, bias=False)
        self.en_passant_projection.weight = mx.random.normal((width, 1)) * 0.01
        self.state_projection = nn.Linear(7, width)
        self.state_projection.weight = mx.random.normal((width, 7)) * 0.01
        self.state_projection.bias = mx.zeros((width,))

    def __call__(self, pieces, en_passant, state):
        encoded = pieces @ self.piece_embedding.weight
        encoded = (
            encoded
            + self.rank_embedding.weight[None, :, None, :]
            + self.file_embedding.weight[None, None, :, :]
        )
        encoded = encoded + self.en_passant_projection(en_passant)
        squares = encoded.reshape(encoded.shape[0], 64, -1)
        return mx.concatenate(
            [squares, self.state_projection(state)[:, None, :]], axis=1
        )


class BoardConditionedModel(nn.Module):
    def __init__(self, backbone, tokenizer, encoder="compact"):
        super().__init__()
        self.backbone = backbone
        self.board_encoder = BoardEncoder(backbone, tokenizer)
        self._board_encoding = dict(ENCODING)
        if encoder == "semantic":
            from semantic_board_encoder import SemanticBoardEncoder

            self.board_encoder = SemanticBoardEncoder(
                backbone, tokenizer, self.board_encoder
            )
            self._board_encoding.update(
                {
                    "version": "board_tensor_semantic_v2",
                    "embedding_slots": self.board_encoder.slot_count,
                    "square_embedding_layout": ["color", "piece_kind", "file", "rank"],
                    "state_label": "Side to move:",
                    "initialization": "separate native word embeddings; numeric live board inputs",
                }
            )
        elif encoder != "compact":
            raise ValueError(f"Unknown board encoder: {encoder}")

    @property
    def board_encoding(self):
        return self._board_encoding

    @property
    def args(self):
        return self.backbone.args

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        return self.backbone.make_cache()

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.backbone(inputs, cache=cache, input_embeddings=input_embeddings)

    def embed_prompt(self, tokens, prompt):
        if prompt.slot_count != self.board_encoding["embedding_slots"]:
            raise ValueError("Board prompt does not match the encoder layout")
        ordinary = self.backbone.model.embed_tokens(tokens)
        board = self.board_encoder(prompt.pieces, prompt.en_passant, prompt.state)
        return mx.concatenate(
            [
                ordinary[:, : prompt.offset],
                board,
                ordinary[:, prompt.offset + prompt.slot_count :],
            ],
            axis=1,
        )


def load_board_policy(path, checkpoint=None, layers=16, seed=2026103110, encoder=None):
    metadata = None
    if checkpoint:
        checkpoint = Path(checkpoint)
        metadata = json.loads((checkpoint / "adapter_config.json").read_text())
        version = metadata.get("board_encoding", {}).get("version")
        recorded = {
            "board_tensor_v1": "compact",
            "board_tensor_semantic_v2": "semantic",
        }.get(version)
        if recorded is None or (encoder is not None and recorded != encoder):
            raise ValueError("Checkpoint has an incompatible board encoder")
        encoder = recorded
    encoder = "compact" if encoder is None else encoder
    mx.random.seed(seed)
    backbone, tokenizer = load(str(path), tokenizer_config={"trust_remote_code": False})
    backbone.set_dtype(mx.float32)
    backbone.freeze()
    lora = {"rank": 8, "scale": 16.0, "dropout": 0.0}
    linear_to_lora_layers(backbone, layers, lora)
    model = BoardConditionedModel(backbone, tokenizer, encoder)
    if checkpoint:
        if (
            metadata.get("board_encoding") != model.board_encoding
            or metadata.get("num_layers") != layers
            or metadata.get("lora_parameters") != lora
        ):
            raise ValueError("Checkpoint board architecture does not match")
        weights = mx.load(str(checkpoint / "adapters.safetensors"))
        expected = dict(tree_flatten(model.trainable_parameters()))
        if set(weights) != set(expected) or any(
            weights[key].shape != expected[key].shape for key in weights
        ):
            raise ValueError(
                "Checkpoint does not contain exactly the encoder and LoRA parameters"
            )
        model.load_weights(list(weights.items()), strict=False)
    model.eval()
    return model, tokenizer
