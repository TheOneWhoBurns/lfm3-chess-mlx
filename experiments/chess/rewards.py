import chess
import numpy as np

REWARD_CONFIG = {
    "version": "stockfish_dominant_exploration_v2",
    "malformed": -2.0,
    "illegal_base": -1.0,
    "legal_min": 1.0,
    "legal_max": 2.0,
    "exact_base": 2.0,
    "exact_bonus": 8.0,
    "fixed_advantage_scale": 2.0,
    "ordinary_advantage_clip": 2.0,
    "match_gradient_share": 0.8,
    "default_shaping_weight": 1.0,
    "illegal_advantage_floor": -1.0,
}


def legality_progress(board, move):
    piece = board.piece_at(move.from_square)
    own = piece is not None and piece.color == board.turn
    geometry = False
    if own:
        dx = abs(
            chess.square_file(move.to_square) - chess.square_file(move.from_square)
        )
        dy = abs(
            chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
        )
        direction = (
            chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
        ) * (1 if piece.color else -1)
        if piece.piece_type == chess.PAWN:
            geometry = (dx == 0 and direction in [1, 2]) or (dx == 1 and direction == 1)
        elif piece.piece_type == chess.KNIGHT:
            geometry = (dx, dy) in [(1, 2), (2, 1)]
        elif piece.piece_type == chess.BISHOP:
            geometry = dx == dy and dx > 0
        elif piece.piece_type == chess.ROOK:
            geometry = (dx == 0) != (dy == 0)
        elif piece.piece_type == chess.QUEEN:
            geometry = ((dx == 0) != (dy == 0)) or (dx == dy and dx > 0)
        else:
            geometry = max(dx, dy) == 1 or board.is_castling(move)
    components = {
        "source_occupied": piece is not None,
        "own_piece": own,
        "movement_geometry": bool(geometry),
        "pseudo_legal": board.is_pseudo_legal(move),
    }
    progress = sum(
        float(components[key]) * weight
        for key, weight in [
            ("source_occupied", 0.1),
            ("own_piece", 0.3),
            ("movement_geometry", 0.3),
            ("pseudo_legal", 0.3),
        ]
    )
    return progress, components


def reward_advantages(responses):
    base = np.array([row["base_reward"] for row in responses], dtype=np.float64)
    if len(base) < 2:
        raise ValueError("At least two sampled responses are required")
    baseline = (base.sum() - base) / (len(base) - 1)
    ordinary = np.clip(
        (base - baseline) / REWARD_CONFIG["fixed_advantage_scale"],
        -REWARD_CONFIG["ordinary_advantage_clip"],
        REWARD_CONFIG["ordinary_advantage_clip"],
    )
    illegal = np.array(
        [not row.get("accepted", row.get("legal", False)) for row in responses]
    )
    ordinary = np.where(
        illegal,
        np.minimum(ordinary, REWARD_CONFIG["illegal_advantage_floor"]),
        ordinary,
    )
    bonus = (
        np.array([row["match_bonus"] for row in responses])
        / REWARD_CONFIG["fixed_advantage_scale"]
    )
    return (ordinary + bonus).astype(np.float32)
