import chess
import mlx.core as mx
import mlx.nn as nn


def word_embeddings(backbone, tokenizer, words):
    ids = []
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Expected one native token for {word!r}, got {tokens}")
        ids.append(tokens[0])
    return mx.stop_gradient(
        backbone.model.embed_tokens(mx.array(ids)).astype(mx.float32)
    )


def align_side(encoder, backbone, tokenizer):
    white, black = word_embeddings(backbone, tokenizer, [" White", " Black"])
    weight = mx.array(encoder.state_projection.weight)
    weight[:, 0] = (white - black) / 2
    encoder.state_projection.weight = weight
    encoder.state_projection.bias = (white + black) / 2


class SemanticBoardEncoder(nn.Module):
    slot_count = 257

    def __init__(self, backbone, tokenizer, original, label_side=True):
        super().__init__()
        width = backbone.args.hidden_size
        self.color_embedding = nn.Embedding(13, width)
        self.kind_embedding = nn.Embedding(13, width)
        self.file_embedding = nn.Embedding(8, width)
        self.rank_embedding = nn.Embedding(8, width)
        self.color_embedding.weight = word_embeddings(
            backbone, tokenizer, [" empty"] + [" White"] * 6 + [" Black"] * 6
        )
        self.kind_embedding.weight = word_embeddings(
            backbone,
            tokenizer,
            [" square"] + [" " + chess.piece_name(kind) for kind in range(1, 7)] * 2,
        )
        self.file_embedding.weight = word_embeddings(
            backbone, tokenizer, [" " + file for file in "abcdefgh"]
        )
        self.rank_embedding.weight = word_embeddings(
            backbone, tokenizer, [str(rank) for rank in range(8, 0, -1)]
        )
        self.en_passant_projection = nn.Linear(1, width, bias=False)
        self.en_passant_projection.weight = mx.array(
            original.en_passant_projection.weight
        )
        self.state_projection = nn.Linear(7, width)
        self.state_projection.update(original.state_projection.parameters())
        align_side(self, backbone, tokenizer)
        self.state_label = None
        if label_side:
            label_ids = tokenizer.encode("\nSide to move:", add_special_tokens=False)
            self.state_label = nn.Embedding(len(label_ids), width)
            self.state_label.weight = mx.stop_gradient(
                backbone.model.embed_tokens(mx.array(label_ids)).astype(mx.float32)
            )
            self.state_label.freeze()
            self.slot_count = 257 + len(label_ids)

    def __call__(self, pieces, en_passant, state):
        colors = pieces @ self.color_embedding.weight
        kinds = pieces @ self.kind_embedding.weight + self.en_passant_projection(
            en_passant
        )
        files = mx.broadcast_to(
            self.file_embedding.weight[None, None, :, :], colors.shape
        )
        ranks = mx.broadcast_to(
            self.rank_embedding.weight[None, :, None, :], colors.shape
        )
        squares = mx.stack([colors, kinds, files, ranks], axis=3).reshape(
            pieces.shape[0], 256, -1
        )
        segments = [squares]
        if self.state_label is not None:
            segments.append(
                mx.broadcast_to(
                    self.state_label.weight[None],
                    (pieces.shape[0], *self.state_label.weight.shape),
                )
            )
        segments.append(self.state_projection(state)[:, None, :])
        return mx.concatenate(segments, axis=1)
