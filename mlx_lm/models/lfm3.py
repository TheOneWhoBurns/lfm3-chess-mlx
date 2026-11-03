from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "lfm3"
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None
    vocab_size: int = 128000
    max_position_embeddings: int = 262144
    norm_eps: float = 1e-5
    conv_L_cache: int = 7
    conv_bias: bool = False
    sliding_window: int = 256
    tie_word_embeddings: bool = True
    use_sandwich_norm: bool = True
    layer_types: Optional[list[str]] = None
    rope_parameters: Optional[dict] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Attention heads must be divisible by KV heads")
        if self.conv_L_cache < 1:
            raise ValueError("conv_L_cache must be positive")
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if i % 4 == 2
                else "sliding_attention"
                if i % 4 == 0 and i > 0
                else "conv"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")
        if not set(self.layer_types) <= {"conv", "full_attention", "sliding_attention"}:
            raise ValueError("Unknown LFM3 layer type")
        if "sliding_attention" in self.layer_types and (
            self.sliding_window is None or self.sliding_window < 1
        ):
            raise ValueError("Sliding attention requires a positive sliding_window")
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
            }
        for kind in set(self.layer_types) - {"conv"}:
            if kind not in self.rope_parameters:
                raise ValueError(f"Missing rotary configuration for {kind}")
            config = self.rope_parameters[kind]
            if config.get("rope_type", "default") != "default":
                raise ValueError("LFM3 currently supports default RoPE only")
            if not 0 < config.get("partial_rotary_factor", 1.0) <= 1:
                raise ValueError("partial_rotary_factor must be in (0, 1]")


class RotaryEmbedding(nn.Module):
    def __init__(self, dims: int, config: dict):
        super().__init__()
        angles = int(config.get("partial_rotary_factor", 1.0) * dims) // 2
        inv_freq = config["rope_theta"] ** (
            -mx.arange(0, 2 * angles, 2, dtype=mx.float32) / dims
        )
        self._inv_freq = mx.concatenate([inv_freq, mx.zeros(dims // 2 - angles)])

    def __call__(self, x, offset=0):
        positions = mx.arange(x.shape[-2], dtype=mx.float32)
        if isinstance(offset, mx.array) and offset.ndim > 0:
            positions = positions[None, :] + offset[:, None]
        else:
            positions = positions[None, :] + offset
        freqs = positions[..., None] * self._inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)[:, None, :, :]
        cos = mx.cos(emb).astype(x.dtype).astype(mx.float32)
        sin = mx.sin(emb).astype(x.dtype).astype(mx.float32)
        first, second = mx.split(x.astype(mx.float32), 2, axis=-1)
        rotated = mx.concatenate([-second, first], axis=-1)
        return (x.astype(mx.float32) * cos + rotated * sin).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self._differentiable_attention = False
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.gate_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, args.rope_parameters[layer_type])

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        )
        k = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        )
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        offset = cache.offset if cache is not None else 0
        q = self.rope(q.transpose(0, 2, 1, 3), offset)
        k = self.rope(k.transpose(0, 2, 1, 3), offset)
        v = v.transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if self._differentiable_attention:
            if not isinstance(k, mx.array) or not isinstance(v, mx.array):
                raise ValueError(
                    "Differentiable attention requires an unquantized cache"
                )
            repeats = self.n_heads // self.n_kv_heads
            keys = mx.repeat(k, repeats, axis=1) if repeats > 1 else k
            values = mx.repeat(v, repeats, axis=1) if repeats > 1 else v
            scores = (q * self.scale) @ keys.swapaxes(-1, -2)
            if isinstance(mask, str):
                if mask != "causal":
                    raise ValueError(f"Unknown attention mask: {mask}")
                queries = mx.arange(k.shape[-2] - q.shape[-2], k.shape[-2])
                mask = queries[:, None] >= mx.arange(k.shape[-2])[None, :]
            if mask is not None:
                scores = (
                    mx.where(mask, scores, -mx.inf)
                    if mask.dtype == mx.bool_
                    else scores + mask
                )
            out = mx.softmax(scores, axis=-1, precise=True) @ values
        else:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )
        out = out.transpose(0, 2, 1, 3)
        out = out * mx.sigmoid(self.gate_proj(x))[..., None]
        return self.o_proj(out.reshape(batch, length, -1))


class ShortConv(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.state_size = args.conv_L_cache - 1
        self.conv = nn.Conv1d(
            args.hidden_size,
            args.hidden_size,
            args.conv_L_cache,
            groups=args.hidden_size,
            bias=args.conv_bias,
        )
        self.in_proj = nn.Linear(
            args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias
        )
        self.out_proj = nn.Linear(
            args.hidden_size, args.hidden_size, bias=args.conv_bias
        )

    def __call__(self, x, mask=None, cache=None):
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        b, c, x = mx.split(self.in_proj(x), 3, axis=-1)
        bx = b * x
        if mask is not None:
            bx = mx.where(mask[..., None], bx, 0)
        if cache is not None and cache[0] is not None:
            state = cache[0]
        else:
            state = mx.zeros((x.shape[0], self.state_size, x.shape[-1]), dtype=x.dtype)
        bx = mx.concatenate([state, bx], axis=1)
        if cache is not None:
            if self.state_size == 0:
                cache[0] = bx[:, :0, :]
            elif cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, x.shape[1])
                positions = (ends[:, None] + mx.arange(self.state_size))[..., None]
                cache[0] = mx.take_along_axis(bx, positions, axis=1)
            else:
                cache[0] = bx[:, -self.state_size :, :]
            cache.advance(x.shape[1])
        return self.out_proj(c * self.conv(bx))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.use_sandwich_norm = args.use_sandwich_norm
        if layer_type == "conv":
            self.conv = ShortConv(args)
        else:
            self.self_attn = Attention(args, layer_type)
        self.feed_forward = MLP(args)
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        operator = self.conv if self.layer_type == "conv" else self.self_attn
        residual = operator(self.operator_norm(x), mask=mask, cache=cache)
        if self.use_sandwich_norm:
            residual = self.post_operator_norm(residual)
        h = x + residual
        residual = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich_norm:
            residual = self.post_ffn_norm(residual)
        return h + residual


class Lfm3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, kind) for kind in args.layer_types]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Cache must contain one entry per layer")
        masks = {}
        for kind in set(self.args.layer_types):
            c = cache[self.args.layer_types.index(kind)]
            if kind == "conv":
                masks[kind] = create_ssm_mask(h, c)
            else:
                window = (
                    self.args.sliding_window if kind == "sliding_attention" else None
                )
                masks[kind] = create_attention_mask(h, c, window_size=window)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=masks[layer.layer_type], cache=c)
        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    @contextmanager
    def differentiable_attention(self):
        attention_layers = [
            layer.self_attn for layer in self.layers if layer.layer_type != "conv"
        ]
        previous = [layer._differentiable_attention for layer in attention_layers]
        for layer in attention_layers:
            layer._differentiable_attention = True
        try:
            yield
        finally:
            for layer, value in zip(attention_layers, previous):
                layer._differentiable_attention = value

    def sanitize(self, weights):
        sanitized = {}
        for name, value in weights.items():
            name = name.removeprefix("language_model.")
            if self.args.tie_word_embeddings and name == "lm_head.weight":
                continue
            if name.endswith("conv.conv.weight") and value.shape[1:] == (
                1,
                self.args.conv_L_cache,
            ):
                value = value.transpose(0, 2, 1)
            sanitized[name] = value
        return sanitized

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(size=1)
            if kind == "conv"
            else RotatingKVCache(max_size=self.args.sliding_window)
            if kind == "sliding_attention"
            else KVCache()
            for kind in self.args.layer_types
        ]
