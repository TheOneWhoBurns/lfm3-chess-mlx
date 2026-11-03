import copy

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from board_encoder import BoardPrompt
from exploration import sequence_terms, response_loss


def clone_cache(cache):
    result = []
    for item in cache:
        cloned = copy.copy(item)
        cloned.state = tree_map(
            lambda value: mx.array(value) if isinstance(value, mx.array) else value,
            item.state,
        )
        result.append(cloned)
    return result


def response_logits(model, prompt, group, evaluate_prefix=False):
    prompt_ids = prompt.tokens if isinstance(prompt, BoardPrompt) else prompt
    inputs = mx.array([prompt_ids])
    embeddings = (
        model.embed_prompt(inputs, prompt) if isinstance(prompt, BoardPrompt) else None
    )
    cache = model.make_cache()
    first = model(inputs, cache=cache, input_embeddings=embeddings)[0, -1]
    if evaluate_prefix:
        mx.eval(first, [item.state for item in cache])
    for row, support, targets, _ in group:
        tokens = row["tokens"]
        if not tokens:
            raise ValueError("Cannot score an empty response")
        logits = first[None]
        if len(tokens) > 1:
            tail = model(mx.array([tokens[:-1]]), cache=clone_cache(cache))[0]
            logits = mx.concatenate([logits, tail], axis=0)
        yield row, support, targets, logits


def shared_prefix_log_probabilities(model, prompt, group):
    probabilities = []
    for row, support, targets, logits in response_logits(model, prompt, group, True):
        selected, _, _ = sequence_terms(logits, support, targets, row)
        probabilities.append(selected.sum())
    result = mx.stack(probabilities)
    mx.eval(result)
    return np.array(result)


def _shared_prefix_policy_loss(
    model, prompt, group, clip=0.2, entropy=0.0, uniform_kl=0.0
):
    losses = []
    for (_, _, _, advantage), (row, support, targets, logits) in zip(
        group, response_logits(model, prompt, group)
    ):
        losses.append(
            response_loss(
                logits, support, targets, row, advantage, clip, entropy, uniform_kl
            )
        )
    return mx.stack(losses).sum()


def shared_prefix_policy_loss(
    model, prompt, group, clip=0.2, entropy=0.0, uniform_kl=0.0
):
    backbone = model.backbone if hasattr(model, "backbone") else model
    with backbone.differentiable_attention():
        return _shared_prefix_policy_loss(
            model, prompt, group, clip, entropy, uniform_kl
        )
