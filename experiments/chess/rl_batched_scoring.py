from collections import defaultdict

import mlx.core as mx
import numpy as np

from batched_policy import expand_cache
from board_encoder import BoardPrompt
from exploration import sequence_terms, response_loss


def project(backbone, hidden):
    if backbone.args.tie_word_embeddings:
        return backbone.model.embed_tokens.as_linear(hidden)
    return backbone.lm_head(hidden)


def response_logits(model, prompt, group, evaluate_prefix=False):
    backbone = model.backbone if hasattr(model, "backbone") else model
    prompt_ids = prompt.tokens if isinstance(prompt, BoardPrompt) else prompt
    inputs = mx.array([prompt_ids])
    embeddings = (
        model.embed_prompt(inputs, prompt) if isinstance(prompt, BoardPrompt) else None
    )
    cache = model.make_cache()
    hidden = backbone.model(inputs, cache=cache, input_embeddings=embeddings)
    first = project(backbone, hidden[:, -1:])[0]
    if evaluate_prefix:
        mx.eval(first, [item.state for item in cache])
    buckets = defaultdict(list)
    for index, (row, _, _, _) in enumerate(group):
        if not row["tokens"]:
            raise ValueError("Cannot score an empty response")
        buckets[len(row["tokens"])].append(index)
    results = {}
    for length, indices in buckets.items():
        if length == 1:
            for index in indices:
                results[index] = first
            continue
        tokens = mx.array([group[index][0]["tokens"][:-1] for index in indices])
        tail = backbone.model(tokens, cache=expand_cache(cache, len(indices)))
        logits = project(backbone, tail)
        for local, index in enumerate(indices):
            results[index] = mx.concatenate([first, logits[local]], axis=0)
    for index, (row, support, targets, _) in enumerate(group):
        yield row, support, targets, results[index]


def batched_prefix_log_probabilities(model, prompt, group):
    probabilities = []
    for row, support, targets, logits in response_logits(model, prompt, group, True):
        selected, _, _ = sequence_terms(logits, support, targets, row)
        probabilities.append(selected.sum())
    result = mx.stack(probabilities)
    mx.eval(result)
    return np.array(result)


def _batched_prefix_policy_loss(
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


def batched_prefix_policy_loss(
    model, prompt, group, clip=0.2, entropy=0.0, uniform_kl=0.0
):
    backbone = model.backbone if hasattr(model, "backbone") else model
    with backbone.differentiable_attention():
        return _batched_prefix_policy_loss(
            model, prompt, group, clip, entropy, uniform_kl
        )
