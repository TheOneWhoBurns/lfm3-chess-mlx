import copy
from dataclasses import replace

import mlx.core as mx
from mlx.utils import tree_map

from board_encoder import BoardPrompt
from mlx_lm.structured import StructuredOutputError
from exploration import behavior_logprobs


def last_logits(model, tokens, cache, embeddings=None):
    backbone = model.backbone
    hidden = backbone.model(tokens, cache=cache, input_embeddings=embeddings)
    hidden = hidden[:, -1:, :]
    projected = (
        backbone.model.embed_tokens.as_linear(hidden)
        if backbone.args.tie_word_embeddings
        else backbone.lm_head(hidden)
    )
    return projected[:, 0, :].astype(mx.float32)


def expand_cache(cache, size):
    result = []
    for item in cache:
        cloned = copy.copy(item)
        cloned.state = tree_map(
            lambda value: (
                mx.repeat(value, size, axis=0)
                if isinstance(value, mx.array) and value.ndim
                else value
            ),
            item.state,
        )
        result.append(cloned)
    return result


def batch_rollout(
    model,
    tokenizer,
    prompts,
    constraint,
    max_tokens=48,
    temperature=0.0,
    exploration_epsilon=0.0,
):
    if not prompts or max_tokens < 1 or temperature < 0:
        raise ValueError(
            "A nonempty board batch and valid decoding budgets are required"
        )
    if not 0 <= exploration_epsilon < 1 or (temperature == 0 and exploration_epsilon):
        raise ValueError(
            "Exploration requires stochastic decoding and epsilon in [0, 1)"
        )
    first = prompts[0]
    if not all(
        isinstance(prompt, BoardPrompt)
        and prompt.tokens == first.tokens
        and prompt.offset == first.offset
        and prompt.slot_count == first.slot_count
        for prompt in prompts
    ):
        raise ValueError(
            "Batched board decoding requires a common prompt/embedding layout"
        )
    batch_size = len(prompts)
    shared = all(prompt is first for prompt in prompts)
    if shared:
        inputs = mx.array([first.tokens])
        embeddings = model.embed_prompt(inputs, first)
    else:
        combined = replace(
            first,
            pieces=mx.concatenate([prompt.pieces for prompt in prompts], axis=0),
            en_passant=mx.concatenate(
                [prompt.en_passant for prompt in prompts], axis=0
            ),
            state=mx.concatenate([prompt.state for prompt in prompts], axis=0),
        )
        inputs = mx.array([prompt.tokens for prompt in prompts])
        embeddings = model.embed_prompt(inputs, combined)
    cache = model.make_cache()
    logits = last_logits(model, inputs, cache, embeddings)
    mx.eval(logits, [item.state for item in cache])
    if shared and batch_size > 1:
        cache = expand_cache(cache, batch_size)
        logits = mx.repeat(logits, batch_size, axis=0)
    tokens, logprobs = [[] for _ in prompts], [[] for _ in prompts]
    ended = [False] * batch_size
    for step in range(max_tokens):
        masked = mx.full(logits.shape, -float("inf"), dtype=mx.float32)
        for index in range(batch_size):
            allowed = (
                [tokenizer.eos_token_id]
                if ended[index]
                else constraint.allowed_tokens(tokens[index])
            )
            if not allowed:
                raise StructuredOutputError("No tokens satisfy the JSON schema")
            allowed = mx.array(allowed)
            masked[index, allowed] = logits[index, allowed]
        distribution = behavior_logprobs(
            masked, temperature or 1.0, exploration_epsilon
        )
        selected = (
            mx.argmax(masked, axis=-1)
            if temperature == 0
            else mx.random.categorical(distribution, axis=-1)
        )
        probabilities = distribution[mx.arange(batch_size), selected]
        mx.eval(selected, probabilities)
        chosen, values = selected.tolist(), probabilities.tolist()
        for index in range(batch_size):
            if not ended[index]:
                tokens[index].append(chosen[index])
                logprobs[index].append(values[index])
                ended[index] = chosen[index] in tokenizer.eos_token_ids
        if all(ended) or step + 1 == max_tokens:
            break
        logits = last_logits(model, selected[:, None], cache)
    responses = []
    for sequence, probability, finished in zip(tokens, logprobs, ended):
        raw = tokenizer.decode(sequence[:-1] if finished else sequence)
        try:
            constraint.validate(raw)
            valid = True
        except StructuredOutputError:
            valid = False
        responses.append(
            {
                "tokens": sequence,
                "old_logprobs": probability,
                "raw": raw,
                "ended": finished,
                "schema_valid": valid,
                "response_format": "json_schema",
                "sampling_temperature": temperature or 1.0,
                "exploration_epsilon": exploration_epsilon,
            }
        )
    return responses
