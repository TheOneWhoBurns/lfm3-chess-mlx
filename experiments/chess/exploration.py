import math

import mlx.core as mx


def behavior_logprobs(values, temperature=1.0, epsilon=0.0):
    if temperature <= 0 or not 0 <= epsilon < 1:
        raise ValueError("Invalid exploration temperature or mixture weight")
    shifted = values.astype(mx.float32) / temperature
    shifted = shifted - mx.max(shifted, axis=-1, keepdims=True)
    result = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
    if epsilon:
        finite = mx.isfinite(values)
        count = mx.sum(finite, axis=-1, keepdims=True)
        uniform = mx.where(finite, math.log(epsilon) - mx.log(count), -float("inf"))
        result = mx.logaddexp(result + math.log1p(-epsilon), uniform)
    return result


def sequence_terms(logits, support, targets, row):
    selected, entropies, uniform_kls = [], [], []
    temperature = row.get("sampling_temperature", 1.0)
    epsilon = row.get("exploration_epsilon", 0.0)
    for index, (allowed, target) in enumerate(zip(support, targets)):
        values = logits[index, allowed].astype(mx.float32)
        base = behavior_logprobs(values)
        selected.append(behavior_logprobs(values, temperature, epsilon)[target])
        if len(allowed) > 1:
            entropies.append(-(mx.exp(base) * base).sum())
            uniform_kls.append(-base.mean() - math.log(len(allowed)))
    zero = mx.array(0.0)
    return (
        mx.stack(selected),
        mx.stack(entropies).mean() if entropies else zero,
        mx.stack(uniform_kls).mean() if uniform_kls else zero,
    )


def response_loss(logits, support, targets, row, advantage, clip, entropy, uniform_kl):
    logprobs, entropies, uniform_kls = sequence_terms(logits, support, targets, row)
    ratio = mx.exp(logprobs - mx.array(row["old_logprobs"]))
    objective = mx.minimum(
        ratio * advantage, mx.clip(ratio, 1 - clip, 1 + clip) * advantage
    )
    return -objective.sum() - entropy * entropies + uniform_kl * uniform_kls
