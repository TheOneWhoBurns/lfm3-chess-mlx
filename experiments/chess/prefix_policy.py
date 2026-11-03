import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.structured import StructuredOutputError
from board_encoder import BoardPrompt
from rl_scoring import clone_cache


class FixedPrefixPolicy:
    def __init__(self, model, tokenizer, prompt, constraint):
        if not isinstance(prompt, BoardPrompt) or prompt.offset < 1:
            raise ValueError("A numeric board prompt with a fixed prefix is required")
        self.model = model
        self.tokenizer = tokenizer
        self.constraint = constraint
        self.prefix = prompt.tokens[: prompt.offset]
        self.encoding = dict(model.board_encoding)
        self.cache = model.make_cache()
        model(mx.array([self.prefix]), cache=self.cache)
        mx.eval([item.state for item in self.cache])

    def rollout(self, prompt, max_tokens=48, temperature=0.0):
        if (
            not isinstance(prompt, BoardPrompt)
            or prompt.tokens[: prompt.offset] != self.prefix
            or self.model.board_encoding != self.encoding
        ):
            raise ValueError(
                "Cached instructions do not match the supplied board prompt"
            )
        tokens = prompt.tokens[prompt.offset :]
        embeddings = self.model.embed_prompt(mx.array([prompt.tokens]), prompt)[
            0, prompt.offset :
        ]
        cache = clone_cache(self.cache)
        sequence, logprobs = [], []
        ended = False
        generator = generate_step(
            mx.array(tokens),
            self.model,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            logits_processors=[self.constraint.processor(len(tokens))],
            input_embeddings=embeddings,
            prompt_cache=cache,
        )
        try:
            for token, probabilities in generator:
                sequence.append(token)
                logprobs.append(float(probabilities[token].item()))
                if token in self.tokenizer.eos_token_ids:
                    ended = True
                    break
        finally:
            generator.close()
        raw = self.tokenizer.decode(sequence[:-1] if ended else sequence)
        try:
            self.constraint.validate(raw)
            valid = True
        except StructuredOutputError:
            valid = False
        return {
            "tokens": sequence,
            "old_logprobs": logprobs,
            "raw": raw,
            "ended": ended,
            "schema_valid": valid,
            "response_format": "json_schema",
        }
