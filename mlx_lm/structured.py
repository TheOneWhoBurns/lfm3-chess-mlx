import json
from functools import lru_cache

import mlx.core as mx


class StructuredOutputError(ValueError):
    pass


def validate_schema(schema):
    try:
        from jsonschema import Draft202012Validator, SchemaError
    except ImportError as error:
        raise ImportError(
            "Install mlx-lm[structured] to use JSON-schema outputs"
        ) from error
    if not isinstance(schema, dict):
        raise ValueError("The JSON schema must be an object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError(f"Invalid JSON schema: {error.message}") from error
    supported = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "enum",
        "const",
        "pattern",
        "title",
        "description",
    }
    unknown = set(schema) - supported
    if unknown:
        raise ValueError(f"Unsupported JSON-schema keywords: {sorted(unknown)}")
    kind = schema.get("type")
    if kind is not None and (
        not isinstance(kind, str)
        or kind
        not in {"object", "array", "string", "integer", "number", "boolean", "null"}
    ):
        raise ValueError("Schema type must be a single supported JSON type")
    if kind is None and "enum" not in schema and "const" not in schema:
        raise ValueError("A schema requires type, enum, or const")
    if kind == "object":
        if schema.get("additionalProperties") is not False:
            raise ValueError("Structured objects require additionalProperties=false")
        for child in schema.get("properties", {}).values():
            validate_schema(child)
    if kind == "array":
        if not isinstance(schema.get("items"), dict):
            raise ValueError("Structured arrays require an items schema")
        validate_schema(schema["items"])


def schema_from_response_format(response_format):
    if response_format is None or response_format == {"type": "text"}:
        return None
    if (
        not isinstance(response_format, dict)
        or response_format.get("type") != "json_schema"
    ):
        raise ValueError("response_format must have type=json_schema or type=text")
    specification = response_format.get("json_schema")
    if not isinstance(specification, dict) or "schema" not in specification:
        raise ValueError("response_format.json_schema.schema is required")
    schema = specification["schema"]
    validate_schema(schema)
    return schema


@lru_cache(maxsize=4)
def _vocabulary(tokenizer):
    try:
        from outlines_core import Vocabulary
    except ImportError as error:
        raise ImportError(
            "Install mlx-lm[structured] to use JSON-schema outputs"
        ) from error
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("Structured output requires a fast ByteLevel tokenizer")
    specification = json.loads(backend.to_str())
    if specification.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError(
            "Structured output currently supports ByteLevel tokenizers only"
        )
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            characters.append(256 + extra)
            extra += 1
    byte_decoder = dict(zip(map(chr, characters), byte_values))
    special = set(tokenizer.all_special_ids)
    model_vocab = specification["model"]["vocab"]
    mapping = {}
    for token, token_id in tokenizer.get_vocab().items():
        if token_id in special:
            continue
        data = (
            bytes(byte_decoder[character] for character in token)
            if token in model_vocab
            else token.encode("utf-8")
        )
        if data:
            mapping.setdefault(data, []).append(token_id)
    return Vocabulary(tokenizer.eos_token_id, mapping)


@lru_cache(maxsize=16)
def _index(tokenizer, serialized_schema):
    from outlines_core import Index
    from outlines_core.json_schema import build_regex_from_schema

    regex = build_regex_from_schema(serialized_schema, whitespace_pattern=r"[ ]?")
    return Index(regex, _vocabulary(tokenizer))


class JSONSchemaConstraint:
    def __init__(self, tokenizer, schema):
        validate_schema(schema)
        from jsonschema import Draft202012Validator
        from outlines_core import Guide
        from .tokenizer_utils import TokenizerWrapper

        self.schema = json.loads(json.dumps(schema))
        self.validator = Draft202012Validator(self.schema)
        self.tokenizer = tokenizer
        self._guide_type = Guide
        inner = (
            tokenizer._tokenizer
            if isinstance(tokenizer, TokenizerWrapper)
            else tokenizer
        )
        self.index = _index(inner, json.dumps(self.schema, sort_keys=True))
        self.eos_token_id = tokenizer.eos_token_id

    def allowed_tokens(self, completion_tokens):
        guide = self._guide_type(self.index)
        for token in completion_tokens:
            if (
                int(token) == self.eos_token_id
                and self.eos_token_id in guide.get_tokens()
            ):
                return [self.eos_token_id]
            guide.advance(int(token), return_tokens=False)
        return guide.get_tokens()

    def processor(self, prompt_length):
        guide = self._guide_type(self.index)
        consumed = prompt_length
        terminated = False

        def process(tokens, logits):
            nonlocal consumed, terminated
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("JSON-schema decoding requires a single sequence")
            if len(tokens) < consumed:
                raise ValueError("A schema processor cannot be reused or rolled back")
            for token in tokens[consumed:].tolist():
                if token == self.eos_token_id and (
                    terminated or self.eos_token_id in guide.get_tokens()
                ):
                    terminated = True
                elif terminated:
                    raise StructuredOutputError(
                        "Tokens appeared after the structured response ended"
                    )
                else:
                    guide.advance(token, return_tokens=False)
            consumed = len(tokens)
            allowed = mx.array(
                [self.eos_token_id] if terminated else guide.get_tokens()
            )
            if not len(allowed):
                raise StructuredOutputError("No tokens satisfy the JSON schema")
            selected = logits[:, allowed]
            if not mx.any(mx.isfinite(selected)).item():
                raise StructuredOutputError(
                    "Other logits processors excluded all schema-valid tokens"
                )
            masked = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
            masked[:, allowed] = selected
            return masked

        return process

    def validate(self, text):
        try:
            value = json.loads(text)
        except (ValueError, TypeError) as error:
            raise StructuredOutputError(
                "Generation ended before a complete JSON response"
            ) from error
        errors = list(self.validator.iter_errors(value))
        if errors:
            raise StructuredOutputError(
                f"Generated JSON violates the schema: {errors[0].message}"
            )
        return value
