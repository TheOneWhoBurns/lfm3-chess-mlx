import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_lm import (
    JSONSchemaConstraint,
    StructuredOutputError,
    generate,
    load,
    stream_generate,
)
from mlx_lm.sample_utils import make_sampler


def response_format(schema):
    return {
        "type": "json_schema",
        "json_schema": {"name": "system_check", "strict": True, "schema": schema},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": False})
    model.set_dtype(mx.float32)
    model.eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return JSON that matches the requested schema."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    schemas = [
        {
            "type": "object",
            "properties": {
                "move": {"type": "string", "pattern": "^[a-h][1-8][a-h][1-8][qrbn]?$"}
            },
            "required": ["move"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "name": {"const": "Andrés ♟"},
                "flags": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["name", "flags"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "count": {"enum": [1, 2, 3]},
                        "label": {"enum": ["yes", "no"]},
                    },
                    "required": ["count", "label"],
                    "additionalProperties": False,
                }
            },
            "required": ["nested"],
            "additionalProperties": False,
        },
    ]
    report = {"model": args.model, "responses": []}
    for index, schema in enumerate(schemas):
        constraint = JSONSchemaConstraint(tokenizer, schema)
        options = {
            "prompt": prompt,
            "max_tokens": 96,
            "sampler": make_sampler(temp=0),
            "response_format": response_format(schema),
        }
        text = generate(model, tokenizer, **options)
        value = constraint.validate(text)
        chunks = list(stream_generate(model, tokenizer, **options))
        streamed = "".join(chunk.text for chunk in chunks)
        if constraint.validate(streamed) != value or streamed != text:
            raise AssertionError(
                "Streaming or independent request state changed the output"
            )
        report["responses"].append(
            {"schema": schema, "text": text, "value": value, "stream_equal": True}
        )
        print(index + 1, text, flush=True)
    try:
        generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1,
            response_format=response_format(schemas[0]),
        )
    except StructuredOutputError:
        report["truncation_rejected"] = True
    else:
        raise AssertionError("Incomplete JSON was returned as successful output")
    illegal = '{"move":"a1a1"}'
    notation = JSONSchemaConstraint(tokenizer, schemas[0])
    prefix = []
    for token in tokenizer.encode(illegal, add_special_tokens=False):
        if token not in notation.allowed_tokens(prefix):
            raise AssertionError(
                "Notation constraint unexpectedly applies chess legality"
            )
        prefix.append(token)
    if tokenizer.eos_token_id not in notation.allowed_tokens(prefix):
        raise AssertionError("Valid JSON could not terminate")
    report["illegal_chess_move_remains_representable"] = True
    try:
        JSONSchemaConstraint(tokenizer, {"type": "string", "minLength": 2})
    except ValueError:
        report["unsupported_keywords_rejected"] = True
    else:
        raise AssertionError("Unsupported schema keyword was silently ignored")
    report["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
