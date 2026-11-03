import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from jsonschema import Draft202012Validator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8097/v1/chat/completions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    schema = {
        "type": "object",
        "properties": {
            "move": {"type": "string", "pattern": "^[a-h][1-8][a-h][1-8][qrbn]?$"}
        },
        "required": ["move"],
        "additionalProperties": False,
    }
    base = {
        "messages": [{"role": "user", "content": "Return one chess move as JSON."}],
        "max_tokens": 48,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "chess_move", "strict": True, "schema": schema},
        },
    }
    records = []

    def request(name, body, expected_status=200):
        payload = urllib.request.Request(
            args.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(payload, timeout=120) as response:
                status, text = response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            status, text = error.code, error.read().decode()
        records.append({"name": name, "status": status, "response": text})
        (args.output / "responses.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        if status != expected_status:
            raise AssertionError(
                f"{name}: expected {expected_status}, received {status}: {text}"
            )
        print(name, status, flush=True)
        return text

    for name in ["ordinary", "cached_repeat"]:
        body = json.loads(request(name, base))
        content = body["choices"][0]["message"]["content"]
        Draft202012Validator(schema).validate(json.loads(content))
        if body["choices"][0]["finish_reason"] != "stop":
            raise AssertionError("Structured request did not finish normally")
    events = request("stream", {**base, "stream": True})
    parts = []
    for line in events.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            value = json.loads(line[6:])
            if "error" in value:
                raise AssertionError(value)
            for choice in value.get("choices", []):
                parts.append(choice.get("delta", {}).get("content") or "")
    Draft202012Validator(schema).validate(json.loads("".join(parts)))
    request("truncation", {**base, "max_tokens": 1}, 422)
    events = request("stream_truncation", {**base, "max_tokens": 1, "stream": True})
    if '"structured_output_error"' not in events:
        raise AssertionError("Streaming truncation was not reported as an error")
    request("custom_stop", {**base, "stop": "}"}, 400)
    request(
        "unsupported_schema",
        {
            **base,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "string", "minLength": 2}},
            },
        },
        400,
    )
    plain = {**base, "response_format": {"type": "text"}, "max_tokens": 8}
    result = json.loads(request("ordinary_text_still_works", plain))
    if not result.get("choices"):
        raise AssertionError("Ordinary text generation failed")
    (args.output / "report.json").write_text(
        json.dumps(
            {"passed": True, "checks": [row["name"] for row in records]}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
