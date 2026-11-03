import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("train_rl.py")),
        "--model",
        str(args.model),
        "--data",
        str(args.data),
        "--encoder",
        "semantic",
        "--groups",
        "3",
        "--samples",
        "8",
        "--save-every",
        "100",
        "--position-offset",
        "3500",
        "--shared-prefix",
        "--batched-rollouts",
        "--batched-scoring",
    ]

    def run(name, extra, interrupt=False):
        output = args.output / name
        with (args.output / f"{name}.log").open("w") as log:
            child = subprocess.Popen(
                common + ["--output", str(output)] + extra,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if interrupt:
                deadline = time.monotonic() + 600
                while child.poll() is None:
                    metrics = output / "metrics.jsonl"
                    if metrics.exists() and metrics.stat().st_size:
                        os.kill(child.pid, signal.SIGTERM)
                        break
                    if time.monotonic() > deadline:
                        child.terminate()
                        raise TimeoutError("Trainer did not produce a completed group")
                    time.sleep(0.05)
            code = child.wait(timeout=600)
        if code:
            raise RuntimeError(f"{name} failed with {code}; inspect its log")
        print(json.dumps({"step": name, "exit_code": code}), flush=True)
        return output

    first = run("interrupted", ["--initialize-from", str(args.checkpoint)], True)
    stopped = json.loads((first / "interrupted.json").read_text())
    saved = stopped["last_complete_checkpoint"]
    assert 0 < saved < 3 and not stopped["unsaved_updates_discarded"], stopped
    assert not (first / "completed.json").exists()
    assert not list((first / "checkpoints").glob(".pending-*"))
    resumed = run("resumed", ["--resume", str(first / "checkpoints" / str(saved))])
    continuous = run("continuous", ["--initialize-from", str(args.checkpoint)])
    import mlx.core as mx

    equality = {}
    for name in ("adapters.safetensors", "optimizer.safetensors"):
        a = mx.load(str(continuous / "checkpoints/3" / name))
        b = mx.load(str(resumed / "checkpoints/3" / name))
        assert a.keys() == b.keys()
        error = max(float(mx.max(mx.abs(a[k] - b[k])).item()) for k in a)
        assert error == 0, (name, error)
        equality[name] = {"tensors": len(a), "max_error": error}

    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    assert rows(continuous / "rollouts.jsonl") == rows(first / "rollouts.jsonl") + rows(
        resumed / "rollouts.jsonl"
    )
    report = {
        "passed": True,
        "scope": "Real trainer CLI, SIGTERM checkpoint at a completed group, fresh-process exact Adam and seeded-mixture continuation; diagnostic fixture only",
        "interrupted_checkpoint": saved,
        "tensor_equality": equality,
        "rollouts_exact": True,
    }
    report["source_hashes"] = {
        str(path.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in [
            Path(__file__).resolve().with_name(name)
            for name in (
                "train_rl.py",
                "exploration.py",
                "rewards.py",
                "rl_policy.py",
                "batched_policy.py",
                "rl_scoring.py",
                "rl_batched_scoring.py",
                "validate_exploration_resume.py",
            )
        ]
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
