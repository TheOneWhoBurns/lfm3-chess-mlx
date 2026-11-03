# Reproducing a new run

Run these commands from the repository root in a dedicated Python 3.12
environment on Apple silicon. They are instructions, not an automatic pipeline.
Downloading archives, labeling millions of positions and model training are
substantial explicit compute steps.

## 1. Install the recorded environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
export PYTHONPATH="$PWD"
```

Install Stockfish 19 separately and ensure `stockfish` is on `PATH`. The scripts
below pass the resolved executable explicitly. Compare its identity and settings
in each run manifest; a different engine binary defines a different reference.

The checked local environment was Python 3.12.13, MLX 0.32.4, Transformers 5.22.1
and PyTorch 2.15.0. `requirements-local.txt` preserves the observed package pins.
The runtime uses an independent version fallback so installation also works from
a source snapshot without upstream Git history. A clean dependency installation
has not been re-executed as part of preparing this source snapshot.

## 2. Obtain and convert the model

Review Liquid AI's model terms before downloading. The following explicitly
downloads the pinned original instruction-tuned model and writes a new local
8-bit conversion. It does not upload anything.

```bash
mkdir -p .artifacts/models
python - <<'PY'
from mlx_lm.convert import convert
convert(
    hf_path="LiquidAI/LFM3-1.7B",
    revision="ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7",
    mlx_path=".artifacts/models/lfm3-1.7b-8bit",
    quantize=True,
    q_bits=8,
    q_group_size=64,
    dtype="float32",
    trust_remote_code=False,
)
PY
```

The chess loader also sets computation to FP32. Runtime/reference examples in
`examples/` are optional further checks; reference checks that load the official
PyTorch implementation explicitly use pinned remote modeling code. They can
allocate significant memory and should run separately from training.

## 3. Build and audit a fresh disjoint corpus

```bash
python experiments/chess/fetch_game_sources.py \
  --months 2016-06 2017-01 --output .artifacts/sources

python experiments/chess/build_position_corpus.py ingest \
  --registry .artifacts/positions.sqlite \
  --sources .artifacts/sources/lichess_db_standard_rated_2016-06.pgn.zst \
            .artifacts/sources/lichess_db_standard_rated_2017-01.pgn.zst

python experiments/chess/build_position_corpus.py export \
  --registry .artifacts/positions.sqlite --output .artifacts/corpus \
  --train 2000000 --valid 50000 --test 100000

python experiments/chess/audit_position_corpus.py \
  --corpus .artifacts/corpus --output .artifacts/corpus-audit
```

The downloader verifies upstream SHA256 checksums. The original run additionally
passed `--exclude` paths for earlier prototype positions; these datasets are not
distributed. Include your own earlier development data in `--exclude` when
applicable. This rebuild retains the disjoint construction rules but does not
claim the historical manifest hash or identical selected positions. The exporter
fails if it cannot meet the requested quotas.

Keep the test split sealed while selecting the recipe. The commands below use
development positions only.

## 4. Label development positions and establish a baseline

```bash
python experiments/chess/label_position_corpus.py \
  --corpus .artifacts/corpus --split valid \
  --database .artifacts/oracle.sqlite --report .artifacts/oracle-report.json \
  --workers 4 --nodes 40000 --engine "$(command -v stockfish)"

python experiments/chess/evaluate_policy.py \
  --model .artifacts/models/lfm3-1.7b-8bit \
  --data .artifacts/corpus --output .artifacts/baseline \
  --split valid --positions 50000 --encoder semantic --batch-size 1 \
  --oracle-cache .artifacts/oracle.sqlite \
  --engine "$(command -v stockfish)" --nodes 40000
```

Preserve this checkout, converted model, corpus and manifests before training.
The report has separate format, legality and exact-match metrics. The baseline
above is a new baseline with the packaged current source; it is not a rerun of
the historical frozen evaluator whose results are summarized in `evidence/`.

## 5. Train a new RL-only stage

This command starts a fresh 10,000-position stage with the corrected recipe.
It does not resume the reported private 3,705-position lineage.

```bash
python experiments/chess/train_rl.py \
  --model .artifacts/models/lfm3-1.7b-8bit \
  --data .artifacts/corpus --output .artifacts/training-001 \
  --encoder semantic --groups 10000 --samples 8 --save-every 25 \
  --shared-prefix --batched-rollouts --batched-scoring \
  --temperature 1.2 --exploration-epsilon 0.08 \
  --entropy 0.01 --uniform-kl 0.02 \
  --oracle-cache .artifacts/oracle.sqlite \
  --engine "$(command -v stockfish)" --nodes 40000
```

SIGINT/SIGTERM requests a checkpoint after the current complete group. Inspect
`interrupted.json` and the checkpoint metadata before resuming. Resume uses the
same data, source, seed and options, a **new** output directory and
`--resume PATH_TO_THE_SAVED_CHECKPOINT`; it restores Adam as well as trainable
weights. Keep `--groups` as the total target, not the number of additional groups.
Changing the objective requires an explicitly new stage, not a silent resume.

The original launchd controller depended on local checkpoint paths and is not
included. These commands run in the foreground; this repository does not install
a persistent background service.

## 6. Compare under the unchanged evaluation protocol

Only after training completes, use its final checkpoint:

```bash
python experiments/chess/evaluate_policy.py \
  --model .artifacts/models/lfm3-1.7b-8bit \
  --adapter .artifacts/training-001/checkpoints/10000 \
  --data .artifacts/corpus --output .artifacts/post-training \
  --split valid --positions 50000 --encoder semantic --batch-size 1 \
  --oracle-cache .artifacts/oracle.sqlite \
  --engine "$(command -v stockfish)" --nodes 40000

python experiments/chess/compare_policy.py \
  --before .artifacts/baseline --after .artifacts/post-training \
  --output .artifacts/comparison
```

The comparator rejects incompatible manifests, incomplete coverage, changed
boards and mismatched decoding. It reports paired results and game-cluster
bootstrap uncertainty. Evaluating the test split is a separate decision after
locking the recipe.

## Shared-prefix benchmark

The historical 3.04× number used an earlier objective. Its matching source
closure is preserved under `experiments/shared_prefix_v1/` rather than silently
combining the old benchmark driver with the changed trainer API.

To run that computation comparison on a newly built **training** example:

```bash
mkdir -p .artifacts/benchmark-input
python - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, "experiments/chess")
from indexed_corpus import IndexedCorpus
corpus = IndexedCorpus(".artifacts/corpus", "train")
try:
    Path(".artifacts/benchmark-input/train.jsonl").write_text(
        json.dumps(corpus[0]) + "\n"
    )
finally:
    corpus.close()
PY

python experiments/shared_prefix_v1/benchmark_rl_scoring.py \
  --model .artifacts/models/lfm3-1.7b-8bit \
  --data .artifacts/benchmark-input --encoder semantic --samples 8 --gradients \
  --output .artifacts/shared-prefix-benchmark
```

This benchmark computes gradients and timing without optimizer updates. A new
board, environment or workload can yield different timings. It does not
reproduce the original fixture hash, establish end-to-end training speed, or
measure chess strength.
