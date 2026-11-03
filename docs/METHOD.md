# Method and experimental contract

## Model and representation

- Instruction-tuned `LiquidAI/LFM3-1.7B`, revision
  `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`.
- Local 8-bit affine conversion, group size 64; FP32 computation in the chess
  runtime. The packed weights remain 8-bit.
- A semantic board encoder consumes an 8×8×13 piece/empty tensor, an 8×8×1
  en-passant plane and seven game-state features. Four vectors per square encode
  color, piece kind, file and rank. A fixed side-to-move label and trainable state
  vector bring the input to 262 slots.
- Native word embeddings initialize the semantic mappings. The board is supplied
  through input embeddings, not serialized through the text tokenizer.
- The encoder and rank-8 LoRA adapters on the last 16 layers learn through rewards.
  The output vocabulary, language head and JSON/UCI interface remain intact.

FEN state does not include repetition history. This is next-move prediction,
not a complete rules-aware game agent.

## Data construction

The source archives are Lichess standard rated games from June 2016 and January
2017. The recorded ingestion scanned 6,136,419 and 10,680,708 games respectively.
Human moves establish reachable board states; they are not supervised targets.

Complete legal mainline UCI sequences are hashed so identical game trajectories
share a split. The first four canonical FEN fields define board-state duplicates;
cross-split collisions are excluded. The historical run also excluded earlier
prototype positions. Development/test splits have at most two positions per game
and balanced side-to-move counts, with opening/middlegame/endgame quotas.

The completed audit covers 2M training, 50K development and 100K test positions.
It reports zero position overlap, zero trajectory overlap, zero overlap with
the earlier excluded positions, and no oracle targets in model inputs. The test
set was audited for hygiene without querying a model or revealing test labels.

## Reward and optimization

Stockfish 19 uses 40,000 nodes, one thread, 64 MB hash and MultiPV=1. Hash state
is cleared and each search starts a fresh game. References are cached under an
engine/settings/source namespace. The oracle is consulted after generation.

| Response | Reward |
| :--- | ---: |
| Malformed or unfinished | −2 |
| Illegal | −1 to 0, from partial-legality shaping |
| Legal nonmatch | 1 to 2, from oracle-assessed move quality |
| Exact oracle match | 10: base 2 + bonus 8 |

Ordinary advantages use leave-one-out centering, fixed scale 2 and clipping to
[−2, 2]. Illegal/incomplete samples retain advantage at most −1 after centering.
The exact-match bonus is added afterward as +4 advantage units, so equal rewards
do not erase the match signal. There is no per-group standard-deviation division.

The current recipe samples eight responses per board at temperature 1.2 with an
8% uniform mixture over syntax-allowed tokens. Likelihood ratios use this actual
behavior distribution. The summed PPO surrogate uses clip 0.2; entropy weight is
0.01 and uniform-to-policy KL weight is 0.02 on branching token positions.

Matched and ordinary gradient components are accumulated separately. The match
component retains at least 80% of their component-norm sum before Adam, when
matches occur. This does not attribute 80% of Adam's final parameter delta to
matches. Global gradient norm is clipped to 1, parameter-update RMS is bounded
by 1e-5, and each sampled response's log-probability change is bounded by 0.5.
Matched groups must increase mean untempered matched likelihood; all-illegal
groups must decrease mean untempered sampled likelihood. Failed proposals roll
back parameters and optimizer state. These are sampled-response guards, not a
global KL guarantee.

## Evaluation and recovery

Evaluation commits one greedy attempt per position, with a 48-token budget and
the same JSON schema. It reports format, legality and exact Stockfish agreement
separately, including color and phase slices. No engine information, legality
filter, retry or repair assists generation. Paired comparisons require matching
source/model/data/protocol hashes and use game-cluster bootstrap uncertainty.

Checkpoints contain encoder/LoRA weights, Adam state, lineage and checksums.
Atomic publication follows synced writes. SIGINT/SIGTERM requests stop after a
complete group and preserve its accepted updates. Strict resume restores
optimizer state; starting a new objective from learned weights records a new
stage with fresh Adam moments.

The corrected November stage retained the previous checkpoint at position 3,500
and reset Adam for the changed objective. It is currently paused after 205 more
groups, at overall position 3,705. A fresh run of the published current recipe is
not that historical continuation.
