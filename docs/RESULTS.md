# Results and limits

Historical evidence was inspected on November 3, 2026. Files in `evidence/` are
small, selected summaries of retained local reports. Each records its original
report SHA256 and extraction scope; they are not newly executed benchmarks.

## Completed observations

| Observation | Result | Interpretation |
| :--- | :--- | :--- |
| Corpus audit | 2M train / 50K dev / 100K test; zero position or trajectory overlap | Dataset construction and split hygiene |
| Original policy on 50K dev boards | 100% valid format; 6.744% legal; 0.472% exact oracle matches | Baseline only; no claim of trained strength |
| Shared-prefix gradient benchmark | 11.983483 s → 3.937871 s; 3.043137× | Specific workload, not full training throughput |
| Gradient parity | Relative L2 error 2.919088e-5; loss error 3.767014e-5 | Agreement of two computation paths |
| Sequence scoring | 3.602644×; maximum log-probability error 5.149841e-5 | Same limited benchmark |
| Interrupted/resumed diagnostic | Exact rollouts, 215 adapter tensors and 432 Adam tensors | Recovery correctness on the checked fixture |

The timing report explicitly notes that concurrent workloads make timing
indicative. It is not a multi-seed performance study. Its historical helper
sources are included under `experiments/shared_prefix_v1/`; the six recorded
source hashes match those files. The fixture data is not distributed.

## Latest saved training status

On November 2, the corrected run saved group 205, overall position 3,705, with
190 accepted updates in that stage. It descends from a prior 3,500-position
checkpoint with 467 updates. Saved adapter and optimizer hashes were verified
at shutdown, no accepted updates were discarded, and the experiment was unloaded
at the owner's request. This is a saved status record, not a live process check.

The planned 10,000-position stage has not completed. The full post-training
50,000-position evaluation and paired comparison are pending. The 100,000-position
test set remains sealed. There is no established Elo, 95% legality result,
broad chess-strength gain, general-capability retention claim or explanation
faithfulness result from this ongoing RL experiment.

Earlier supervised and small development pilots are not part of the current
RL-only lineage. Their metrics are not presented as results of the current run.

## Evidence index

- [Corpus audit](../evidence/corpus-audit.json)
- [Baseline summary](../evidence/baseline.json)
- [Shared-prefix computation benchmark](../evidence/shared-prefix-benchmark.json)
- [Exact resume diagnostic](../evidence/resume-validation.json)
- [Latest saved status](../evidence/latest-status.json)
- [Model and method provenance](../evidence/model-and-method.json)
- [Packaged source provenance](../SOURCE_PROVENANCE.json)
