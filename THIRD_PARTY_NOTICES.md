# Attribution and third-party notices

## Included source

The `mlx_lm` runtime is derived from `ml-explore/mlx-lm`, commit
`736ab0d8c1d0edc79c3d798044fe6904e2fa23be`.

The original **MIT License, Copyright © 2023 Apple Inc.** is preserved in
`LICENSE`. Existing source copyright headers and `ACKNOWLEDGMENTS.md` are retained.
The local LFM3 architecture, JSON-schema integration and chess experiments are
identified separately in `SOURCE_PROVENANCE.json`; the entire upstream runtime
must not be represented as original project work.

## External model, data and dependencies

These components are referenced or installed separately; their implementations,
weights, datasets and binaries are not vendored in this repository.

| Component | Attribution / license information |
| :--- | :--- |
| LFM3-1.7B | Liquid AI. Obtain weights from its model repository at the pinned revision and review that repository's model terms. This repository does not distribute or relicense the weights. |
| Lichess standard rated archives | Lichess; the source download records identify CC0 and link to `database.lichess.org`. Raw PGN and derived position datasets are not included. |
| MLX and MLX Metal | Apple / MLX contributors; installed metadata identifies MIT. |
| python-chess / chess | Niklas Fiekas and contributors; installed metadata identifies GPL-3.0-or-later. Imported as an external dependency; no library source is copied here. |
| Stockfish | The Stockfish developers. External engine executable, not bundled; see its distribution's GPLv3 license and source obligations. |
| Outlines Core | dottxt / contributors; installed metadata identifies Apache-2.0. External dependency. |

Other dependencies retain their own license terms. The MIT notice for this source
tree does not relicense those dependencies or remove obligations that apply when
redistributing a combined package. This staging copy is a source repository,
not a distribution of the full installed Python environment or engine binaries.

Reference links: [MLX LM](https://github.com/ml-explore/mlx-lm),
[Liquid AI model](https://huggingface.co/LiquidAI/LFM3-1.7B),
[Lichess](https://database.lichess.org/),
[Stockfish license](https://github.com/official-stockfish/Stockfish/blob/master/Copying.txt),
[python-chess](https://github.com/niklasf/python-chess),
[Outlines Core](https://github.com/dottxt-ai/outlines-core).
