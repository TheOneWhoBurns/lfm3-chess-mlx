import hashlib
import json
import random
from pathlib import Path

import numpy as np

from indexed_corpus import IndexedCorpus


class OrderedCorpus:
    def __init__(self, root, split, seed):
        self.corpus = IndexedCorpus(root, split)
        self.order = np.random.default_rng(seed).permutation(len(self.corpus))

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.corpus[int(i)] for i in self.order[index]]
        return self.corpus[int(self.order[index])]

    def close(self):
        self.corpus.close()


def corpus_signature(root, split):
    root = Path(root)
    path = root / f"{split}.jsonl"
    if not path.exists():
        path = root / "manifest.json"
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_positions(root, split, seed):
    root = Path(root)
    path = root / f"{split}.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        random.Random(seed).shuffle(rows)
        return rows, "legacy_jsonl_python_shuffle"
    return OrderedCorpus(root, split, seed), "indexed_jsonl_pcg64_permutation"
