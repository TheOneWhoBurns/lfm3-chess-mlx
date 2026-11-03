import bisect
import json
import hashlib
import mmap
from collections import OrderedDict
from pathlib import Path


class IndexedCorpus:
    def __init__(self, root, split, max_open_shards=4):
        self.root = Path(root)
        self.split = split
        self.max_open_shards = max_open_shards
        manifest = json.loads((self.root / "manifest.json").read_text())
        if not manifest["complete"]:
            raise ValueError("Corpus must have a complete frozen manifest")
        self.metadata = manifest["splits"][split]
        self.shards = self.metadata["shards"]
        self.ends = []
        total = 0
        for shard in self.shards:
            total += shard["positions"]
            self.ends.append(total)
        if total != self.metadata["positions"] or max_open_shards < 1:
            raise ValueError("Invalid corpus shard manifest or file-handle budget")
        self.handles = OrderedDict()
        self.verified = set()

    def __len__(self):
        return self.metadata["positions"]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[shard_index - 1] if shard_index else 0)
        if shard_index not in self.handles:
            shard = self.shards[shard_index]
            data = (self.root / self.split / shard["file"]).open("rb")
            index_file = (self.root / self.split / shard["index"]).open("rb")
            if shard_index not in self.verified:
                data_hash = hashlib.file_digest(data, "sha256").hexdigest()
                index_hash = hashlib.file_digest(index_file, "sha256").hexdigest()
                if data_hash != shard["sha256"] or index_hash != shard["index_sha256"]:
                    index_file.close()
                    data.close()
                    raise ValueError("Frozen corpus shard checksum mismatch")
                self.verified.add(shard_index)
            offsets = mmap.mmap(index_file.fileno(), 0, access=mmap.ACCESS_READ)
            if len(offsets) != shard["positions"] * 8:
                offsets.close()
                index_file.close()
                data.close()
                raise ValueError("Shard index length does not match manifest")
            self.handles[shard_index] = (data, index_file, offsets)
            while len(self.handles) > self.max_open_shards:
                _, (old_data, old_index, old_offsets) = self.handles.popitem(last=False)
                old_offsets.close()
                old_index.close()
                old_data.close()
        self.handles.move_to_end(shard_index)
        data, _, offsets = self.handles[shard_index]
        offset = int.from_bytes(offsets[local * 8 : (local + 1) * 8], "little")
        data.seek(offset)
        return json.loads(data.readline())

    def close(self):
        for data, index_file, offsets in self.handles.values():
            offsets.close()
            index_file.close()
            data.close()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
