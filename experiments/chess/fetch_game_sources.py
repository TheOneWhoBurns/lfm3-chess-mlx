import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import requests


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month) for month in args.months):
        parser.error("Months must be YYYY-MM")
    args.output.mkdir(parents=True, exist_ok=True)
    base = "https://database.lichess.org/standard/"
    response = requests.get(base + "sha256sums.txt", timeout=(15, 60))
    response.raise_for_status()
    checksums = {
        line.split()[1]: line.split()[0] for line in response.text.splitlines()
    }
    (args.output / "upstream-sha256sums.txt").write_text(response.text)
    for month in args.months:
        name = f"lichess_db_standard_rated_{month}.pgn.zst"
        expected = checksums[name]
        destination = args.output / name
        metadata_path = args.output / f"{name}.json"
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError(f"Existing source checksum mismatch: {destination}")
            print("verified existing", name, flush=True)
            continue
        partial = args.output / f"{name}.partial"
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with requests.get(
            base + name, headers=headers, stream=True, timeout=(15, 60)
        ) as stream:
            stream.raise_for_status()
            if offset and (
                stream.status_code != 206
                or not stream.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                )
            ):
                raise ValueError("Server did not honor a safe partial-download resume")
            remaining = int(stream.headers.get("Content-Length", 0))
            if (
                not remaining
                or shutil.disk_usage(args.output).free < remaining + 5 * 1024**3
            ):
                raise ValueError("Source size unknown or insufficient free disk space")
            metadata = {
                "url": base + name,
                "license": "CC0",
                "license_source": "https://database.lichess.org/",
                "expected_sha256": expected,
                "bytes": offset,
                "expected_bytes": offset + remaining,
                "complete": False,
            }
            started, last_print = time.monotonic(), 0
            with partial.open("ab" if offset else "wb") as handle:
                for chunk in stream.iter_content(chunk_size=4 * 1024**2):
                    handle.write(chunk)
                    metadata["bytes"] += len(chunk)
                    if time.monotonic() - last_print >= 10:
                        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                        print(
                            name,
                            metadata["bytes"],
                            "/",
                            metadata["expected_bytes"],
                            flush=True,
                        )
                        last_print = time.monotonic()
            actual = digest(partial)
            if actual != expected:
                raise ValueError(f"Download checksum mismatch for {name}: {actual}")
            partial.rename(destination)
            metadata.update(
                complete=True, sha256=actual, seconds=time.monotonic() - started
            )
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print("verified", name, actual, flush=True)


if __name__ == "__main__":
    main()
