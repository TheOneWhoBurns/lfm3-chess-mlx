import hashlib
import json
import sqlite3
from pathlib import Path


def oracle_identity(oracle):
    identity = dict(oracle.identity)
    with Path(__file__).with_name("oracle.py").open("rb") as handle:
        identity["oracle_source_sha256"] = hashlib.file_digest(
            handle, "sha256"
        ).hexdigest()
    return identity


def oracle_key(oracle):
    return hashlib.sha256(
        json.dumps(oracle_identity(oracle), sort_keys=True).encode()
    ).hexdigest()


class SharedOracleCache:
    def __init__(self, path, oracle):
        self.path = Path(path).resolve()
        self.key = oracle_key(oracle)
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        row = self.connection.execute(
            "SELECT metadata FROM engines WHERE id=?", (self.key,)
        ).fetchone()
        if row is None or json.loads(row[0]) != oracle_identity(oracle):
            self.connection.close()
            raise ValueError(
                "Shared oracle cache does not contain the requested engine configuration and source version"
            )

    def get(self, fen):
        row = self.connection.execute(
            "SELECT analysis FROM labels WHERE engine=? AND fen=?", (self.key, fen)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.connection.close()
