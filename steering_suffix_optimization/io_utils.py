from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ArtifactWriter:
    def __init__(self, output: Path, commit: Callable[[], None] | None = None):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.commit = commit or (lambda: None)

    def json(self, name: str, payload: Any) -> None:
        atomic_json(self.output / name, payload)
        self.commit()

    def jsonl(self, name: str, row: Any) -> None:
        with (self.output / name).open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.commit()
