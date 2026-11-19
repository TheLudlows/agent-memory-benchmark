"""Read files from the checked-in evaluation data bundle."""

import gzip
import json
import lzma
from pathlib import Path


def data_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2].joinpath("data", *parts)


def read_json(path: Path):
    opener = {".gz": gzip.open, ".xz": lzma.open}.get(path.suffix, open)
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)
