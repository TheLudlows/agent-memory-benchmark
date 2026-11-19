"""Publish complete JSON files for readers in other processes."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

_REPLACE_DELAYS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


def is_windows_file_contention(error: PermissionError) -> bool:
    # Access denied is also reported when an open Windows handle blocks replacement.
    return getattr(error, "winerror", None) in {5, 32, 33}


def _replace_with_retry(source: Path, target: Path) -> None:
    for delay in _REPLACE_DELAYS:
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            if not is_windows_file_contention(exc):
                raise
            sleep(delay)
    source.replace(target)


def write_json(path: Path, data: dict, *, ensure_ascii: bool = False, indent: int | None = None) -> None:
    temporary = None
    try:
        # The temporary file is private and on the same volume as the destination.
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.stem}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=ensure_ascii, indent=indent)
        _replace_with_retry(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError as exc:
                if not is_windows_file_contention(exc):
                    raise
