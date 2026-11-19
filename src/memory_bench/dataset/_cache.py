import errno
import os
from contextlib import contextmanager
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen
from uuid import uuid4


def dataset_cache_dir(name: str) -> Path:
    """Return .datasets/<name> at the project root, creating it if needed."""
    # Walk up from this file until we find pyproject.toml
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            path = parent / ".datasets" / name
            path.mkdir(parents=True, exist_ok=True)
            return path
    if (Path.cwd() / "pyproject.toml").is_file():
        path = Path.cwd() / ".datasets" / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    # Fallback: next to this file
    path = here.parent / ".datasets" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def staging_file(path: Path):
    """Publish the dataset cache only after a complete download."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_or_download(dataset: str, filename: str, url: str, sha256: str | None = None) -> Path:
    """Resolve a cached file, with pinned downloads available as an opt-in."""
    path = dataset_cache_dir(dataset) / filename
    if path.is_file():
        return path
    if os.environ.get("AMB_ALLOW_DATASET_DOWNLOADS") != "1":
        raise FileNotFoundError(
            errno.ENOENT, f"评测数据缺失。请使用完整的 AMB 数据包或提前准备本地文件（{dataset}/{filename}）。",
            path,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with staging_file(path) as temporary:
        with urlopen(url, timeout=90) as source, temporary.open("wb") as target:
            copyfileobj(source, target)
        if sha256:
            with temporary.open("rb") as source:
                if file_digest(source, "sha256").hexdigest() != sha256:
                    raise ValueError(f"Dataset download checksum failed for {dataset}/{filename}")
    return path
