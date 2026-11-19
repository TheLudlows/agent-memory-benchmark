from contextlib import contextmanager
from pathlib import Path
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
