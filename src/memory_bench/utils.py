import functools
import lzma
from base64 import b64decode
from pathlib import Path

import tiktoken

CHUNK_SIZE = 512  # tokens
_TOKENIZER_PATH = Path(__file__).with_name("assets") / "cl100k_base" / "cl100k_base.tiktoken.xz"


@functools.lru_cache(maxsize=1)
def _enc():
    # Use the cl100k_base definition from tiktoken 0.12.0 with its bundled vocabulary.
    with lzma.open(_TOKENIZER_PATH, "rb") as vocab:
        ranks = {b64decode(token): int(rank) for token, rank in map(bytes.split, vocab)}
    return tiktoken.Encoding(
        "cl100k_base",
        pat_str=(
            r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"
            r"| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
        ),
        mergeable_ranks=ranks,
        special_tokens={
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    )


def count_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into token-sized chunks, decoded back to strings."""
    enc = _enc()
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= size:
        return [text]
    return [enc.decode(tokens[i:i + size]) for i in range(0, len(tokens), size)]
