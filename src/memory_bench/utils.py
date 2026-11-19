import functools
import hashlib
import lzma
from base64 import b64decode

import tiktoken

from .data_files import data_path

CHUNK_SIZE = 512  # tokens


@functools.lru_cache(maxsize=1)
def _enc():
    # Same pattern, special tokens, and BPE ranks as tiktoken's cl100k_base.
    path = data_path("offline", "tokenizer", "cl100k_base.tiktoken.xz")
    with lzma.open(path, "rb") as stream:
        contents = stream.read()
    if hashlib.sha256(contents).hexdigest() != "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7":
        raise ValueError(f"Corrupt bundled tokenizer data: {path}")
    ranks = {}
    for line in contents.splitlines():
        token, rank = line.split()
        ranks[b64decode(token)] = int(rank)
    return tiktoken.Encoding(
        "cl100k_base",
        pat_str=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s",
        mergeable_ranks=ranks,
        special_tokens={
            "<|endoftext|>": 100257, "<|fim_prefix|>": 100258, "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260, "<|endofprompt|>": 100276,
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
