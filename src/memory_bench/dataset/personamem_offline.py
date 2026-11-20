"""PersonaMem v1 with the exact history prefix from each published question."""

import re
from bisect import bisect_right
from collections import defaultdict
from functools import cached_property

from ..data_files import data_path, read_json
from ..models import Document, Query
from .personamem import PersonaMemDataset

TOTAL_QUERIES = {"32k": 589, "128k": 2727, "1M": 2674}
_TURN_HEADER = re.compile(r"(?:^|\n\n)\[(SYSTEM|USER|ASSISTANT)\] ")


class OfflinePersonaMemDataset(PersonaMemDataset):
    isolation_unit = "user"
    incremental_history = True

    def __init__(self, split: str) -> None:
        if split not in self.splits:
            raise ValueError(f"Unknown PersonaMem split: {split}")
        self._split = split

    @cached_property
    def _history_index(self) -> dict:
        return read_json(data_path("offline", "personamem", f"{self._split}.json.gz"))

    def categories(self, split: str) -> list[str]:
        return sorted({q.meta["question_type"] for q in self.load_queries(split)})

    @staticmethod
    def _export(split: str, filename: str):
        return read_json(data_path("personamem", split, f"{filename}.json.gz"))

    def load_queries(
        self, split: str, category: str | None = None, limit: int | None = None,
    ) -> list[Query]:
        cutoffs = self._history_index["questions"]
        rows = []
        for row in self._export(split, "queries"):
            if category and row["meta"]["question_type"] != category:
                continue
            row["meta"] = {**row["meta"], "context_end": cutoffs[row["id"]]}
            # The original cutoff counts turns. The exported inferred date may
            # refer to a sentence inside the session after that cutoff.
            row["meta"].pop("query_timestamp", None)
            rows.append(Query(**row))
        return rows[:limit] if limit else rows

    def load_documents(
        self, split: str, category: str | None = None, limit: int | None = None,
        ids: set[str] | None = None, user_ids: set[str] | None = None,
    ) -> list[Document]:
        cuts: dict[str, set[int]] = defaultdict(set)
        for q in self.load_queries(split):
            cuts[q.user_id].add(q.meta["context_end"])
        boundaries = {uid: sorted(cutoffs) for uid, cutoffs in cuts.items()}

        documents = []
        sessions = self._history_index["sessions"]
        for row in self._export(split, "documents"):
            doc_id, uid = row["id"], row["user_id"]
            if uid not in boundaries or (user_ids is not None and uid not in user_ids):
                continue
            if ids is not None and doc_id not in ids:
                continue
            start, *omitted = sessions[doc_id]
            parts = _TURN_HEADER.split(row["content"])
            turns = list(zip(parts[1::2], parts[2::2], strict=True))
            stop = start + len(turns) + len(omitted)
            cuts_for_session = boundaries[uid]
            first = bisect_right(cuts_for_session, start)
            last = bisect_right(cuts_for_session, stop)
            segment_start = start
            for end in sorted(set(cuts_for_session[first:last]) | {stop}):
                if end > cuts_for_session[-1]:
                    break
                left = segment_start - start - bisect_right(omitted, segment_start - 1)
                right = end - start - bisect_right(omitted, end - 1)
                segment = turns[left:right]
                segment_start = end
                if not segment:
                    continue
                documents.append(Document(
                    id=f"{doc_id}:{end}", user_id=uid,
                    content="\n\n".join(f"[{role}] {text}" for role, text in segment),
                    messages=[{"role": role.lower(), "content": text} for role, text in segment],
                ))
        return documents[:limit] if limit and ids is None else documents

    @staticmethod
    def document_history_position(document: Document) -> int:
        return int(document.id.rsplit(":", 1)[1])

    @staticmethod
    def query_history_position(query: Query) -> int:
        return query.meta["context_end"]
