"""
LifeBench dataset (https://github.com/1754955896/LifeBench).
Paper: https://arxiv.org/abs/2603.03781 (ICLR 2026 Memory Agent Workshop)

A benchmark for long-horizon multi-source personalized memory evaluation.
Simulates a full year of daily life for 10 fictional users with rich
multi-modal digital traces (SMS, calls, photos, calendar, fitness, etc.)
and 2,003 English QA pairs derived from them.

The English questions and sessions are bundled. LIFEBENCH_DATA_PATH can
point at a local copy of the original our_en.json.

Structure
---------
10 users × ~200 QA pairs = 2,003 queries. Single "en" split.

The data uses a LoCoMo-compatible format:
  - sample_id: user name
  - conversation: dict with session_1..session_364 (one per calendar day),
                  each with session_N_date_time timestamp
  - qa: list of {question, answer, evidence, category}

Documents  = one per session per user
             ID = "{sample_id}_{session_key}" (e.g. "alice_session_1")

Queries    = one per QA pair per user
             gold_ids = sessions referenced in evidence

Categories (query-level, by category string):
  "0" → information-extraction   (direct fact retrieval)
  "1" → multi-hop                (reasoning over multiple sources)
  "2" → nondeclarative           (habits, preferences, emotions)
  "3" → temporal-updating        (temporal reasoning, evolving info)
  "4" → unanswerable             (questions with no answer in data)
"""
import json
import os
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..data_files import data_path, read_json
from ..models import Document, Query
from ._cache import cached_or_download
from .base import Dataset

_DATA_URL = (
    "https://raw.githubusercontent.com/1754955896/LifeBench/37be8131b602001b2662b6d02c6eccadad099182"
    "/life_bench_data/version1/locomo_format/our_en.json"
)
_DATA_SHA256 = "9f10d933a313e9c7af9a9882888c055915dcc63968d4fc642bcc50fccaf49330"

SPLITS = ["en"]
TOTAL_QUERIES = 2003

_CATEGORY_NAMES: dict[str, str] = {
    "0": "information-extraction",
    "1": "multi-hop",
    "2": "nondeclarative",
    "3": "temporal-updating",
    "4": "unanswerable",
}

_CATEGORY_LABELS: dict[str, str] = {
    "information-extraction": "Information Extraction",
    "multi-hop": "Multi-hop Reasoning",
    "temporal-updating": "Temporal & Knowledge Updating",
    "nondeclarative": "Nondeclarative Memory",
    "unanswerable": "Unanswerable",
}


class LifeBenchDataset(Dataset):
    """
    LifeBench — long-horizon multi-source personalized memory benchmark.

    Set LIFEBENCH_DATA_PATH to override the repository exports with a raw file.
    """

    name = "lifebench"
    published = True
    description = "Long-horizon multi-source personalized memory benchmark across 10 users."
    splits = SPLITS
    task_type = "open"
    isolation_unit = "user"
    links = [
        {"label": "Paper", "url": "https://arxiv.org/abs/2603.03781"},
        {"label": "GitHub", "url": "https://github.com/1754955896/LifeBench"},
    ]

    def __init__(self) -> None:
        env = os.environ.get("LIFEBENCH_DATA_PATH")
        self._local_path: Path | None = Path(env).expanduser() if env else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bundle(self, split: str, filename: str) -> Path | None:
        if split not in self.splits:
            raise ValueError(f"Unknown LifeBench split: {split}")
        path = data_path(self.name, split, filename)
        return path if self._local_path is None and path.is_file() else None

    def _load_raw(self) -> list[dict]:
        path = self._local_path or cached_or_download(self.name, "our_en.json", _DATA_URL, _DATA_SHA256)
        return read_json(path)

    @staticmethod
    def _session_keys(conv: dict) -> list[str]:
        """Return sorted session keys (e.g. ['session_1', 'session_2', ...])."""
        return sorted(
            (k for k in conv
             if k.startswith("session_") and not k.endswith("_date_time")
             and isinstance(conv[k], list)),
            key=lambda k: int(k.split("_", 1)[1]),
        )

    @staticmethod
    def _parse_date(date_string: str | None) -> str | None:
        """Parse various date formats → ISO-8601 UTC string."""
        if not date_string:
            return None
        for fmt in [
            "%I:%M %p on %d %B, %Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ]:
            try:
                dt = datetime.strptime(date_string, fmt)
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except (ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _build_dia_to_session(conv: dict, session_keys: list[str]) -> dict[str, str]:
        """Return {dia_id: session_key} for all turns in a conversation."""
        mapping: dict[str, str] = {}
        for key in session_keys:
            for turn in conv.get(key, []):
                if isinstance(turn, dict) and "dia_id" in turn:
                    mapping[str(turn["dia_id"])] = key
        return mapping

    def _sessions(self, split: str) -> Iterator[tuple[str, str, str | None, str | list]]:
        bundle = self._bundle(split, "documents.json.gz")
        if bundle:
            for row in read_json(bundle):
                user = row["user_id"]
                session = row["id"].removeprefix(f"{user}_")
                yield user, session, row["timestamp"], row["content"]
            return

        for item in self._load_raw():
            conversation = item["conversation"]
            for session in self._session_keys(conversation):
                turns = conversation[session]
                if turns:
                    timestamp = self._parse_date(conversation.get(f"{session}_date_time"))
                    yield item["sample_id"], session, timestamp, turns

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def categories(self, split: str) -> list[str] | None:
        # Doc categories: one per user (isolates document space)
        # Query categories: question types (same docs, different query filter)
        bundle = self._bundle(split, "queries.json.gz")
        if bundle:
            users = dict.fromkeys(row["user_id"] for row in read_json(bundle))
        else:
            users = [item["sample_id"] for item in self._load_raw()]
        return [*users, *_CATEGORY_NAMES.values()]

    def should_judge_empty_context(self, meta: dict) -> bool:
        return meta.get("category") == "unanswerable"

    def category_type(self, split: str, category: str) -> str:
        return "query" if category in _CATEGORY_NAMES.values() else "doc"

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        axes: dict[str, list[str]] = {}
        if meta.get("sample_id"):
            axes["User"] = [meta["sample_id"]]
        if meta.get("category"):
            axes["Question Type"] = [meta["category"]]
        return axes

    def build_rag_prompt(
        self,
        query: str,
        context: str,
        task_type: str,
        split: str,
        category: str | None = None,
        meta: dict | None = None,
    ) -> str:
        meta = meta or {}
        query_timestamp = meta.get("query_timestamp")
        date_str = ""
        if query_timestamp:
            date_str = f"\n# CURRENT DATE:\nThe question is being asked on: {query_timestamp} UTC\n"
        raw = meta.get("_raw_response")
        ctx = json.dumps(raw) if raw else context
        return f"""\
# CONTEXT:
You have access to facts and entities from a user's personal life data.
{date_str}
# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information or multiple instances of an event, say them all
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. If the answer is not explicitly stated in the memories, use logical reasoning based on the information available to answer (e.g. calculate duration of an event from different memories).

Context:

{ctx}

Question: {query}
"""

    def build_judge_prompt(self, query: str, gold_answers: list[str], answer: str) -> str:
        gold_str = gold_answers[0] if gold_answers else ""
        return f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question about a user's personal information and life events,
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The gold answer will usually be a concise answer that includes the key fact.
The generated answer might be much longer, but you should be generous with your grading — as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.
For time-related questions, the gold answer will be a specific date or period. The generated answer is CORRECT if it refers to the same date or period, even if formatted differently.
If the gold answer indicates the question is unanswerable from the data, and the generated answer also says it cannot be answered, count it as CORRECT.

Question: {query}
Gold answer: {gold_str}
Generated answer: {answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true.
"""

    def load_queries(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Query]:
        # Determine filter type
        cat_filter: str | None = None    # question-type filter (category name)
        user_filter: str | None = None   # user filter (sample_id)
        if category:
            if category in _CATEGORY_NAMES.values():
                cat_filter = category
            else:
                user_filter = category

        bundle = self._bundle(split, "queries.json.gz")
        if bundle:
            queries = [
                Query(**row) for row in read_json(bundle)
                if (user_filter is None or row["user_id"] == user_filter)
                and (cat_filter is None or row["meta"]["category"] == cat_filter)
            ]
            return queries[:limit] if limit else queries

        queries: list[Query] = []
        for item in self._load_raw():
            sample_id = item["sample_id"]
            if user_filter is not None and sample_id != user_filter:
                continue

            conv = item["conversation"]
            session_keys = self._session_keys(conv)
            dia_to_session = self._build_dia_to_session(conv, session_keys)

            # Query timestamp = date of the last session
            last_session_ts: str | None = None
            for sk in reversed(session_keys):
                last_session_ts = self._parse_date(conv.get(f"{sk}_date_time"))
                if last_session_ts:
                    break

            for qi, qa in enumerate(item.get("qa", [])):
                cat_str = str(qa.get("category", ""))
                cat_name = _CATEGORY_NAMES.get(cat_str, cat_str)
                if cat_filter is not None and cat_name != cat_filter:
                    continue

                question = qa.get("question", "")
                answer = qa.get("answer", "")
                evidence = qa.get("evidence") or []

                # Map evidence dia_ids → unique session doc IDs
                gold_session_keys = {
                    dia_to_session[str(eid)]
                    for eid in evidence
                    if str(eid) in dia_to_session
                }
                gold_ids = [f"{sample_id}_{sk}" for sk in sorted(gold_session_keys)]

                queries.append(Query(
                    id=f"{sample_id}_q{qi}",
                    query=question,
                    gold_ids=gold_ids,
                    gold_answers=[answer],
                    user_id=sample_id,
                    meta={
                        "sample_id": sample_id,
                        "category": cat_name,
                        **({"query_timestamp": last_session_ts} if last_session_ts else {}),
                    },
                ))

        if limit:
            queries = queries[:limit]
        return queries

    def load_documents(
        self,
        split: str,
        category: str | None = None,
        limit: int | None = None,
        ids: set[str] | None = None,
        user_ids: set[str] | None = None,
    ) -> list[Document]:
        documents: list[Document] = []

        # Doc-category filter: restrict to a single user
        user_filter = category if category and category not in _CATEGORY_NAMES.values() else None

        for user, session, timestamp, turns in self._sessions(split):
            if user_filter is not None and user != user_filter:
                continue
            if user_ids is not None and user not in user_ids:
                continue
            doc_id = f"{user}_{session}"
            if ids is not None and doc_id not in ids:
                continue
            documents.append(Document(
                id=doc_id, user_id=user, timestamp=timestamp,
                content=turns if isinstance(turns, str) else json.dumps(turns),
                context=f"User '{user}' — {session} ({timestamp or 'unknown date'})",
            ))
        return documents[:limit] if limit and ids is None else documents

    def dataset_stats(self, console: Console, **_) -> None:
        table = Table(title="LifeBench dataset stats")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        queries = self.load_queries("en")
        cat_counts = Counter(q.meta["category"] for q in queries)
        documents = self.load_documents("en")
        table.add_row("Users", str(len({q.user_id for q in queries} | {d.user_id for d in documents})))
        table.add_row("Sessions (docs)", str(len(documents)))
        table.add_row("QA pairs", str(len(queries)))
        for cat_name in _CATEGORY_NAMES.values():
            label = _CATEGORY_LABELS.get(cat_name, cat_name)
            table.add_row(f"  {label}", str(cat_counts.get(cat_name, 0)))
        console.print(table)
