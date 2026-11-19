"""LifeBench 2.0 Chinese multi-source dataset, pinned to the official release."""

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from ..data_files import data_path, read_json
from ._cache import cached_or_download
from .base import Dataset
from ..models import Document, Query

SPLITS = ["zh"]
TOTAL_QUERIES = 3380
_RAW_URL = (
    "https://raw.githubusercontent.com/1754955896/LifeBench/"
    "b4adbe3d0afa66efa4ab362e9d67160089f860dd/life_bench_data/version2/data/"
)
_USERS = (
    "fenghaoran", "leimingxuan", "lumingqiang", "maxiulan", "songyajing",
    "sunyuwei", "yemingxuan", "yinhao", "yuxiaowei", "yuxiaowen",
)
_SOURCES = {
    "persona.json": "用户档案", "daily_event.json": "日常事件",
    "phone_data/agent_chat.json": "助手对话", "phone_data/calendar.json": "日历",
    "phone_data/call.json": "通话记录", "phone_data/contact.json": "联系人",
    "phone_data/fitness_health.json": "运动健康", "phone_data/note.json": "笔记",
    "phone_data/photo.json": "照片记录", "phone_data/push.json": "通知",
    "phone_data/sms.json": "短信",
}
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_DOC_CHAR_LIMIT = 6000


def _day(record: dict) -> str:
    value = record.get("datetime") or record.get("date") or record.get("日期") or ""
    if isinstance(value, list):
        value = value[0] if value else ""
    match = _DAY.search(value)
    return match[0] if match else ""


def _timestamp(day: str) -> str | None:
    return f"{day}T00:00:00+08:00" if day else None


@lru_cache(maxsize=1)
def _bundled_user(user: str) -> dict:
    return read_json(data_path("offline", "lifebench-v2", f"{user}.json.xz"))


class LifeBenchV2Dataset(Dataset):
    name = "lifebench-v2"
    description = "LifeBench 2.0 Chinese multi-source memory: 10 users, 3,380 questions."
    splits = SPLITS
    isolation_unit = "user"
    query_timestamp_cutoff = True
    links = [{"label": "Dataset", "url": "https://huggingface.co/datasets/C1754955896/Lifebenchv2.0"}]

    def _read(self, user: str, file: str) -> list | dict:
        local_root = os.environ.get("LIFEBENCH_V2_DATA_PATH")
        if local_root:
            return read_json(Path(local_root).expanduser() / user / file)
        if data_path("offline", self.name, f"{user}.json.xz").is_file():
            bundled = _bundled_user(user)
            if file in bundled:
                return bundled[file]
        filename = f"{user}/{file}"
        return read_json(cached_or_download(self.name, filename, _RAW_URL + filename))

    def load_queries(
        self, split: str, category: str | None = None, limit: int | None = None,
    ) -> list[Query]:
        if split not in self.splits:
            raise ValueError(f"Unknown LifeBench 2.0 split: {split}")
        queries = []
        for user in _USERS:
            for index, row in enumerate(self._read(user, "QA_all/QA.json"), 1):
                types = row["question_type"]
                if category and category not in types:
                    continue
                queries.append(Query(
                    id=f"{user}-{index}", query=row["question"], user_id=user,
                    gold_ids=[], gold_answers=[row["answer"]],
                    meta={
                        "question_types": types,
                        "abstention": "Unanswerable" in types,
                        "ask_time": row["ask_time"],
                        "query_timestamp": _timestamp(row["ask_time"]),
                        "rubric": row["score_points"],
                    },
                ))
                if limit is not None and len(queries) >= limit:
                    return queries
        return queries

    def load_documents(
        self, split: str, category: str | None = None, limit: int | None = None,
        ids: set[str] | None = None, user_ids: set[str] | None = None,
    ) -> list[Document]:
        if split not in self.splits:
            raise ValueError(f"Unknown LifeBench 2.0 split: {split}")
        docs = []
        for user in _USERS:
            if user_ids is not None and user not in user_ids:
                continue
            user_docs = []
            for file, label in _SOURCES.items():
                records = self._read(user, file)
                if isinstance(records, dict):
                    # The profile stores each relationship history in a single list.
                    # Keep individual relationships intact when splitting the profile.
                    profile = {key: value for key, value in records.items() if key != "relation"}
                    records = [profile, *({"relation": relation} for relation in records.get("relation", []))]
                by_day: dict[str, list[str]] = defaultdict(list)
                for record in records:
                    by_day[_day(record)].append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                source = Path(file).stem
                for day, lines in by_day.items():
                    header = f"来源：{label}" + (f"\n日期：{day}" if day else "")
                    content = header
                    index = 1
                    for line in lines:
                        if content != header and len(content) + len(line) + 1 > _DOC_CHAR_LIMIT:
                            user_docs.append(self._document(user, source, day, index, content))
                            content = header
                            index += 1
                        content += "\n" + line
                    user_docs.append(self._document(user, source, day, index, content))
            docs.extend(doc for doc in sorted(user_docs, key=lambda doc: (doc.timestamp or "", doc.id))
                        if ids is None or doc.id in ids)
        return docs[:limit] if limit is not None else docs

    @staticmethod
    def _document(user: str, source: str, day: str, index: int, content: str) -> Document:
        return Document(
            id=f"{user}-{source}-{day or 'undated'}-{index}", content=content,
            user_id=user, timestamp=_timestamp(day),
        )

    def categories(self, split: str) -> list[str]:
        return sorted({label for query in self.load_queries(split) for label in query.meta["question_types"]})

    def category_type(self, split: str, category: str) -> str:
        return "query"

    def dataset_stats(self, console, **_) -> None:
        queries = self.load_queries("zh")
        counts = Counter(label for query in queries for label in query.meta["question_types"])
        console.print(f"LifeBench 2.0 (zh): {len(queries):,} questions across {len(_USERS)} users")
        for label, count in sorted(counts.items()):
            console.print(f"  {label}: {count:,}")

    def supports_oracle(self) -> bool:
        return False

    def get_result_categories(self, meta: dict) -> dict[str, list[str]]:
        return {"题型": meta["question_types"]}

    def should_judge_empty_context(self, meta: dict) -> bool:
        return bool(meta.get("abstention"))

    def build_rag_prompt(self, query, context, task_type, split, category=None, meta=None) -> str:
        ask_time = (meta or {}).get("ask_time", "")
        return (
            "根据提供的资料，用中文直接回答问题。只使用提问日期及以前的信息，"
            "注意在不同来源之间核对细节。如果证据不足，说明无法确定。\n\n"
            f"提问日期：{ask_time}\n\n资料：\n{context}\n\n问题：{query}"
        )

    def get_judge_prompt_fn(self, category: str | None, meta: dict):
        def prompt(query: str, gold_answers: list[str], answer: str) -> str:
            instructions = (
                "这题证据不足。只有回答明确指出无法确定且没有编造结论时才算正确。"
                if meta["abstention"]
                else "以参考答案为准，按评分要点核对关键事实。可以有同义表达；缺失或矛盾的关键事实判错。"
            )
            rubric = json.dumps(meta["rubric"], ensure_ascii=False)
            return (
                f"判断回答是否正确，并给出理由。{instructions}\n"
                f"问题：{query}\n参考答案：{gold_answers[0]}\n评分要点：{rubric}\n回答：{answer}"
            )
        return prompt
