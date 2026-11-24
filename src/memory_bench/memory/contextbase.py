import json
import os
import re
import time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx

from ..models import Document
from .base import MemoryProvider, RetrievalError
from .hindsight import _as_recall_response, _build_docs, _deduplicate_results


BENCH_ADMIN_TOKEN_ENV = "CONTEXTBASE_BENCH_ADMIN_TOKEN"


def knowledge_bank_id(run_id: str) -> str:
    return f"amb_rag_{run_id}"


class ContextBaseMemoryProvider(MemoryProvider):
    name = "contextbase"
    description = "ContextBase Retain and Recall over HTTP using pre-authorized Banks."
    kind = "cloud"
    supports_incremental_ingest = True

    def __init__(self, *, single_bank_id: str | None = None) -> None:
        url = os.environ.get("CONTEXTBASE_API_URL", "").strip()
        token = os.environ.get("CONTEXTBASE_API_TOKEN", "").strip()
        if not url or not token:
            raise ValueError("Set CONTEXTBASE_API_URL and CONTEXTBASE_API_TOKEN for ContextBase.")
        if single_bank_id is None:
            self._bank_id = os.environ.get("CONTEXTBASE_BANK_ID", "").strip()
            self._bank_map = json.loads(os.environ.get("CONTEXTBASE_BENCH_BANKS") or "{}")
            if not isinstance(self._bank_map, dict) or not all(
                isinstance(bank, str) and bank.strip() for bank in self._bank_map.values()
            ):
                raise ValueError("CONTEXTBASE_BENCH_BANKS must map isolation unit IDs to Bank IDs.")
            if bool(self._bank_id) == bool(self._bank_map):
                raise ValueError("Set either CONTEXTBASE_BANK_ID or CONTEXTBASE_BENCH_BANKS.")
        else:
            self._bank_id = single_bank_id
            self._bank_map = {}

        self._precision_documents: dict[str, dict[str, str]] = {}
        self._configured_run_id = os.environ.get("CONTEXTBASE_BENCH_RUN_ID")
        if self._configured_run_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", self._configured_run_id):
            raise ValueError("CONTEXTBASE_BENCH_RUN_ID must use 1–48 letters, digits, underscores, or hyphens.")
        self.ingestion_run_id = self._configured_run_id or uuid4().hex
        self._retain_timeout = int(os.environ.get("CONTEXTBASE_BENCH_RETAIN_TIMEOUT_S", "1800"))
        if self._retain_timeout <= 0:
            raise ValueError("CONTEXTBASE_BENCH_RETAIN_TIMEOUT_S must be positive.")
        self._max_tokens = int(os.environ.get("AMB_RECALL_MAX_TOKENS") or "32768")
        self._max_chunk_tokens = int(os.environ.get("AMB_RECALL_MAX_CHUNK_TOKENS") or "16384")
        self._types = [kind.strip() for kind in os.environ.get("AMB_RECALL_TYPES", "").split(",") if kind.strip()]
        self.concurrency = int(os.environ.get("AMB_CONCURRENCY") or str(self.concurrency))
        self.unit_concurrency = int(os.environ.get("AMB_UNIT_CONCURRENCY") or "1")
        if self.concurrency <= 0 or self.unit_concurrency <= 0:
            raise ValueError("AMB_CONCURRENCY and AMB_UNIT_CONCURRENCY must be positive.")
        self._http = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=60)

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._precision_documents.clear()
        if not unit_ids:
            raise ValueError("No isolation units matched the benchmark selection.")
        missing = unit_ids - self._bank_map.keys() if self._bank_map else set()
        if missing:
            raise ValueError(f"Missing ContextBase Bank mappings for: {', '.join(sorted(missing))}")
        if self._bank_map and len({self._bank_map[unit] for unit in unit_ids}) != len(unit_ids):
            raise ValueError("Assign a distinct ContextBase Bank to each mapped isolation unit.")
        if reset:
            self.ingestion_run_id = uuid4().hex
        elif not self._configured_run_id:
            raise ValueError("--skip-ingestion needs the original CONTEXTBASE_BENCH_RUN_ID.")
        else:
            self.ingestion_run_id = self._configured_run_id
        self._ingestion_dir = store_dir / self.name / self.ingestion_run_id
        print(f"ContextBase run ID: {self.ingestion_run_id}")

    def cleanup(self) -> None:
        self._http.close()

    def _scope(self, unit_id: str | None) -> tuple[str, list[str]]:
        if not unit_id:
            raise ValueError("ContextBase requires an isolation unit ID for Retain and Recall.")
        bank_id = self._bank_map[unit_id] if self._bank_map else self._bank_id
        return f"v1/default/banks/{quote(bank_id, safe='')}", [f"amb-run:{self.ingestion_run_id}", f"amb-unit:{unit_id}"]

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self._http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def _retained_id(self, document_id: str) -> str:
        return f"amb-{self.ingestion_run_id}-{document_id}"

    def _items_by_unit(self, documents: list[Document]) -> dict[str, list[dict]]:
        items_by_unit: dict[str, list[dict]] = {}
        for doc in documents:
            unit_id = doc.user_id
            if not unit_id:
                raise ValueError("ContextBase requires an isolation unit ID for each retained document.")
            _, tags = self._scope(unit_id)
            item = {
                "content": doc.content.replace("\x00", ""),
                "document_id": self._retained_id(doc.id),
                "metadata": {"doc_id": doc.id},
                "tags": tags,
                "update_mode": "replace",
            }
            if doc.timestamp:
                item["timestamp"] = doc.timestamp
            if doc.context:
                item["context"] = doc.context
            items_by_unit.setdefault(unit_id, []).append(item)
        return items_by_unit

    def _checkpoint(self, unit_id: str) -> Path:
        return self._ingestion_dir / f"{quote(unit_id, safe='')}.sha256"

    def _ingestion_fingerprint(self, path: str, items: list[dict]) -> str:
        payload = json.dumps([str(self._http.base_url), path, items], sort_keys=True)
        return sha256(payload.encode("utf-8")).hexdigest()

    def validate_ingestion(self, documents: list[Document]) -> None:
        items_by_unit = self._items_by_unit(documents)
        expected_by_bank: dict[str, set[str]] = {}
        for unit_id, items in items_by_unit.items():
            checkpoint = self._checkpoint(unit_id)
            path, _ = self._scope(unit_id)
            if not checkpoint.is_file():
                raise ValueError(
                    f"ContextBase run {self.ingestion_run_id} has no completed ingest for {unit_id}. "
                    "Use the original output directory, run name, and selection, or run ingestion again."
                )
            if checkpoint.read_text(encoding="ascii") != self._ingestion_fingerprint(path, items):
                raise ValueError(f"ContextBase run {self.ingestion_run_id}: endpoint, Bank, or documents changed for {unit_id}.")
            expected_by_bank.setdefault(path, set()).update(item["document_id"] for item in items)

        for path, missing in expected_by_bank.items():
            offset = 0
            while missing:
                result = self._request("GET", f"{path}/documents", params={
                    "q": f"amb-{self.ingestion_run_id}-", "limit": 100, "offset": offset,
                })
                rows, total = result.get("items"), result.get("total")
                if not isinstance(rows, list) or not isinstance(total, int):
                    raise RuntimeError(f"ContextBase returned an invalid document listing for {path}")
                missing.difference_update(row["id"] for row in rows)
                offset += len(rows)
                if not rows or offset >= total:
                    break
            if missing:
                raise ValueError(f"ContextBase run {self.ingestion_run_id} is missing {len(missing)} document(s) in {path}.")
        self._record_precision_documents(items_by_unit)

    def _submit_retain(self, path: str, items: list[dict]) -> list[str]:
        operation_id = str(uuid4())
        try:
            result = self._request("POST", f"{path}/memories", json={
                "items": items, "async": True, "operation_id": operation_id,
            })
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 413 or len(items) == 1:
                raise
            # A gateway can reject a large batch of contracts while accepting each document.
            middle = len(items) // 2
            return self._submit_retain(path, items[:middle]) + self._submit_retain(path, items[middle:])
        except httpx.TransportError as exc:
            raise RuntimeError(
                f"ContextBase Retain submission status unknown for {path}, operation {operation_id}"
            ) from exc
        if result.get("success") is not True:
            raise RuntimeError(f"ContextBase Retain failed for {path}, operation {operation_id}")
        if result.get("items_count") != len(items):
            raise RuntimeError(
                f"ContextBase Retain accepted an unexpected number of items for {path}, operation {operation_id}"
            )
        operation_ids = result.get("operation_ids") or [result.get("operation_id")]
        if result.get("async") is not True or not all(operation_ids):
            raise RuntimeError(f"ContextBase Retain returned no async operation ID for {path}")
        return operation_ids

    def ingest(self, documents: list[Document]) -> None:
        items_by_unit = self._items_by_unit(documents)
        for unit_id in items_by_unit:
            self._checkpoint(unit_id).unlink(missing_ok=True)

        pending: set[tuple[str, str]] = set()
        batch_size = 20
        for unit_id, items in items_by_unit.items():
            path, _ = self._scope(unit_id)
            for start in range(0, len(items), batch_size):
                operation_ids = self._submit_retain(path, items[start:start + batch_size])
                pending.update((path, op_id) for op_id in operation_ids)

        deadline = time.monotonic() + self._retain_timeout
        while pending:
            for path, operation_id in sorted(pending):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"ContextBase Retain timed out waiting for {sorted(pending)}")
                op = self._request("GET", f"{path}/operations/{quote(operation_id, safe='')}", timeout=min(remaining, 60))
                status = op.get("status")
                if status == "completed":
                    pending.remove((path, operation_id))
                elif status not in {"pending", "processing"}:
                    raise RuntimeError(f"ContextBase Retain {operation_id}: {status}: {op.get('error_message') or ''}")
            if pending:
                time.sleep(min(2, max(0, deadline - time.monotonic())))

        for unit_id, items in items_by_unit.items():
            path, _ = self._scope(unit_id)
            checkpoint = self._checkpoint(unit_id)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(f".{uuid4().hex}.tmp")
            try:
                temporary.write_text(self._ingestion_fingerprint(path, items), encoding="ascii")
                temporary.replace(checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
        self._record_precision_documents(items_by_unit)

    def _record_precision_documents(self, items_by_unit: dict[str, list[dict]]) -> None:
        if self.dataset_name == "precisionmembench":
            for unit_id, items in items_by_unit.items():
                self._precision_documents.setdefault(unit_id, {}).update(
                    (item["document_id"], item["metadata"]["doc_id"]) for item in items
                )

    def _precision_source_id(self, fact: dict, documents: dict[str, str], scope: list[str]) -> str | None:
        fact_tags = set(fact.get("tags") or [])
        source_scope = {tag for tag in fact_tags if tag.startswith(("amb-run:", "amb-unit:"))}
        if source_scope.difference(scope):
            return None

        document_id = fact.get("document_id") or ""
        if document_id in documents:
            return documents[document_id]
        if document_id.startswith("amb-") or scope[0] not in fact_tags:
            return None

        # Metadata and context carry fixture IDs that repeat across runs.
        # Accept them only for a fact tagged with this run.
        source_id = (fact.get("metadata") or {}).get("doc_id")
        if not source_id:
            marker = re.match(r"beliefId=(\S+) scope=", fact.get("context") or "")
            source_id = marker[1] if marker else None
        return source_id if source_id in documents.values() else None

    def _precision_sources(
        self, result: dict, source_facts: dict, documents: dict[str, str], scope: list[str],
    ) -> list[str]:
        references = result.get("source_fact_ids") or []
        facts = (source_facts.get(fact_id) or {} for fact_id in references) if references else [result]
        sources = []
        for fact in facts:
            source = self._precision_source_id(fact, documents, scope)
            if not source:
                # A partial observation cannot be scored against mustExclude.
                return []
            sources.append(source)
        return list(dict.fromkeys(sources))

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict]:
        path, tags = self._scope(user_id)
        include: dict = {"entities": None}
        if self.dataset_name == "precisionmembench":
            include["source_facts"] = {"max_tokens": max(self._max_tokens, 4096)}
        if self._max_chunk_tokens:
            include["chunks"] = {"max_tokens": self._max_chunk_tokens}
        payload = {
            "query": query[:1900], "budget": "high", "max_tokens": self._max_tokens,
            "include": include, "tags": tags, "tags_match": "all_strict",
        }
        if query_timestamp:
            payload["query_timestamp"] = query_timestamp
        if self._types:
            payload["types"] = self._types
        raw = self._request("POST", f"{path}/memories/recall", json=payload)
        if not isinstance(raw.get("results"), list):
            raise RuntimeError(f"ContextBase Recall returned an invalid results field for {path}")
        recall = _as_recall_response(raw)
        docs = _build_docs(_deduplicate_results(recall.results), recall.chunks)
        if self.dataset_name == "precisionmembench":
            results = {}
            for result in raw["results"]:
                results.setdefault(result["id"], result)
            source_facts = raw.get("source_facts") or {}
            documents = self._precision_documents.get(user_id, {})
            for doc in docs:
                doc.source_ids = self._precision_sources(results[doc.id], source_facts, documents, tags)
            return docs, raw

        prefix = f"amb-{self.ingestion_run_id}-"
        for doc in docs:
            if doc.source_ids:
                doc.source_ids = list(dict.fromkeys(source.removeprefix(prefix) for source in doc.source_ids))
        return docs, raw


class ContextBaseKnowledgeProvider(ContextBaseMemoryProvider):
    """Rank benchmark source documents or passages using scoped ContextBase Recall."""

    name = "contextbase-knowledge"
    description = "ContextBase Retain and Recall over a tagged text corpus, scored by source document or passage."
    provider = "contextbase"
    variant = "knowledge"
    supports_incremental_ingest = False
    supports_filters = True
    supported_datasets = ("vidore-v3-markdown", "browsecomp-plus", "legalbench-rag")
    _unit = "knowledge"

    def __init__(self, *, auto_bank_id: str = "") -> None:
        bank_id = os.environ.get("CONTEXTBASE_BANK_ID", "").strip()
        self._auto_bank = not bank_id
        self._auto_bank_id = auto_bank_id
        self._admin_token = os.environ.get(BENCH_ADMIN_TOKEN_ENV, "").strip() if self._auto_bank else ""
        if self._auto_bank and not self._admin_token:
            raise ValueError(f"Set CONTEXTBASE_BANK_ID or {BENCH_ADMIN_TOKEN_ENV} for ContextBase knowledge retrieval.")
        if self._admin_token and self._admin_token == os.environ.get("CONTEXTBASE_API_TOKEN", "").strip():
            raise ValueError("ContextBase auto Bank creation requires separate administrator and business API keys.")
        super().__init__(single_bank_id=bank_id)
        self._source_ids: dict[str, str] = {}

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._source_ids.clear()
        if self._auto_bank and not reset:
            raise ValueError("--skip-ingestion needs the original CONTEXTBASE_BANK_ID.")
        super().prepare(store_dir, {self._unit}, reset=reset)
        if self._auto_bank:
            bank_id = self._auto_bank_id or knowledge_bank_id(self.ingestion_run_id)
            config = {"enable_observations": False}
            if self.dataset_name == "legalbench-rag":
                config["retain_extraction_mode"] = "chunks"
            print(f"Creating ContextBase Bank {bank_id} with observations disabled.", flush=True)
            try:
                self._request(
                    "PUT", f"v1/default/banks/{quote(bank_id, safe='')}",
                    headers={"Authorization": f"Bearer {self._admin_token}", "If-None-Match": "*"},
                    json=config,
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                details = f" Check the administrator API key in {BENCH_ADMIN_TOKEN_ENV}." if status in {401, 403} else ""
                raise RuntimeError(f"ContextBase could not create Bank {bank_id} (HTTP {status}).{details}") from exc
            self._bank_id = bank_id

    def _items_by_unit(self, documents: list[Document]) -> dict[str, list[dict]]:
        if not documents or any(doc.path or not doc.content.replace("\x00", "").strip() for doc in documents):
            raise ValueError("ContextBase knowledge retrieval needs a nonempty text corpus.")
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("ContextBase knowledge retrieval needs unique source document IDs.")
        return super()._items_by_unit([replace(doc, user_id=self._unit) for doc in documents])

    def ingest(self, documents: list[Document]) -> None:
        self._source_ids.clear()
        try:
            super().ingest(documents)
        except httpx.HTTPStatusError as exc:
            if self._auto_bank and exc.response.status_code in {401, 403}:
                raise RuntimeError(
                    f"The ContextBase business API key needs read/write access to the newly created Bank {self._bank_id}."
                ) from exc
            raise
        self._source_ids = {self._retained_id(doc.id): doc.id for doc in documents}

    def validate_ingestion(self, documents: list[Document]) -> None:
        self._source_ids.clear()
        super().validate_ingestion(documents)
        self._source_ids = {self._retained_id(doc.id): doc.id for doc in documents}

    def _source_document_id(self, result: dict) -> str:
        source = result.get("document_id") if isinstance(result, dict) else None
        if not isinstance(source, str) or source not in self._source_ids:
            raise RetrievalError("ContextBase Recall returned a hit without a known benchmark source document.")
        return self._source_ids[source]

    def _ranked_chunks(self, results: list[dict], chunks: dict | None, k: int) -> list[Document]:
        if not results and not chunks:
            return []
        if not isinstance(chunks, dict) or not chunks:
            raise RetrievalError("ContextBase Recall returned hits without original chunks for LegalBench-RAG.")

        ranked = {}
        for result in results:
            doc_id = self._source_document_id(result)
            chunk_id = result.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise RetrievalError("ContextBase Recall returned a LegalBench-RAG hit without a source chunk ID.")
            # The chunk budget can omit lower-ranked chunks, and several facts can point to one chunk.
            if chunk_id not in chunks or chunk_id in ranked:
                continue
            chunk = chunks[chunk_id]
            text = chunk.get("text") if isinstance(chunk, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise RetrievalError("ContextBase Recall returned an empty or invalid LegalBench-RAG chunk.")
            ranked[chunk_id] = Document(id=doc_id, content=text, source_ids=[doc_id])
            if len(ranked) == k:
                break
        if not ranked:
            raise RetrievalError("ContextBase Recall did not link any LegalBench-RAG hits to the returned chunks.")
        return list(ranked.values())

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                 query_timestamp: str | None = None, filters: dict | None = None) -> tuple[list[Document], dict]:
        if not self._source_ids:
            raise RetrievalError("ContextBase has no completed benchmark ingest.")
        legalbench = self.dataset_name == "legalbench-rag"
        include = {"entities": None}
        max_tokens = self._max_tokens
        if legalbench:
            chunk_budget = (filters or {}).get("max_tokens", 4096)
            if k <= 0 or chunk_budget <= 0:
                return [], {}
            include["chunks"] = {"max_tokens": chunk_budget}
            max_tokens = max(max_tokens, chunk_budget)

        path, tags = self._scope(self._unit)
        raw = self._request("POST", f"{path}/memories/recall", json={
            "query": query[:1900], "budget": "high", "max_tokens": max_tokens,
            "include": include, "types": ["world", "experience"],
            "tags": tags, "tags_match": "all_strict",
        })
        results = raw.get("results")
        if not isinstance(results, list):
            raise RetrievalError("ContextBase Recall did not return a result list.")
        if legalbench:
            return self._ranked_chunks(results, raw.get("chunks"), k), raw

        ranked = {}
        for result in results:
            doc_id = self._source_document_id(result)
            text = result.get("text")
            if not isinstance(text, str):
                raise RetrievalError("ContextBase Recall returned an invalid hit text.")
            if doc_id not in ranked:
                ranked[doc_id] = Document(id=doc_id, content=text, source_ids=[doc_id])
        return list(ranked.values())[:k], raw
