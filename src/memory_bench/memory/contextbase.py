import json
import os
import re
import time
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx

from ..models import Document
from .base import MemoryProvider
from .hindsight import _as_recall_response, _build_docs, _deduplicate_results


class ContextBaseMemoryProvider(MemoryProvider):
    name = "contextbase"
    description = "ContextBase Retain and Recall over HTTP using pre-authorized Banks."
    kind = "cloud"
    supports_incremental_ingest = True

    def __init__(self) -> None:
        url = os.environ.get("CONTEXTBASE_API_URL", "").strip()
        token = os.environ.get("CONTEXTBASE_API_TOKEN", "").strip()
        if not url or not token:
            raise ValueError("Set CONTEXTBASE_API_URL and CONTEXTBASE_API_TOKEN for ContextBase.")
        self._bank_id = os.environ.get("CONTEXTBASE_BANK_ID", "").strip()
        self._bank_map = json.loads(os.environ.get("CONTEXTBASE_BENCH_BANKS") or "{}")
        if not isinstance(self._bank_map, dict) or not all(
            isinstance(bank, str) and bank.strip() for bank in self._bank_map.values()
        ):
            raise ValueError("CONTEXTBASE_BENCH_BANKS must map isolation unit IDs to Bank IDs.")
        if bool(self._bank_id) == bool(self._bank_map):
            raise ValueError("Set either CONTEXTBASE_BANK_ID or CONTEXTBASE_BENCH_BANKS.")

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

    def _items_by_unit(self, documents: list[Document]) -> dict[str, list[dict]]:
        items_by_unit: dict[str, list[dict]] = {}
        for doc in documents:
            unit_id = doc.user_id
            if not unit_id:
                raise ValueError("ContextBase requires an isolation unit ID for each retained document.")
            _, tags = self._scope(unit_id)
            item = {
                "content": doc.content.replace("\x00", ""),
                "document_id": f"amb-{self.ingestion_run_id}-{doc.id}",
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
        expected_by_bank: dict[str, set[str]] = {}
        for unit_id, items in self._items_by_unit(documents).items():
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

    def ingest(self, documents: list[Document]) -> None:
        items_by_unit = self._items_by_unit(documents)
        for unit_id in items_by_unit:
            self._checkpoint(unit_id).unlink(missing_ok=True)

        pending: set[tuple[str, str]] = set()
        batch_size = 20
        for unit_id, items in items_by_unit.items():
            path, _ = self._scope(unit_id)
            for start in range(0, len(items), batch_size):
                batch = items[start:start + batch_size]
                operation_id = str(uuid4())
                try:
                    result = self._request("POST", f"{path}/memories", json={
                        "items": batch, "async": True, "operation_id": operation_id,
                    })
                except httpx.TransportError as exc:
                    raise RuntimeError(
                        f"ContextBase Retain submission status unknown for {path}, operation {operation_id}"
                    ) from exc
                if result.get("success") is not True:
                    raise RuntimeError(f"ContextBase Retain failed for {path}, operation {operation_id}")
                if result.get("items_count") != len(batch):
                    raise RuntimeError(
                        f"ContextBase Retain accepted an unexpected number of items for {path}, operation {operation_id}"
                    )
                operation_ids = result.get("operation_ids") or [result.get("operation_id")]
                if result.get("async") is not True or not all(operation_ids):
                    raise RuntimeError(f"ContextBase Retain returned no async operation ID for {path}")
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

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> tuple[list[Document], dict]:
        path, tags = self._scope(user_id)
        include: dict = {"entities": None}
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
        prefix = f"amb-{self.ingestion_run_id}-"
        for doc in docs:
            if doc.source_ids:
                doc.source_ids = list(dict.fromkeys(source.removeprefix(prefix) for source in doc.source_ids))
        return docs, raw
