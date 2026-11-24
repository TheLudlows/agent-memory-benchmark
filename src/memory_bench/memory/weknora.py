"""WeKnora document knowledge bases and ranked retrieval over HTTP."""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

from ..json_files import write_json
from ..models import Document
from ..utils import chunk_text, count_tokens
from .base import MemoryProvider, RetrievalError

_log = logging.getLogger(__name__)
_UPLOAD_WORKERS = 4
_POLL_BATCH_SIZE = 50
_POLL_INTERVAL = 1.0
_PENDING = {"pending", "processing", "finalizing"}
_TOP_K_LIMIT = 200


def _legal_upload_padding(documents: list[Document]) -> dict[str, str]:
    # WeKnora deduplicates file bytes. Trailing newlines distinguish repeated contracts
    # without shifting any scored source offsets or introducing searchable text.
    occupied = {document.content for document in documents}
    seen = set()
    padding = {}
    for document in documents:
        content = document.content
        if content in seen:
            suffix = "\n"
            while content + suffix in occupied:
                suffix += "\n"
            padding[document.id] = suffix
            occupied.add(content + suffix)
        seen.add(content)
    return padding


def _literal_span(source: str, passage: str, bounds: list[tuple[int, int]]) -> tuple[int, int] | None:
    position = None
    for start, end in bounds:
        first = source.find(passage, start, end)
        if first < 0:
            continue
        if (position is not None and first != position) or source.find(passage, first + 1, end) >= 0:
            return None
        position = first
    return (position, position + len(passage)) if position is not None else None


def _source_locators(hit: dict, kind: str) -> list[dict]:
    locators = hit.get("source_locators")
    if locators is None:
        return []
    if not isinstance(locators, list) or any(not isinstance(loc, dict) for loc in locators):
        raise RetrievalError("WeKnora returned an invalid source locator list.")
    return [loc for loc in locators if loc.get("type") == kind]


def _legal_spans(source: str, hit: dict) -> list[tuple[int, int]]:
    content = hit.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RetrievalError("WeKnora returned a LegalBench hit without source text.")
    passage = content.strip()
    # Text locators can cover an entire paragraph even when a hit quotes only a part of it.
    # Use them to place returned literal text, never as character coverage by themselves.
    bom = int(source.startswith("\ufeff"))
    locators = _source_locators(hit, "text")
    bounds = [
        (start + bom, end + bom)
        for loc in locators
        if type(start := loc.get("start", 0)) is int and type(end := loc.get("end")) is int
        and 0 <= start < end <= len(source) - bom
    ]
    global_bounds = [(0, len(source))]
    span = _literal_span(source, passage, bounds) or _literal_span(source, passage, global_bounds)
    if span is not None:
        return [span]

    # Chunker offsets can be relative to transformed Markdown. Use them only when the
    # exact file locators cover both ends and the corresponding original text agrees.
    start, end = hit.get("start_at", 0), hit.get("end_at")
    if type(start) is int and type(end) is int and start < end:
        if any(low <= start < high for low, high in bounds) and any(low < end <= high for low, high in bounds):
            original = source[start:end]
            if original.strip() == passage:
                start += len(original) - len(original.lstrip())
                return [(start, start + len(passage))]

    # The retrieval pipeline can join separated chunks of a document with a blank line.
    spans = []
    for piece in (part.strip() for part in passage.split("\n\n") if part.strip()):
        span = _literal_span(source, piece, bounds) or _literal_span(source, piece, global_bounds)
        if span is None or (spans and span[0] < spans[-1][1]):
            raise RetrievalError("WeKnora returned LegalBench source text that could not be placed unambiguously.")
        spans.append(span)
    return spans


def _legal_evidence(source: str, spans: list[tuple[int, int]], budget: int) -> tuple[str, list[tuple[int, int]], int]:
    content = "\n\n".join(source[start:end] for start, end in spans)
    tokens = count_tokens(content)
    if tokens <= budget:
        return content, spans, budget - tokens

    prefix = chunk_text(content, budget)[0]
    # A token boundary can end inside the UTF-8 bytes of one character.
    while not content.startswith(prefix) or count_tokens(prefix) > budget:
        prefix = prefix[:-1]
    covered = []
    characters = len(prefix)
    for start, end in spans:
        if characters <= 0:
            break
        covered.append((start, min(end, start + characters)))
        characters -= end - start + 2  # the separator between literal source spans
    return prefix, covered, 0


class WeKnoraMemoryProvider(MemoryProvider):
    name = "weknora"
    description = "WeKnora knowledge base upload and hybrid retrieval over HTTP."
    kind = "cloud"
    link = "https://github.com/Tencent/WeKnora"
    supports_filters = True

    def __init__(self) -> None:
        self._url = os.getenv("WEKNORA_API_URL", "").strip().rstrip("/")
        if self._url and not self._url.endswith("/api/v1"):
            self._url += "/api/v1"
        self._key = os.getenv("WEKNORA_API_KEY", "").strip()
        self._embedding_model = os.getenv("WEKNORA_EMBEDDING_MODEL_ID", "").strip()
        self._vlm_model = os.getenv("WEKNORA_VLM_MODEL_ID", "").strip()
        self._http: httpx.Client | None = None
        self._state_path: Path | None = None
        self._kb_id = ""
        self._document_ids: dict[str, str] = {}
        self._legal_texts: dict[str, str] = {}

    def initialize(self) -> None:
        required = {
            "WEKNORA_API_URL": self._url,
            "WEKNORA_API_KEY": self._key,
            "WEKNORA_EMBEDDING_MODEL_ID": self._embedding_model,
        }
        if self.dataset_name == "miracl-vision":
            required["WEKNORA_VLM_MODEL_ID"] = self._vlm_model
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"WeKnora requires {', '.join(missing)}.")
        try:
            address = urlsplit(self._url)
            _ = address.port
        except ValueError:
            raise ValueError("WEKNORA_API_URL must be an HTTP(S) API address.") from None
        if address.scheme not in {"http", "https"} or not address.hostname or address.username is not None or address.query or address.fragment:
            raise ValueError("WEKNORA_API_URL must be an HTTP(S) API address without credentials, a query or a fragment.")
        self._http = httpx.Client(base_url=self._url, headers={"X-API-Key": self._key}, timeout=120.0)

    def cleanup(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = store_dir / "weknora_knowledge_base.json"
        self._kb_id = ""
        self._document_ids = {}
        self._legal_texts = {}
        if reset:
            return
        if not self._state_path.is_file():
            raise ValueError("WeKnora has no recorded knowledge base to reuse. Run ingestion first.")
        state = json.loads(self._state_path.read_text(encoding="utf-8"))
        if state["url"] != self._url or state["dataset"] != self.dataset_name:
            raise ValueError("WeKnora knowledge base belongs to a different connection or dataset.")
        if state["models"] != self._models():
            raise ValueError("WeKnora model configuration changed. Run ingestion again.")
        self._kb_id = state["knowledge_base_id"]
        self._document_ids = state["document_ids"]

    def validate_ingestion(self, documents: list[Document]) -> None:
        if set(self._document_ids.values()) != {doc.id for doc in documents}:
            raise ValueError("WeKnora knowledge base does not contain the same benchmark documents. Run ingestion again.")
        self._set_legal_texts(documents)

    def _set_legal_texts(self, documents: list[Document]) -> None:
        if self.dataset_name == "legalbench-rag":
            self._legal_texts = {doc.id: doc.content for doc in documents}

    def _models(self) -> dict[str, str]:
        return {
            "embedding": self._embedding_model,
            "vlm": self._vlm_model if self.dataset_name in {"miracl-vision", "vidore-v3-pdf"} else "",
        }

    def _request(self, method: str, route: str, operation: str, **kwargs) -> dict:
        if self._http is None:
            raise RuntimeError("WeKnora connection has not been initialized.")
        try:
            response = self._http.request(method, route, **kwargs)
        except httpx.RequestError as exc:
            raise RuntimeError(f"WeKnora {operation} failed ({type(exc).__name__}).") from None
        if not response.is_success:
            raise RuntimeError(f"WeKnora {operation} failed {self._http_error(response)}")
        try:
            envelope = response.json()
        except ValueError:
            raise RuntimeError(f"WeKnora {operation} returned invalid JSON.") from None
        if not isinstance(envelope, dict) or envelope.get("success") is not True or "data" not in envelope:
            raise RuntimeError(f"WeKnora {operation} returned an invalid response.")
        return envelope

    def _http_error(self, response: httpx.Response) -> str:
        status = f"HTTP {response.status_code}"
        if not response.is_client_error:
            return f"({status})."

        request_id = response.headers.get("X-Request-ID", "")
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        payload = payload if isinstance(payload, dict) else {}
        error = payload.get("error")
        if isinstance(error, dict):
            code, message = error.get("code"), error.get("message")
        else:
            code, message = payload.get("code"), payload.get("message") or error

        parts = [status]
        if isinstance(code, (int, str)) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", str(code)):
            parts.append(f"code {code}")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", request_id):
            parts.append(f"request ID {request_id}")
        if isinstance(message, str):
            if self._key:
                message = message.replace(self._key, "[redacted]")
            message = " ".join("".join(c if c.isprintable() else " " for c in message).split())
        else:
            message = ""
        rendered = f"({', '.join(parts)})." + (f" {message[:300]}" if message else "")
        return rendered.replace(self._key, "[redacted]") if self._key else rendered

    @staticmethod
    def _identifier(record: object, operation: str) -> str:
        value = record.get("id") if isinstance(record, dict) else None
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"WeKnora {operation} did not return an ID.")
        return value

    def _create_knowledge_base(self) -> str:
        payload = {
            "name": f"amb-{self.dataset_name}-{uuid4().hex[:12]}",
            "type": "document",
            "embedding_model_id": self._embedding_model,
        }
        if vlm := self._models()["vlm"]:
            payload["vlm_config"] = {"enabled": True, "model_id": vlm}
        if self.dataset_name == "miracl-vision":
            payload["image_processing_config"] = {"image_vector_enabled": True}
        record = self._request("POST", "knowledge-bases", "create knowledge base", json=payload)["data"]
        kb_id = self._identifier(record, "create knowledge base")
        image_config = record.get("image_processing_config") or {}
        if self.dataset_name == "miracl-vision" and image_config.get("image_vector_enabled") is not True:
            raise RuntimeError("WeKnora did not enable image vector indexing for MIRACL-Vision.")
        return kb_id

    def _upload(self, document: Document, padding: str) -> str:
        route = f"knowledge-bases/{quote(self._kb_id, safe='')}/knowledge"
        operation = f"upload document {document.id}"
        options = {"summary_enabled": False}
        if document.path:
            form = {"process_config": json.dumps(options)}
            if self._models()["vlm"]:
                form["enable_multimodel"] = "true"
            with Path(document.path).open("rb") as stream:
                file = (document.filename or Path(document.path).name, stream)
                record = self._request("POST", f"{route}/file", operation, files={"file": file}, data=form)["data"]
        elif self.dataset_name == "legalbench-rag":
            file = (document.filename or "document.txt", (document.content + padding).encode("utf-8"), "text/plain")
            record = self._request(
                "POST", f"{route}/file", operation, files={"file": file}, data={"process_config": json.dumps(options)},
            )["data"]
        else:
            # The manual endpoint preserves distinct corpus IDs for documents with identical bytes.
            record = self._request("POST", f"{route}/manual", operation, json={
                "title": document.filename or document.id,
                "content": document.content,
                "status": "publish",
                "process_config": options,
            })["data"]
        return self._identifier(record, operation)

    def _wait_for_index(self, documents: dict[str, str], deadline: float) -> None:
        pending = documents.copy()
        while pending:
            if monotonic() >= deadline:
                raise TimeoutError(f"WeKnora knowledge base {self._kb_id} still has documents waiting for indexing.")
            rows = self._request("GET", "knowledge/batch", "poll indexing", params={
                "ids": list(pending), "kb_id": self._kb_id,
            })["data"]
            if not isinstance(rows, list):
                raise RuntimeError("WeKnora indexing status response did not contain a document list.")
            status_by_id = {self._identifier(row, "poll indexing"): row.get("parse_status") for row in rows}
            if status_by_id.keys() != pending.keys():
                raise RuntimeError("WeKnora indexing status response did not match the uploaded documents.")
            for remote_id, status in status_by_id.items():
                if status == "completed":
                    del pending[remote_id]
                elif not isinstance(status, str) or status not in _PENDING:
                    status = status if isinstance(status, str) else ""
                    known_failure = status if status in {"failed", "cancelled", "deleting", "draft"} else "unexpected status"
                    raise RuntimeError(f"WeKnora could not index document {pending[remote_id]} ({known_failure}).")
            if pending:
                sleep(min(_POLL_INTERVAL, max(0, deadline - monotonic())))

    def _check_image_index(self, remote_id: str) -> None:
        chunks = self._request("GET", f"chunks/{quote(remote_id, safe='')}", "verify image index", params={
            "chunk_type": "image_vector", "page_size": 1,
        })["data"]
        if not isinstance(chunks, list) or not any(isinstance(chunk, dict) and chunk.get("chunk_type") == "image_vector" for chunk in chunks):
            raise RuntimeError("WeKnora did not index the first image as an image vector. Check its embedding and VLM models.")

    def ingest(self, documents: list[Document]) -> None:
        if self._state_path is None:
            raise RuntimeError("WeKnora run storage has not been prepared.")
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("WeKnora requires unique benchmark document IDs.")
        if not documents:
            raise ValueError("WeKnora requires a nonempty benchmark corpus.")
        deadline = monotonic() + float(os.getenv("AMB_KNOWLEDGE_INGEST_TIMEOUT", str(6 * 3600)))
        self._kb_id = self._create_knowledge_base()
        _log.info("WeKnora knowledge base %s created for %s", self._kb_id, self.dataset_name)
        document_ids = {}
        padding = _legal_upload_padding(documents) if self.dataset_name == "legalbench-rag" else {}
        with ThreadPoolExecutor(max_workers=_UPLOAD_WORKERS) as pool:
            for start in range(0, len(documents), _POLL_BATCH_SIZE):
                batch = documents[start:start + _POLL_BATCH_SIZE]
                pending = {}
                uploads = pool.map(self._upload, batch, (padding.get(document.id, "") for document in batch))
                for document, remote_id in zip(batch, uploads, strict=True):
                    if remote_id in document_ids:
                        raise RuntimeError("WeKnora assigned the same source ID to distinct benchmark documents.")
                    document_ids[remote_id] = document.id
                    pending[remote_id] = document.id
                self._wait_for_index(pending, deadline)
                if start == 0 and self.dataset_name == "miracl-vision":
                    self._check_image_index(next(iter(pending)))
        write_json(self._state_path, {
            "url": self._url,
            "dataset": self.dataset_name,
            "models": self._models(),
            "knowledge_base_id": self._kb_id,
            "document_ids": document_ids,
        })
        self._document_ids = document_ids
        self._set_legal_texts(documents)

    def _benchmark_id(self, hit: object) -> str:
        if not isinstance(hit, dict) or hit.get("knowledge_base_id", self._kb_id) != self._kb_id:
            raise RetrievalError("WeKnora returned a hit outside the benchmark knowledge base.")
        remote_id = hit.get("knowledge_id")
        if not isinstance(remote_id, str) or remote_id not in self._document_ids:
            raise RetrievalError("WeKnora returned a hit without a known benchmark source ID.")
        return self._document_ids[remote_id]

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None,
                 query_timestamp: str | None = None, filters: dict | None = None) -> tuple[list[Document], dict | None]:
        if not self._kb_id or not self._document_ids:
            raise RetrievalError("WeKnora has no indexed benchmark corpus.")
        if k <= 0:
            return [], None
        try:
            result = self._request("POST", "knowledge-search", "search", json={
                "query": query,
                "knowledge_base_ids": [self._kb_id],
                "match_count": min(k, _TOP_K_LIMIT),
            })
        except RuntimeError as exc:
            raise RetrievalError(str(exc)) from None
        rerank = ((result.get("meta") or {}).get("rerank") or {}).get("outcome")
        if rerank in {"model_error", "model_unavailable"}:
            raise RetrievalError(f"WeKnora reranking failed ({rerank}).")
        hits = result["data"]
        if hits is None:  # An empty Go slice can be returned as JSON null.
            return [], None
        if not isinstance(hits, list):
            raise RetrievalError("WeKnora search did not return a ranked list.")

        documents = []
        budget = (filters or {}).get("max_tokens", 4096)
        for hit in hits[:min(k, _TOP_K_LIMIT)]:
            if self.dataset_name == "legalbench-rag" and budget <= 0:
                break
            document_id = self._benchmark_id(hit)
            if self.dataset_name == "vidore-v3-pdf":
                pages = dict.fromkeys(
                    page for loc in _source_locators(hit, "pdf")
                    if type(page := loc.get("page")) is int and page > 0
                )
                if not pages:
                    raise RetrievalError("WeKnora did not return a PDF page source locator. Check the WeKnora version and PDF parser.")
                documents.extend(Document(id=document_id, content="", page=page) for page in pages)
            elif self.dataset_name == "legalbench-rag":
                source = self._legal_texts[document_id]
                spans = _legal_spans(source, hit)
                content, spans, budget = _legal_evidence(source, spans, budget)
                if spans:
                    documents.append(Document(id=document_id, content=content, spans=spans))
            else:
                documents.append(Document(id=document_id, content=""))
        return documents, None
