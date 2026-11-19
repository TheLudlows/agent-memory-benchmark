import os
import time
import uuid
from pathlib import Path

from rich.console import Console

from ..models import Document
from .base import MemoryProvider

_console = Console()


class Mem0CloudMemoryProvider(MemoryProvider):
    name = "mem0-cloud"
    description = "Mem0 cloud API. Async indexing — waits for indexing to complete before eval. k=20."
    kind = "cloud"
    supports_incremental_ingest = True
    provider = "mem0"
    variant = "cloud"

    def __init__(self, k: int = 20, *, user_prefix: str = ""):
        self.k = k
        self._client = None
        self._configured_user_prefix = user_prefix
        self._user_prefix = user_prefix
        self._default_user_id = f"bench_{uuid.uuid4().hex[:8]}"

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        self._user_prefix = self._configured_user_prefix

    def _ensure_client(self):
        if self._client is None:
            from mem0 import MemoryClient
            api_key = os.environ.get("MEM0_CLOUD_KEY") or os.environ["MEM0_API_KEY"]
            self._client = MemoryClient(api_key=api_key, host=os.environ.get("MEM0_CLOUD_URL") or None)
        return self._client

    def ingest(self, documents: list[Document]) -> None:
        client = self._ensure_client()

        user_ids = {self._user_id(doc.user_id) for doc in documents}
        if not self._user_prefix:
            # Preserve CLI replay behavior for runs with stable user IDs.
            for uid in user_ids:
                try:
                    client.delete_all(user_id=uid)
                except Exception:
                    pass

        for doc in documents:
            messages = doc.messages or [{"role": "user", "content": doc.content}]
            client.add(
                messages=messages, user_id=self._user_id(doc.user_id),
                metadata={"doc_id": doc.id}, async_mode=not self._user_prefix,
            )

        if user_ids and not self._user_prefix:
            self._wait_for_indexing(client, user_ids)

    async def async_ingest_increment(self, documents: list[Document], *, first: bool) -> None:
        if not self._user_prefix:
            # A temporal CLI run also needs fresh storage and synchronous indexing.
            self._user_prefix = f"amb-temporal-{uuid.uuid4().hex}-"
        await self.async_ingest(documents)

    def _user_id(self, user_id: str | None) -> str:
        return self._user_prefix + (user_id or self._default_user_id)

    def _wait_for_indexing(self, client, user_ids: set[str], timeout: int = 300) -> None:
        """Poll until at least one memory is searchable, then wait proportionally."""
        sample_uid = next(iter(user_ids))
        deadline = time.time() + timeout
        _console.print("[dim]  Waiting for mem0 cloud indexing...[/dim]")
        while time.time() < deadline:
            try:
                result = client.get_all(filters={"user_id": sample_uid}, limit=1)
                entries = result.get("results", result) if isinstance(result, dict) else result
                if entries:
                    _console.print(f"[dim]  Indexed.[/dim]")
                    return
            except Exception:
                pass
            time.sleep(5)
        _console.print("[yellow]  Warning: timed out waiting for mem0 indexing[/yellow]")

    def retrieve(
        self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None
    ) -> tuple[list[Document], dict | None]:
        uid = self._user_id(user_id)
        results = self._ensure_client().search(query, filters={"user_id": uid}, top_k=self.k)
        entries = results.get("results", results) if isinstance(results, dict) else results
        raw = results if isinstance(results, dict) else {"results": results}
        docs = []
        for r in entries:
            lines = [r["memory"]]
            if r.get("score") is not None:
                lines.append(f"score: {r['score']:.3f}")
            if r.get("created_at"):
                lines.append(f"created: {r['created_at']}")
            if r.get("updated_at"):
                lines.append(f"updated: {r['updated_at']}")
            meta = r.get("metadata") or {}
            if meta:
                lines.append(f"metadata: {meta}")
            doc_id = meta.get("doc_id")
            docs.append(Document(id=r["id"], content="\n".join(lines),
                                 source_ids=[doc_id] if doc_id else None))
        return docs, raw
