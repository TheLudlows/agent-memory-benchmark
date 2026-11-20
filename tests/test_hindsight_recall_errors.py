import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from memory_bench.memory.base import RetrievalError
from memory_bench.memory.hindsight import HindsightMemoryProvider


@pytest.fixture(params=["sync", "async"])
def retrieve(request):
    if request.param == "sync":
        return lambda provider: provider.retrieve("query")
    return lambda provider: asyncio.run(provider.async_retrieve("query"))


@pytest.mark.parametrize("dataset", ["lifebench", "locomo"])
def test_recall_failure_is_only_an_error_for_strict_datasets(monkeypatch, retrieve, dataset):
    monkeypatch.setattr("memory_bench.memory.hindsight.time.sleep", Mock())
    monkeypatch.setattr("memory_bench.memory.hindsight.asyncio.sleep", AsyncMock())
    provider = HindsightMemoryProvider()
    provider._dataset = dataset
    request = httpx.Request("POST", "http://memory.test/recall")
    response = httpx.Response(403, request=request)
    denied = httpx.HTTPStatusError("Recall forbidden", request=request, response=response)
    provider._client = SimpleNamespace(recall=Mock(side_effect=denied), arecall=AsyncMock(side_effect=denied))

    if dataset == "lifebench":
        with pytest.raises(RetrievalError, match=r"Hindsight Recall failed \(403\)") as error:
            retrieve(provider)
        assert error.value.__cause__ is denied
    else:
        assert retrieve(provider) == ([], None)


def test_lifebench_empty_recall_remains_valid(retrieve):
    provider = HindsightMemoryProvider()
    provider._dataset = "lifebench"
    raw = {"results": []}
    empty = SimpleNamespace(results=[], chunks=None, model_dump=lambda: raw)
    provider._client = SimpleNamespace(recall=Mock(return_value=empty), arecall=AsyncMock(return_value=empty))

    assert retrieve(provider) == ([], raw)


def test_lifebench_sync_http_recall_rejects_malformed_result(monkeypatch):
    monkeypatch.setattr("memory_bench.memory.hindsight.time.sleep", Mock())
    provider = HindsightMemoryProvider()
    provider._dataset = "lifebench"
    provider._cloud_base_url = "http://memory.test"
    monkeypatch.setattr(provider, "_needs_http_recall", lambda _: True)
    request = httpx.Request("POST", "http://memory.test/recall")
    response = httpx.Response(200, json={}, request=request)
    monkeypatch.setattr(httpx, "post", Mock(return_value=response))

    with pytest.raises(RetrievalError, match="ValueError") as error:
        provider.retrieve("query")
    assert str(error.value.__cause__) == "Hindsight Recall response is missing a result list"
