from json import JSONDecodeError
from unittest.mock import Mock

import pytest
from mem0.exceptions import AuthenticationError, NetworkError

from memory_bench.memory.base import RetrievalError
from memory_bench.memory.mem0_cloud import Mem0CloudMemoryProvider


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_recall_accepts_both_results_shapes(wrapped, empty):
    rows = [] if empty else [{"id": "memory-1", "memory": "A memory", "metadata": {"doc_id": "doc-1"}}]
    provider = Mem0CloudMemoryProvider(k=3, user_prefix="run-")
    provider._client = Mock()
    raw_results = {"results": rows, "metadata": {"page": 1}} if wrapped else {"results": rows}
    provider._client.search.return_value = raw_results if wrapped else rows

    documents, raw = provider.retrieve("question", user_id="user-1")

    assert raw == raw_results
    if empty:
        assert documents == []
    else:
        assert documents[0].id == "memory-1"
        assert documents[0].source_ids == ["doc-1"]
        assert documents[0].content.startswith("A memory")
    provider._client.search.assert_called_once_with("question", filters={"user_id": "run-user-1"}, top_k=3)


@pytest.mark.parametrize("payload, reason", [
    ({}, "invalid results list"),
    ({"error": "upstream unavailable"}, "invalid results list"),
    ({"results": None}, "invalid results list"),
    ({"results": {}}, "invalid results list"),
    ({"results": [{}]}, "invalid memory entry"),
    ([{"id": "memory-1", "memory": "A memory", "score": "bad"}], "invalid memory score"),
    ([{"id": "memory-1", "memory": "A memory", "metadata": []}], "invalid memory metadata"),
])
def test_recall_rejects_malformed_successes(payload, reason):
    provider = Mem0CloudMemoryProvider()
    provider._client = Mock()
    provider._client.search.return_value = payload

    with pytest.raises(RetrievalError, match=reason):
        provider.retrieve("question")


def test_recall_rejects_an_unparseable_success_body():
    provider = Mem0CloudMemoryProvider()
    provider._client = Mock()
    provider._client.search.side_effect = JSONDecodeError("Expected JSON", "", 0)

    with pytest.raises(RetrievalError, match="invalid JSON"):
        provider.retrieve("question")


@pytest.mark.parametrize("error_type, debug_info, failure", [
    (AuthenticationError, {"status_code": 401}, "HTTP 401"),
    (NetworkError, {"error_type": "timeout"}, "NetworkError"),
])
def test_recall_reports_sdk_failures_without_returning_their_body(error_type, debug_info, failure):
    provider = Mem0CloudMemoryProvider()
    provider._client = Mock()
    provider._client.search.side_effect = error_type("private response body", "TEST", debug_info=debug_info)

    with pytest.raises(RetrievalError, match=failure) as error:
        provider.retrieve("question")

    assert "private response body" not in str(error.value)
