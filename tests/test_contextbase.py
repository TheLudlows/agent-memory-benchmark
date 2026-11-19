import json

import httpx
import pytest
from typer.testing import CliRunner

from memory_bench import cli, llm, runner
from memory_bench.dataset.locomo import LoComoDataset
from memory_bench.memory import get_memory_provider
from memory_bench.memory import contextbase
from memory_bench.models import Document


def _configure_contextbase(monkeypatch, handler):
    monkeypatch.setenv("CONTEXTBASE_API_URL", "https://example.test/tenant-api/")
    monkeypatch.setenv("CONTEXTBASE_API_TOKEN", "test-token")
    monkeypatch.setenv("CONTEXTBASE_BANK_ID", "test-bank")
    monkeypatch.setenv("AMB_UNIT_CONCURRENCY", "1")
    monkeypatch.delenv("CONTEXTBASE_BENCH_RUN_ID", raising=False)
    monkeypatch.delenv("CONTEXTBASE_BENCH_BANKS", raising=False)
    client = httpx.Client
    monkeypatch.setattr(contextbase.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))


def _provider(monkeypatch, tmp_path, handler):
    _configure_contextbase(monkeypatch, handler)
    provider = get_memory_provider("contextbase")
    provider.prepare(tmp_path, {"conv-26"})
    return provider


def test_retain_waits_for_every_operation_and_recall_keeps_scope_and_source(monkeypatch, tmp_path):
    statuses = {"first": ["processing", "completed"], "second": ["completed"]}
    timestamp = "2023-10-22T09:55:00+00:00"
    monkeypatch.setattr(contextbase.time, "sleep", lambda _: None)

    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.url.path.startswith("/tenant-api/v1/default/banks/test-bank/")
        if request.url.path.endswith("/memories"):
            body = json.loads(request.content)
            assert body["async"] is True
            assert body["items"] == [{
                "content": "a conversation", "document_id": f"amb-{provider.ingestion_run_id}-conv-26_session_1",
                "metadata": {"doc_id": "conv-26_session_1"}, "tags": scopes, "update_mode": "replace",
                "timestamp": timestamp, "context": "session 1",
            }]
            return httpx.Response(200, json={
                "success": True, "async": True, "items_count": 1, "operation_id": "first",
                "operation_ids": ["first", "second"],
            })
        if "/operations/" in request.url.path:
            operation = request.url.path.rsplit("/", 1)[1]
            return httpx.Response(200, json={"status": statuses[operation].pop(0)})
        assert request.url.path.endswith("/memories/recall")
        assert all(not status for status in statuses.values())
        body = json.loads(request.content)
        assert body["tags"] == scopes and body["tags_match"] == "all_strict"
        assert body["query_timestamp"] == timestamp
        return httpx.Response(200, json={"results": [{
            "id": "a-fact", "text": "a memory", "chunk_id": "source-chunk",
            "metadata": {"doc_id": "conv-26_session_1"},
        }], "chunks": {"source-chunk": {"text": "original conversation text"}}})

    provider = _provider(monkeypatch, tmp_path, handler)
    scopes = [f"amb-run:{provider.ingestion_run_id}", "amb-unit:conv-26"]
    provider.ingest([Document("conv-26_session_1", "a conversation", "conv-26", timestamp=timestamp, context="session 1")])
    docs, raw = provider.retrieve("a question", user_id="conv-26", query_timestamp=timestamp)
    assert docs[0].source_ids == ["conv-26_session_1"]
    assert "original conversation text" in docs[0].content
    assert raw["results"][0]["id"] == "a-fact"
    provider.cleanup()


def test_failed_operation_and_http_recall_error_propagate(monkeypatch, tmp_path):
    def handler(request):
        if request.url.path.endswith("/memories"):
            return httpx.Response(200, json={"success": True, "async": True, "items_count": 1, "operation_id": "failed-op"})
        if request.url.path.endswith("/operations/failed-op"):
            return httpx.Response(200, json={"status": "failed", "error_message": "extraction failed"})
        return httpx.Response(403)

    provider = _provider(monkeypatch, tmp_path, handler)
    docs = [Document("conv-26_session_1", "content", "conv-26")]
    with pytest.raises(RuntimeError, match="failed-op: failed: extraction failed"):
        provider.ingest(docs)
    with pytest.raises(ValueError, match="no completed ingest"):
        provider.validate_ingestion(docs)
    with pytest.raises(httpx.HTTPStatusError, match="403"):
        provider.retrieve("a question", user_id="conv-26")
    provider.cleanup()


def test_ingestion_reuse_checks_checkpoint_input_and_each_remote_document(monkeypatch, tmp_path):
    retained = []
    offsets = []

    def handler(request):
        if request.url.path.endswith("/memories"):
            body = json.loads(request.content)
            retained.extend(item["document_id"] for item in body["items"])
            return httpx.Response(200, json={
                "success": True, "async": True, "items_count": len(body["items"]), "operation_id": body["operation_id"],
            })
        if "/operations/" in request.url.path:
            return httpx.Response(200, json={"status": "completed"})
        assert request.url.path.endswith("/documents")
        assert request.url.params["q"] == f"amb-{run_id}-"
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        return httpx.Response(200, json={"items": [{"id": item} for item in retained[offset:offset + 1]], "total": len(retained)})

    docs = [Document(f"conv-26_session_{i}", f"session {i}", "conv-26") for i in (1, 2)]
    provider = _provider(monkeypatch, tmp_path, handler)
    run_id = provider.ingestion_run_id
    provider.ingest(docs)
    provider.cleanup()

    monkeypatch.setenv("CONTEXTBASE_BENCH_RUN_ID", run_id)
    resumed = get_memory_provider("contextbase")
    resumed.prepare(tmp_path, {"conv-26"}, reset=False)
    resumed.validate_ingestion(docs)
    assert offsets == [0, 1]
    with pytest.raises(ValueError, match="documents changed"):
        resumed.validate_ingestion(docs[:1])
    retained.pop()
    with pytest.raises(ValueError, match=r"missing 1 document\(s\)"):
        resumed.validate_ingestion(docs)
    resumed.cleanup()

    monkeypatch.setenv("CONTEXTBASE_BANK_ID", "different-bank")
    wrong_bank = get_memory_provider("contextbase")
    wrong_bank.prepare(tmp_path, {"conv-26"}, reset=False)
    with pytest.raises(ValueError, match="Bank, or documents changed"):
        wrong_bank.validate_ingestion(docs)
    wrong_bank.cleanup()


def test_failed_rerun_and_invalid_ingestion_id_do_not_mix_results(monkeypatch, tmp_path):
    fail_judgment = True

    class Model:
        model_id = "test-model"

        def generate(self, prompt, schema):
            if "answer" in schema.properties:
                return {"answer": "test answer", "reasoning": "test"}
            return {"correct": not (fail_judgment and "conv-30" in prompt), "reason": "test"}

    def raw_dataset(self):
        return [
            {
                "sample_id": unit,
                "conversation": {
                    "speaker_a": "A", "speaker_b": "B",
                    "session_1_date_time": "1:00 pm on 1 May, 2023",
                    "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "a conversation"}],
                },
                "qa": [{"question": f"Question for {unit}?", "answer": "test answer", "category": 1, "evidence": ["D1:1"]}],
            }
            for unit in ("conv-26", "conv-30")
        ]

    fail_second_unit = False
    retained = set()
    retain_calls = 0

    def handler(request):
        nonlocal retain_calls
        if request.url.path.endswith("/memories"):
            retain_calls += 1
            body = json.loads(request.content)
            if fail_second_unit and "amb-unit:conv-30" in body["items"][0]["tags"]:
                return httpx.Response(403)
            retained.update(item["document_id"] for item in body["items"])
            return httpx.Response(200, json={
                "success": True, "async": True, "items_count": len(body["items"]), "operation_id": body["operation_id"],
            })
        if "/operations/" in request.url.path:
            return httpx.Response(200, json={"status": "completed"})
        if request.url.path.endswith("/documents"):
            rows = [{"id": doc_id} for doc_id in retained if doc_id.startswith(request.url.params["q"])]
            return httpx.Response(200, json={"items": rows, "total": len(rows)})
        assert request.url.path.endswith("/memories/recall")
        return httpx.Response(200, json={"results": [{"id": "fact", "text": "new" if fail_second_unit else "old"}]})

    _configure_contextbase(monkeypatch, handler)
    monkeypatch.setenv("GEMINI_API_KEY", "stub")
    monkeypatch.setattr(LoComoDataset, "_load_raw", raw_dataset)
    monkeypatch.setattr(cli, "get_answer_llm", Model)
    monkeypatch.setattr(llm, "get_judge_llm", Model)
    monkeypatch.setattr(runner, "count_tokens", len)
    args = ["run", "--dataset", "locomo", "--split", "locomo10", "--memory", "contextbase", "--output-dir", str(tmp_path)]
    output = tmp_path / "locomo/contextbase/rag/locomo10.json"
    first = CliRunner().invoke(cli.app, args)
    assert first.exit_code == 0, first.exception
    original = json.loads(output.read_text())
    assert original["total_queries"] == 2 and original["correct"] == 1

    monkeypatch.setenv("CONTEXTBASE_BENCH_RUN_ID", original["ingestion_run_id"])
    reused = CliRunner().invoke(cli.app, [*args, "--skip-ingestion"])
    assert reused.exit_code == 0, reused.exception
    assert retain_calls == 2
    assert json.loads(output.read_text())["ingestion_run_id"] == original["ingestion_run_id"]

    fail_judgment = False
    retried = CliRunner().invoke(cli.app, [*args, "--skip-ingestion", "--only-failed"])
    assert retried.exit_code == 0, retried.exception
    assert retain_calls == 2
    retried_output = json.loads(output.read_text())
    assert retried_output["total_queries"] == 2 and retried_output["correct"] == 2

    fail_second_unit = True
    second = CliRunner().invoke(cli.app, args)
    assert isinstance(second.exception, httpx.HTTPStatusError)
    partial = json.loads(output.read_text())
    assert partial["ingestion_run_id"] != original["ingestion_run_id"]
    assert partial["total_queries"] == 1
    assert [row["query_id"] for row in partial["results"]] == ["conv-26_q0"]
    assert partial["results"][0]["context"].endswith("new")

    for run_id in ("unknown-run", partial["ingestion_run_id"]):
        monkeypatch.setenv("CONTEXTBASE_BENCH_RUN_ID", run_id)
        invalid = CliRunner().invoke(cli.app, [*args, "--skip-ingestion"])
        assert isinstance(invalid.exception, ValueError)
        assert "no completed ingest" in str(invalid.exception)
        assert json.loads(output.read_text()) == partial
