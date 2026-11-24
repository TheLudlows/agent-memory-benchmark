import json

import httpx
import pytest
from typer.testing import CliRunner

from memory_bench import cli, llm, runner
from memory_bench.dataset import knowledge_data as data
from memory_bench.dataset.knowledge import BrowseCompPlusDataset, LegalBenchRagDataset, VidoreMarkdownDataset
from memory_bench.dataset.locomo import LoComoDataset
from memory_bench.memory import get_memory_provider
from memory_bench.memory import contextbase
from memory_bench.models import Document
from memory_bench.workbench import SCORING_INCOMPLETE_EXIT_CODE
from memory_bench.workbench_worker import run


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


@pytest.mark.parametrize("dataset_name", ["locomo", "precisionmembench"])
def test_ingestion_reuse_checks_checkpoint_input_and_each_remote_document(monkeypatch, tmp_path, dataset_name):
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
        if request.url.path.endswith("/memories/recall"):
            include = json.loads(request.content)["include"]
            assert ("source_facts" in include) == (dataset_name == "precisionmembench")
            return httpx.Response(200, json={
                "results": [{"id": "fact", "text": "A fact", "document_id": retained[0]}],
            })
        assert request.url.path.endswith("/documents")
        assert request.url.params["q"] == f"amb-{run_id}-"
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        return httpx.Response(200, json={"items": [{"id": item} for item in retained[offset:offset + 1]], "total": len(retained)})

    docs = [Document(f"conv-26_session_{i}", f"session {i}", "conv-26") for i in (1, 2)]
    provider = _provider(monkeypatch, tmp_path, handler)
    provider.dataset_name = dataset_name
    run_id = provider.ingestion_run_id
    provider.ingest(docs)
    provider.cleanup()

    monkeypatch.setenv("CONTEXTBASE_BENCH_RUN_ID", run_id)
    resumed = get_memory_provider("contextbase")
    resumed.dataset_name = dataset_name
    resumed.prepare(tmp_path, {"conv-26"}, reset=False)
    resumed.validate_ingestion(docs)
    assert offsets == [0, 1]
    hits, _ = resumed.retrieve("test", user_id="conv-26")
    assert hits[0].source_ids == [docs[0].id]
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


@pytest.mark.parametrize(("dataset", "split", "foreign_source", "auto_bank"), [
    (VidoreMarkdownDataset, "computer_science", False, False),
    (VidoreMarkdownDataset, "computer_science", False, True),
    (BrowseCompPlusDataset, "evidence", False, False),
    (VidoreMarkdownDataset, "computer_science", True, False),
])
def test_contextbase_knowledge_worker_scores_original_documents(
    monkeypatch, tmp_path, dataset, split, foreign_source, auto_bank,
):
    monkeypatch.setattr(dataset, "_load", lambda self, split: data.RankedSuite(
        name="scope",
        items=(
            data.TextItem(id="page-one", text="first source", filename="page-one.md"),
            data.TextItem(id="page-two", text="second source", filename="page-two.md"),
        ),
        queries=(data.RankedQuery(id="q1", text="Query", group="english", relevant={"page-two": 1}),),
    ))
    calls, retained = [], {}
    run_id = "09a6147e16b94919a908f98863235a51c"
    bank_id = f"amb_rag_{run_id}" if auto_bank else "test-bank"

    def handler(request):
        path = request.url.path.removeprefix(f"/tenant-api/v1/default/banks/{bank_id}")
        calls.append(path)
        if request.method == "PUT":
            assert auto_bank and calls == [""]
            assert request.headers["Authorization"] == "Bearer test-admin-token"
            assert request.headers["If-None-Match"] == "*"
            assert json.loads(request.content) == {"enable_observations": False}
            return httpx.Response(200, json={"bank_id": bank_id})
        assert request.headers["Authorization"] == "Bearer test-token"
        if path == "/memories":
            payload = json.loads(request.content)
            assert payload["async"] is True
            for item in payload["items"]:
                assert item["document_id"].startswith("amb-")
                retained[item["metadata"]["doc_id"]] = item
            assert {item["content"] for item in retained.values()} == {"first source", "second source"}
            return httpx.Response(200, json={
                "success": True, "async": True, "items_count": 2, "operation_ids": ["retain-1", "retain-2"],
            })
        if path in {"/operations/retain-1", "/operations/retain-2"}:
            assert request.method == "GET"
            return httpx.Response(200, json={"status": "completed"})
        assert path == "/memories/recall"
        assert calls == ([""] if auto_bank else []) + [
            "/memories", "/operations/retain-1", "/operations/retain-2", "/memories/recall",
        ]
        body = json.loads(request.content)
        assert body["tags_match"] == "all_strict"
        assert body["tags"] == retained["page-one"]["tags"] == retained["page-two"]["tags"]
        assert body["types"] == ["world", "experience"]
        source = "amb-previous-run-page-two" if foreign_source else retained["page-two"]["document_id"]
        return httpx.Response(200, json={"results": [
            {"document_id": source, "text": "best passage"},
            {"document_id": source, "text": "another passage on the same page"},
            {"document_id": retained["page-one"]["document_id"], "text": "less relevant"},
        ]})

    _configure_contextbase(monkeypatch, handler)
    monkeypatch.setenv("CONTEXTBASE_BENCH_ADMIN_TOKEN", "test-admin-token")
    if auto_bank:
        monkeypatch.delenv("CONTEXTBASE_BANK_ID")
    (tmp_path / "run.json").write_text(json.dumps({
        "id": run_id, "name": "ContextBase document ranking",
        "selection": {"backend": "contextbase-knowledge", "dataset": dataset.name, "split": split, "query_limit": 1},
        "target": {"bank_id": bank_id, "auto_bank": auto_bank},
        "models": {},
    }), encoding="utf-8")

    expected_exit = SCORING_INCOMPLETE_EXIT_CODE if foreign_source else 0
    assert run(tmp_path) == expected_exit
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["score"] == (None if foreign_source else 1.0)
    assert summary["score_type"] == "ndcg@10"
    assert summary["ingested_docs"] == 2
    assert summary["answer_llm"] is None and summary["judge_llm"] is None


@pytest.mark.parametrize("include_chunks", [True, False])
def test_contextbase_legalbench_scores_ranked_original_chunks(monkeypatch, tmp_path, include_chunks):
    evidence = ["Disclosure requires notice.", "Return all copies.", "The agreement lasts two years."]
    distractors = ["First distractor.", "Second distractor.", "Third distractor."]
    related = "\n\n".join(evidence)
    unrelated = "\n\n".join(distractors)
    monkeypatch.setattr(LegalBenchRagDataset, "_load", lambda self, split: data.RankedSuite(
        name="legalbench",
        items=(
            data.TextItem(id="contract-a", text=related, filename="a.txt"),
            data.TextItem(id="contract-b", text=unrelated, filename="b.txt"),
        ),
        queries=(data.RankedQuery(id="q1", text="What conditions apply?", group="MAUD", relevant={"contract-a": 1}),),
        spans={"q1": [
            ("contract-a", related.index(text), related.index(text) + len(text)) for text in evidence
        ]},
    ))
    run_id = "09a6147e16b94919a908f98863235a51c"
    bank_id = f"amb_rag_{run_id}"
    retained = {}
    submitted, completed = set(), set()

    def handler(request):
        path = request.url.path.removeprefix(f"/tenant-api/v1/default/banks/{bank_id}")
        if request.method == "PUT":
            assert path == ""
            assert request.headers["Authorization"] == "Bearer test-admin-token"
            assert json.loads(request.content) == {"enable_observations": False, "retain_extraction_mode": "chunks"}
            return httpx.Response(200, json={"bank_id": bank_id})
        assert request.headers["Authorization"] == "Bearer test-token"
        if path == "/memories":
            assert request.method == "POST"
            body = json.loads(request.content)
            if len(body["items"]) > 1:
                return httpx.Response(413)
            item = body["items"][0]
            retained[item["metadata"]["doc_id"]] = item
            operation_id = body["operation_id"]
            assert operation_id not in submitted
            submitted.add(operation_id)
            return httpx.Response(200, json={
                "success": True, "async": True, "items_count": 1, "operation_id": operation_id,
            })
        if path.startswith("/operations/"):
            assert request.method == "GET"
            completed.add(path.removeprefix("/operations/"))
            return httpx.Response(200, json={"status": "completed"})
        assert request.method == "POST" and path == "/memories/recall"
        assert len(submitted) == 2 and completed == submitted
        assert {item["content"] for item in retained.values()} == {related, unrelated}
        body = json.loads(request.content)
        assert body["include"] == {"entities": None, "chunks": {"max_tokens": 4096}}
        assert body["tags_match"] == "all_strict"
        assert body["tags"] == retained["contract-a"]["tags"] == retained["contract-b"]["tags"]
        ranked = [
            ("contract-a", "a-1", evidence[0]),
            ("contract-a", "a-1", evidence[0]),
            ("contract-b", "b-1", distractors[0]),
            ("contract-b", "b-2", distractors[1]),
            ("contract-b", "b-3", distractors[2]),
            ("contract-a", "a-2", evidence[1]),
            ("contract-a", "a-3", evidence[2]),
        ]
        return httpx.Response(200, json={
            "results": [
                {"document_id": retained[source]["document_id"], "chunk_id": chunk, "text": "An extracted fact."}
                for source, chunk, _ in ranked
            ],
            "chunks": {chunk: {"text": text} for _, chunk, text in reversed(ranked)} if include_chunks else None,
        })

    _configure_contextbase(monkeypatch, handler)
    monkeypatch.delenv("CONTEXTBASE_BANK_ID")
    monkeypatch.setenv("CONTEXTBASE_BENCH_ADMIN_TOKEN", "test-admin-token")
    (tmp_path / "run.json").write_text(json.dumps({
        "id": run_id, "name": "ContextBase contract evidence",
        "selection": {"backend": "contextbase-knowledge", "dataset": "legalbench-rag", "split": "mini", "query_limit": 1},
        "target": {"bank_id": bank_id, "auto_bank": True},
        "models": {},
    }), encoding="utf-8")

    assert run(tmp_path) == (0 if include_chunks else SCORING_INCOMPLETE_EXIT_CODE)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["score_type"] == "recall@5"
    assert summary["ingested_docs"] == 2
    if include_chunks:
        assert summary["score"] == pytest.approx((len(evidence[0]) + len(evidence[1])) / sum(map(len, evidence)))
    else:
        assert summary["score"] is None
        assert summary["scoring_failed_queries"] == 1


def test_contextbase_knowledge_auto_bank_requires_fresh_ingestion_and_admin_access(monkeypatch, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"error": "Forbidden"})

    _configure_contextbase(monkeypatch, handler)
    monkeypatch.delenv("CONTEXTBASE_BANK_ID")
    monkeypatch.setenv("CONTEXTBASE_BENCH_ADMIN_TOKEN", "test-admin-token")
    provider = get_memory_provider("contextbase-knowledge")
    try:
        with pytest.raises(ValueError, match="original CONTEXTBASE_BANK_ID"):
            provider.prepare(tmp_path, reset=False)
        assert calls == []
        with pytest.raises(RuntimeError, match=r"HTTP 403.*CONTEXTBASE_BENCH_ADMIN_TOKEN"):
            provider.prepare(tmp_path)
        assert [(request.method, request.url.path) for request in calls] == [
            ("PUT", f"/tenant-api/v1/default/banks/amb_rag_{provider.ingestion_run_id}"),
        ]
    finally:
        provider.cleanup()


def test_cli_rejects_unsupported_contextbase_knowledge_dataset():
    result = CliRunner().invoke(cli.app, [
        "run", "--memory", "contextbase-knowledge", "--mode", "retrieval",
        "--dataset", "vidore-v3-pdf", "--split", "computer_science",
    ])
    assert result.exit_code == 2
    assert "ContextBase knowledge retrieval supports" in result.output
