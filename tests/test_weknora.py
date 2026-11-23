"""Exercise the Workbench worker and WeKnora HTTP contract with small synthetic corpora."""

import json

import httpx
import pytest

from memory_bench.dataset import knowledge_data as data
from memory_bench.dataset.knowledge import (
    BrowseCompPlusDataset,
    LegalBenchRagDataset,
    MiraclVisionDataset,
    VidoreMarkdownDataset,
    VidorePdfDataset,
)
from memory_bench.memory import weknora
from memory_bench.workbench import SCORING_INCOMPLETE_EXIT_CODE
from memory_bench.workbench_worker import run


@pytest.fixture
def weknora_http(monkeypatch):
    monkeypatch.setenv("WEKNORA_API_URL", "http://weknora.test/api/v1/")
    monkeypatch.setenv("WEKNORA_API_KEY", "test-secret")
    monkeypatch.setenv("WEKNORA_EMBEDDING_MODEL_ID", "embed-id")
    monkeypatch.setenv("WEKNORA_VLM_MODEL_ID", "vlm-id")
    client = httpx.Client

    def configure(handler):
        def handle(request):
            assert request.url.host == "weknora.test"
            assert request.headers["X-API-Key"] == "test-secret"
            assert request.url.path.startswith("/api/v1/")
            return handler(request)

        def mock_client(*args, **kwargs):
            return client(*args, transport=httpx.MockTransport(handle), **kwargs)

        monkeypatch.setattr(httpx, "Client", mock_client)

    return configure


def _job(run_dir, dataset, split):
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "id": "weknora-smoke",
        "name": "WeKnora benchmark smoke",
        "selection": {"backend": "weknora", "dataset": dataset.name, "split": split, "query_limit": 1},
        "models": {},
    }), encoding="utf-8")


def _success(data):
    return httpx.Response(200, json={"success": True, "data": data})


@pytest.mark.parametrize(("dataset_class", "split"), [
    (VidoreMarkdownDataset, "hr"),
    (BrowseCompPlusDataset, "evidence"),
])
def test_weknora_scores_ranked_text_after_the_index_is_ready(dataset_class, split, weknora_http, monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_class, "_load", lambda self, split: data.RankedSuite(
        name="text",
        items=(
            data.TextItem(id="first-page", text="Repeated page text", filename="page1.md"),
            data.TextItem(id="second-page", text="Repeated page text", filename="page2.md"),
        ),
        queries=(data.RankedQuery(id="q1", text="A query", group="english", relevant={"second-page": 1}),),
    ))
    uploaded = {}
    batch_reads = []
    monkeypatch.setattr(weknora, "sleep", lambda seconds: None)

    def respond(request):
        path = request.url.path.removeprefix("/api/v1")
        if path == "/knowledge-bases":
            assert request.method == "POST"
            body = json.loads(request.content)
            assert body["name"].startswith("amb-")
            assert body["embedding_model_id"] == "embed-id"
            return _success({"id": "kb-id"})
        if path == "/knowledge-bases/kb-id/knowledge/manual":
            body = json.loads(request.content)
            assert body["content"] == "Repeated page text"
            assert body["status"] == "publish"
            remote_id = "wk-" + body["title"]
            uploaded[remote_id] = body
            return _success({"id": remote_id})
        if path == "/knowledge/batch":
            returned = request.url.params.get_list("ids")
            assert set(returned) == set(uploaded)
            assert request.url.params["kb_id"] == "kb-id"
            batch_reads.append(returned)
            status = "finalizing" if len(batch_reads) == 1 else "completed"
            return _success([{"id": item, "parse_status": status} for item in returned])
        if path == "/knowledge-search":
            assert len(batch_reads) == 2 and len(uploaded) == 2
            assert json.loads(request.content) == {
                "query": "A query", "knowledge_base_ids": ["kb-id"], "match_count": 200,
            }
            return _success([
                {"knowledge_id": "wk-page2.md", "knowledge_base_id": "kb-id", "score": 0.99},
                {"knowledge_id": "wk-page1.md", "knowledge_base_id": "kb-id", "score": 0.1},
            ])
        pytest.fail(f"Unexpected WeKnora request {request.method} {path}")

    weknora_http(respond)
    run_dir = tmp_path / "run"
    _job(run_dir, dataset_class, split)
    assert run(run_dir) == 0
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["score"] == 1.0
    assert summary["score_type"] == "ndcg@10"
    assert summary["ingested_docs"] == 2
    assert summary["p95_retrieve_time_ms"] is not None
    assert summary["answer_llm"] is None and summary["judge_llm"] is None


@pytest.mark.parametrize(("kind", "has_page", "expected_score"), [
    ("pdf", True, 1.0),
    ("pdf", False, None),
    ("image", False, 1.0),
    ("legal", False, 1.0),
])
def test_weknora_maps_file_results_to_the_benchmark_sources(
    kind, has_page, expected_score, weknora_http, monkeypatch, tmp_path,
):
    contract_text = "条款 ABC / 条款 ABC"
    if kind == "legal":
        dataset_class, split = LegalBenchRagDataset, "mini"
        item = data.TextItem(id="original-doc", text=contract_text, filename="contract.txt")
        duplicate = data.TextItem(id="duplicate-doc", text=contract_text, filename="duplicate.txt")
        gold = {"duplicate-doc": 1}
        extra = {"spans": {"q1": [("duplicate-doc", 9, 15)]}}
        upload_bytes = contract_text.encode()
    else:
        if kind == "pdf":
            import pypdfium2

            dataset_class, split = VidorePdfDataset, "hr"
            source = tmp_path / "report.pdf"
            pdf = pypdfium2.PdfDocument.new()
            for _ in range(2):
                pdf.new_page(120, 120).close()
            pdf.save(source)
            pdf.close()
            gold = {"page-2": 1}
            extra = {"page_ids": {"original-doc": ("page-1", "page-2")}}
        else:
            from PIL import Image

            dataset_class, split = MiraclVisionDataset, "zh"
            source = tmp_path / "article.jpg"
            Image.new("RGB", (2, 2)).save(source)
            gold = {"original-doc": 1}
            extra = {}
        item = data.FileItem(id="original-doc", path=source)
        upload_bytes = source.read_bytes()

    monkeypatch.setattr(dataset_class, "_load", lambda self, split: data.RankedSuite(
        name=kind,
        items=(item, duplicate) if kind == "legal" else (item,),
        queries=(data.RankedQuery(id="q1", text="A query", group="test", relevant=gold),),
        **extra,
    ))
    uploaded = set()

    def respond(request):
        path = request.url.path.removeprefix("/api/v1")
        if path == "/knowledge-bases":
            body = json.loads(request.content)
            assert body["embedding_model_id"] == "embed-id"
            if kind == "image":
                assert body["vlm_config"]["model_id"] == "vlm-id"
                assert body["image_processing_config"]["image_vector_enabled"] is True
                return _success({"id": "kb-id", "image_processing_config": body["image_processing_config"]})
            return _success({"id": "kb-id"})
        if path == "/knowledge-bases/kb-id/knowledge/file":
            assert "multipart/form-data" in request.headers["Content-Type"]
            body = request.read()
            assert upload_bytes in body
            remote_id = "remote-doc"
            if kind == "legal":
                if b'filename="duplicate.txt"' in body:
                    assert upload_bytes + b"\n\r\n--" in body
                    remote_id = "remote-duplicate"
                else:
                    assert b'filename="contract.txt"' in body
                    assert upload_bytes + b"\r\n--" in body
            uploaded.add(remote_id)
            return _success({"id": remote_id})
        if path == "/knowledge/batch":
            assert set(request.url.params.get_list("ids")) == uploaded
            return _success([{"id": item, "parse_status": "completed"} for item in uploaded])
        if path == "/chunks/remote-doc":
            assert kind == "image" and uploaded
            assert request.url.params["chunk_type"] == "image_vector"
            assert request.url.params["page_size"] == "1"
            return _success([{"chunk_type": "image_vector"}])
        if path == "/knowledge-search":
            assert uploaded
            body = json.loads(request.content)
            assert body["knowledge_base_ids"] == ["kb-id"]
            result = {"knowledge_id": "remote-doc", "knowledge_base_id": "kb-id", "content": "a matching passage"}
            if has_page:
                result["source_locators"] = [{"type": "pdf", "page": 2}]
            if kind == "legal":
                assert body["match_count"] == 5
                assert uploaded == {"remote-doc", "remote-duplicate"}
                result.update(knowledge_id="remote-duplicate", content="条款 ABC", start_at=9, end_at=15)
                result["source_locators"] = [{"type": "text", "start": 9, "end": 15}]
            return _success([result])
        pytest.fail(f"Unexpected WeKnora request {request.method} {path}")

    weknora_http(respond)
    run_dir = tmp_path / "run"
    _job(run_dir, dataset_class, split)
    assert run(run_dir) == (0 if expected_score is not None else SCORING_INCOMPLETE_EXIT_CODE)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["score"] == expected_score
    assert summary["scoring_failed_queries"] == int(expected_score is None)
