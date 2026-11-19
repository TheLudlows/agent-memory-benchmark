import asyncio
import io
import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from memory_bench.dataset.beam import BEAMDataset
from memory_bench.judge import ScoringError
from memory_bench.memory.base import MemoryProvider
from memory_bench.memory.hindsight import HindsightHTTPMemoryProvider
from memory_bench.memory.mem0_cloud import Mem0CloudMemoryProvider
from memory_bench.models import AnswerResult, Document, QueryResult
from memory_bench.modes.rag import RAGMode
from memory_bench.runner import EvaluationProgress, EvalRunner
from memory_bench.workbench import create_app


@pytest.fixture(autouse=True)
def private_workbench_config(monkeypatch, tmp_path):
    directory = tmp_path / "private-config"
    monkeypatch.setenv("AMB_WORKBENCH_CONFIG_DIR", str(directory))
    return directory


@pytest.fixture
def mock_services():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, data, status=200):
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            calls.append((self.path, {}))
            self.send_json({"status": "completed"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/chat/completions"):
                calls.append((self.path, {**body, "authorization": self.headers.get("Authorization")}))
                if body["model"] == "judge-fail":
                    self.send_json({"error": {"message": "test judge unavailable"}}, status=401)
                    return
                if body["model"] == "judge-probe-auth":
                    self.send_json({"error": {"message": self.headers.get("Authorization")}}, status=401)
                    return
                schema = body["response_format"]["json_schema"]["schema"]["properties"]
                prompt = body["messages"][0]["content"]
                values = {"answer": "Paris", "reasoning": "From memory", "score": 1.0, "correct": True, "reason": "Match"}
                if body["model"] == "judge-probe-invalid":
                    values["correct"] = "true"
                if "What did they say about pets?" in prompt:
                    values["answer"] = "I don't have enough information to answer this question."
                    values["score"] = float(values["answer"] in prompt)
                elif "What is their salary?" in prompt:
                    values["answer"] = "$120,000"
                    values["score"] = 0.0
                self.send_json({"choices": [{"message": {"content": json.dumps({key: values[key] for key in schema})}}]})
                return
            calls.append((self.path, body))
            if self.path.endswith("/memories"):
                if self.headers.get("Authorization") != "Bearer test-bearer-value" or "/banks/ui-bank/" not in self.path:
                    self.send_json({"error": "Incorrect Workbench backend configuration"}, status=403)
                    return
                self.send_json({"success": True, "async": True, "items_count": len(body["items"]), "operation_id": "test-op"})
            elif self.path.endswith("/memories/recall"):
                empty_queries = {"No memory?", "What did they say about pets?", "What is their salary?"}
                matches = [] if body["query"] in empty_queries else [{"id": "fact", "text": "The location is Paris"}]
                self.send_json({"results": matches})
            else:
                self.send_json({}, status=404)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", calls
        finally:
            server.shutdown()
            thread.join()


def await_result(client, response):
    assert response.status_code == 201, response.text
    detail_url = f"/api/runs/{response.json()['id']}"
    deadline = time.monotonic() + 40
    while True:
        detail = client.get(detail_url).json()
        if detail["status"] != "running" or time.monotonic() >= deadline:
            return detail_url, detail
        time.sleep(0.1)


def test_standard_question_counts_without_loading_datasets(monkeypatch, tmp_path):
    for variable in ("LOCOMO_DATA_PATH", "LONGMEMEVAL_DATA_PATH", "BEAM_DATA_PATH", "BEAM_DATA_SPLIT",
                     "BEAM_100K_DATA_PATH", "BEAM_500K_DATA_PATH", "BEAM_1M_DATA_PATH", "BEAM_10M_DATA_PATH"):
        monkeypatch.delenv(variable, raising=False)

    with TestClient(create_app(tmp_path / "runs"), base_url="http://127.0.0.1") as client:
        for dataset, split, count in (
            ("locomo", "locomo10", 1540), ("longmemeval", "s", 500),
            ("beam", "100k", 400), ("beam", "500k", 700), ("beam", "1m", 700), ("beam", "10m", 200),
        ):
            response = client.get(f"/api/datasets/{dataset}/{split}/query-count")
            assert response.status_code == 200, response.text
            assert response.json() == {"query_count": count}
        assert client.get("/api/datasets/locomo/s/query-count").status_code == 404
        monkeypatch.setenv("LOCOMO_DATA_PATH", str(tmp_path / "missing.json"))
        assert client.get("/api/datasets/locomo/locomo10/query-count").status_code == 422


def test_datasets_run_through_local_api_and_worker(monkeypatch, tmp_path, mock_services):
    endpoint, calls = mock_services
    options = {
        "OPENAI_API_KEY": "test-model-key",
        "OPENAI_BASE_URL": endpoint + "/v1",
        "OMB_ANSWER_LLM": "openai",
        "OMB_ANSWER_MODEL": "answer-stub",
        "OMB_JUDGE_LLM": "openai",
        "OMB_JUDGE_MODEL": "judge-stub",
    }
    for key, value in options.items():
        monkeypatch.setenv(key, value)
    for name in ("CONTEXTBASE_API_URL", "CONTEXTBASE_API_TOKEN", "CONTEXTBASE_BANK_ID", "CONTEXTBASE_BENCH_BANKS"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))

    turn = {"role": "user", "content": "The location is Paris", "has_answer": True}
    fixtures = [
        ("locomo", "locomo10", "LOCOMO_DATA_PATH", {
            "sample_id": "conv-test",
            "conversation": {
                "speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1:00 pm on 1 May, 2023",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": turn["content"]}],
            },
            "qa": [
                {"question": "Where?", "answer": "Paris", "category": 1, "evidence": ["D1:1"]},
                {"question": "Unknown?", "category": 5},
            ],
        }),
        ("beam", "100k", "BEAM_100K_DATA_PATH", {
            "conversation_id": "beam-test", "chat": [[{**turn, "time_anchor": "May-1-2023"}]],
            "probing_questions": {
                "abstention": [
                    {"question": "What did they say about pets?",
                     "rubric": ["States that the conversation does not say anything about pets."]},
                    {"question": "What is their salary?"},
                ],
                "information_extraction": [
                    {"question": "Where?", "answer": "Paris"},
                    {"question": "No memory?", "answer": "Paris"},
                ],
            },
        }),
        ("longmemeval", "s", "LONGMEMEVAL_DATA_PATH", {
            "question_id": "long-test", "question_type": "single-session-user", "question": "Where?", "answer": "Paris",
            "haystack_sessions": [[turn], [turn]], "haystack_dates": ["2023/05/01", "2023/05/02"],
            "haystack_session_ids": ["same", "same"],
        }),
    ]

    results_dir = tmp_path / "runs"
    with TestClient(create_app(results_dir), base_url="http://127.0.0.1") as client:
        assert client.get("/").status_code == 200
        assert client.get("/assets/workbench.js").status_code == 200
        config = client.get("/api/config").json()
        assert set(config["datasets"]) == {case[0] for case in fixtures}
        assert set(config["backends"]) == {"contextbase", "mem0-cloud", "hindsight-http"}
        assert config["backends"]["contextbase"]["missing"] == ["API 地址", "API Key", "Bank ID"]

        connection = {"api_url": endpoint, "api_key": "test-bearer-value", "bank_id": "ui-bank"}
        reused = {"api_url": endpoint, "bank_id": "ui-bank", "use_saved_key": True}
        saved_fingerprints = {}
        document_fingerprints = {}
        for dataset, split, variable, data in fixtures:
            dataset_path = tmp_path / f"{dataset}.json"
            dataset_path.write_text(json.dumps([data]))
            monkeypatch.setenv(variable, str(dataset_path))
            question_count = 4 if dataset == "beam" else 1
            count_response = client.get(f"/api/datasets/{dataset}/{split}/query-count")
            assert count_response.status_code == 200, count_response.text
            assert count_response.json() == {"query_count": question_count}
            selection = {
                "name": f"{dataset} baseline", "dataset": dataset, "split": split,
                "backend": "contextbase", "query_limit": None,
                "connection": connection if dataset == "locomo" else reused,
            }
            assert client.post("/api/runs", json={**selection, "split": "../invalid"}).status_code == 422
            assert client.post("/api/runs", json=selection, headers={"Origin": "https://elsewhere.test"}).status_code == 403
            response = client.post("/api/runs", json=selection)
            assert response.status_code == 201, response.text
            assert response.json()["backend_config"]["has_key"]
            assert "test-bearer-value" not in response.text
            assert client.post("/api/runs", json=selection).status_code == 409
            detail_url, detail = await_result(client, response)
            assert detail["status"] == "completed", detail["log"]
            document_count = 2 if dataset == "longmemeval" else 1
            assert detail["progress"] == {
                "stage": "finalizing", "completed": question_count, "total": question_count,
                "ingested_docs": document_count, "total_docs": document_count,
            }
            assert detail["summary"]["score"] == (0.5 if dataset == "beam" else 1.0)
            assert detail["summary"]["judge_llm"] == "openai:judge-stub"
            assert detail["summary"]["ingested_docs"] == (2 if dataset == "longmemeval" else 1)
            assert detail["result_available"]
            assert detail["name"] == selection["name"]
            assert detail["target"] == {"api_url": endpoint, "bank_id": "ui-bank"}
            metrics = client.get(detail_url + "/metrics").json()
            assert metrics["summary"] == detail["summary"] and "log" not in metrics
            assert metrics["summary"]["p95_retrieve_time_ms"] >= 0
            saved_fingerprints[dataset] = metrics["summary"]["question_fingerprint"]
            document_fingerprints[dataset] = metrics["summary"]["document_fingerprint"]
            assert metrics["summary"]["ingestion_wall_time_ms"] == metrics["summary"]["ingestion_time_ms"]
            assert "test-bearer-value" not in json.dumps(detail)
            assert "test-bearer-value" not in json.dumps(metrics)
            result = client.get(detail_url + "/result")
            assert "attachment" in result.headers["Content-Disposition"]
            rows = result.json()["results"]
            if dataset == "beam":
                assert result.json()["accuracy"] == 0.5
                assert [row["score"] for row in rows] == [1.0, 0.0, 1.0, 0.0]
                assert rows[0]["context"] == rows[1]["context"] == ""
                assert rows[0]["answer"] == "I don't have enough information to answer this question."
                assert rows[1]["answer"] == "$120,000"
            else:
                assert rows[0]["answer"] == "Paris"

        ingests = [body for path, body in calls if path.endswith("/memories")]
        recalls = [body for path, body in calls if path.endswith("/memories/recall")]
        assert len({body["tags"][0] for body in recalls}) == len(fixtures)
        assert all(body["tags_match"] == "all_strict" for body in recalls)
        assert len({item["document_id"] for item in ingests[-1]["items"]}) == 2
        assert len(client.get("/api/runs").json()) == len(fixtures)

        beam_path = tmp_path / "beam.json"
        beam_rows = json.loads(beam_path.read_text())
        beam_rows[0]["chat"][0][0]["content"] = "The input conversation now says London."
        beam_path.write_text(json.dumps(beam_rows))
        monkeypatch.setenv("OMB_JUDGE_MODEL", "judge-fail")
        detail_url, detail = await_result(client, client.post("/api/runs", json={
            "name": "BEAM scoring offline", "connection": connection,
            "dataset": "beam", "split": "100k", "backend": "contextbase", "query_limit": 4,
        }))
        assert detail["status"] == "scoring_incomplete", detail["log"]
        assert detail["summary"]["scoring_failed_queries"] == 3
        assert detail["summary"]["score"] is None
        assert detail["summary"]["question_fingerprint"] == saved_fingerprints["beam"]
        assert detail["summary"]["document_fingerprint"] != document_fingerprints["beam"]
        assert len(set(saved_fingerprints.values())) == len(fixtures)
        assert client.get("/api/config").json()["backends"]["contextbase"]["api_url"] == endpoint
        result = client.get(detail_url + "/result").json()
        assert result["accuracy"] is None and result["scoring_failed_queries"] == 3
        rows = result["results"]
        assert [(r["score"], r["scoring_failed"]) for r in rows] == [(None, True)] * 3 + [(0.0, False)]
        assert rows[2]["answer"] == "Paris"

        response = client.post("/api/runs", json={
            "name": "Cancel test", "connection": connection,
            "dataset": "locomo", "split": "locomo10", "backend": "contextbase", "query_limit": 1,
        })
        assert response.status_code == 201
        detail_url = f"/api/runs/{response.json()['id']}"
        assert client.post(detail_url + "/cancel").status_code == 202
        deadline = time.monotonic() + 10
        while client.get(detail_url).json()["status"] != "cancelled" and time.monotonic() < deadline:
            time.sleep(0.1)
        assert client.get(detail_url).json()["status"] == "cancelled"

    with TestClient(create_app(results_dir), base_url="http://127.0.0.1") as restarted:
        history = restarted.get("/api/runs").json()
        assert len(history) == len(fixtures) + 2
        assert [job["status"] for job in history[:2]] == ["cancelled", "scoring_incomplete"]
        assert restarted.get("/api/runs?limit=1&offset=1").json() == [history[1]]
        assert history[0]["name"] == "Cancel test"
        old_metrics = restarted.get(f"/api/runs/{history[-1]['id']}/metrics").json()
        assert old_metrics["summary"]["question_fingerprint"] == saved_fingerprints["locomo"]


def test_workbench_hindsight_and_mem0_use_fresh_namespaces(monkeypatch, tmp_path):
    from hindsight_client import Hindsight

    monkeypatch.setattr(Hindsight, "__init__", lambda self, **kwargs: None)
    store = tmp_path / "beam" / "web-test" / "_store" / "100k" / "all"
    hindsight = HindsightHTTPMemoryProvider(bank_prefix="amb-web-test-")
    hindsight.prepare(store, {"conversation"})
    assert hindsight._bank_id_for("conversation") == "amb-web-test-beam-100k-uconversation"
    assert "retain_mission" in hindsight._bank_kwargs()

    calls = []

    class Mem0Client:
        def add(self, **kwargs):
            calls.append(kwargs)

        def search(self, query, **kwargs):
            calls.append(kwargs)
            return {"results": []}

    mem0 = Mem0CloudMemoryProvider(user_prefix="amb-web-test-")
    mem0._client = Mem0Client()
    mem0.ingest([Document("doc-1", "text", user_id="conversation")])
    mem0.retrieve("Where?", user_id="conversation")
    assert calls[0]["user_id"] == calls[1]["filters"]["user_id"] == "amb-web-test-conversation"
    assert calls[0]["async_mode"] is False


def test_beam_workbench_requires_a_split_for_legacy_local_data(monkeypatch, tmp_path):
    for key in ("BEAM_100K_DATA_PATH", "BEAM_500K_DATA_PATH", "BEAM_1M_DATA_PATH", "BEAM_DATA_SPLIT"):
        monkeypatch.delenv(key, raising=False)
    local = tmp_path / "500k.json"
    local.write_text('[{"conversation_id":"local-500k"}]')
    monkeypatch.setenv("BEAM_DATA_PATH", str(local))
    dataset = BEAMDataset(require_local_split=True)

    def download(split, path):
        path.write_text(json.dumps([{"conversation_id": "download-" + split}]))

    monkeypatch.setattr(dataset, "_cache_path", lambda split: tmp_path / f"cached-{split}.json")
    monkeypatch.setattr(dataset, "_download", download)
    with TestClient(create_app(tmp_path / "runs"), base_url="http://127.0.0.1") as client:
        errors = client.get("/api/config").json()["split_errors"]["beam"]
        assert set(errors) == {"100k", "500k", "1m"}
        response = client.post("/api/runs", json={
            "name": "BEAM local", "connection": {"api_url": "https://example.test"},
            "dataset": "beam", "split": "500k", "backend": "contextbase", "query_limit": 5,
        })
        assert response.status_code == 422 and "BEAM_DATA_SPLIT" in response.json()["detail"]
        assert client.get("/api/runs").json() == []

        monkeypatch.setenv("BEAM_DATA_SPLIT", "500k")
        assert client.get("/api/config").json()["split_errors"]["beam"] == {}
    assert dataset._load_raw("500k")[0]["conversation_id"] == "local-500k"
    assert dataset._load_raw("100k")[0]["conversation_id"] == "download-100k"

    split_file = tmp_path / "1m.json"
    split_file.write_text('[{"conversation_id":"local-1m"}]')
    monkeypatch.setenv("BEAM_1M_DATA_PATH", str(split_file))
    assert dataset._load_raw("1m")[0]["conversation_id"] == "local-1m"


@pytest.mark.parametrize("payload", [{}, {"score": True}, {"score": "NaN"}, {"score": 2}])
def test_beam_rejects_an_invalid_judge_score(payload):
    class Judge:
        def generate(self, prompt, schema):
            return payload

    with pytest.raises(ScoringError):
        BEAMDataset._rubric_item_score("Question", "Answer", "Criterion", Judge())


@pytest.mark.parametrize("replace_on", ["http.response.start", "http.response.body"])
def test_result_download_during_checkpoint_replacement(tmp_path, replace_on):
    run_id = "0" * 32
    results = tmp_path / "runs"
    directory = results / run_id
    path = directory / "results/beam" / f"web-{run_id}" / "rag/100k.json"
    path.parent.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({
        "id": run_id, "status": "completed",
        "selection": {"dataset": "beam", "split": "100k", "backend": "contextbase"},
    }))
    old = json.dumps({"answer": "a" * 70_000}).encode()
    new = json.dumps({"answer": "b" * 80_000}).encode()
    path.write_bytes(old)
    app = create_app(results)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/api/runs/{run_id}/result")

    async def download():
        headers = {}
        chunks = []
        replaced = False

        async def send(message):
            nonlocal replaced
            if message["type"] == "http.response.start":
                headers.update(message["headers"])
            else:
                chunks.append(message["body"])
            if message["type"] == replace_on and not replaced:
                updated = path.with_suffix(".tmp")
                updated.write_bytes(new)
                updated.replace(path)
                replaced = True

        scope = {"type": "http", "method": "GET", "headers": [], "asgi": {"spec_version": "2.4"}}
        await endpoint(run_id)(scope, None, send)
        body = b"".join(chunks)
        assert body == (new if replace_on == "http.response.start" else old)
        if b"content-length" in headers:
            assert int(headers[b"content-length"]) == len(body)

    asyncio.run(download())


def test_beam_can_rejudge_a_legacy_empty_context_abstention(monkeypatch, tmp_path):
    dataset = BEAMDataset()
    monkeypatch.setattr(dataset, "_load_raw", lambda split: [{
        "conversation_id": "conv",
        "chat": [[{"role": "user", "content": "I live in Paris."}]],
        "probing_questions": {"abstention": [{"question": "What is my salary?"}]},
    }])
    refusal = "I don't have enough information to answer this question."

    class Judge:
        model_id = "judge-stub"

        def generate(self, prompt, schema):
            assert f"LLM RESPONSE:\n{refusal}" in prompt
            assert "does not provide a specific answer" in prompt
            return {"score": 1.0, "reason": "States that the information is unavailable"}

    class Memory(MemoryProvider):
        name = "test-memory"

        def ingest(self, documents):
            pytest.fail("cached rejudging should not ingest documents")

        def retrieve(self, query, k=10, user_id=None, query_timestamp=None):
            pytest.fail("cached rejudging should not retrieve")

    judge = Judge()
    monkeypatch.setattr(dataset, "default_judge_llm", lambda: judge)
    legacy = QueryResult(
        query_id="conv_abstention_0", query="What is my salary?", answer=refusal, reasoning="",
        context="", context_tokens=0, retrieve_time_ms=0.0, gold_answers=[], correct=False,
        judge_reason="empty context — no memories retrieved", score=0.0,
        meta={"question_category": "abstention"},
    )
    result = tmp_path / "beam" / Memory.name / "rag" / "100k.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"accuracy": 0.0, "results": [asdict(legacy)]}))

    summary = EvalRunner(tmp_path).run(
        dataset=dataset, split="100k", memory=Memory(), mode=RAGMode(judge),
        skip_answer=True, skip_ingestion=True, only_failed=True,
    )
    assert summary.accuracy == 1.0
    assert summary.scoring_failed_queries == 0
    assert summary.results[0].correct


def test_workbench_remembers_connections_by_backend_and_url(monkeypatch, tmp_path, private_workbench_config):
    monkeypatch.setenv("HINDSIGHT_HTTP_URL", "http://127.0.0.1:8888/")
    monkeypatch.setenv("HINDSIGHT_HTTP_KEY", "environment-secret")
    monkeypatch.setenv("CONTEXTBASE_API_URL", "https://env.example/context")
    monkeypatch.setenv("CONTEXTBASE_API_TOKEN", "context-environment-secret")
    monkeypatch.setenv("CONTEXTBASE_BANK_ID", "environment-bank")
    monkeypatch.setenv("OMB_ANSWER_LLM", "gemini")
    monkeypatch.setenv("OMB_JUDGE_LLM", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "model-test-key")
    directory = tmp_path / "runs"
    environments = []
    first = "https://service.example/a"
    second = "https://service.example/b"
    selection = {"name": "Connection history", "dataset": "locomo", "split": "locomo10"}

    def launch(*_, env, **__):
        environments.append(env)
        return SimpleNamespace(stdout=io.StringIO(), wait=lambda timeout=None: 0, poll=lambda: 0)

    monkeypatch.setattr("memory_bench.workbench.subprocess.Popen", launch)

    def start(client, backend, connection):
        response = client.post("/api/runs", json={**selection, "backend": backend, "connection": connection})
        _, detail = await_result(client, response)
        assert detail["status"] == "completed"
        assert "backend_config" not in detail
        return response.json()["backend_config"]

    with TestClient(create_app(directory), base_url="http://127.0.0.1") as client:
        config = client.get("/api/config")
        assert "environment-secret" not in config.text
        assert config.json()["backends"]["hindsight-http"]["has_key"]
        saved = start(client, "contextbase", {"api_url": first + "/", "api_key": "context-a-secret", "bank_id": "bank-a"})
        assert saved["api_url"] == first and saved["has_key"]
        assert environments[-1]["CONTEXTBASE_API_TOKEN"] == "context-a-secret"
        saved = start(client, "contextbase", {"api_url": second, "api_key": "context-b-secret", "bank_id": "bank-b"})
        assert [(item["api_url"], item["bank_id"]) for item in saved["connections"]] == [
            (second, "bank-b"), (first, "bank-a"), ("https://env.example/context", "environment-bank"),
        ]
        start(client, "mem0-cloud", {"api_url": first, "api_key": "mem0-private-key"})
        assert environments[-1]["MEM0_CLOUD_KEY"] == "mem0-private-key"

        saved = start(client, "hindsight-http", {"api_url": "http://127.0.0.1:9250"})
        assert not saved["has_key"] and environments[-1]["HINDSIGHT_HTTP_KEY"] == ""
        assert saved["connections"][1]["api_url"] == "http://127.0.0.1:8888"
        assert saved["connections"][1]["has_key"]

    with TestClient(create_app(directory), base_url="http://127.0.0.1") as client:
        config = client.get("/api/config")
        assert config.json()["backends"]["contextbase"]["api_url"] == second
        effective = {"api_url": first, "bank_id": "bank-a", "use_saved_key": True}
        saved = start(client, "contextbase", effective)
        assert saved["api_url"] == first
        assert environments[-1]["CONTEXTBASE_API_TOKEN"] == "context-a-secret"
        assert environments[-1]["CONTEXTBASE_BANK_ID"] == "bank-a"
        start(client, "mem0-cloud", {"api_url": first, "use_saved_key": True})
        assert environments[-1]["MEM0_CLOUD_KEY"] == "mem0-private-key"
        saved = start(client, "contextbase", {"api_url": first + "/", "api_key": "rotated-secret", "bank_id": "bank-updated"})
        assert [item["api_url"] for item in saved["connections"]] == [first, second, "https://env.example/context"]
        assert saved["bank_id"] == "bank-updated"
        assert client.get("/api/config").json()["backends"]["contextbase"] == saved
        history = client.get("/api/runs").text
        for key in ("environment-secret", "context-a-secret", "context-b-secret", "mem0-private-key", "rotated-secret"):
            assert key not in config.text and key not in history and key not in json.dumps(saved)
        storage, = private_workbench_config.glob("*/connections.json")
        assert json.loads(storage.read_text(encoding="utf-8"))["contextbase"][0]["api_key"] == "rotated-secret"
        assert not (directory / ".connections.json").exists()


def test_mem0_client_uses_configured_host(monkeypatch):
    import mem0

    monkeypatch.setenv("MEM0_CLOUD_URL", "https://example.test/mem0")
    monkeypatch.setenv("MEM0_CLOUD_KEY", "test-api-key")
    calls = []
    monkeypatch.setattr(mem0, "MemoryClient", lambda **kwargs: calls.append(kwargs))
    Mem0CloudMemoryProvider()._ensure_client()
    assert calls == [{"api_key": "test-api-key", "host": "https://example.test/mem0"}]


def test_workbench_requires_per_run_details_and_never_echoes_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXTBASE_API_URL", "https://default.example")
    monkeypatch.setenv("CONTEXTBASE_API_TOKEN", "saved-round-key")
    monkeypatch.setenv("CONTEXTBASE_BANK_ID", "default-bank")
    basic = {"dataset": "locomo", "split": "locomo10", "backend": "contextbase"}
    with TestClient(create_app(tmp_path / "runs"), base_url="http://127.0.0.1") as client:
        connection = {"api_url": "https://default.example", "bank_id": "test-bank"}
        invalid = client.post("/api/runs", json={**basic, "connection": {**connection, "api_key": "secret-in-invalid-form"}})
        assert invalid.status_code == 422 and "secret-in-invalid-form" not in invalid.text
        invalid = client.post("/api/runs", json={**basic, "name": "   ", "connection": connection})
        assert invalid.status_code == 422
        invalid = client.post("/api/runs", json={**basic, "name": "A", "connection": connection})
        assert invalid.status_code == 422 and "API Key" in invalid.json()["detail"]
        invalid = client.post("/api/runs", json={**basic, "name": "A", "connection": {
            **connection, "api_url": "https://other.example", "use_saved_key": True,
        }})
        assert invalid.status_code == 422 and "对应的 API 地址" in invalid.json()["detail"]
        assert "saved-round-key" not in invalid.text
        assert client.get("/api/runs").json() == []


def test_progress_tracks_parallel_writes_and_answers_independently(monkeypatch, tmp_path):
    dataset = BEAMDataset()
    turn = {"role": "user", "content": "Paris"}
    question = {"question": "Where?", "answer": "Paris"}
    monkeypatch.setattr(dataset, "_load_raw", lambda split: [
        {
            "conversation_id": unit, "chat": [[turn]] * documents,
            "probing_questions": {"information_extraction": [question]},
        }
        for unit, documents in (("first", 2), ("second", 1))
    ])
    started_second_write = asyncio.Event()
    finish_second_write = asyncio.Event()
    clock = 0.0

    def perf_counter():
        return clock

    monkeypatch.setattr("memory_bench.runner.time.perf_counter", perf_counter)

    class Memory(MemoryProvider):
        name = "parallel-test"
        unit_concurrency = 2

        def ingest(self, documents):
            pytest.fail("expected asynchronous ingestion")

        async def async_ingest(self, documents):
            nonlocal clock
            if documents[0].user_id == "second":
                started_second_write.set()
                await asyncio.wait_for(finish_second_write.wait(), timeout=5)
                clock = 0.2
            else:
                await asyncio.wait_for(started_second_write.wait(), timeout=5)
                clock = 0.15

        def retrieve(self, query, k=10, user_id=None, query_timestamp=None):
            pytest.fail("the stub answers directly")

    class Judge:
        model_id = "test-judge"

        def generate(self, prompt, schema):
            return {"score": 1.0, "reason": "matches"}

    class Mode(RAGMode):
        async def async_answer(self, query, memory, task_type="open", user_id=None, meta=None):
            if user_id == "first":
                await asyncio.wait_for(started_second_write.wait(), timeout=5)
                finish_second_write.set()
            return AnswerResult(answer="Paris", reasoning="", context="Paris", retrieve_time_ms=1.0)

    monkeypatch.setattr(dataset, "default_judge_llm", Judge)
    updates = []
    runner = EvalRunner(tmp_path, on_progress=updates.append)
    summary = runner.run(dataset, "100k", Memory(), Mode(Judge()))

    assert updates[0] == EvaluationProgress("ingesting", 0, 2, 0, 3)
    assert EvaluationProgress("ingesting_and_evaluating", 0, 2, 2, 3) in updates
    assert [update.completed for update in updates] == sorted(update.completed for update in updates)
    assert [update.ingested_docs for update in updates] == sorted(update.ingested_docs for update in updates)
    assert updates[-1] == EvaluationProgress("finalizing", 2, 2, 3, 3)
    assert summary.ingestion_time_ms == 350.0
    assert summary.ingestion_wall_time_ms == 200.0


def test_workbench_migrates_legacy_credentials_out_of_results(tmp_path, private_workbench_config):
    results = tmp_path / "amb-results" / "workbench"
    legacy = results / ".connections.json"
    results.mkdir(parents=True)
    key = "mock-key-used-for-migration"
    legacy.write_text(json.dumps({
        "contextbase": {"api_url": "https://example.test", "api_key": key, "bank_id": "test-bank"},
    }), encoding="utf-8")

    with TestClient(create_app(results), base_url="http://127.0.0.1") as client:
        config = client.get("/api/config")
        saved = config.json()["backends"]["contextbase"]
        assert saved["has_key"] and saved["saved"]
        assert saved["api_url"] == "https://example.test" and saved["bank_id"] == "test-bank"
        assert key not in config.text
    assert not legacy.exists()
    private, = private_workbench_config.glob("*/connections.json")
    assert json.loads(private.read_text(encoding="utf-8"))["contextbase"][0]["api_key"] == key

    archive = io.BytesIO()
    with ZipFile(archive, "w") as exported:
        for path in results.parent.rglob("*"):
            if path.is_file():
                exported.write(path, path.relative_to(results.parent))
    with ZipFile(archive) as exported:
        assert all(key.encode() not in exported.read(name) for name in exported.namelist())

    with TestClient(create_app(results), base_url="http://127.0.0.1") as restarted:
        assert restarted.get("/api/config").json()["backends"]["contextbase"] == saved

    # Existing private settings from before address history remain selectable.
    private.write_text(json.dumps({"contextbase": {
        "api_url": "https://example.test", "api_key": key, "bank_id": "test-bank",
    }}), encoding="utf-8")
    with TestClient(create_app(results), base_url="http://127.0.0.1") as restarted:
        assert restarted.get("/api/config").json()["backends"]["contextbase"] == saved


def test_workbench_refuses_config_directory_inside_results(monkeypatch, tmp_path):
    results = tmp_path / "runs"
    monkeypatch.setenv("AMB_WORKBENCH_CONFIG_DIR", str(results / "config"))
    with pytest.raises(ValueError, match="outside the Workbench results directory"):
        create_app(results)


def test_workbench_runs_use_saved_global_models(monkeypatch, tmp_path, mock_services, private_workbench_config):
    endpoint, calls = mock_services
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMB_ANSWER_LLM", "gemini")
    monkeypatch.setenv("OMB_JUDGE_LLM", "gemini")
    monkeypatch.delenv("OMB_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", endpoint + "/environment/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-model-key")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    dataset_path = tmp_path / "long.json"
    dataset_path.write_text(json.dumps([{
        "question_id": "round", "question_type": "single-session-user",
        "question": "Where?", "answer": "Paris", "haystack_dates": ["2023/05/01"],
        "haystack_sessions": [[{"role": "user", "content": "The location is Paris", "has_answer": True}]],
        "haystack_session_ids": ["session"],
    }]))
    monkeypatch.setenv("LONGMEMEVAL_DATA_PATH", str(dataset_path))

    directory = tmp_path / "runs"
    selection = {
        "name": "Model settings", "backend": "contextbase", "dataset": "longmemeval", "split": "s",
        "query_limit": 1, "connection": {"api_url": endpoint, "api_key": "test-bearer-value", "bank_id": "ui-bank"},
    }

    def config(model, service, key=None):
        return {"provider": "openai", "model": model, "api_url": f"{endpoint}/{service}/v1", "api_key": key}

    def assert_run_models(client, response, answer, judge):
        url, detail = await_result(client, response)
        assert detail["status"] == "completed", detail["log"]
        for role, (name, service, key) in {"answer": answer, "judge": judge}.items():
            assert detail["summary"][f"{role}_llm"] == f"openai:{name}"
            assert detail["models"][role] == {
                "provider": "openai", "model": name, "api_url": f"{endpoint}/{service}/v1",
            }
            assert any(path == f"/{service}/v1/chat/completions" and
                       call["model"] == name and call["authorization"] == f"Bearer {key}" for path, call in calls)
        for path in (url, url + "/result", "/api/config", "/api/runs"):
            text = client.get(path).text
            for key in (answer[2], judge[2]):
                assert key not in text
        return url

    with TestClient(create_app(directory), base_url="http://127.0.0.1") as client:
        invalid = client.post("/api/models/answer", json=config("bad", "answer", {"secret": "invalid-model-key"}))
        assert invalid.status_code == 422 and "invalid-model-key" not in invalid.text
        no_key = client.post("/api/models/judge", json=config("judge-default", "judge"))
        assert no_key.status_code == 422 and "API Key" in no_key.json()["detail"]
        saved = client.post("/api/models/judge", json=config("judge-default", "judge", "saved-judge-key"))
        assert saved.status_code == 200 and saved.json()["has_key"]
        assert "saved-judge-key" not in saved.text
        assert client.post("/api/runs", json=selection).status_code == 422

        answer = client.post("/api/models/answer", json=config("answer-default", "environment"))
        assert answer.status_code == 200 and answer.json()["has_key"]

        override = config("answer-round", "answer-round", "override-answer-key")
        invalid = client.post("/api/runs", json={**selection, "models": {"answer": override}})
        assert invalid.status_code == 422 and "override-answer-key" not in invalid.text
        assert client.get("/api/runs").json() == []

        run = client.post("/api/runs", json=selection)
        assert run.status_code == 201, run.text
        assert client.post("/api/models/judge", json=config("judge-updated", "judge")).status_code == 200
        first_url = assert_run_models(client, run, ("answer-default", "environment", "environment-model-key"),
                                      ("judge-default", "judge", "saved-judge-key"))

    with TestClient(create_app(directory), base_url="http://127.0.0.1") as client:
        models = client.get("/api/config").json()["models"]
        assert models["answer"]["model"] == "answer-default"
        assert models["judge"]["model"] == "judge-updated"
        assert client.get(first_url).json()["models"]["judge"]["model"] == "judge-default"
        response = client.post("/api/runs", json=selection)
        assert_run_models(client, response, ("answer-default", "environment", "environment-model-key"),
                          ("judge-updated", "judge", "saved-judge-key"))
        assert client.post("/api/models/judge/reset").json()["provider"] == "gemini"
        storage, = private_workbench_config.glob("*/models.json")
        assert storage.stat().st_mode & 0o077 == 0
        assert "override-answer-key" not in storage.read_text(encoding="utf-8")
        assert "saved-judge-key" not in storage.read_text(encoding="utf-8")
        assert "environment-model-key" not in storage.read_text(encoding="utf-8")
        assert not list(directory.rglob("models.json"))


def test_workbench_keeps_beam_judge_default(monkeypatch, tmp_path):
    from memory_bench.workbench_models import ModelSettings

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OMB_ANSWER_LLM", "gemini")
    monkeypatch.setenv("OMB_JUDGE_LLM", "gemini")
    monkeypatch.delenv("OMB_ANSWER_MODEL", raising=False)
    monkeypatch.delenv("OMB_JUDGE_MODEL", raising=False)
    settings = ModelSettings(tmp_path / "models.json")
    beam, _ = settings.for_run("beam")
    locomo, _ = settings.for_run("locomo")
    assert beam["judge"]["model"] == "gemini-3.5-flash"
    assert beam["answer"]["model"] == locomo["judge"]["model"] == "gemini-2.5-flash-lite"


def test_workbench_scores_a_probe_on_the_server_with_unsaved_settings(monkeypatch, tmp_path, mock_services):
    endpoint, calls = mock_services
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", endpoint + "/environment/v1")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)

    saved = {"provider": "openai", "model": "judge-default", "api_url": endpoint + "/saved/v1", "api_key": "saved-judge-key"}
    candidate = {"provider": "openai", "model": "judge-candidate", "api_url": endpoint + "/draft/v1", "api_key": "draft-judge-key"}
    path = "/api/models/judge/test"
    with TestClient(create_app(tmp_path / "runs"), base_url="http://127.0.0.1") as client:
        assert client.post("/api/models/judge", json=saved).status_code == 200
        assert client.post(path, json={**candidate, "api_key": None}).status_code == 422
        assert client.post(path, json=candidate, headers={"Origin": "https://elsewhere.example"}).status_code == 403
        assert calls == []

        response = client.post(path, json=candidate)
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "openai:judge-candidate"
        assert response.json()["elapsed_ms"] >= 0
        request_path, call = calls.pop()
        assert request_path == "/draft/v1/chat/completions"
        assert call["model"] == "judge-candidate"
        assert call["authorization"] == "Bearer draft-judge-key"
        assert "What is 2 + 2?" in call["messages"][0]["content"]
        assert call["response_format"]["json_schema"]["schema"]["required"] == ["correct", "reason"]

        response = client.post(path, json={**saved, "api_key": None})
        assert response.status_code == 200, response.text
        request_path, call = calls.pop()
        assert request_path == "/saved/v1/chat/completions"
        assert call["authorization"] == "Bearer saved-judge-key"

        error = client.post(path, json={**candidate, "model": "judge-probe-auth"})
        assert error.status_code == 502 and "鉴权失败" in error.json()["detail"]
        assert "draft-judge-key" not in error.text
        malformed = client.post(path, json={**candidate, "model": "judge-probe-invalid"})
        assert malformed.status_code == 502 and "格式" in malformed.json()["detail"]
        assert client.get("/api/config").json()["models"]["judge"]["model"] == "judge-default"
        assert client.get("/api/runs").json() == []

        monkeypatch.setattr("memory_bench.workbench.JUDGE_PROBE_TIMEOUT", 0.05)
        timeout = client.post(path, json=candidate)
        assert timeout.status_code == 504 and "未完成" in timeout.json()["detail"]
