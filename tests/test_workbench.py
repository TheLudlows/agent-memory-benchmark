import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memory_bench.memory.hindsight import HindsightHTTPMemoryProvider
from memory_bench.memory.mem0_cloud import Mem0CloudMemoryProvider
from memory_bench.models import Document
from memory_bench.workbench import create_app


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
            calls.append((self.path, body))
            if self.path == "/v1/chat/completions":
                schema = body["response_format"]["json_schema"]["schema"]["properties"]
                values = {"answer": "Paris", "reasoning": "From memory", "score": 1.0, "correct": True, "reason": "Match"}
                self.send_json({"choices": [{"message": {"content": json.dumps({key: values[key] for key in schema})}}]})
            elif self.path.endswith("/memories"):
                self.send_json({"success": True, "async": True, "items_count": len(body["items"]), "operation_id": "test-op"})
            elif self.path.endswith("/memories/recall"):
                self.send_json({"results": [{"id": "fact", "text": "The location is Paris"}]})
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


def test_datasets_run_through_local_api_and_worker(monkeypatch, tmp_path, mock_services):
    endpoint, calls = mock_services
    options = {
        "CONTEXTBASE_API_URL": endpoint,
        "CONTEXTBASE_API_TOKEN": "test-bearer-value",
        "CONTEXTBASE_BANK_ID": "bench-bank",
        "OPENAI_API_KEY": "test-model-key",
        "OPENAI_BASE_URL": endpoint + "/v1",
        "OMB_ANSWER_LLM": "openai",
        "OMB_ANSWER_MODEL": "answer-stub",
        "OMB_JUDGE_LLM": "openai",
        "OMB_JUDGE_MODEL": "judge-stub",
    }
    for key, value in options.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("CONTEXTBASE_BENCH_BANKS", raising=False)
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
            "qa": [{"question": "Where?", "answer": "Paris", "category": 1, "evidence": ["D1:1"]}],
        }),
        ("beam", "100k", "BEAM_DATA_PATH", {
            "conversation_id": "beam-test", "chat": [[{**turn, "time_anchor": "May-1-2023"}]],
            "probing_questions": {"information_extraction": [{"question": "Where?", "answer": "Paris"}]},
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
        assert options["CONTEXTBASE_API_TOKEN"] not in json.dumps(config)

        for dataset, split, variable, data in fixtures:
            dataset_path = tmp_path / f"{dataset}.json"
            dataset_path.write_text(json.dumps([data]))
            monkeypatch.setenv(variable, str(dataset_path))
            selection = {"dataset": dataset, "split": split, "backend": "contextbase", "query_limit": 1}
            assert client.post("/api/runs", json={**selection, "split": "../invalid"}).status_code == 422
            assert client.post("/api/runs", json=selection, headers={"Origin": "https://elsewhere.test"}).status_code == 403
            response = client.post("/api/runs", json=selection)
            assert response.status_code == 201, response.text
            run = response.json()
            assert client.post("/api/runs", json=selection).status_code == 409

            detail_url = f"/api/runs/{run['id']}"
            deadline = time.monotonic() + 40
            while True:
                detail = client.get(detail_url).json()
                if detail["status"] != "running" or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            assert detail["status"] == "completed", detail["log"]
            assert detail["progress"] == {"stage": "evaluating", "completed": 1, "total": 1}
            assert detail["summary"]["score"] == 1.0
            assert detail["summary"]["judge_llm"] == "openai:judge-stub"
            assert detail["summary"]["ingested_docs"] == (2 if dataset == "longmemeval" else 1)
            assert detail["result_available"]
            result = client.get(detail_url + "/result")
            assert result.json()["results"][0]["answer"] == "Paris"
            assert "attachment" in result.headers["Content-Disposition"]

        ingests = [body for path, body in calls if path.endswith("/memories")]
        recalls = [body for path, body in calls if path.endswith("/memories/recall")]
        assert len({body["tags"][0] for body in recalls}) == len(fixtures)
        assert all(body["tags_match"] == "all_strict" for body in recalls)
        assert len({item["document_id"] for item in ingests[-1]["items"]}) == 2
        assert len(client.get("/api/runs").json()) == len(fixtures)

        response = client.post("/api/runs", json={
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
        assert len(history) == len(fixtures) + 1
        assert history[0]["status"] == "cancelled"


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
