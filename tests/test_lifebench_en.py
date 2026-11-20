import json
import socket
from collections import Counter

import pytest

from memory_bench.dataset import _cache, lifebench


def test_bundled_english_lifebench_offline(monkeypatch):
    monkeypatch.delenv("LIFEBENCH_DATA_PATH", raising=False)
    monkeypatch.delenv("AMB_ALLOW_DATASET_DOWNLOADS", raising=False)

    def blocked(*args, **kwargs):
        pytest.fail("The English bundle must load without network access")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    dataset = lifebench.LifeBenchDataset()
    questions = dataset.load_queries("en")
    documents = dataset.load_documents("en")
    users = {q.user_id for q in questions}
    assert len(users) == 10
    assert len(questions) == len({q.id for q in questions}) == 2003
    assert len(documents) == len({d.id for d in documents}) == 3605
    assert {d.user_id for d in documents} == users
    assert all(d.timestamp and d.context and d.content for d in documents)
    assert all(q.gold_answers and q.meta["query_timestamp"] for q in questions)
    assert Counter(q.meta["category"] for q in questions) == {
        "information-extraction": 718, "multi-hop": 597, "nondeclarative": 429,
        "temporal-updating": 229, "unanswerable": 30,
    }
    first = documents[0]
    assert first.context == f"User '{first.user_id}' — session_1 ({first.timestamp})"
    assert dataset.load_queries("en", category=first.user_id, limit=1) == questions[:1]
    selected = dataset.load_documents("en", user_ids={first.user_id}, ids={first.id, documents[-1].id})
    assert selected == [first]


def test_english_lifebench_prefers_raw_override_and_preserves_evidence(monkeypatch, tmp_path):
    path = tmp_path / "our_en.json"
    path.write_text(json.dumps([{
        "sample_id": "custom", "conversation": {
            "session_1_date_time": "2025-04-01", "session_1": [{"dia_id": "turn1", "text": "A walk."}],
            "session_2_date_time": "2025-04-02", "session_2": [],
        },
        "qa": [
            {"question": "What happened?", "answer": "A walk.", "evidence": ["turn1"], "category": "0"},
            {"question": "What next?", "answer": "Unknown.", "evidence": [], "category": "4"},
        ],
    }]), encoding="utf-8")
    monkeypatch.setenv("LIFEBENCH_DATA_PATH", str(path))
    dataset = lifebench.LifeBenchDataset()
    questions = dataset.load_queries("en")
    docs = dataset.load_documents("en")
    assert [q.user_id for q in questions] == ["custom", "custom"]
    assert questions[0].gold_ids == ["custom_session_1"] == [d.id for d in docs]
    assert questions[0].meta["query_timestamp"] == "2025-04-02T00:00:00+00:00"
    assert docs[0].timestamp == "2025-04-01T00:00:00+00:00"
    assert dataset.load_queries("en", category="unanswerable") == questions[1:]
    assert not dataset.should_judge_empty_context(questions[0].meta)
    assert dataset.should_judge_empty_context(questions[1].meta)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        dataset.load_queries("en")


def test_missing_english_bundle_respects_download_policy(monkeypatch, tmp_path):
    monkeypatch.delenv("LIFEBENCH_DATA_PATH", raising=False)
    monkeypatch.delenv("AMB_ALLOW_DATASET_DOWNLOADS", raising=False)
    monkeypatch.setattr(lifebench, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(_cache, "dataset_cache_dir", lambda _: tmp_path)
    monkeypatch.setattr(_cache, "urlopen", lambda *a, **kw: pytest.fail("Unexpected download"))
    dataset = lifebench.LifeBenchDataset()
    with pytest.raises(FileNotFoundError) as error:
        dataset.load_queries("en")
    assert str(error.value.filename) == str(tmp_path / "our_en.json")
    (tmp_path / "our_en.json").write_text("[]", encoding="utf-8")
    assert dataset.load_queries("en") == []
