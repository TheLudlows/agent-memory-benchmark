import socket

import pytest

from memory_bench.dataset import get_dataset
from memory_bench.utils import _enc, count_tokens


@pytest.fixture
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Bundled data and tokenization must not access the network")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    for name in ("LOCOMO_DATA_PATH", "LONGMEMEVAL_DATA_PATH", "BEAM_DATA_PATH", "BEAM_100K_DATA_PATH",
                 "BEAM_500K_DATA_PATH", "BEAM_1M_DATA_PATH", "BEAM_10M_DATA_PATH",
                 "LIFEBENCH_V2_DATA_PATH", "PRECISIONMEMBENCH_DATA_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(("name", "split", "count"), [
    ("locomo", "locomo10", 1540), ("longmemeval", "s", 500), ("beam", "100k", 400),
    ("beam", "500k", 700), ("beam", "1m", 700), ("beam", "10m", 200),
    ("lifebench-v2", "zh", 3380), ("precisionmembench", "single-turn", 77),
])
def test_workbench_bundles_load_without_network(offline, name, split, count):
    dataset = get_dataset(name)
    queries = dataset.load_queries(split)
    assert len(queries) == count
    assert len({q.id for q in queries}) == count
    user_ids = {queries[0].user_id}
    documents = dataset.load_documents(split, user_ids=user_ids)
    assert documents and {doc.user_id for doc in documents} == user_ids
    assert all(doc.content for doc in documents)
    if name == "beam":
        assert all(doc.timestamp and doc.context for doc in documents)
        assert len(documents) > 1


def test_tokenizer_uses_bundled_ranks_without_cache(offline, monkeypatch, tmp_path):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    _enc.cache_clear()
    assert _enc().encode("hello world") == [15339, 1917]
    assert count_tokens("这是一次离线测试") == 8
    assert not list(tmp_path.iterdir())


def test_missing_local_data_reports_the_path(offline, monkeypatch, tmp_path):
    path = tmp_path / "locomo.json"
    monkeypatch.setenv("LOCOMO_DATA_PATH", str(path))
    with pytest.raises(FileNotFoundError) as exc:
        get_dataset("locomo").load_queries("locomo10")
    assert exc.value.filename == str(path)


def test_installed_cli_can_load_data_from_the_repository_directory(offline, monkeypatch, tmp_path):
    from memory_bench import data_files

    repository = data_files.data_path().parent
    monkeypatch.chdir(repository)
    monkeypatch.setattr(data_files, "__file__", str(tmp_path / "site-packages" / "memory_bench" / "data_files.py"))
    assert len(get_dataset("locomo").load_queries("locomo10")) == 1540
