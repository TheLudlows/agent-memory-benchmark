import gzip
import hashlib
import io
import socket

import pytest

from memory_bench.dataset import get_dataset
from memory_bench.dataset import _cache, beam, locomo, longmemeval
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


def test_missing_bundle_uses_cache_then_downloads_once(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCOMO_DATA_PATH", raising=False)
    monkeypatch.setenv("AMB_ALLOW_DATASET_DOWNLOADS", "1")
    monkeypatch.setattr(locomo, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(_cache, "dataset_cache_dir", lambda _: tmp_path)
    raw = b"[]"
    monkeypatch.setattr(locomo, "_RAW_SHA256", hashlib.sha256(raw).hexdigest())
    calls = []

    def fetch(url, *, timeout):
        calls.append(url)
        return io.BytesIO(raw)

    monkeypatch.setattr(_cache, "urlopen", fetch)
    cached_path = tmp_path / "locomo10.json"
    cached_path.write_bytes(raw)
    assert get_dataset("locomo")._load_raw() == []
    assert not calls

    cached_path.unlink()
    assert get_dataset("locomo")._load_raw() == []
    assert cached_path.read_bytes() == raw
    assert get_dataset("locomo")._load_raw() == []
    assert calls == [locomo._RAW_URL]


def test_failed_checksum_does_not_publish_the_dataset(monkeypatch, tmp_path):
    monkeypatch.setenv("AMB_ALLOW_DATASET_DOWNLOADS", "1")
    monkeypatch.setattr(_cache, "dataset_cache_dir", lambda _: tmp_path)
    monkeypatch.setattr(_cache, "urlopen", lambda *_, **__: io.BytesIO(b"bad payload"))
    with pytest.raises(ValueError, match="checksum"):
        _cache.cached_or_download("locomo", "locomo10.json", locomo._RAW_URL, locomo._RAW_SHA256)
    assert not list(tmp_path.iterdir())


def test_missing_bundle_reports_the_missing_file_without_network(offline, monkeypatch, tmp_path):
    monkeypatch.delenv("AMB_ALLOW_DATASET_DOWNLOADS", raising=False)
    monkeypatch.setattr(locomo, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(_cache, "dataset_cache_dir", lambda _: tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        get_dataset("locomo").load_queries("locomo10")
    assert str(exc.value.filename) == str(tmp_path / "locomo10.json")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(("module", "name", "split"), [(beam, "beam", "100k"), (longmemeval, "longmemeval", "s")])
def test_local_query_bundle_is_enough_to_load_queries(module, name, split, offline, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    source = tmp_path / name / split / "queries.json.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(gzip.compress(b"[]"))
    assert get_dataset(name).load_queries(split) == []


def test_beam_downloads_only_the_missing_raw_shard(monkeypatch, tmp_path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    monkeypatch.setenv("AMB_ALLOW_DATASET_DOWNLOADS", "1")
    monkeypatch.delenv("BEAM_10M_DATA_PATH", raising=False)
    monkeypatch.setattr(beam, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(beam, "dataset_cache_dir", lambda _: tmp_path)
    monkeypatch.setattr(_cache, "dataset_cache_dir", lambda _: tmp_path)

    folder = tmp_path / "10m"
    folder.mkdir()
    parquet.write_table(arrow.table({"conversation_id": ["first"]}), folder / "10M-00000-of-00002.parquet")
    missing = io.BytesIO()
    parquet.write_table(arrow.table({"conversation_id": ["second"]}), missing)
    raw = missing.getvalue()
    monkeypatch.setitem(beam._RAW_SHA256, "10m", ("unused", hashlib.sha256(raw).hexdigest()))
    calls = []

    def fetch(url, *, timeout):
        calls.append(url)
        return io.BytesIO(raw)

    monkeypatch.setattr(_cache, "urlopen", fetch)
    dataset = beam.BEAMDataset()
    expected = [{"conversation_id": "first"}, {"conversation_id": "second"}]
    assert list(dataset._load_raw("10m", columns=["conversation_id"])) == expected
    assert list(dataset._load_raw("10m", columns=["conversation_id"])) == expected
    assert calls == [beam._RAW_10M_URL + "10M-00001-of-00002.parquet"]
