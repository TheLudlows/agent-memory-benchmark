"""Backend settings for the local Workbench."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .json_files import write_json


BackendId = Literal["contextbase", "mem0-cloud", "hindsight-http", "hindsight-knowledge", "weknora"]


@dataclass(frozen=True)
class Backend:
    label: str
    url_env: str
    key_env: str
    default_url: str = ""
    key_optional: bool = False


BACKENDS: dict[BackendId, Backend] = {
    "hindsight-knowledge": Backend("Hindsight Knowledge", "HINDSIGHT_KNOWLEDGE_URL", "HINDSIGHT_KNOWLEDGE_KEY",
                                   "http://127.0.0.1:8888", key_optional=True),
    "weknora": Backend("WeKnora", "WEKNORA_API_URL", "WEKNORA_API_KEY"),
    "contextbase": Backend("ContextBase", "CONTEXTBASE_API_URL", "CONTEXTBASE_API_TOKEN"),
    "mem0-cloud": Backend("Mem0 Cloud", "MEM0_CLOUD_URL", "MEM0_CLOUD_KEY", "https://api.mem0.ai"),
    "hindsight-http": Backend("Hindsight HTTP", "HINDSIGHT_HTTP_URL", "HINDSIGHT_HTTP_KEY",
                              "http://127.0.0.1:8888", key_optional=True),
}


class RunConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_url: str = Field(max_length=2048)
    api_key: SecretStr | None = Field(default=None, max_length=4096)
    bank_id: str = Field(default="", max_length=256)
    use_saved_key: bool = False
    embedding_model_id: str | None = Field(default=None, max_length=256)
    vlm_model_id: str | None = Field(default=None, max_length=256)

    def supplied_key(self) -> str:
        return clean_api_key(self.api_key)


def clean_api_key(value: SecretStr | None) -> str:
    key = value.get_secret_value().strip() if value else ""
    if any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise ValueError("API Key 不能包含换行或控制字符。")
    return key


def validate_api_url(value: str) -> str:
    value = value.strip()
    invalid = "请填写 HTTP 或 HTTPS API 地址，地址中不能包含用户名、密码、查询参数或片段。"
    try:
        address = urlsplit(value)
        _ = address.port
    except ValueError as exc:
        raise ValueError(invalid) from exc
    if (
        address.scheme not in {"http", "https"} or not address.hostname
        or "@" in address.netloc or "?" in value or "#" in value or "\\" in value
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(invalid)
    return value.rstrip("/")


@dataclass(frozen=True)
class Connection:
    api_url: str
    api_key: str = field(default="", repr=False)
    bank_id: str = ""
    bank_mapping: str = ""
    embedding_model_id: str = ""
    vlm_model_id: str = ""

    def public(self) -> dict:
        result = {
            "api_url": self.api_url, "bank_id": self.bank_id,
            "has_bank_mapping": bool(self.bank_mapping), "has_key": bool(self.api_key),
        }
        if self.embedding_model_id or self.vlm_model_id:
            result.update(embedding_model_id=self.embedding_model_id, vlm_model_id=self.vlm_model_id)
        return result

    def missing(self, backend: BackendId, *, requires_vlm: bool = False) -> list[str]:
        missing = []
        if not self.api_url:
            missing.append("API 地址")
        if not BACKENDS[backend].key_optional and not self.api_key:
            missing.append("API Key")
        if backend == "contextbase" and not (self.bank_id or self.bank_mapping):
            missing.append("Bank ID")
        if backend == "weknora":
            if not self.embedding_model_id:
                missing.append("Embedding 模型 ID")
            if requires_vlm and not self.vlm_model_id:
                missing.append("视觉模型 ID")
        return missing

    def worker_env(self, backend: BackendId) -> dict[str, str]:
        spec = BACKENDS[backend]
        env = {spec.url_env: self.api_url, spec.key_env: self.api_key}
        if backend == "contextbase":
            env.update(CONTEXTBASE_BANK_ID=self.bank_id, CONTEXTBASE_BENCH_BANKS=self.bank_mapping)
        elif backend == "weknora":
            env.update(WEKNORA_EMBEDDING_MODEL_ID=self.embedding_model_id, WEKNORA_VLM_MODEL_ID=self.vlm_model_id)
        return env


def _environment_connection(backend: BackendId) -> Connection:
    spec = BACKENDS[backend]
    try:
        api_url = validate_api_url(os.getenv(spec.url_env, "").strip() or spec.default_url)
    except ValueError:
        api_url = ""
    key = os.getenv(spec.key_env, "").strip()
    if backend == "mem0-cloud":
        key = key or os.getenv("MEM0_API_KEY", "").strip()
    if backend == "contextbase":
        return Connection(api_url, key, os.getenv("CONTEXTBASE_BANK_ID", "").strip(),
                          os.getenv("CONTEXTBASE_BENCH_BANKS", "").strip())
    if backend == "weknora":
        return Connection(api_url, key, embedding_model_id=os.getenv("WEKNORA_EMBEDDING_MODEL_ID", "").strip(),
                          vlm_model_id=os.getenv("WEKNORA_VLM_MODEL_ID", "").strip())
    return Connection(api_url, key)


def write_private_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    write_json(path, data)


class Connections:
    def __init__(self, path: Path, *, legacy_path: Path | None = None):
        self.path = path
        self.lock = Lock()
        self.saved = self._load(path)
        if legacy_path is not None and legacy_path.exists():
            for backend, entries in self._load(legacy_path).items():
                recent = self.saved.setdefault(backend, {})
                for url, connection in entries.items():
                    recent.setdefault(url, connection)
            self._write(self.saved)
            legacy_path.unlink()

    @staticmethod
    def _load(path: Path) -> dict[BackendId, dict[str, Connection]]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        history = {}
        for backend in BACKENDS:
            entries = raw.get(backend, [])
            # The previous format stored one connection object per backend.
            if isinstance(entries, dict):
                entries = [entries]
            if entries:
                history[backend] = {entry["api_url"]: Connection(**entry) for entry in entries}
        return history

    def _public(self, backend: BackendId) -> dict:
        saved = self.saved.get(backend, {})
        environment = _environment_connection(backend)
        entries = list(saved.values())
        if environment.api_url and environment.api_url not in saved:
            entries.append(environment)
        default = entries[0] if entries else environment
        spec = BACKENDS[backend]
        return {
            **default.public(), "label": spec.label, "key_optional": spec.key_optional,
            "saved": bool(saved), "missing": default.missing(backend),
            "connections": [connection.public() for connection in entries],
        }

    def public(self) -> dict:
        with self.lock:
            return {backend: self._public(backend) for backend in BACKENDS}

    def _write(self, connections: dict[BackendId, dict[str, Connection]]) -> None:
        write_private_json(self.path, {
            backend: [asdict(connection) for connection in history.values()]
            for backend, history in connections.items()
        })
        self.saved = connections

    def remember(self, backend: BackendId, connection: Connection) -> dict:
        url = connection.api_url
        with self.lock:
            recent = {url: connection}
            for address, saved in self.saved.get(backend, {}).items():
                recent.setdefault(address, saved)
            self._write({**self.saved, backend: recent})
            return self._public(backend)

    def for_run(self, backend: BackendId, request: RunConnection, *, requires_vlm: bool = False) -> Connection:
        api_url = validate_api_url(request.api_url)
        key = request.supplied_key()
        with self.lock:
            current = self.saved.get(backend, {}).get(api_url)
        if current is None:
            environment = _environment_connection(backend)
            current = environment if environment.api_url == api_url else Connection(api_url)
        if request.use_saved_key:
            if key:
                raise ValueError("请输入本轮 Key 或选择使用已保存的 Key。")
            if not current.api_key:
                raise ValueError("已保存的 Key 仅可用于对应的 API 地址。请填写本轮 Key。")
            key = current.api_key
        if backend != "contextbase" and request.bank_id:
            raise ValueError("只有 ContextBase 需要 Bank ID。")
        if backend != "weknora" and (request.embedding_model_id is not None or request.vlm_model_id is not None):
            raise ValueError("只有 WeKnora 使用知识库模型 ID。")
        bank_id = request.bank_id.strip()
        mapping = current.bank_mapping if not bank_id else ""
        embedding_model_id = (current.embedding_model_id if request.embedding_model_id is None
                              else request.embedding_model_id.strip())
        vlm_model_id = current.vlm_model_id if request.vlm_model_id is None else request.vlm_model_id.strip()
        connection = Connection(api_url, key, bank_id, mapping, embedding_model_id, vlm_model_id)
        if missing := connection.missing(backend, requires_vlm=requires_vlm):
            raise ValueError("请填写本轮评测的 " + "、".join(missing))
        return connection
