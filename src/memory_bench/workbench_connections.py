"""Backend settings for the local Workbench."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr


BackendId = Literal["contextbase", "mem0-cloud", "hindsight-http"]


@dataclass(frozen=True)
class Backend:
    label: str
    url_env: str
    key_env: str
    default_url: str = ""
    key_optional: bool = False


BACKENDS: dict[BackendId, Backend] = {
    "contextbase": Backend("ContextBase", "CONTEXTBASE_API_URL", "CONTEXTBASE_API_TOKEN"),
    "mem0-cloud": Backend("Mem0 Cloud", "MEM0_CLOUD_URL", "MEM0_CLOUD_KEY", "https://api.mem0.ai"),
    "hindsight-http": Backend("Hindsight HTTP", "HINDSIGHT_HTTP_URL", "HINDSIGHT_HTTP_KEY",
                              "http://127.0.0.1:8888", key_optional=True),
}


class ConnectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_url: str = Field(max_length=2048)
    api_key: SecretStr | None = Field(default=None, max_length=4096)
    bank_id: str = Field(default="", max_length=256)
    clear_key: bool = False


def _api_url(value: str) -> str:
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

    def missing(self, backend: BackendId) -> list[str]:
        missing = []
        if not self.api_url:
            missing.append("API 地址")
        if not BACKENDS[backend].key_optional and not self.api_key:
            missing.append("API Key")
        if backend == "contextbase" and not (self.bank_id or self.bank_mapping):
            missing.append("Bank ID")
        return missing


def _environment_connection(backend: BackendId) -> Connection:
    spec = BACKENDS[backend]
    try:
        api_url = _api_url(os.getenv(spec.url_env, "").strip() or spec.default_url)
    except ValueError:
        api_url = ""
    key = os.getenv(spec.key_env, "").strip()
    if backend == "mem0-cloud":
        key = key or os.getenv("MEM0_API_KEY", "").strip()
    if backend == "contextbase":
        return Connection(api_url, key, os.getenv("CONTEXTBASE_BANK_ID", "").strip(),
                          os.getenv("CONTEXTBASE_BENCH_BANKS", "").strip())
    return Connection(api_url, key)


class Connections:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        self.saved: dict[BackendId, Connection] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.saved = {backend: Connection(**raw[backend]) for backend in BACKENDS if backend in raw}

    def _get(self, backend: BackendId) -> Connection:
        return self.saved.get(backend) or _environment_connection(backend)

    def _public(self, backend: BackendId) -> dict:
        connection = self._get(backend)
        spec = BACKENDS[backend]
        return {
            "label": spec.label, "api_url": connection.api_url, "bank_id": connection.bank_id,
            "has_bank_mapping": bool(connection.bank_mapping), "has_key": bool(connection.api_key),
            "key_optional": spec.key_optional, "saved": backend in self.saved,
            "missing": connection.missing(backend),
        }

    def public(self) -> dict:
        with self.lock:
            return {backend: self._public(backend) for backend in BACKENDS}

    def _write(self, connections: dict[BackendId, Connection]) -> None:
        # The file is outside the public assets directory. NamedTemporaryFile creates it with
        # owner-only permissions on POSIX and keeps those permissions after the atomic rename.
        data = {name: asdict(connection) for name, connection in connections.items()}
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                    prefix=".connections-", delete=False) as fh:
                temporary = Path(fh.name)
                json.dump(data, fh, ensure_ascii=False)
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.saved = connections

    def save(self, backend: BackendId, update: ConnectionUpdate) -> dict:
        api_url = _api_url(update.api_url)
        supplied_key = update.api_key.get_secret_value().strip() if update.api_key else ""
        if any(ord(char) < 32 or ord(char) == 127 for char in supplied_key):
            raise ValueError("API Key 不能包含换行或控制字符。")
        if update.clear_key and (not BACKENDS[backend].key_optional or supplied_key):
            raise ValueError("只有 Hindsight 可以清除 Key，清除时请留空 Key 输入框。")

        with self.lock:
            current = self._get(backend)
            same_url = api_url == current.api_url
            if not same_url and current.api_key and not supplied_key and not update.clear_key:
                raise ValueError("API 地址已变更，请重新填写 API Key。若 Hindsight 无需鉴权，可清除已保存的 Key。")
            if backend != "contextbase" and update.bank_id:
                raise ValueError("只有 ContextBase 需要 Bank ID。")
            bank_id = update.bank_id.strip()
            mapping = current.bank_mapping if not bank_id and same_url else ""
            key = "" if update.clear_key else supplied_key or (current.api_key if same_url else "")
            connection = Connection(api_url, key, bank_id, mapping)
            if missing := connection.missing(backend):
                raise ValueError("请填写 " + "、".join(missing))
            self._write({**self.saved, backend: connection})
            return self._public(backend)

    def reset(self, backend: BackendId) -> dict:
        with self.lock:
            if backend in self.saved:
                self._write({name: value for name, value in self.saved.items() if name != backend})
            return self._public(backend)

    def worker_env(self, backend: BackendId) -> dict[str, str]:
        with self.lock:
            connection = self._get(backend)
        if missing := connection.missing(backend):
            raise ValueError(f"请先在评测台中配置 {BACKENDS[backend].label} 的 " + "、".join(missing))
        spec = BACKENDS[backend]
        env = {spec.url_env: connection.api_url, spec.key_env: connection.api_key}
        if backend == "contextbase":
            env.update(CONTEXTBASE_BANK_ID=connection.bank_id, CONTEXTBASE_BENCH_BANKS=connection.bank_mapping)
        return env
