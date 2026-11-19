"""Resolve the answer and judge models for the local Workbench."""

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints

from .dataset.beam import DEFAULT_JUDGE_MODEL
from .llm import LLM, REGISTRY, OpenAILLM
from .workbench_connections import clean_api_key, validate_api_url, write_private_json

Role = Literal["answer", "judge"]
Provider = Literal["gemini", "openai", "groq"]
ROLES: dict[Role, str] = {"answer": "回答模型", "judge": "评分模型"}
LLM_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
}
OPENAI_URL = "https://api.openai.com/v1"


class ModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Provider
    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
    api_url: str = Field(default="", max_length=2048)
    api_key: SecretStr | None = Field(default=None, max_length=4096)


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str = ""
    api_url: str = ""
    api_key: str = field(default="", repr=False)

    def connection(self) -> tuple[str, str]:
        return self.provider, self.api_url

    def metadata(self) -> dict:
        return {"provider": self.provider, "model": self.model, "api_url": self.api_url}


def _environment_service(provider: str) -> ModelConfig:
    key = next((value for name in LLM_KEYS.get(provider, ()) if (value := os.getenv(name, "").strip())), "")
    url = ""
    if provider == "openai":
        try:
            url = validate_api_url(os.getenv("OPENAI_BASE_URL", "").strip() or OPENAI_URL)
        except ValueError:
            key = ""
    return ModelConfig(provider, api_url=url, api_key=key)


def _worker_key(role: Role) -> str:
    return f"AMB_WORKBENCH_{role.upper()}_API_KEY"


def worker_model(role: Role, config: dict) -> LLM:
    provider = config["provider"]
    model = config["model"]
    key = os.environ[_worker_key(role)]
    if provider == "openai":
        return OpenAILLM(model, api_key=key, base_url=config["api_url"])
    return REGISTRY[provider](model, api_key=key)


class ModelSettings:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        try:
            self.saved = {role: ModelConfig(**data) for role, data in json.loads(path.read_text(encoding="utf-8")).items()}
        except FileNotFoundError:
            self.saved = {}

    def _get(self, role: Role) -> ModelConfig:
        if role in self.saved:
            return self.saved[role]
        provider = os.getenv(f"OMB_{role.upper()}_LLM", "gemini").strip()
        return replace(_environment_service(provider), model=os.getenv(f"OMB_{role.upper()}_MODEL", "").strip())

    @staticmethod
    def _key(config: ModelConfig) -> str:
        if config.api_key:
            return config.api_key
        service = _environment_service(config.provider)
        return service.api_key if config.connection() == service.connection() else ""

    def _public(self, role: Role) -> dict:
        config = self._get(role)
        return {**config.metadata(), "has_key": bool(self._key(config)), "saved": role in self.saved}

    def public(self) -> dict:
        with self.lock:
            return {role: self._public(role) for role in ROLES}

    @staticmethod
    def services() -> dict:
        services = {}
        for provider, client in REGISTRY.items():
            service = _environment_service(provider)
            services[provider] = {
                "api_url": service.api_url, "has_key": bool(service.api_key),
                "default_model": client.DEFAULT_MODEL,
            }
        return services

    def _updated_config(self, role: Role, update: ModelInput) -> ModelConfig:
        current = self._get(role)
        provider = update.provider
        url = update.api_url.strip()
        if provider == "openai":
            default = current.api_url if current.provider == provider else _environment_service(provider).api_url
            url = validate_api_url(url or default)
        elif url:
            raise ValueError(f"{ROLES[role]}的自定义 API 地址仅支持 OpenAI 兼容服务。")
        config = ModelConfig(provider, update.model, url)
        key = clean_api_key(update.api_key) or (current.api_key if config.connection() == current.connection() else "")
        return replace(config, api_key=key)

    def _check_key(self, role: Role, config: ModelConfig) -> str:
        if config.provider not in LLM_KEYS:
            raise ValueError(f"请选择有效的{ROLES[role]}提供方。")
        if config.provider == "openai" and not config.api_url:
            raise ValueError(f"请填写{ROLES[role]}的 API 地址。")
        key = self._key(config)
        if not key:
            raise ValueError(f"请填写{ROLES[role]}的 API Key。已配置的 Key 仅用于原提供方和地址。")
        return key

    def _write(self, saved: dict) -> None:
        write_private_json(self.path, {role: asdict(config) for role, config in saved.items()})
        self.saved = saved

    def save(self, role: Role, update: ModelInput) -> dict:
        with self.lock:
            config = self._updated_config(role, update)
            self._check_key(role, config)
            self._write({**self.saved, role: config})
            return self._public(role)

    def reset(self, role: Role) -> dict:
        with self.lock:
            if role in self.saved:
                self._write({name: config for name, config in self.saved.items() if name != role})
            return self._public(role)

    def for_run(self, dataset: str) -> tuple[dict, dict[str, str]]:
        metadata = {}
        env = {}
        with self.lock:
            for role in ROLES:
                config = self._get(role)
                env[_worker_key(role)] = self._check_key(role, config)
                fallback = (
                    DEFAULT_JUDGE_MODEL if role == "judge" and dataset == "beam" and config.provider == "gemini"
                    else REGISTRY[config.provider].DEFAULT_MODEL
                )
                metadata[role] = {**config.metadata(), "model": config.model or fallback}
        return metadata, env
