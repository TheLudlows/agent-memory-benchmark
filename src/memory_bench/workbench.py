"""API for starting and inspecting AMB runs."""

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from statistics import mean
from threading import Lock, Thread
from time import monotonic
from typing import Annotated
from uuid import uuid4

import ijson
from anyio import open_file
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .dataset import get_dataset
from .dataset.beam import BEAMDataset, DEFAULT_JUDGE_MODEL, SPLITS as BEAM_SPLITS, local_data_path as beam_local_data_path
from .dataset.knowledge import WORKBENCH_KNOWLEDGE_DATASETS
from .dataset.knowledge_data import workbench_cache_files
from .dataset.lifebench import SPLITS as LIFEBENCH_SPLITS, TOTAL_QUERIES as LIFEBENCH_QUERIES
from .dataset.lifebench_v2 import SPLITS as LIFEBENCH_V2_SPLITS, TOTAL_QUERIES as LIFEBENCH_V2_QUERIES
from .dataset.locomo import SPLITS as LOCOMO_SPLITS
from .dataset.longmemeval import SPLITS as LONGMEMEVAL_SPLITS
from .dataset.personamem import SPLITS as PERSONAMEM_SPLITS
from .dataset.personamem_offline import TOTAL_QUERIES as PERSONAMEM_QUERIES
from .dataset.precisionmembench import SPLITS as PRECISIONMEMBENCH_SPLITS, TOTAL_QUERIES as PRECISIONMEMBENCH_QUERIES
from .json_files import write_json
from .memory.contextbase import BENCH_ADMIN_TOKEN_ENV, ContextBaseKnowledgeProvider, knowledge_bank_id
from .workbench_connections import BackendId, Connections, RunConnection
from .workbench_models import ROLES, ModelInput, ModelProbeExit, ModelProbeInput, ModelSettings, Role

DATASETS = {
    "locomo": LOCOMO_SPLITS,
    "beam": BEAM_SPLITS,
    "longmemeval": LONGMEMEVAL_SPLITS,
    "lifebench": LIFEBENCH_SPLITS,
    "lifebench-v2": LIFEBENCH_V2_SPLITS,
    "personamem": PERSONAMEM_SPLITS,
    "precisionmembench": PRECISIONMEMBENCH_SPLITS,
}
KNOWLEDGE_SPLITS = {name: cls.splits for name, cls in WORKBENCH_KNOWLEDGE_DATASETS.items()}
KNOWLEDGE_BACKENDS: tuple[BackendId, ...] = ("hindsight-knowledge", "weknora", "contextbase-knowledge")
KNOWLEDGE_BACKEND_DATASETS: dict[BackendId, tuple[str, ...]] = {
    "contextbase-knowledge": ContextBaseKnowledgeProvider.supported_datasets,
}
RETRIEVAL_DATASETS = {"precisionmembench"}
CHOICE_DATASETS = {"personamem"}
QUESTION_TYPE_FIELDS = {
    "locomo": ("Question Type", "category"),
    "beam": ("Question Category", "question_category"),
    "longmemeval": ("Question Type", "question_type"),
    "lifebench": ("Question Type", "category"),
    "lifebench-v2": ("题型", "question_types"),
    "personamem": ("Question Type", "question_type"),
    "precisionmembench": ("Category", "category"),
}
# Counts from data/<dataset>/<split>/stats.json.gz. Previews of the standard
# splits should be available before the evaluation downloads their corpora.
STANDARD_QUERY_COUNTS = {
    "locomo": {"locomo10": 1540},
    "beam": {"100k": 400, "500k": 700, "1m": 700, "10m": 200},
    "longmemeval": {"s": 500},
    "lifebench": {"en": LIFEBENCH_QUERIES},
    "lifebench-v2": {"zh": LIFEBENCH_V2_QUERIES},
    "personamem": PERSONAMEM_QUERIES,
    "precisionmembench": {"single-turn": PRECISIONMEMBENCH_QUERIES},
}
RUN_ID = re.compile(r"[0-9a-f]{32}")
SCORING_INCOMPLETE_EXIT_CODE = 2
MODEL_PROBE_TIMEOUT = 20
ASSETS = Path(__file__).with_name("workbench_ui")


class RunSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    split: str
    backend: BackendId
    query_limit: int | None = Field(default=5, ge=1, strict=True)

    @property
    def is_knowledge(self) -> bool:
        return self.dataset in KNOWLEDGE_SPLITS

    @property
    def mode(self) -> str:
        return "retrieval" if self.is_knowledge or self.dataset in RETRIEVAL_DATASETS else "rag"

    @model_validator(mode="after")
    def check_split(self):
        datasets = KNOWLEDGE_SPLITS if self.is_knowledge else DATASETS
        if self.split not in datasets.get(self.dataset, ()):
            raise ValueError(f"Split {self.split!r} is unavailable for {self.dataset}.")
        if self.is_knowledge != (self.backend in KNOWLEDGE_BACKENDS):
            raise ValueError("请选择此评测类型支持的后端。")
        if (supported := KNOWLEDGE_BACKEND_DATASETS.get(self.backend)) and self.dataset not in supported:
            raise ValueError("此后端不支持所选知识库数据集。")
        return self


class RunRequest(RunSelection):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    connection: RunConnection


def _dataset_error(dataset: str, split: str) -> str | None:
    if dataset in KNOWLEDGE_SPLITS:
        if any(not path.is_file() for path in workbench_cache_files(dataset, split)):
            return f"该语料尚未准备。请先运行 amb prepare-knowledge --dataset {dataset} --split {split}。"
    if dataset == "beam":
        try:
            beam_local_data_path(split, require_split=True)
        except ValueError as exc:
            return str(exc)
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _settings_path(output_dir: Path, filename: str) -> Path:
    configured_dir = os.environ.get("AMB_WORKBENCH_CONFIG_DIR")
    root = Path(configured_dir).expanduser() if configured_dir else Path.home() / ".config" / "amb" / "workbench"
    key = sha256(os.fsencode(output_dir)).hexdigest()
    path = (root / key / filename).resolve()
    if path.is_relative_to(output_dir):
        raise ValueError("AMB_WORKBENCH_CONFIG_DIR must be outside the Workbench results directory.")
    return path


@dataclass
class ActiveRun:
    process: subprocess.Popen
    monitor: Thread
    cancelling: bool = False


class RunManager:
    def __init__(self, root: Path):
        limit = os.getenv("AMB_WORKBENCH_MAX_CONCURRENT_RUNS", "2")
        try:
            self.max_concurrent_runs = int(limit)
        except ValueError as exc:
            raise ValueError("AMB_WORKBENCH_MAX_CONCURRENT_RUNS 必须是正整数。") from exc
        if self.max_concurrent_runs < 1:
            raise ValueError("AMB_WORKBENCH_MAX_CONCURRENT_RUNS 必须是正整数。")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.connections = Connections(
            _settings_path(self.root, "connections.json"), legacy_path=self.root / ".connections.json",
        )
        self.models = ModelSettings(_settings_path(self.root, "models.json"))
        self.lock = Lock()
        self.active: dict[str, ActiveRun] = {}
        for path in self.root.glob("*/run.json"):
            job = _read_json(path)
            if job.get("status") in {"running", "cancelling"}:
                write_json(path, {**job, "status": "interrupted", "ended_at": _now()})

    def capacity(self) -> dict:
        with self.lock:
            return {"active": len(self.active), "limit": self.max_concurrent_runs}

    def start(self, selection: RunRequest) -> dict:
        if error := _dataset_error(selection.dataset, selection.split):
            raise HTTPException(422, error)
        with self.lock:
            if len(self.active) >= self.max_concurrent_runs:
                raise HTTPException(
                    409, f"已达到同时运行 {self.max_concurrent_runs} 轮评测的上限。请等待一轮结束后再启动。",
                )
            try:
                connection = self.connections.for_run(
                    selection.backend, selection.connection, requires_vlm=selection.dataset == "miracl-vision",
                )
                models, model_env = {}, {}
                if selection.mode == "rag":
                    models, model_env = self.models.for_run(selection.dataset, needs_judge=selection.dataset not in CHOICE_DATASETS)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            try:
                backend_config = self.connections.remember(selection.backend, connection)
            except OSError as exc:
                raise HTTPException(500, "无法保存本轮连接，请检查配置目录。") from exc
            run_id = uuid4().hex
            directory = self.root / run_id
            directory.mkdir(mode=0o700)
            auto_bank = selection.backend == "contextbase-knowledge" and not connection.bank_id
            target = {"api_url": connection.api_url, "bank_id": connection.bank_id}
            if auto_bank:
                target.update(bank_id=knowledge_bank_id(run_id), auto_bank=True)
            if selection.backend == "weknora":
                target.update(embedding_model_id=connection.embedding_model_id, vlm_model_id=connection.vlm_model_id)
                if connection.tenant_id:
                    target["tenant_id"] = connection.tenant_id
            job = {
                "id": run_id,
                "name": selection.name,
                "selection": selection.model_dump(exclude={"name", "connection"}),
                "target": target,
                "models": models,
                "status": "running",
                "started_at": _now(),
            }
            write_json(directory / "run.json", job)
            env = {
                **os.environ, **connection.worker_env(selection.backend), **model_env,
                "NO_COLOR": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
            }
            if selection.is_knowledge:
                env["AMB_KNOWLEDGE_OFFLINE"] = "1"
            if not auto_bank:
                env.pop(BENCH_ADMIN_TOKEN_ENV, None)
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "memory_bench.workbench_worker", str(directory)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, text=True, encoding="utf-8", errors="replace",
                )
            except OSError as exc:
                write_json(directory / "run.json", {**job, "status": "failed", "ended_at": _now()})
                raise HTTPException(500, "无法启动 AMB 进程。") from exc
            secrets = sorted(
                {value for key, value in env.items() if value and key.endswith(("_KEY", "_TOKEN", "_SECRET"))},
                key=len, reverse=True,
            )
            monitor = Thread(target=self._monitor, args=(job, process, secrets), daemon=True)
            self.active[run_id] = ActiveRun(process, monitor)
            try:
                monitor.start()
            except RuntimeError as exc:
                del self.active[run_id]
                process.kill()
                process.wait()
                process.stdout.close()
                write_json(directory / "run.json", {**job, "status": "failed", "ended_at": _now()})
                raise HTTPException(500, "无法监控 AMB 进程。") from exc
            return {**job, "backend_config": backend_config}

    def _monitor(self, job: dict, process: subprocess.Popen, secrets: list[str]) -> None:
        directory = self.root / job["id"]
        try:
            with process.stdout as output, (directory / "run.log").open("w", encoding="utf-8", buffering=1) as log:
                for line in output:
                    for secret in secrets:
                        line = line.replace(secret, "[redacted]")
                    log.write(line)
        except OSError:
            process.kill()
        finally:
            code = process.wait()
            with self.lock:
                if self.active[job["id"]].cancelling:
                    status = "cancelled"
                else:
                    status = {0: "completed", SCORING_INCOMPLETE_EXIT_CODE: "scoring_incomplete"}.get(code, "failed")
                job.update(status=status, ended_at=_now(), exit_code=code)
                try:
                    write_json(directory / "run.json", job)
                finally:
                    del self.active[job["id"]]

    def cancel(self, run_id: str) -> None:
        with self.lock:
            run = self.active.get(run_id)
            if run is None:
                raise HTTPException(409, "此评测已结束。")
            if run.cancelling:
                return
            if run.process.poll() is not None:
                raise HTTPException(409, "此评测已结束。")
            path = self.root / run_id / "run.json"
            write_json(path, {**_read_json(path), "status": "cancelling"})
            run.cancelling = True
            run.process.terminate()

    def stop(self) -> None:
        with self.lock:
            runs = list(self.active.values())
            for run in runs:
                if not run.cancelling and run.process.poll() is None:
                    run.cancelling = True
                    run.process.terminate()
        deadline = monotonic() + 5
        for run in runs:
            run.monitor.join(timeout=max(0, deadline - monotonic()))
        for run in runs:
            if run.process.poll() is None:
                run.process.kill()
        deadline = monotonic() + 5
        for run in runs:
            run.monitor.join(timeout=max(0, deadline - monotonic()))

    def directory(self, run_id: str, knowledge: bool) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise HTTPException(404, "评测不存在。")
        path = self.root / run_id
        job = _read_json(path / "run.json")
        if not job or is_knowledge_job(job) != knowledge:
            raise HTTPException(404, "评测不存在。")
        return path


def is_knowledge_job(job: dict) -> bool:
    return job["selection"]["dataset"] in KNOWLEDGE_SPLITS


def is_knowledge_request(request: Request) -> bool:
    return request.url.path.startswith("/knowledge/")


def _result_path(directory: Path, job: dict) -> Path:
    selection = RunSelection.model_validate(job["selection"])
    return directory / "results" / selection.dataset / f"web-{job['id']}" / selection.mode / f"{selection.split}.json"


@lru_cache(maxsize=16)
def _context_token_report(path: Path) -> tuple[float | None, list[dict]]:
    # Completed run files are immutable. Read one result at a time and cache
    # only the fields used in the table, without retaining the full contexts.
    try:
        with path.open("rb") as source:
            rows = [
                {"query_id": row["query_id"], "query": row.get("query", ""), "context_tokens": row.get("context_tokens")}
                for row in ijson.items(source, "results.item")
            ]
    except FileNotFoundError:
        return None, []
    counts = [row["context_tokens"] for row in rows if row["context_tokens"] is not None]
    average = round(mean(counts), 1) if counts else None
    return average, rows


def summarize_question_types(dataset: str, rows: Iterable[dict]) -> list[dict]:
    if dataset in KNOWLEDGE_SPLITS:
        return []
    axis, meta_key = QUESTION_TYPE_FIELDS[dataset]
    by_type: dict[str, dict] = {}
    for row in rows:
        labels = (row.get("category_axes") or {}).get(axis) or (row.get("meta") or {}).get(meta_key)
        if isinstance(labels, str):
            labels = [labels]
        correct = row.get("correct") is True
        score = float(correct)
        failed = bool(row.get("scoring_failed"))
        if dataset == "beam" and not failed:
            raw_score = row.get("score")
            if raw_score is not None:
                score = float(raw_score)
            elif (row.get("judge_reason") or "").startswith("empty context"):
                # Older BEAM runs left the score unset for a confirmed empty recall.
                score = 0.0
            else:
                failed = True
        for label in set(labels or [""]):
            stats = by_type.setdefault(label, {"total": 0, "correct": 0, "scoring_failed": 0, "score_sum": 0.0})
            stats["total"] += 1
            if failed:
                stats["scoring_failed"] += 1
            else:
                stats["correct"] += int(correct)
                stats["score_sum"] += score

    result = []
    for label, stats in sorted(by_type.items(), key=lambda item: (not item[0], item[0])):
        scored = stats["total"] - stats["scoring_failed"]
        score_sum = stats.pop("score_sum")
        result.append({"label": label, **stats, "score": score_sum / scored if scored else None})
    return result


@lru_cache(maxsize=16)
def _legacy_question_type_report(path: Path, dataset: str) -> list[dict]:
    with path.open("rb") as source:
        return summarize_question_types(dataset, ijson.items(source, "results.item"))


def _run_summary(directory: Path, job: dict) -> dict:
    summary = _read_json(directory / "summary.json")
    if summary and "avg_context_tokens" not in summary and job["status"] in {"completed", "scoring_incomplete"}:
        average, _ = _context_token_report(_result_path(directory, job))
        summary["avg_context_tokens"] = average
    return summary


async def _stream_result(path: Path):
    # Keep one file open for the whole download, even if the runner replaces the
    # checkpoint. The response does not advertise a size from a separate stat.
    async with await open_file(path, "rb") as result:
        while chunk := await result.read(64 * 1024):
            yield chunk


def _log_tail(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as log:
        log.seek(max(0, path.stat().st_size - 24_000))
        return log.read(24_000).decode("utf-8", errors="replace")


def create_app(output_dir: Path) -> FastAPI:
    allowed_hosts = ["127.0.0.1", "localhost"]
    allowed_hosts.extend(
        host.strip().lower()
        for host in os.getenv("AMB_WORKBENCH_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    )
    if any("*" in host for host in allowed_hosts):
        raise ValueError("AMB_WORKBENCH_ALLOWED_HOSTS must list explicit IP addresses or hostnames.")

    manager = RunManager(output_dir)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts, www_redirect=False)
    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
    evaluation = APIRouter()

    @app.middleware("http")
    async def local_writes(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin != f"http://{request.headers.get('host')}":
            return JSONResponse({"detail": "请求来源与此服务不一致。"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.method == "POST" and request.url.path in {"/api/runs", "/knowledge/api/runs"}:
            return JSONResponse({"detail": "评测配置无效，请检查名称、类型和本轮连接。"}, status_code=422)
        if request.url.path.startswith("/api/models/"):
            return JSONResponse({"detail": "模型配置格式无效，请检查提供方、模型 ID、地址和 Key。"}, status_code=422)
        return await request_validation_exception_handler(request, exc)

    @app.get("/")
    @app.get("/knowledge")
    def home():
        return FileResponse(ASSETS / "index.html")

    @evaluation.get("/config")
    def config(request: Request):
        backends = manager.connections.public()
        if is_knowledge_request(request):
            return {
                "datasets": KNOWLEDGE_SPLITS,
                "retrieval_datasets": list(KNOWLEDGE_SPLITS),
                "score_types": {name: cls.score_type for name, cls in WORKBENCH_KNOWLEDGE_DATASETS.items()},
                "split_errors": {
                    name: {split: error for split in splits if (error := _dataset_error(name, split))}
                    for name, splits in KNOWLEDGE_SPLITS.items()
                },
                "backends": {key: backends[key] for key in KNOWLEDGE_BACKENDS},
                "backend_datasets": KNOWLEDGE_BACKEND_DATASETS,
            }
        return {
            "datasets": DATASETS,
            "retrieval_datasets": sorted(RETRIEVAL_DATASETS),
            "choice_datasets": sorted(CHOICE_DATASETS),
            "split_errors": {"beam": {
                split: error for split in BEAM_SPLITS if (error := _dataset_error("beam", split))
            }},
            "backends": {key: backends[key] for key in backends if key not in KNOWLEDGE_BACKENDS},
            "models": manager.models.public(),
            "model_services": manager.models.services(),
            "beam_default_judge_model": DEFAULT_JUDGE_MODEL,
        }

    @app.get("/api/datasets/{dataset}/{split}/query-count")
    def dataset_query_count(dataset: str, split: str):
        if split not in DATASETS.get(dataset, ()):
            raise HTTPException(404, "评测数据集或 split 不可用。")
        if dataset == "personamem":
            return {"query_count": STANDARD_QUERY_COUNTS[dataset][split]}
        try:
            if dataset == "beam":
                local_path = beam_local_data_path(split, require_split=True)
                loader = BEAMDataset(require_local_split=True)
            else:
                local_path = os.getenv(f"{dataset.replace('-', '_').upper()}_DATA_PATH")
                loader = get_dataset(dataset)
            count = len(loader.load_queries(split)) if local_path else STANDARD_QUERY_COUNTS[dataset][split]
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise HTTPException(422, "无法读取本地评测数据的题数。") from exc
        return {"query_count": count}

    @app.post("/api/models/{role}")
    def save_model(role: Role, update: ModelInput):
        try:
            return manager.models.save(role, update)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/models/{role}/reset")
    def reset_model(role: Role):
        return manager.models.reset(role)

    @app.post("/api/models/{role}/test")
    def test_model(role: Role, update: ModelProbeInput):
        label = ROLES[role]
        try:
            model, model_env = manager.models.for_probe(role, update)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        env = {**os.environ, **model_env}
        # Evaluation debugging can log the provider's raw response. Keep probes out of that log.
        env.pop("AMB_BAD_JSON_LOG", None)
        if update.bypass_proxy:
            env = {name: value for name, value in env.items()
                   if name.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
            env.update(NO_PROXY="*", no_proxy="*")
        started = monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "memory_bench.workbench_worker", "--probe-model", role],
                input=json.dumps(model), encoding="utf-8", env=env, timeout=MODEL_PROBE_TIMEOUT,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(504, f"{label}测试在 {MODEL_PROBE_TIMEOUT} 秒内未完成。请检查后台网络或服务状态。") from None
        except OSError:
            raise HTTPException(500, f"后台无法启动{label}测试。") from None

        if result.returncode:
            messages = {
                ModelProbeExit.INVALID_OUTPUT: (
                    "模型返回的格式不正确，或评分结果与预期不符。" if role == "judge" else "模型返回的回答格式不正确。"
                ),
                ModelProbeExit.AUTH_FAILED: f"{label}鉴权失败，请检查 API Key 和访问权限。",
                ModelProbeExit.NOT_FOUND: f"未找到{label}或 API 路径，请检查模型 ID 和地址。",
                ModelProbeExit.REQUEST_REJECTED: "模型拒绝了请求，请检查模型配置和结构化输出支持。",
            }
            raise HTTPException(502, messages.get(result.returncode, f"后台调用{label}失败，请检查 API 地址和服务状态。"))
        return {"model": f"{model['provider']}:{model['model']}", "elapsed_ms": round((monotonic() - started) * 1000)}

    @evaluation.get("/runs")
    def runs(request: Request, limit: Annotated[int, Query(ge=1, le=100)] = 30,
             offset: Annotated[int, Query(ge=0)] = 0):
        knowledge = is_knowledge_request(request)
        jobs = [job for path in manager.root.glob("*/run.json")
                if (job := _read_json(path)) and is_knowledge_job(job) == knowledge]
        ordered = sorted(jobs, key=lambda job: (job["started_at"], job["id"]), reverse=True)
        return ordered[offset:offset + limit]

    @evaluation.post("/runs", status_code=201)
    def start(request: Request, selection: RunRequest):
        if is_knowledge_request(request) != selection.is_knowledge:
            raise HTTPException(422, "当前入口不支持该评测数据集。")
        return manager.start(selection)

    @evaluation.get("/runs/capacity")
    def run_capacity():
        return manager.capacity()

    @evaluation.get("/runs/{run_id}")
    def detail(request: Request, run_id: str):
        directory = manager.directory(run_id, is_knowledge_request(request))
        job = _read_json(directory / "run.json")
        return {
            **job,
            "result_available": _result_path(directory, job).is_file(),
            "progress": _read_json(directory / "progress.json"),
            "summary": _run_summary(directory, job),
            "log": _log_tail(directory / "run.log"),
        }

    @evaluation.get("/runs/{run_id}/metrics")
    def metrics(request: Request, run_id: str):
        directory = manager.directory(run_id, is_knowledge_request(request))
        job = _read_json(directory / "run.json")
        return {**job, "summary": _run_summary(directory, job)}

    @evaluation.get("/runs/{run_id}/context-tokens")
    def context_tokens(request: Request, run_id: str, offset: Annotated[int, Query(ge=0)] = 0,
                       limit: Annotated[int, Query(ge=1, le=100)] = 20):
        directory = manager.directory(run_id, is_knowledge_request(request))
        job = _read_json(directory / "run.json")
        if job["status"] not in {"completed", "scoring_incomplete"}:
            raise HTTPException(404, "当前还没有完整的逐题上下文指标。")
        average, rows = _context_token_report(_result_path(directory, job))
        return {"average": average, "total": len(rows), "rows": rows[offset:offset + limit]}

    @evaluation.get("/runs/{run_id}/question-types")
    def question_types(request: Request, run_id: str):
        directory = manager.directory(run_id, is_knowledge_request(request))
        job = _read_json(directory / "run.json")
        if job["status"] not in {"completed", "scoring_incomplete"}:
            raise HTTPException(404, "评测完成后可查看题型统计。")
        rows = _read_json(directory / "summary.json").get("question_types")
        if rows is None:
            try:
                rows = _legacy_question_type_report(_result_path(directory, job), job["selection"]["dataset"])
            except FileNotFoundError:
                raise HTTPException(404, "当前还没有评测结果。") from None
        return {"rows": rows}

    @evaluation.post("/runs/{run_id}/cancel", status_code=202)
    def cancel(request: Request, run_id: str):
        manager.directory(run_id, is_knowledge_request(request))
        manager.cancel(run_id)
        return {"status": "cancelling"}

    @evaluation.get("/runs/{run_id}/result")
    def result(request: Request, run_id: str):
        directory = manager.directory(run_id, is_knowledge_request(request))
        job = _read_json(directory / "run.json")
        selection = job["selection"]
        path = _result_path(directory, job)
        if not path.is_file():
            raise HTTPException(404, "当前还没有评测结果。")
        filename = f"{selection['dataset']}-{selection['split']}-{selection['backend']}-{run_id[:8]}.json"
        return StreamingResponse(
            _stream_result(path), media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    app.include_router(evaluation, prefix="/api")
    app.include_router(evaluation, prefix="/knowledge/api")
    return app
