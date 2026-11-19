"""Local-only API for starting and inspecting AMB runs."""

import json
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock, Thread
from time import monotonic
from typing import Annotated, Literal
from uuid import uuid4

from anyio import open_file
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .dataset import get_dataset
from .dataset.beam import BEAMDataset, DEFAULT_JUDGE_MODEL, SPLITS as BEAM_SPLITS, local_data_path as beam_local_data_path
from .dataset.locomo import SPLITS as LOCOMO_SPLITS
from .dataset.longmemeval import SPLITS as LONGMEMEVAL_SPLITS
from .workbench_connections import BackendId, Connections, RunConnection
from .workbench_models import JudgeProbeExit, ModelInput, ModelSettings, Role

DATASETS = {
    "locomo": LOCOMO_SPLITS,
    "beam": BEAM_SPLITS,
    "longmemeval": LONGMEMEVAL_SPLITS,
}
# Counts from data/<dataset>/<split>/stats.json.gz. Previews of the standard
# splits should be available before the evaluation downloads their corpora.
STANDARD_QUERY_COUNTS = {
    "locomo": {"locomo10": 1540},
    "beam": {"100k": 400, "500k": 700, "1m": 700, "10m": 200},
    "longmemeval": {"s": 500},
}
RUN_ID = re.compile(r"[0-9a-f]{32}")
SCORING_INCOMPLETE_EXIT_CODE = 2
JUDGE_PROBE_TIMEOUT = 20
ASSETS = Path(__file__).with_name("workbench_ui")


class RunSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["locomo", "beam", "longmemeval"]
    split: str
    backend: BackendId
    query_limit: int | None = Field(default=5, ge=1, strict=True)

    @model_validator(mode="after")
    def check_split(self):
        if self.split not in DATASETS[self.dataset]:
            raise ValueError(f"Split {self.split!r} is unavailable for {self.dataset}.")
        return self


class RunRequest(RunSelection):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    connection: RunConnection


def _dataset_error(dataset: str, split: str) -> str | None:
    if dataset == "beam":
        try:
            beam_local_data_path(split, require_split=True)
        except ValueError as exc:
            return str(exc)
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


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


class RunManager:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.connections = Connections(
            _settings_path(self.root, "connections.json"), legacy_path=self.root / ".connections.json",
        )
        self.models = ModelSettings(_settings_path(self.root, "models.json"))
        self.lock = Lock()
        self.active: tuple[str, subprocess.Popen] | None = None
        self.cancelling = False
        for path in self.root.glob("*/run.json"):
            job = _read_json(path)
            if job.get("status") in {"running", "cancelling"}:
                write_json(path, {**job, "status": "interrupted", "ended_at": _now()})

    def start(self, selection: RunRequest) -> dict:
        if error := _dataset_error(selection.dataset, selection.split):
            raise HTTPException(422, error)
        with self.lock:
            if self.active:
                raise HTTPException(409, "已有评测正在运行。")
            try:
                connection = self.connections.for_run(selection.backend, selection.connection)
                models, model_env = self.models.for_run(selection.dataset)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            try:
                backend_config = self.connections.remember(selection.backend, connection)
            except OSError as exc:
                raise HTTPException(500, "无法保存本轮连接，请检查配置目录。") from exc
            run_id = uuid4().hex
            directory = self.root / run_id
            directory.mkdir(mode=0o700)
            job = {
                "id": run_id,
                "name": selection.name,
                "selection": selection.model_dump(exclude={"name", "connection"}),
                "target": {"api_url": connection.api_url, "bank_id": connection.bank_id},
                "models": models,
                "status": "running",
                "started_at": _now(),
            }
            write_json(directory / "run.json", job)
            env = {
                **os.environ, **connection.worker_env(selection.backend), **model_env,
                "NO_COLOR": "1", "PYTHONUNBUFFERED": "1",
            }
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "memory_bench.workbench_worker", str(directory)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, text=True, encoding="utf-8", errors="replace",
                )
            except OSError as exc:
                write_json(directory / "run.json", {**job, "status": "failed", "ended_at": _now()})
                raise HTTPException(500, "无法启动 AMB 进程。") from exc
            self.active = (run_id, process)
            secrets = sorted(
                {value for key, value in env.items() if value and key.endswith(("_KEY", "_TOKEN", "_SECRET"))},
                key=len, reverse=True,
            )
            Thread(target=self._monitor, args=(job, process, secrets), daemon=True).start()
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
                if self.cancelling:
                    status = "cancelled"
                else:
                    status = {0: "completed", SCORING_INCOMPLETE_EXIT_CODE: "scoring_incomplete"}.get(code, "failed")
                job.update(status=status, ended_at=_now(), exit_code=code)
                try:
                    write_json(directory / "run.json", job)
                finally:
                    self.active = None
                    self.cancelling = False

    def cancel(self, run_id: str) -> None:
        with self.lock:
            if not self.active or self.active[0] != run_id:
                raise HTTPException(409, "此评测已结束。")
            if self.cancelling:
                return
            process = self.active[1]
            if process.poll() is not None:
                raise HTTPException(409, "此评测已结束。")
            self.cancelling = True
            path = self.root / run_id / "run.json"
            write_json(path, {**_read_json(path), "status": "cancelling"})
            process.terminate()

    def stop(self) -> None:
        with self.lock:
            if not self.active:
                return
            _, process = self.active
            self.cancelling = process.poll() is None
            if self.cancelling:
                process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def directory(self, run_id: str) -> Path:
        path = self.root / run_id
        if not RUN_ID.fullmatch(run_id) or not (path / "run.json").is_file():
            raise HTTPException(404, "评测不存在。")
        return path


def _result_path(directory: Path, job: dict) -> Path:
    selection = job["selection"]
    return directory / "results" / selection["dataset"] / f"web-{job['id']}" / "rag" / f"{selection['split']}.json"


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
    manager = RunManager(output_dir)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")

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
        if request.method == "POST" and request.url.path == "/api/runs":
            return JSONResponse({"detail": "评测配置无效，请检查名称、类型和本轮连接。"}, status_code=422)
        if request.url.path.startswith("/api/models/"):
            return JSONResponse({"detail": "模型配置格式无效，请检查提供方、模型 ID、地址和 Key。"}, status_code=422)
        return await request_validation_exception_handler(request, exc)

    @app.get("/")
    def home():
        return FileResponse(ASSETS / "index.html")

    @app.get("/api/config")
    def config():
        return {
            "datasets": DATASETS,
            "split_errors": {"beam": {
                split: error for split in BEAM_SPLITS if (error := _dataset_error("beam", split))
            }},
            "backends": manager.connections.public(),
            "models": manager.models.public(),
            "model_services": manager.models.services(),
            "beam_default_judge_model": DEFAULT_JUDGE_MODEL,
        }

    @app.get("/api/datasets/{dataset}/{split}/query-count")
    def dataset_query_count(dataset: str, split: str):
        if split not in DATASETS.get(dataset, ()):
            raise HTTPException(404, "评测数据集或 split 不可用。")
        try:
            if dataset == "beam":
                local_path = beam_local_data_path(split, require_split=True)
                loader = BEAMDataset(require_local_split=True)
            else:
                local_path = os.getenv(f"{dataset.upper()}_DATA_PATH")
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

    @app.post("/api/models/judge/test")
    def test_judge_model(update: ModelInput):
        try:
            model, model_env = manager.models.for_judge_probe(update)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        env = {**os.environ, **model_env}
        # Evaluation debugging can log the provider's raw response. Keep probes out of that log.
        env.pop("AMB_BAD_JSON_LOG", None)
        started = monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "memory_bench.workbench_worker", "--probe-judge"],
                input=json.dumps(model), encoding="utf-8", env=env, timeout=JUDGE_PROBE_TIMEOUT,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(504, f"评分模型测试在 {JUDGE_PROBE_TIMEOUT} 秒内未完成。请检查后台网络或服务状态。") from None
        except OSError:
            raise HTTPException(500, "后台无法启动评分模型测试。") from None

        if result.returncode:
            messages = {
                JudgeProbeExit.INVALID_SCORE: "模型返回的格式不正确，或评分结果与预期不符。",
                JudgeProbeExit.AUTH_FAILED: "评分模型鉴权失败，请检查 API Key 和访问权限。",
                JudgeProbeExit.NOT_FOUND: "未找到评分模型或 API 路径，请检查模型 ID 和地址。",
                JudgeProbeExit.REQUEST_REJECTED: "评分请求被拒绝，请检查模型配置和结构化输出支持。",
            }
            raise HTTPException(502, messages.get(result.returncode, "后台调用评分模型失败，请检查 API 地址和服务状态。"))
        return {"model": f"{model['provider']}:{model['model']}", "elapsed_ms": round((monotonic() - started) * 1000)}

    @app.get("/api/runs")
    def runs(limit: Annotated[int, Query(ge=1, le=100)] = 30, offset: Annotated[int, Query(ge=0)] = 0):
        jobs = [_read_json(path) for path in manager.root.glob("*/run.json")]
        ordered = sorted(jobs, key=lambda job: (job["started_at"], job["id"]), reverse=True)
        return ordered[offset:offset + limit]

    @app.post("/api/runs", status_code=201)
    def start(selection: RunRequest):
        return manager.start(selection)

    @app.get("/api/runs/{run_id}")
    def detail(run_id: str):
        directory = manager.directory(run_id)
        job = _read_json(directory / "run.json")
        return {
            **job,
            "result_available": _result_path(directory, job).is_file(),
            "progress": _read_json(directory / "progress.json"),
            "summary": _read_json(directory / "summary.json"),
            "log": _log_tail(directory / "run.log"),
        }

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str):
        directory = manager.directory(run_id)
        return {**_read_json(directory / "run.json"), "summary": _read_json(directory / "summary.json")}

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    def cancel(run_id: str):
        manager.directory(run_id)
        manager.cancel(run_id)
        return {"status": "cancelling"}

    @app.get("/api/runs/{run_id}/result")
    def result(run_id: str):
        directory = manager.directory(run_id)
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

    return app
