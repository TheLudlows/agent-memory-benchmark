"""Local-only API for starting and inspecting AMB runs."""

import json
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
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

from .dataset.beam import SPLITS as BEAM_SPLITS, local_data_path as beam_local_data_path
from .dataset.locomo import SPLITS as LOCOMO_SPLITS
from .dataset.longmemeval import SPLITS as LONGMEMEVAL_SPLITS
from .workbench_connections import BackendId, Connections, ConnectionUpdate, RunConnection

DATASETS = {
    "locomo": LOCOMO_SPLITS,
    "beam": BEAM_SPLITS,
    "longmemeval": LONGMEMEVAL_SPLITS,
}
LLM_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
}
RUN_ID = re.compile(r"[0-9a-f]{32}")
SCORING_INCOMPLETE_EXIT_CODE = 2
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


def _configured(*keys: str) -> bool:
    return any(os.environ.get(key, "").strip() for key in keys)


def _models() -> dict:
    return {
        role.lower(): {"provider": os.getenv(f"OMB_{role}_LLM", "gemini"), "model": os.getenv(f"OMB_{role}_MODEL")}
        for role in ("ANSWER", "JUDGE")
    }


def _missing_models() -> list[str]:
    missing = []
    for role, config in _models().items():
        keys = LLM_KEYS.get(config["provider"])
        if not keys:
            missing.append(f"OMB_{role.upper()}_LLM (gemini / openai / groq)")
        elif not _configured(*keys):
            missing.append(" / ".join(keys))
    return list(dict.fromkeys(missing))


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


class RunManager:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.connections = Connections(self.root / ".connections.json")
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
        try:
            connection = self.connections.for_run(selection.backend, selection.connection)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if missing := _missing_models():
            raise HTTPException(422, "请先在服务器的 .env 中配置 " + "、".join(missing))
        with self.lock:
            if self.active:
                raise HTTPException(409, "已有评测正在运行。")
            run_id = uuid4().hex
            directory = self.root / run_id
            directory.mkdir(mode=0o700)
            job = {
                "id": run_id,
                "name": selection.name,
                "selection": selection.model_dump(exclude={"name", "connection"}),
                "target": {"api_url": connection.api_url, "bank_id": connection.bank_id},
                "models": _models(),
                "status": "running",
                "started_at": _now(),
            }
            write_json(directory / "run.json", job)
            env = {**os.environ, **connection.worker_env(selection.backend), "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
            env.setdefault("OMB_ANSWER_LLM", "gemini")
            env.setdefault("OMB_JUDGE_LLM", "gemini")
            if env.get("GEMINI_API_KEY"):
                env["GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]
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
            return job

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
        if request.url.path.startswith("/api/connections/"):
            # FastAPI normally returns the invalid input, which may contain a key.
            return JSONResponse({"detail": "后端配置格式无效，请检查输入。"}, status_code=422)
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
            "models": _models(),
            "missing_models": _missing_models(),
        }

    @app.post("/api/connections/{backend}")
    def save_connection(backend: BackendId, update: ConnectionUpdate):
        try:
            return manager.connections.save(backend, update)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/connections/{backend}/reset")
    def reset_connection(backend: BackendId):
        return manager.connections.reset(backend)

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
