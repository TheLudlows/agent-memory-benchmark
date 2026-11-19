"""Evaluate one Workbench run in a separate process."""

import json
import logging
import sys
import traceback
from collections.abc import Iterable
from dataclasses import asdict
from hashlib import sha256
from math import ceil
from pathlib import Path
from statistics import mean
from time import monotonic

from .dataset import get_dataset
from .dataset.beam import BEAMDataset
from .json_files import is_windows_file_contention, write_json
from .judge import GeminiJudge, ScoringError
from .memory import get_memory_provider
from .memory.hindsight import HindsightHTTPMemoryProvider
from .memory.mem0_cloud import Mem0CloudMemoryProvider
from .models import Document
from .modes import get_mode
from .modes.rag import RAGMode
from .run_diagnostics import error_location, error_reason, evaluation_step
from .runner import EvaluationProgress, EvalRunner
from .workbench import SCORING_INCOMPLETE_EXIT_CODE, RunSelection, _read_json
from .workbench_models import ModelProbeExit, Role, worker_model

logger = logging.getLogger(__name__)


def run(directory: Path) -> int:
    job = _read_json(directory / "run.json")
    spec = RunSelection.model_validate(job["selection"])
    run_id = job["id"]
    progress_path = directory / "progress.json"
    expected_queries = 0
    retry_progress_at = 0.0

    def progress(update: EvaluationProgress) -> None:
        nonlocal expected_queries, retry_progress_at
        expected_queries = update.total
        if update.stage != "finalizing" and monotonic() < retry_progress_at:
            return
        try:
            write_json(progress_path, asdict(update))
        except PermissionError as exc:
            if not is_windows_file_contention(exc):
                raise
            if not retry_progress_at:
                logger.warning("Could not update the progress file; the evaluation will continue: %s", exc)
            retry_progress_at = monotonic() + 2
        else:
            retry_progress_at = 0.0

    progress(EvaluationProgress("loading"))
    prefix = f"amb-web-{run_id}-"
    if spec.backend == "hindsight-http":
        memory = HindsightHTTPMemoryProvider(bank_prefix=prefix)
    elif spec.backend == "mem0-cloud":
        memory = Mem0CloudMemoryProvider(user_prefix=prefix)
    else:
        memory = get_memory_provider(spec.backend)

    document_digest = sha256()

    def record_documents(documents: Iterable[Document]) -> None:
        for document in documents:
            item = json.dumps(asdict(document), ensure_ascii=False, sort_keys=True)
            document_digest.update((item + "\n").encode("utf-8"))

    models = job["models"]
    with evaluation_step("初始化回答模型"):
        answer_llm = worker_model("answer", models["answer"]) if spec.mode == "rag" else None
    with evaluation_step("初始化评分模型"):
        judge_llm = worker_model("judge", models["judge"]) if spec.mode == "rag" else None
    runner = EvalRunner(
        output_dir=directory / "results", on_progress=progress, on_documents=record_documents,
        judge_llm=judge_llm,
    )
    dataset = BEAMDataset(require_local_split=True) if spec.dataset == "beam" else get_dataset(spec.dataset)
    summary = runner.run(
        dataset=dataset, split=spec.split, memory=memory,
        mode=get_mode(spec.mode, llm=answer_llm), query_limit=spec.query_limit, run_name="web-" + run_id,
        description=job.get("name"),
    )
    if not expected_queries or summary.total_queries != expected_queries:
        raise ValueError(
            f"Evaluated {summary.total_queries} of {expected_queries} selected questions. Check the dataset and local data file."
        )
    times = sorted(result.retrieve_time_ms for result in summary.results if result.retrieve_time_ms is not None)
    question_digest = sha256()
    for row in sorted(summary.results, key=lambda row: row.query_id):
        question = [
            row.query_id, row.query, row.gold_answers, row.category_axes,
            row.meta.get("rubric"), row.meta.get("query_timestamp"),
        ]
        if spec.mode == "retrieval":
            question.extend(row.meta.get(key) for key in ("scope", "budget", "expect"))
            question.append("source_attribution_required")
        if dataset.query_timestamp_cutoff:
            question.append("ingest_until_query_time")
        question_digest.update((json.dumps(question, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    write_json(directory / "summary.json", {
        "score": summary.accuracy,
        "correct": summary.correct,
        "total_queries": summary.total_queries,
        "scoring_failed_queries": summary.scoring_failed_queries,
        "ingested_docs": summary.ingested_docs,
        "ingestion_time_ms": summary.ingestion_time_ms,
        "ingestion_wall_time_ms": summary.ingestion_wall_time_ms,
        "avg_retrieve_time_ms": mean(times) if times else None,
        "p95_retrieve_time_ms": times[ceil(0.95 * len(times)) - 1] if times else None,
        "question_fingerprint": question_digest.hexdigest(),
        "document_fingerprint": document_digest.hexdigest(),
        "answer_llm": summary.answer_llm,
        "judge_llm": summary.judge_llm,
        **dataset.summary_metrics(summary.results),
    })
    return SCORING_INCOMPLETE_EXIT_CODE if summary.scoring_failed_queries else 0


def probe_model(role: Role) -> int:
    try:
        llm = worker_model(role, json.load(sys.stdin))
        if role == "answer":
            result = RAGMode(llm=llm).answer_from_context("Which city did Lina visit?", "Lina visited Paris.")
            valid = (
                isinstance(result.answer, str) and bool(result.answer.strip()) and isinstance(result.reasoning, str)
            )
        else:
            score = GeminiJudge(llm=llm).score("What is 2 + 2?", "4", ["4"], strict=True)
            valid = score.correct
        return 0 if valid else ModelProbeExit.INVALID_OUTPUT
    except (ScoringError, json.JSONDecodeError, KeyError, TypeError):
        return ModelProbeExit.INVALID_OUTPUT
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (401, 403):
            return ModelProbeExit.AUTH_FAILED
        if status == 404:
            return ModelProbeExit.NOT_FOUND
        if status in (400, 422):
            return ModelProbeExit.REQUEST_REJECTED
        return ModelProbeExit.REQUEST_FAILED


def run_with_diagnostics(directory: Path) -> int:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return run(directory)
    except Exception as exc:
        try:
            stage = _read_json(directory / "progress.json").get("stage")
        except (OSError, ValueError):
            stage = None
        default = {
            "loading": "加载评测数据",
            "ingesting": "写入记忆",
            "evaluating": "检索、回答或评分",
            "ingesting_and_evaluating": "写入或答题",
            "finalizing": "保存评测结果",
        }.get(stage, "启动评测")
        print("调用位置：", file=sys.stderr)
        traceback.print_tb(exc.__traceback__)
        print(f"[评测失败] {error_location(exc, default)}：{error_reason(exc)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    if sys.argv[1] == "--probe-model":
        raise SystemExit(probe_model(sys.argv[2]))
    raise SystemExit(run_with_diagnostics(Path(sys.argv[1])))
