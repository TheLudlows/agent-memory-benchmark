"""Evaluate one Workbench run in a separate process."""

import json
import sys
from collections.abc import Iterable
from dataclasses import asdict
from hashlib import sha256
from math import ceil
from pathlib import Path
from statistics import mean

from .dataset import get_dataset
from .dataset.beam import BEAMDataset
from .judge import GeminiJudge, ScoringError
from .memory import get_memory_provider
from .memory.hindsight import HindsightHTTPMemoryProvider
from .memory.mem0_cloud import Mem0CloudMemoryProvider
from .models import Document
from .modes import get_mode
from .modes.rag import RAGMode
from .runner import EvaluationProgress, EvalRunner
from .workbench import SCORING_INCOMPLETE_EXIT_CODE, RunSelection, _read_json, write_json
from .workbench_models import ModelProbeExit, Role, worker_model


def run(directory: Path) -> int:
    job = _read_json(directory / "run.json")
    spec = RunSelection.model_validate(job["selection"])
    run_id = job["id"]
    progress_path = directory / "progress.json"
    write_json(progress_path, asdict(EvaluationProgress("loading")))
    prefix = f"amb-web-{run_id}-"
    if spec.backend == "hindsight-http":
        memory = HindsightHTTPMemoryProvider(bank_prefix=prefix)
    elif spec.backend == "mem0-cloud":
        memory = Mem0CloudMemoryProvider(user_prefix=prefix)
    else:
        memory = get_memory_provider(spec.backend)

    expected_queries = 0

    def progress(update: EvaluationProgress) -> None:
        nonlocal expected_queries
        expected_queries = update.total
        write_json(progress_path, asdict(update))

    document_digest = sha256()

    def record_documents(documents: Iterable[Document]) -> None:
        for document in documents:
            item = json.dumps(asdict(document), ensure_ascii=False, sort_keys=True)
            document_digest.update((item + "\n").encode("utf-8"))

    models = job["models"]
    runner = EvalRunner(
        output_dir=directory / "results", on_progress=progress, on_documents=record_documents,
        judge_llm=worker_model("judge", models["judge"]),
    )
    dataset = BEAMDataset(require_local_split=True) if spec.dataset == "beam" else get_dataset(spec.dataset)
    summary = runner.run(
        dataset=dataset, split=spec.split, memory=memory,
        mode=get_mode("rag", llm=worker_model("answer", models["answer"])), query_limit=spec.query_limit, run_name="web-" + run_id,
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


if __name__ == "__main__":
    if sys.argv[1] == "--probe-model":
        raise SystemExit(probe_model(sys.argv[2]))
    raise SystemExit(run(Path(sys.argv[1])))
