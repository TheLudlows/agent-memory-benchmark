"""Evaluate one Workbench run in a separate process."""

import json
import sys
from hashlib import sha256
from math import ceil
from pathlib import Path
from statistics import mean

from .dataset import get_dataset
from .dataset.beam import BEAMDataset
from .llm import get_answer_llm
from .memory import get_memory_provider
from .memory.hindsight import HindsightHTTPMemoryProvider
from .memory.mem0_cloud import Mem0CloudMemoryProvider
from .modes import get_mode
from .runner import EvalRunner
from .workbench import SCORING_INCOMPLETE_EXIT_CODE, RunSelection, _read_json, write_json


def run(directory: Path) -> int:
    job = _read_json(directory / "run.json")
    spec = RunSelection.model_validate(job["selection"])
    run_id = job["id"]
    prefix = f"amb-web-{run_id}-"
    if spec.backend == "hindsight-http":
        memory = HindsightHTTPMemoryProvider(bank_prefix=prefix)
    elif spec.backend == "mem0-cloud":
        memory = Mem0CloudMemoryProvider(user_prefix=prefix)
    else:
        memory = get_memory_provider(spec.backend)

    expected_queries = 0

    def progress(stage: str, completed: int, total: int) -> None:
        nonlocal expected_queries
        expected_queries = total
        write_json(directory / "progress.json", {"stage": stage, "completed": completed, "total": total})

    progress("loading", 0, 0)
    runner = EvalRunner(output_dir=directory / "results", on_progress=progress)
    dataset = BEAMDataset(require_local_split=True) if spec.dataset == "beam" else get_dataset(spec.dataset)
    summary = runner.run(
        dataset=dataset, split=spec.split, memory=memory,
        mode=get_mode("rag", llm=get_answer_llm()), query_limit=spec.query_limit, run_name="web-" + run_id,
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
        "avg_retrieve_time_ms": mean(times) if times else None,
        "p95_retrieve_time_ms": times[ceil(0.95 * len(times)) - 1] if times else None,
        "question_fingerprint": question_digest.hexdigest(),
        "answer_llm": summary.answer_llm,
        "judge_llm": summary.judge_llm,
    })
    return SCORING_INCOMPLETE_EXIT_CODE if summary.scoring_failed_queries else 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
