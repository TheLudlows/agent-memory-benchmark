"""Evaluate one Workbench run in a separate process."""

import sys
from pathlib import Path
from statistics import mean

from .dataset import get_dataset
from .llm import get_answer_llm
from .memory import get_memory_provider
from .memory.hindsight import HindsightHTTPMemoryProvider
from .memory.mem0_cloud import Mem0CloudMemoryProvider
from .modes import get_mode
from .runner import EvalRunner
from .workbench import RunSelection, _read_json, write_json


def run(directory: Path) -> None:
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
    summary = runner.run(
        dataset=get_dataset(spec.dataset), split=spec.split, memory=memory,
        mode=get_mode("rag", llm=get_answer_llm()), query_limit=spec.query_limit, run_name="web-" + run_id,
    )
    if not expected_queries or summary.total_queries != expected_queries:
        raise ValueError(
            f"Evaluated {summary.total_queries} of {expected_queries} selected questions. Check the dataset and local data file."
        )
    times = [result.retrieve_time_ms for result in summary.results if result.retrieve_time_ms is not None]
    write_json(directory / "summary.json", {
        "score": summary.accuracy,
        "correct": summary.correct,
        "total_queries": summary.total_queries,
        "ingested_docs": summary.ingested_docs,
        "ingestion_time_ms": summary.ingestion_time_ms,
        "avg_retrieve_time_ms": mean(times) if times else None,
        "answer_llm": summary.answer_llm,
        "judge_llm": summary.judge_llm,
    })


if __name__ == "__main__":
    run(Path(sys.argv[1]))
