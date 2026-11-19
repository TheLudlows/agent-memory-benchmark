# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.9.18-python3.12-bookworm-slim AS cli
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    OMB_ANSWER_LLM=gemini \
    OMB_JUDGE_LLM=gemini

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group http-runner
COPY src/ ./src/
COPY data/offline/ ./data/offline/
COPY data/beam/ ./data/beam/
COPY data/longmemeval/ ./data/longmemeval/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps --editable .
ENTRYPOINT ["amb"]
CMD ["run", "--dataset", "locomo", "--split", "locomo10", "--memory", "hindsight-http", "--mode", "rag"]


FROM cli AS workbench
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group http-runner --only-group workbench \
    && uv pip install --no-deps --editable .
CMD ["workbench", "--docker"]

FROM cli AS default
