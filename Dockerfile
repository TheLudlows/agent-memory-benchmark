# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.9.18-python3.12-bookworm-slim AS cli
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    LOCOMO_DATA_PATH=/opt/locomo/locomo10.json \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    OMB_ANSWER_LLM=gemini \
    OMB_JUDGE_LLM=gemini

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group http-runner
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps --editable . \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chmod -R a+rX /opt/tiktoken

# Bundle the original, unmodified dialogs and the upstream dataset license.
ADD --chmod=0444 --checksum=sha256:79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4 https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json /opt/locomo/locomo10.json
ADD --chmod=0444 --checksum=sha256:41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0 https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt /opt/locomo/LICENSE.txt
ENTRYPOINT ["amb"]
CMD ["run", "--dataset", "locomo", "--split", "locomo10", "--memory", "hindsight-http", "--mode", "rag"]


FROM cli AS workbench
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group http-runner --only-group workbench \
    && uv pip install --no-deps --editable .
CMD ["workbench", "--docker"]

FROM cli AS default
