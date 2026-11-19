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


FROM scratch AS workbench-data
ADD --chmod=0444 --checksum=sha256:c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace https://huggingface.co/datasets/Mohammadta/BEAM/resolve/3205395e897e7318c7b094ef4e6047b9b82dbb03/data/100K-00000-of-00001.parquet /beam/100k/100K-00000-of-00001.parquet
ADD --chmod=0444 --checksum=sha256:af05921c979355038e1761b7cde3d2dd713200dd3071b278de0200f6c7f30122 https://huggingface.co/datasets/Mohammadta/BEAM/resolve/3205395e897e7318c7b094ef4e6047b9b82dbb03/data/500K-00000-of-00001.parquet /beam/500k/500K-00000-of-00001.parquet
ADD --chmod=0444 --checksum=sha256:41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533 https://huggingface.co/datasets/Mohammadta/BEAM/resolve/3205395e897e7318c7b094ef4e6047b9b82dbb03/data/1M-00000-of-00001.parquet /beam/1m/1M-00000-of-00001.parquet
ADD --chmod=0444 --checksum=sha256:31d96fd47ec56221d202e68792f26c00e49467dd4b36ee105c36ebd19ef78ad5 https://huggingface.co/datasets/Mohammadta/BEAM-10M/resolve/9b2096193fe74e2837e4713e483351e19817773c/data/10M-00000-of-00002.parquet /beam/10m/10M-00000-of-00002.parquet
ADD --chmod=0444 --checksum=sha256:a4f13fe25af51d57405ae41008689c31d1421377f3efde56a024b441deb2ee65 https://huggingface.co/datasets/Mohammadta/BEAM-10M/resolve/9b2096193fe74e2837e4713e483351e19817773c/data/10M-00001-of-00002.parquet /beam/10m/10M-00001-of-00002.parquet
ADD --chmod=0444 --checksum=sha256:d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442 https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json /longmemeval/longmemeval_s_cleaned.json
ADD --chmod=0444 https://huggingface.co/datasets/Mohammadta/BEAM/resolve/3205395e897e7318c7b094ef4e6047b9b82dbb03/README.md /beam/BEAM-dataset-card.md
ADD --chmod=0444 https://huggingface.co/datasets/Mohammadta/BEAM-10M/resolve/9b2096193fe74e2837e4713e483351e19817773c/README.md /beam/BEAM-10M-dataset-card.md
ADD --chmod=0444 https://raw.githubusercontent.com/xiaowu0162/LongMemEval/24cdd24b3f67843016cb779d7e7e4dba9be65042/LICENSE /longmemeval/LICENSE.txt

FROM cli AS workbench
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --only-group http-runner --only-group workbench \
    && uv pip install --no-deps --editable .
COPY --from=workbench-data / /opt/amb-datasets/
ENV BEAM_BUNDLED_DATA_DIR=/opt/amb-datasets/beam \
    LONGMEMEVAL_DATA_PATH=/opt/amb-datasets/longmemeval/longmemeval_s_cleaned.json
EXPOSE 8090
CMD ["workbench", "--docker"]

FROM cli AS default
