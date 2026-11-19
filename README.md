# AMB — Agent Memory Benchmark

We built AMB because we wanted to be honest about how Hindsight performs — and because no existing benchmark gave us the full picture. AMB is fully open: datasets, prompts, scoring logic, and results.

Live leaderboard: **[agentmemorybenchmark.ai](https://agentmemorybenchmark.ai)**

## Docker quick start · 一键运行 LoCoMo

在已克隆的仓库中使用 Docker Compose 2.24 或更新版本。镜像内置 LoCoMo 原始数据和分词表，连接已启动的 Hindsight 测试 API。首次构建需要联网。

1. 将仓库根目录的 `.env.example` 复制为 `.env`，填写 `GEMINI_API_KEY`。默认由 Gemini 回答和评分。
2. 将 `HINDSIGHT_HTTP_URL` 指向测试实例，启用鉴权时填写 `HINDSIGHT_HTTP_KEY`。宿主机服务用 `http://host.docker.internal:8888`，远端服务用实际 URL。Linux 上宿主机服务的监听地址须允许容器网桥访问。
3. 在仓库根目录运行完整评测。默认会重建同名评测 Bank，请使用专用测试实例。

```sh
docker compose run --rm --build amb
```

全量为 10 段对话、1,540 道题，结果保存在 `amb-results/locomo/hindsight-http/rag/locomo10.json`。要先验证服务连通性、写入、问答和评分，可仅跑 5 题，单独保存结果。

```sh
docker compose run --rm --build amb run --dataset locomo --split locomo10 --memory hindsight-http --mode rag --query-limit 5 --name smoke
```

只检查镜像和数据可运行 `docker compose run --rm --build amb dataset-stats --dataset locomo`，无需 Key。容器中的 `localhost` 指向容器自身。此镜像用于 HTTP 评测；ContextBase 的普通业务 Key 还需适配预建 Bank 的写入流程。

主分支镜像工作流成功后，亦可在仓库的 Packages 页面获取 `ghcr.io/theludlows/agent-memory-benchmark:latest` 或对应的 `sha-<commit>` 标签。镜像内 LoCoMo 的来源是 [snap-research/locomo](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)，授权文件位于 `/opt/locomo/LICENSE.txt`。

## The problem with existing benchmarks

LoComo and LongMemEval are solid datasets, but they were designed for an era of 32k context windows. State-of-the-art models now have million-token context windows — on most instances, a naive "dump everything into context" approach scores competitively, not because it's a good memory architecture, but because retrieval has become the easy part. The benchmarks can no longer tell them apart.

Both datasets were also built around chatbot use cases. Agents today don't just answer questions about conversation history — they research, plan, execute multi-step tasks, and build knowledge across many interactions. AMB adds datasets that focus on agentic tasks: memory across tool calls, knowledge built from document research, preferences applied to multi-step decisions.

## What AMB measures

A memory system that scores 90% accuracy but costs $10 per user per day is not better than one that scores 82% and costs $0.10. AMB starts from accuracy because it's the hardest to fake, and tracks speed and token cost alongside it.

The only credible benchmark result is one you can reproduce yourself. AMB publishes everything: the evaluation harness, judge prompts, answer generation prompts, and the exact models used. Small changes to any of these can swing accuracy scores by double digits — we publish all of them.

## How it works

1. **Ingest** — documents from a dataset are loaded into a memory provider
2. **Retrieve** — for each query the memory provider retrieves relevant context
3. **Generate** — a Gemini model produces an answer from the retrieved context
4. **Judge** — a second Gemini call scores the answer against gold answers

Retrieval time is tracked separately from generation; ingestion time is also recorded.

One dataset breaks this shape on purpose. **PrecisionMemBench** ([upstream](https://github.com/tenurehq/precisionmembench)) asserts, per case, exactly which memories must and must not come back — noise is a hard failure, not an invisible cost — so it runs with `--mode retrieval`: no answer LLM, no judge, and the score is the fraction of cases whose returned belief-ID set satisfies every assertion.

```bash
uv run amb run --dataset precisionmembench --split single-turn --memory hindsight-cloud --mode retrieval
```

**Knowledge** datasets (ViDoRe V3 as markdown or PDFs, MIRACL-Vision, video, BrowseComp-Plus, ATM-Bench) are search only too: a corpus goes into a Hindsight knowledge bank and each query's ranking is scored with the leaderboard's own metric, NDCG@10 or Recall@10 (the dataset's `score_type`, which the viewer shows instead of "accuracy"). Corpora are cached in `~/.cache/hindsight/knowledge-evals`; `--query-limit` cuts queries, never the corpus.

```bash
HINDSIGHT_KNOWLEDGE_URL=http://localhost:8888 HINDSIGHT_KNOWLEDGE_KEY=... \
  uv run amb run --dataset vidore-v3-markdown --split hr --memory hindsight-knowledge --mode retrieval
```

## Setup

```bash
# Copy and fill in your API key
cp .env.example .env   # or just create .env with:
# GEMINI_API_KEY=...
```

## Usage

```bash
# List available datasets, memory providers, and modes
uv run amb providers

# List domains for a dataset
uv run amb domains --dataset personamem

# Run a benchmark
uv run amb run --dataset personamem --domain 32k --memory bm25

# Limit scale for a quick test
uv run amb run --dataset personamem --domain 32k --memory bm25 --query-limit 20

# Oracle mode: ingest only gold documents (tests generation quality in isolation)
uv run amb run --dataset personamem --domain 32k --memory bm25 --oracle

# Dataset statistics
uv run amb dataset-stats --dataset personamem

# Browse results in the browser
uv run amb view
```

## Results

Results are saved to `outputs/{dataset}/{memory}/{mode}/{domain}.json` and can be explored with `uv run amb view`.

## Requirements

- Python ≥ 3.11
- `GEMINI_API_KEY` in `.env` or environment
- For MemBench: set `MEMBENCH_DATA_PATH` to your local data directory
