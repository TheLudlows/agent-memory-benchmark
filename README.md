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

只检查镜像和数据可运行 `docker compose run --rm --build amb dataset-stats --dataset locomo`，无需 Key。容器中的 `localhost` 指向容器自身。ContextBase 使用同一个镜像和预建的测试 Bank，见 [LoCoMo on ContextBase](#locomo-on-contextbase)。

PR 的 GitHub Actions 构建成功后，也可下载 linux/amd64 镜像产物 `amb-locomo-linux-amd64`。解压下载的 ZIP，执行 `docker load -i amb-locomo-linux-amd64.tar.gz`，然后在仓库目录用 `docker compose run --rm amb` 启动。

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

### AMB Workbench · 浏览器评测

评测台可搭配 LoCoMo、BEAM 或 LongMemEval 和 ContextBase、Mem0 Cloud 或 Hindsight HTTP。
每轮填写评测名称，选择数据集、split 和后端，默认跑 5 题，也可跑全量。题数上限只限制问答，
选中题目对应的会话历史仍会写入。运行详情会显示当前阶段、已用时、已写入的文档数和已处理的题数。
写入进度每组文档处理完后更新，答题进度逐题更新。结束后可查看得分、写入耗时、平均检索延迟，
或下载 AMB 逐题结果 JSON。完成的评测会保存得分、题数、写入耗时、文档数、平均检索延迟和检索 P95。
在「运行历史」勾选 2 至 4 轮，即可用图表对比得分与耗时。页面会提示题目、写入资料或回答、评分模型不一致的情况。
新评测的写入实际耗时只统计至少有一个写入任务在执行的时间，并行区间计一次。旧记录保留原累计耗时口径，
在另一个图表中展示。

1. 复制 `.env.example` 为 `.env`，填写回答和评分模型所用的 API Key。默认两者使用 Gemini。
2. 启动评测台并打开 http://127.0.0.1:8090：

   ```sh
   docker compose up --build amb-web
   ```

3. 在「本轮连接」填写 API 地址和 Key。若要复用同一地址已有的 Key，显式勾选「使用该地址已保存的 Key」。
   每轮的历史记录包含名称、数据集、后端和实际使用的地址。单次输入的 Key 不写入历史或结果。

   | 后端 | 页面配置 |
   |---|---|
   | ContextBase | API 地址、可读写的业务 API Key、已授权的测试 Bank ID |
   | Mem0 Cloud | API 地址（默认 `https://api.mem0.ai`）和 API Key，替换地址时需兼容 Mem0 Cloud API |
   | Hindsight HTTP | API 地址，开启鉴权时填写 API Key |

点击「保存为默认连接」可为后续评测预填地址。也可以通过 `.env` 提供默认值：ContextBase 使用
`CONTEXTBASE_API_URL`、`CONTEXTBASE_API_TOKEN`、`CONTEXTBASE_BANK_ID`；Mem0 使用 `MEM0_CLOUD_URL`、
`MEM0_CLOUD_KEY`（或 `MEM0_API_KEY`）；Hindsight 使用 `HINDSIGHT_HTTP_URL`、`HINDSIGHT_HTTP_KEY`。
页面的「恢复初始配置」会删除该后端保存的连接，再读取这些默认值。默认连接保存在结果目录之外，
直接运行时使用 `~/.config/amb/workbench/`（可用 `AMB_WORKBENCH_CONFIG_DIR` 指定私有配置目录），
Docker Compose 使用独立的 `amb-workbench-config` 卷。升级后，旧的 `.connections.json` 会自动迁移并从
结果目录移除。页面只显示 Key 是否已配置，改地址时需要重新填写 Key。

也可以在仓库根目录直接运行 `uv run amb workbench`。本地直跑时，同机 Hindsight 的地址通常是
`http://127.0.0.1:8888`。Docker 中连接宿主机用 `http://host.docker.internal:8888`，并确保 API
监听容器可访问的地址。

使用 OpenAI 兼容模型服务时，将 `OMB_ANSWER_LLM` 和 `OMB_JUDGE_LLM` 设置为 `openai`，并配置
`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OMB_ANSWER_MODEL` 和 `OMB_JUDGE_MODEL`。模型端点需要支持
Chat Completions 的 JSON Schema 输出，`OPENAI_BASE_URL` 应包含服务的 API 前缀（通常是 `/v1`）。ContextBase 和 Hindsight 在各自服务内配置写入和检索使用的模型。

可选 split 为 LoCoMo `locomo10`、BEAM `100k / 500k / 1m / 10m`、LongMemEval `s`。Compose 镜像
内置 LoCoMo，首次运行其他数据集时会下载数据。直接运行时可通过 `LOCOMO_DATA_PATH`、
`LONGMEMEVAL_DATA_PATH` 指定本地数据。BEAM 按 split 使用 `BEAM_100K_DATA_PATH`、
`BEAM_500K_DATA_PATH`、`BEAM_1M_DATA_PATH` 或 `BEAM_10M_DATA_PATH`。如果评测台继续使用
`BEAM_DATA_PATH`，需要同时用 `BEAM_DATA_SPLIT` 指明文件所属的标准 split。没有指定本地文件的 split
会使用对应的官方数据。比较不同后端时，请使用相同的 split、题数和回答及评分模型。

BEAM 有答案的题目如果没有召回内容，以 0 分计入平均分。数据集标注的拒答题（`abstention`）即使
没有召回内容，也会按评分标准检查模型是否说明信息不足。评分服务失败时，运行标为「评分未完成」并显示失败题数。
导出的 JSON 保留逐题结果，失败题标记为 `scoring_failed`，总分 `accuracy` 为 `null`。

评测台供本机单用户使用，一次运行一个评测。每次运行都有独立的 Hindsight Bank 名称或 Mem0 用户 ID；
ContextBase 使用预授权的测试 Bank，并以运行和题目所属的隔离单元标签分开数据。停止评测会终止本地
评测进程，之前已提交给后端的写入可能仍在处理。历史、日志和结果均保存在
`amb-results/workbench/`（Compose）或 `outputs/workbench/`（直接运行）。

### LoCoMo on ContextBase

Use a dedicated empty test Bank and a ContextBase business API key with read/write
access to that Bank. Configure the following values in `.env` or your environment:

```dotenv
CONTEXTBASE_API_URL=https://contextbase.example.com
CONTEXTBASE_API_TOKEN=<business-api-key>
CONTEXTBASE_BANK_ID=<test-bank-id>
GEMINI_API_KEY=<amb-model-key>
OMB_ANSWER_LLM=gemini
OMB_ANSWER_MODEL=gemini-2.5-flash-lite
OMB_JUDGE_LLM=gemini
OMB_JUDGE_MODEL=gemini-2.5-flash-lite
AMB_RECALL_TYPES=world,experience
```

Run five questions from one conversation, with all of that conversation's sessions ingested:

```bash
uv run amb run --dataset locomo --split locomo10 --memory contextbase --mode rag \
  --category conv-26 --query-limit 5 --name contextbase-locomo-smoke
```

The Docker equivalent writes results to `amb-results/` on the host:

```bash
docker compose run --rm --build amb run --dataset locomo --split locomo10 \
  --memory contextbase --mode rag --category conv-26 --query-limit 5 \
  --name contextbase-locomo-smoke
```

For a full run, omit `--category` and `--query-limit` and choose a new `--name`.
For one physical Bank per conversation, set `CONTEXTBASE_BENCH_BANKS` to a JSON
object mapping every selected conversation ID to a distinct pre-authorized Bank ID
and leave `CONTEXTBASE_BANK_ID` unset. With one Bank, Recall uses strict tags for
both the conversation and the run. Use separate empty Banks for cold-ingest or
latency comparisons against `hindsight-http`, which uses one Bank per conversation.

The adapter submits asynchronous Retain batches and waits for each operation before
asking questions. `CONTEXTBASE_BENCH_RETAIN_TIMEOUT_S` defaults to 1800 seconds per
ingest call. The adapter leaves test documents in their Banks for inspection. To
reuse an ingested run, set `CONTEXTBASE_BENCH_RUN_ID` to the printed run ID or the
result's `ingestion_run_id`. Pass `--skip-ingestion` with the same endpoint, Banks,
dataset selection, run name, and output directory. AMB checks its completed ingest
records in `_store` and verifies the documents still exist before scoring. Runs
from before this check need one fresh ingest to create the records. Fresh runs
get a new ingestion ID. Use empty Banks for a cold-ingest measurement.

For a Hindsight comparison, keep the dataset, question IDs, answer and judge models,
Recall token budgets, memory types, observation configuration, and concurrency the
same on both sides. Set `LOCOMO_DATA_PATH` to a local copy of `locomo10.json` to pin
the input file.

## Results

Results are saved to `outputs/{dataset}/{memory}/{mode}/{domain}.json` and can be explored with `uv run amb view`.

## Requirements

- Python ≥ 3.11
- `GEMINI_API_KEY` in `.env` or environment
- For MemBench: set `MEMBENCH_DATA_PATH` to your local data directory
