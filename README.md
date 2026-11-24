# AMB — Agent Memory Benchmark

We built AMB because we wanted to be honest about how Hindsight performs — and because no existing benchmark gave us the full picture. AMB is fully open: datasets, prompts, scoring logic, and results.

Live leaderboard: **[agentmemorybenchmark.ai](https://agentmemorybenchmark.ai)**

## Docker quick start · 一键运行 LoCoMo

在已克隆的仓库中使用 Docker Compose 2.24 或更新版本。镜像内置评测台的全部数据集和分词表。首次构建需安装依赖，运行时只需能访问配置的评测后端和模型服务。

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

主分支镜像工作流成功后，亦可在仓库的 Packages 页面获取 `ghcr.io/theludlows/agent-memory-benchmark:latest` 或对应的 `sha-<commit>` 标签。镜像内 LoCoMo 的来源是 [snap-research/locomo](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)，授权文件位于 `data/offline/locomo/LICENSE.txt`。

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

**Knowledge** datasets (ViDoRe V3 as markdown or PDFs, MIRACL-Vision, video, BrowseComp-Plus, ATM-Bench, LegalBench-RAG mini) are search only too: a corpus goes into a knowledge base and each query's ranking is scored with the leaderboard's own metric, nDCG@10, Recall@10 or character Recall@5 (the dataset's `score_type`, which the viewer shows instead of "accuracy"). Corpora are cached in `~/.cache/hindsight/knowledge-evals`; `--query-limit` cuts queries, never the corpus.

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

评测台支持 LoCoMo、BEAM、LongMemEval、LifeBench 1.0 英文版、LifeBench 2.0 中文版、PersonaMem v1 和 PrecisionMemBench，可搭配 ContextBase、Mem0 Cloud 或 Hindsight HTTP。
每轮填写评测名称，选择数据集、split 和后端，默认跑 5 题，也可跑全量。题数上限只限制问答，
LoCoMo、BEAM 和 LongMemEval 仍会写入所选题目对应的会话历史。PrecisionMemBench 始终写入完整测试数据。运行详情会显示当前阶段、已用时、已写入的文档数和已处理的题数。
写入进度每组文档处理完后更新，答题进度逐题更新。结束后可查看得分、写入耗时、平均检索延迟，
或下载 AMB 逐题结果 JSON。完成的评测会保存得分、题数、写入耗时、文档数、平均检索延迟和检索 P95。
在「运行历史」勾选 2 至 4 轮，即可用图表对比得分与耗时。页面会提示题目、写入资料或回答、评分模型不一致的情况。
新评测的写入实际耗时只统计至少有一个写入任务在执行的时间，并行区间计一次。旧记录保留原累计耗时口径，
在另一个图表中展示。

1. 如需通过环境变量预设配置，复制 `.env.example` 为 `.env`。模型 Key 也可在启动后通过页面填写。
2. 启动评测台并打开 http://127.0.0.1:8090：

   ```sh
   docker compose up --build amb-web
   ```

3. 在「本轮连接」选择已有地址，或填写新地址和 Key。发起评测时按后端和地址自动保存连接，下次默认选中
   该后端最近使用的地址。选择其他地址会带出对应的 Bank ID 和 Key 设置，填写新 Key 可更新该地址的 Key。
   每轮的历史记录包含名称、数据集、后端和实际使用的地址。Key 保存在服务端的连接配置中，评测历史和结果不含 Key。

   | 后端 | 页面配置 |
   |---|---|
   | ContextBase | API 地址、可读写的业务 API Key、已授权的测试 Bank ID |
   | Mem0 Cloud | API 地址（默认 `https://api.mem0.ai`）和 API Key，替换地址时需兼容 Mem0 Cloud API |
   | Hindsight HTTP | API 地址，开启鉴权时填写 API Key |

也可以通过 `.env` 提供初始连接。ContextBase 使用
`CONTEXTBASE_API_URL`、`CONTEXTBASE_API_TOKEN`、`CONTEXTBASE_BANK_ID`；Mem0 使用 `MEM0_CLOUD_URL`、
`MEM0_CLOUD_KEY`（或 `MEM0_API_KEY`）；Hindsight 使用 `HINDSIGHT_HTTP_URL`、`HINDSIGHT_HTTP_KEY`。
连接配置保存在结果目录之外，
直接运行时使用 `~/.config/amb/workbench/`（可用 `AMB_WORKBENCH_CONFIG_DIR` 指定私有配置目录），
Docker Compose 使用独立的 `amb-workbench-config` 卷。升级后，旧的 `.connections.json` 会自动迁移并从
结果目录移除。选择「输入新地址」后，可填写适用于该地址的 Key。Hindsight 无需鉴权时可取消使用该地址的 Key。

也可以在仓库根目录直接运行 `uv run amb workbench`。本地直跑时，同机 Hindsight 的地址通常是
`http://127.0.0.1:8888`。Docker 中连接宿主机用 `http://host.docker.internal:8888`，并确保 API
监听容器可访问的地址。

展开「全局模型设置」，分别配置回答模型和评分模型。回答模型根据检索结果生成答案，评分模型根据参考答案或评分规则打分。选择 Gemini、OpenAI 兼容或 Groq，填写模型 ID 和 Key。OpenAI 兼容服务需填写 Chat Completions 的 API 地址及前缀（通常是 `/v1`）。AMB 优先通过 API 请求 JSON Schema 输出，网关忽略或拒绝这一参数时，会改用提示词要求模型返回符合字段定义的 JSON。

点击「测试回答模型」或「测试评分模型」，评测台后台会用当前填写的配置处理一道固定样题并检查结果。可以在保存前测试。留空 Key 时，后台按同样的规则复用已有 Key。测试默认沿用后台的代理设置。可勾选「测试时跳过代理」让这次测试直连模型地址。测试最多等待 20 秒，结果显示在该模型设置下方。

每轮问答评测在启动时读取全局回答模型和评分模型的配置，运行详情记录当时实际使用的模型。修改设置后，新启动的评测使用新配置。编辑全局模型时，留空 Key 可复用同一提供方和 API 地址已配置的 Key。全局配置在重启后保留，并与默认连接一起存放在私有配置目录。

没有保存网页默认配置时，系统使用 `.env` 中的 `OMB_ANSWER_LLM`、`OMB_ANSWER_MODEL`、`OMB_JUDGE_LLM`、`OMB_JUDGE_MODEL` 和相应提供方的 Key。页面可随时恢复这些默认值。ContextBase 和 Hindsight 在各自服务内配置写入和检索使用的模型。

可选 split 为 LoCoMo `locomo10`、BEAM `100k / 500k / 1m / 10m`、LongMemEval `s`、
LifeBench 1.0 `en`（2,003 题）、LifeBench 2.0 `zh`（3,380 题）、PersonaMem v1 `32k / 128k / 1M`（589 / 2,727 / 2,674 题）、PrecisionMemBench `single-turn`（77 个用例）。以上数据和分词表都随仓库及
镜像提供。使用完整仓库并提前安装依赖，或提前导入构建好的镜像，运行时从本地读取数据和词表，只需能访问配置的记忆后端和模型端点。
本地文件缺失时，默认直接报告缺失项。在可联网的环境中准备缓存时，可以设置 `AMB_ALLOW_DATASET_DOWNLOADS=1`，让 AMB 下载缺失的上游文件。
仍可通过 `LOCOMO_DATA_PATH`、
`LONGMEMEVAL_DATA_PATH` 指定本地数据。BEAM 按 split 使用 `BEAM_100K_DATA_PATH`、
`BEAM_500K_DATA_PATH`、`BEAM_1M_DATA_PATH` 或 `BEAM_10M_DATA_PATH`。如果评测台继续使用
`BEAM_DATA_PATH`，需要同时用 `BEAM_DATA_SPLIT` 指明文件所属的标准 split。没有指定本地文件的 split
会按上述顺序加载数据。比较不同后端时，请使用相同的 split、题数和回答及评分模型。

- **LifeBench 1.0 英文版**（`lifebench/en`）沿用 AMB 的评测方式。先写入所选用户的完整历史，再回答该用户的问题。全量包含 10 位用户的 3,605 个非空日会话。结果按信息提取、多跳推理、非陈述性记忆、时间与知识更新、不可回答五类统计。不可回答题即使召回为空，也会将模型回答交给评分模型。可用 `LIFEBENCH_DATA_PATH` 指向官方英文文件 `our_en.json`。
- **LifeBench 2.0 中文版**使用官方原始中文数据。每个用户的档案、日常事件和九类手机记录按日期和来源写入，
  AMB 用中文回答，并根据参考答案和评分要点判定正误。可用 `LIFEBENCH_V2_DATA_PATH` 指向官方中文 `data/`
  目录，该目录直接包含各用户文件夹。评测按用户和提问日期推进，每批只补写截至该日期的记录，
  并等待写入完成才作答。CLI 运行需要支持增量写入的后端，每轮从新的时间线开始。
- **PersonaMem v1** 是英文四选一评测。评测台按原始题目的对话截止位置增量写入记录，使用全局回答模型作答并按答案字母判分。三档题目和会话随 Workbench 镜像提供。
- **PrecisionMemBench**直接用返回的记忆比对预期 ID，评测台无需配置回答或评分模型。
  结果展示总通过率、实际检索用例通过数、平均检索精确率和召回率。可用
  `PRECISIONMEMBENCH_DATA_PATH` 指向含 `beliefs.seed.json` 和 `retrieval.cases.json` 的目录。
  检索报错或无法映射来源的用例会使总分标记为未完成。Hindsight HTTP 使用原有的共享 Bank、用户标签和召回过滤，
  服务端需支持对应的 Bank 配置和标签筛选。

BEAM 有答案的题目如果没有召回内容，以 0 分计入平均分。数据集标注的拒答题（`abstention`）即使
没有召回内容，也会按评分标准检查模型是否说明信息不足。评分服务失败时，运行标为「评分未完成」并显示失败题数。
导出的 JSON 保留逐题结果，失败题标记为 `scoring_failed`，总分 `accuracy` 为 `null`。

同一个评测台默认最多同时运行两轮评测，在「运行历史」中切换查看。达到上限时，等待任一轮结束后即可发起新评测。
可在 `.env` 中设置 `AMB_WORKBENCH_MAX_CONCURRENT_RUNS`（正整数），直接用 Docker 启动时可通过 `-e` 设置。内存有限且需要运行 BEAM 10m 时，可将上限设为 `1`。
每轮使用独立的 Hindsight Bank 名称或 Mem0 用户 ID。
ContextBase 使用预授权的测试 Bank，并以运行和题目所属的隔离单元标签分开数据。点击「停止评测」会终止当前选中的本地
评测进程，之前已提交给后端的写入可能仍在处理。并行运行会共享后端资源和模型 API 配额。历史、日志和结果均保存在
`amb-results/workbench/`（Compose）或 `outputs/workbench/`（直接运行）。


#### 知识库检索评测

在顶部选择「知识库检索」进入 `/knowledge`。这页独立保存连接和知识库评测历史，记忆评测仍从首页进入。每轮导入完整语料，等待写入完成后执行检索。AMB 无需配置回答模型或评分模型。

Hindsight Knowledge 和 WeKnora 每轮创建一个评测知识库。**ContextBase** 支持 ViDoRe V3 Markdown 和 BrowseComp-Plus。选择其中一个数据集后，填写 ContextBase API 地址、业务 API Key 和已授权读写的测试 Bank ID。AMB 在该 Bank 内用独立标签隔离每轮语料，通过 Retain 写入，并按 Recall 返回的来源文档 ID 评分。评测数据会保留在 Bank 中。结果反映该 Bank 的 Retain 配置和 Recall 排序。要评测原文分块检索，可由管理员把测试 Bank 的提取模式设为 `chunks`。

选择 WeKnora 后，填写服务地址、API Key 和该 WeKnora 服务中的 Embedding Model ID。使用平台级 API Key 时填写目标工作空间的数字 ID，AMB 会发送 `X-Tenant-ID`。使用空间级 API Key 时可留空。MIRACL-Vision 需要支持图像输入的 Embedding 模型和支持图像识别的 VLM Model ID。ViDoRe PDF 可选填 VLM Model ID。API Key 需要创建知识库、写入文档和检索的权限（`manage_kbs`、`ingest`、`retrieve`），并能访问本轮创建的知识库。ViDoRe PDF 需要 WeKnora 检索接口返回 PDF 页码来源。

| 数据集 | CLI 数据集名 | 分组示例 | 主指标 |
| --- | --- | --- | --- |
| ViDoRe V3 Markdown | `vidore-v3-markdown` | `hr` | nDCG@10 |
| ViDoRe V3 PDF | `vidore-v3-pdf` | `hr` | nDCG@10 |
| BrowseComp-Plus | `browsecomp-plus` | `evidence` 或 `gold` | nDCG@10 |
| LegalBench-RAG mini | `legalbench-rag` | `mini` | 字符级 Recall@5 |
| MIRACL-Vision | `miracl-vision` | `zh` | nDCG@10 |

ViDoRe V3 Markdown 的 `computer_science` 分组随仓库和 Workbench 镜像提供。选中该分组即可离线加载语料和题目。[数据来源与许可](data/offline/vidore-v3-markdown/README.md)

其他分组首次使用前，在联网环境准备语料。例如：

```bash
uv run --group workbench amb prepare-knowledge --dataset legalbench-rag --split mini
```

Docker Compose 使用同一个语料卷准备和运行：

```bash
docker compose --profile web run --rm amb-web prepare-knowledge --dataset legalbench-rag --split mini
docker compose --profile web up -d amb-web
```

语料默认保存在 `~/.cache/hindsight/knowledge-evals`，可通过 `AMB_KNOWLEDGE_CACHE` 指定目录。Workbench 镜像默认使用 `/app/.datasets/knowledge`。断网部署时，复制准备好的缓存并将其挂载到此目录。评测只读取随仓库提供的语料或本地缓存，仍需能连接所选后端，并由后端配置适用于文本、PDF 或图像的模型。Hindsight 服务须启用 Knowledge Bank。首次准备可能需要下载数 GB 数据。修改题数只影响查询，ViDoRe 按语言、LegalBench 按合同类别应用上限。

也可通过 CLI 运行 ContextBase 检索评测。先配置 `CONTEXTBASE_API_URL`、`CONTEXTBASE_API_TOKEN` 和 `CONTEXTBASE_BANK_ID`，再运行：

```bash
uv run --group workbench amb run --dataset vidore-v3-markdown --split computer_science --memory contextbase-knowledge --mode retrieval --query-limit 5
```

也可通过 CLI 运行 WeKnora 检索评测。先设置 `WEKNORA_API_URL`、`WEKNORA_API_KEY`、`WEKNORA_EMBEDDING_MODEL_ID`。使用平台级 API Key 时设置 `WEKNORA_TENANT_ID`，使用图像语料时设置 `WEKNORA_VLM_MODEL_ID`。

```bash
uv run --group workbench amb run --dataset legalbench-rag --split mini \
  --memory weknora --mode retrieval --query-limit 5
```

#### 在华为云 ECS 上启动 Workbench

[AMB Workbench image](https://github.com/TheLudlows/agent-memory-benchmark/actions/workflows/workbench-image.yml)
在通过断网测试后，将 Linux x86_64 服务镜像发布到
`ghcr.io/theludlows/agent-memory-benchmark:workbench`。
需要固定版本时，将标签换成 `workbench-sha-<完整提交 SHA>`。

1. 创建能访问 GHCR 的 x86_64 ECS，选择带 Cloud-init 的 Ubuntu 镜像。在「实例自定义数据」中粘贴
   [启动脚本](deploy/ecs-user-data.sh)。首次启动时，脚本安装 Docker（如需要）、启动 Workbench，并持久化评测结果和配置。
2. 从本地建立 SSH 隧道，然后打开 [Workbench](http://127.0.0.1:8090)。请按所选系统镜像替换登录用户名。

   ```sh
   ssh -N -L 8090:127.0.0.1:8090 ubuntu@ECS_IP
   ```

已安装 Docker 的断网主机，可以从上述工作流下载 `amb-workbench-linux-amd64`，将解压后的文件复制到主机，再执行以下命令。

```sh
sha256sum -c amb-workbench-linux-amd64.tar.gz.sha256
sudo docker load -i amb-workbench-linux-amd64.tar.gz
sudo AMB_IMAGE=amb-workbench:local AMB_PULL=never bash ecs-user-data.sh
```

#### 通过宿主机 IP 访问评测台

在受信任的网络中，可以用 Docker 将评测台发布到宿主机的 `0.0.0.0:8090`。
将示例 IP `192.0.2.10` 换成浏览器实际使用的宿主机 IP，然后打开 `http://宿主机IP:8090`。

```sh
docker run -d --name amb-workbench --init --restart unless-stopped \
  -p 0.0.0.0:8090:8090 \
  -e AMB_WORKBENCH_ALLOWED_HOSTS=192.0.2.10 \
  --add-host=host.docker.internal:host-gateway \
  -v amb-workbench-results:/app/outputs \
  -v amb-workbench-config:/app/workbench-config \
  amb-workbench:local
```

使用 Compose 时，在 `.env` 中设置 `AMB_WORKBENCH_BIND=0.0.0.0` 和 `AMB_WORKBENCH_ALLOWED_HOSTS=192.0.2.10`。
ECS 启动脚本也接受这两个环境变量。允许的 Host 可以是 IP 或域名，多个值用逗号分隔，填写时省略协议和端口。
评测台没有登录鉴权。使用宿主机 IP 直连时，建议通过 ECS 安全组或防火墙限制可访问 8090 的来源。

启动后，在页面填写容器可访问的记忆服务地址，并按评测需要配置回答和评分模型。无公网时，可以使用同机或内网的 API。

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
