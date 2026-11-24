const $ = (id) => document.getElementById(id);
const knowledge = location.pathname === "/knowledge";
const apiBase = knowledge ? "/knowledge/api" : "/api";
const knowledgeMetricLabels = {
  "ndcg@10": "nDCG@10", "recall@10": "Recall@10", "recall@5": "字符级 Recall@5",
  "precision@5": "字符级 Precision@5",
};
const names = {
  locomo: "LoCoMo", beam: "BEAM", longmemeval: "LongMemEval", personamem: "PersonaMem v1",
  lifebench: "LifeBench 1.0 英文版", "lifebench-v2": "LifeBench 2.0 中文版",
  precisionmembench: "PrecisionMemBench",
  "vidore-v3-markdown": "ViDoRe V3 · Markdown",
  "vidore-v3-pdf": "ViDoRe V3 · PDF",
  "browsecomp-plus": "BrowseComp-Plus",
  "legalbench-rag": "LegalBench-RAG mini",
  "miracl-vision": "MIRACL-Vision",
};
const statuses = { running: "运行中", cancelling: "停止中", completed: "已完成", scoring_incomplete: "评分未完成", failed: "失败", cancelled: "已停止", interrupted: "已中断" };
const retrievalOnly = (dataset) => settings?.retrieval_datasets?.includes(dataset);
const choiceOnly = (dataset) => settings?.choice_datasets?.includes(dataset);
const modelLabels = (job) => {
  const dataset = job.selection.dataset;
  if (retrievalOnly(dataset)) return "纯检索评测";
  const answer = `回答 ${storedModelDetail(job, "answer")}`;
  return choiceOnly(dataset) ? answer : `${answer} · 评分 ${storedModelDetail(job, "judge")}`;
};
const stages = {
  loading: "加载数据集", ingesting: "写入记忆", evaluating: "检索、回答与评分",
  ingesting_and_evaluating: "同时写入和答题", finalizing: "整理评测结果",
};
const isActive = (job) => job && ["running", "cancelling"].includes(job.status);
const formatDate = (value) => new Date(value).toLocaleString("zh-CN", { hour12: false });

const pageSize = 12;
const contextTokenPageSize = 20;
const compareLimit = 4;
const compareIds = new Set();
let comparisonVersion = 0;
let settings;
let selected;
let jobs = [];
let historyOffset = 0;
let hasNextPage = false;
let runCapacity;
let refreshVersion = 0;
let submitting = false;
let questionCountVersion = 0;
let contextTokenRunId;
let contextTokenOffset = 0;
let contextTokenRequest = 0;
let questionTypesRunId;
let questionTypesRequest = 0;

const questionTypeLabels = new Map(Object.entries({
  "single-hop": "单跳", "multi-hop": "多跳", temporal: "时间推理", "open-domain": "开放域",
  abstention: "拒答", contradiction_resolution: "矛盾处理", event_ordering: "事件排序",
  information_extraction: "信息提取", instruction_following: "指令遵循", knowledge_update: "知识更新",
  multi_session_reasoning: "跨会话推理", preference_following: "偏好遵循", summarization: "摘要",
  temporal_reasoning: "时间推理", "single-session-user": "单会话（用户）",
  "single-session-assistant": "单会话（助手）", "single-session-preference": "单会话（偏好）",
  "information-extraction": "信息提取", "temporal-updating": "时间推理与知识更新",
  nondeclarative: "非陈述性记忆", unanswerable: "不可回答",
  "multi-session": "跨会话", "knowledge-update": "知识更新", "temporal-reasoning": "时间推理",
  recall_user_shared_facts: "回忆用户分享的事实", recalling_facts_mentioned_by_the_user: "回忆用户提到的细节",
  generalizing_to_new_scenarios: "迁移到新场景", generalize_to_new_scenarios: "迁移到新场景",
  provide_preference_aligned_recommendations: "按偏好推荐", suggest_new_ideas: "提出新点子",
  acknowledge_latest_user_preferences: "识别最新偏好", track_full_preference_evolution: "追踪偏好演变",
  track_full_preference_updates: "追踪偏好演变", revisit_reasons_behind_preference_updates: "回忆偏好变化的原因",
  recalling_the_reasons_behind_previous_updates: "回忆偏好变化的原因",
}));

const modelRoles = { answer: "回答模型", judge: "评分模型" };
const providerLabels = { gemini: "Gemini", openai: "OpenAI 兼容", groq: "Groq" };
const modelEditors = {};
const savingModels = new Set();

async function api(path, options) {
  const response = await fetch(`${apiBase}${path}`, options);
  const data = await response.json();
  if (!response.ok) {
    const message = typeof data.detail === "string" ? data.detail : "请求失败，请检查评测配置。";
    throw new Error(message);
  }
  return data;
}

function setError(id, error) {
  $(id).textContent = error || "";
  $(id).hidden = !error;
}

function options(id, values) {
  $(id).replaceChildren(...values.map(([value, name]) => new Option(name, value)));
}

function updateSplits() {
  options("split", settings.datasets[$("dataset").value].map((item) => [item, item]));
  const backend = $("backend").value;
  const choices = Object.entries(settings.backends).filter(([key]) => {
    const datasets = settings.backend_datasets?.[key];
    return !datasets || datasets.includes($("dataset").value);
  });
  options("backend", choices.map(([key, value]) => [key, value.label]));
  if (choices.some(([key]) => key === backend)) {
    $("backend").value = backend;
    updateForm();
  } else {
    loadConnection();
  }
  updateQuestionCount();
}

async function updateQuestionCount() {
  const version = ++questionCountVersion;
  const full = $("full").checked;
  const count = $("full-count");
  $("limit").disabled = full;
  $("limit").hidden = full;
  count.hidden = !full;
  if (!full || !settings) return;

  const dataset = $("dataset").value;
  const split = $("split").value;
  if (knowledge) {
    count.textContent = "全部查询";
    return;
  }
  count.textContent = "正在读取题数…";
  try {
    const { query_count: total } = await api(`/datasets/${dataset}/${split}/query-count`);
    if (version === questionCountVersion) count.textContent = `${total.toLocaleString("zh-CN")} 题`;
  } catch {
    if (version === questionCountVersion) count.textContent = "无法读取题数";
  }
}

function loadConnection() {
  const backend = $("backend").value;
  const { connections } = settings.backends[backend];
  const addresses = connections.map(({ api_url }) => [api_url, api_url]);
  options("connection-address", [...addresses, ["", "输入新地址"]]);
  $("connection-panel").hidden = false;
  $("connection-address-row").hidden = addresses.length === 0;
  $("connection-bank-row").hidden = !backend.startsWith("contextbase");
  $("connection-weknora-models").hidden = backend !== "weknora";
  selectConnection();
}

function selectedConnection() {
  const { connections } = settings.backends[$("backend").value];
  return connections.find(({ api_url }) => api_url === $("connection-address").value);
}

function selectConnection() {
  const connection = selectedConnection();
  $("connection-url-row").hidden = Boolean(connection);
  $("connection-url").value = "";
  $("connection-bank").placeholder = connection?.has_bank_mapping ? "使用已有 Bank 映射" : "已授权的测试 Bank";
  $("connection-bank").value = connection?.bank_id || "";
  $("connection-tenant").value = connection?.tenant_id || "";
  $("connection-embedding-model").value = connection?.embedding_model_id || "";
  $("connection-vlm-model").value = connection?.vlm_model_id || "";
  $("connection-key").value = "";
  $("connection-reuse").checked = Boolean(connection?.has_key);
  updateForm();
}

function connectionInput() {
  const backend = $("backend").value;
  const connection = {
    api_url: $("connection-address").value || $("connection-url").value.trim(),
    bank_id: backend.startsWith("contextbase") ? $("connection-bank").value.trim() : "",
    api_key: $("connection-key").value.trim() || null,
  };
  if (backend === "weknora") {
    connection.tenant_id = $("connection-tenant").value.trim();
    connection.embedding_model_id = $("connection-embedding-model").value.trim();
    connection.vlm_model_id = $("connection-vlm-model").value.trim();
  }
  return connection;
}

function defaultModel(role, provider) {
  return role === "judge" && provider === "gemini" && $("dataset").value === "beam"
    ? settings.beam_default_judge_model : settings.model_services[provider]?.default_model || "";
}

function loadModelEditor(editor, provider = settings.models[editor.role].provider) {
  const config = settings.models[editor.role];
  const service = settings.model_services[provider];
  editor.provider.value = provider;
  editor.model.value = (provider === config.provider && config.model) || service?.default_model || "";
  editor.url.value = (provider === config.provider && config.api_url) || service?.api_url || "";
  editor.key.value = "";
  setError(editor.message.id, "");
}

function modelInput(editor) {
  return {
    provider: editor.provider.value,
    model: editor.model.value.trim(),
    api_url: editor.provider.value === "openai" ? editor.url.value.trim() : "",
    api_key: editor.key.value.trim() || null,
  };
}

function modelHasKey(role, input) {
  const config = settings.models[role];
  const service = settings.model_services[input.provider];
  const url = input.api_url.replace(/\/+$/, "");
  return Boolean(input.api_key ||
    (input.provider === config.provider && url === config.api_url && config.has_key) ||
    (url === service?.api_url && service.has_key));
}

function updateModelEditor(editor) {
  const input = modelInput(editor);
  const busy = savingModels.has(editor.role) || submitting || editor.testing;
  editor.urlRow.hidden = input.provider !== "openai";
  for (const field of [editor.provider, editor.model, editor.key]) field.disabled = busy;
  editor.url.disabled = busy || editor.urlRow.hidden;
  editor.bypassProxy.disabled = busy;
  const hasKey = modelHasKey(editor.role, { ...input, api_key: null });
  editor.key.placeholder = hasKey ? "已配置，留空继续使用" : "填写模型 API Key";
  editor.reset.hidden = !settings.models[editor.role].saved;
  editor.reset.disabled = busy;
  const invalid = Boolean(modelInputError(editor, input));
  editor.save.disabled = busy || invalid;
  editor.test.disabled = busy || invalid;
  editor.test.textContent = editor.testing ? "测试中…" : `测试${modelRoles[editor.role]}`;

  const config = settings.models[editor.role];
  const model = config.model || defaultModel(editor.role, config.provider);
  const label = providerLabels[config.provider];
  editor.summary.textContent = label && model
    ? `全局默认：${label} · ${model}${config.has_key ? "" : "（缺少 Key）"}`
    : "请选择提供方并保存默认模型。";
}

function modelInputError(editor, input) {
  const label = modelRoles[editor.role];
  if (!input.provider || !input.model) return `请填写${label}的提供方和模型 ID。`;
  if (input.provider === "openai" && (!input.api_url || !editor.url.validity.valid)) {
    return `请填写${label}的 API 地址。`;
  }
  return modelHasKey(editor.role, input) ? "" : `请填写${label}的 API Key。`;
}

async function saveModel(editor, reset = false) {
  const { role } = editor;
  const input = modelInput(editor);
  if (savingModels.has(role) || (!reset && modelInputError(editor, input))) return;
  savingModels.add(role);
  updateForm();
  try {
    const request = reset ? {} : { headers: {"Content-Type": "application/json"}, body: JSON.stringify(input) };
    settings.models[role] = await api(`/models/${role}${reset ? "/reset" : ""}`, { method: "POST", ...request });
    loadModelEditor(editor);
    setError(editor.message.id, reset ? "已恢复 .env 默认配置。" : "已保存全局默认模型。");
  } catch (error) {
    setError(editor.message.id, error.message);
  } finally {
    savingModels.delete(role);
    updateForm();
  }
}

async function testModel(editor) {
  const input = modelInput(editor);
  if (editor.test.disabled || modelInputError(editor, input)) return;
  const label = modelRoles[editor.role];
  const bypassProxy = editor.bypassProxy.checked;
  editor.testing = true;
  setError(editor.message.id, `正在由后台测试${label}…`);
  updateForm();
  try {
    const result = await api(`/models/${editor.role}/test`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ ...input, bypass_proxy: bypassProxy }),
    });
    const connection = bypassProxy ? " · 直连" : "";
    setError(editor.message.id, `${label}测试通过 · ${result.model} · ${(result.elapsed_ms / 1000).toFixed(1)} 秒${connection}`);
  } catch (error) {
    setError(editor.message.id, `${label}测试失败。${error.message}`);
  } finally {
    editor.testing = false;
    updateForm();
  }
}

function createModelEditor(role) {
  const template = $("model-editor").content.cloneNode(true);
  const editor = { role, testing: false };
  for (const part of template.querySelectorAll("[data-part]")) {
    const name = part.dataset.part;
    part.id = `global-${role}-${name}`;
    const property = name.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    editor[property] = part;
  }
  for (const label of template.querySelectorAll("[data-for]")) label.htmlFor = editor[label.dataset.for].id;
  editor.title.textContent = modelRoles[role];
  const providers = Object.keys(settings.model_services).map((provider) => new Option(providerLabels[provider], provider));
  editor.provider.replaceChildren(...providers);
  $("global-models").append(template);
  loadModelEditor(editor);
  editor.provider.addEventListener("change", () => {
    loadModelEditor(editor, editor.provider.value);
    updateForm();
  });
  for (const input of [editor.model, editor.url, editor.key]) {
    input.addEventListener("input", () => {
      setError(editor.message.id, "");
      updateForm();
    });
  }
  editor.bypassProxy.addEventListener("change", () => setError(editor.message.id, ""));
  editor.test.addEventListener("click", () => testModel(editor));
  editor.save.addEventListener("click", () => saveModel(editor));
  editor.reset.addEventListener("click", () => saveModel(editor, true));
  modelEditors[role] = editor;
}

function updateForm() {
  if (!settings) return;
  const backend = $("backend").value;
  const config = settings.backends[backend];
  const connection = connectionInput();
  const saved = selectedConnection();
  const descriptions = {
    "contextbase": "使用预先授权的测试 Bank。",
    "contextbase-knowledge": "使用有读写权限的测试 Bank，按原始文档来源评分。",
    "mem0-cloud": "使用 Mem0 Cloud 兼容 API。",
    "hindsight-http": "连接已启动的 Hindsight API。",
    "hindsight-knowledge": "连接已启用知识库的 Hindsight API。每轮创建一个评测知识库。",
    "weknora": "连接 WeKnora API。每轮创建一个评测知识库。",
  };
  $("backend-help").textContent = descriptions[backend];
  $("connection-reuse-row").hidden = !saved?.has_key;
  $("connection-key").placeholder = saved?.has_key ? "填写新 Key 可更新" : config.key_optional ? "选填" : "请输入 API Key";
  const usingSaved = saved?.has_key && $("connection-reuse").checked;
  const splitError = settings.split_errors?.[$("dataset").value]?.[$("split").value];
  for (const editor of Object.values(modelEditors)) updateModelEditor(editor);
  const dataset = $("dataset").value;
  const weknora = backend === "weknora";
  const requiresVlm = weknora && dataset === "miracl-vision";
  $("connection-embedding-model").required = weknora;
  $("connection-embedding-model").placeholder = requiresVlm ? "需支持图片向量化" : "";
  $("connection-vlm-model").required = requiresVlm;
  $("connection-vlm-model").placeholder = requiresVlm ? "MIRACL-Vision 必填" : "选填";
  const retrieval = retrievalOnly(dataset);
  const roles = choiceOnly(dataset) ? ["answer"] : Object.keys(modelRoles);
  const missing = retrieval ? [] : roles.filter((role) => !settings.models[role].has_key);
  const modelError = !retrieval && missing.length
    ? `请在「全局模型设置」配置${missing.map((role) => modelRoles[role]).join("和")}。` : "";
  $("dataset-coverage").textContent = knowledge
    ? (["vidore-v3-markdown", "vidore-v3-pdf", "legalbench-rag"].includes(dataset)
      ? "每种查询语言或合同类别分别应用题数上限，始终写入完整语料。"
      : "始终写入完整语料，题数上限只影响查询。")
    : retrieval ? "纯检索评分，写入完整测试数据。"
    : dataset === "lifebench" ? "按用户写入完整会话历史，再回答所选题目。"
    : choiceOnly(dataset) ? "按每题可见的对话范围增量写入，四选一按答案字母判分。"
    : dataset === "lifebench-v2" ? "按提问日期写入记录，之后的资料在该题作答时不可见。"
    : "限制题数时，仍会写入所选题目对应的会话历史。";
  const ready = $("evaluation-name").value.trim() && connection.api_url &&
    (config.key_optional || connection.api_key || usingSaved) &&
    (!backend.startsWith("contextbase") || connection.bank_id ||
      (backend === "contextbase" && saved?.has_bank_mapping)) &&
    (!weknora || (connection.embedding_model_id && (!requiresVlm || connection.vlm_model_id)));
  setError("config-error", splitError || modelError);
  for (const id of ["backend", "connection-address", "connection-bank", "connection-key", "connection-reuse"]) {
    $(id).disabled = submitting;
  }
  $("connection-tenant").disabled = submitting || !weknora;
  $("connection-embedding-model").disabled = submitting || !weknora;
  $("connection-vlm-model").disabled = submitting || !weknora;
  $("connection-url").disabled = submitting || Boolean(saved);
  const capacityFull = runCapacity && runCapacity.active >= runCapacity.limit;
  const capacityHint = $("run-capacity");
  capacityHint.hidden = !capacityFull;
  const capacityMessage = capacityFull
    ? `已有 ${runCapacity.active} / ${runCapacity.limit} 轮评测在运行或停止中。等一轮结束后可启动新评测。` : "";
  if (capacityMessage !== capacityHint.textContent) capacityHint.textContent = capacityMessage;
  $("start").disabled = submitting || savingModels.size > 0 || !runCapacity || capacityFull || !ready || Boolean(splitError || modelError);
  $("start").textContent = submitting ? "正在启动…" : capacityFull ? "并行名额已满" : "开始评测";
}

function runType(job) {
  const { dataset, split, backend } = job.selection;
  return `${names[dataset]} · ${split} · ${settings.backends[backend].label}`;
}

function title(job) {
  return job.name || runType(job);
}

const number = (value, unit = "") => value == null ? "—" : `${value.toFixed(1)}${unit}`;
const scoreText = (score) => score == null ? "—" : number(score * 100, "%");
const seconds = (ms) => ms == null ? "—" : number(ms / 1000, " s");
const tokenFormat = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 });
const knowledgeScore = (score) => score == null ? "—" : score.toFixed(4);
const tokensText = (value) => value == null ? "—" : tokenFormat.format(value);

function bars(container, rows, maximum, format) {
  container.replaceChildren(...rows.map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "chart-row";
    const name = document.createElement("span");
    name.textContent = label;
    const bar = document.createElement(value == null ? "span" : "meter");
    if (value != null) {
      bar.min = 0;
      bar.max = maximum;
      bar.value = value;
      bar.setAttribute("aria-label", label);
    }
    const text = document.createElement("span");
    text.textContent = format(value);
    row.append(name, bar, text);
    return row;
  }));
}

function storedModelLabel(job, role) {
  const saved = job.models?.[role];
  return job.summary?.[`${role}_llm`] || (saved?.model ? `${saved.provider}:${saved.model}` : null);
}

function storedModelDetail(job, role) {
  return storedModelLabel(job, role) || job.models?.[role]?.provider || "—";
}

function comparisonWarning(runs) {
  const differs = (values) => new Set(values.filter((value) => value != null && value !== "")).size > 1;
  const questions = runs.map((run) => run.summary.question_fingerprint);
  const documents = runs.map((run) => run.summary.document_fingerprint);
  const answerRuns = runs.filter((run) => !retrievalOnly(run.selection.dataset));
  const answerModels = answerRuns.map((run) => storedModelLabel(run, "answer"));
  const judgeRuns = answerRuns.filter((run) => !choiceOnly(run.selection.dataset));
  const judgeModels = judgeRuns.map((run) => storedModelLabel(run, "judge"));
  const endpoints = (group, role) => group.map((run) => run.models?.[role]?.api_url);
  const missingEndpointFor = (group, role) => group.some((run) =>
    storedModelLabel(run, role)?.startsWith("openai:") && !run.models?.[role]?.api_url);
  const selectionDiffers = ["dataset", "split"].some((key) => differs(runs.map((run) => run.selection[key])));
  const mismatches = [];
  if (selectionDiffers || differs(questions) || differs(runs.map((run) => run.summary.total_queries))) {
    mismatches.push("题目");
  }
  if (differs(documents)) mismatches.push("写入资料");
  if (differs(answerModels) || differs(endpoints(answerRuns, "answer"))) mismatches.push("回答模型");
  if (differs(judgeModels) || differs(endpoints(judgeRuns, "judge"))) mismatches.push("评分模型");
  const missingEndpoint = missingEndpointFor(answerRuns, "answer") || missingEndpointFor(judgeRuns, "judge");
  const fields = [questions, documents, answerModels, judgeModels];
  const missing = missingEndpoint || fields
    .some((values) => values.some((value) => !value));
  const unknown = "部分评测缺少题目、写入资料或模型信息，无法确认这些评测的口径。";
  if (mismatches.length) {
    return `${mismatches.join("、")}不同，分数不可直接比较。` + (missing ? unknown : "");
  }
  if (missing) return unknown;
  if (judgeRuns.length) return "题目、写入资料、回答模型和评分模型一致。";
  return answerRuns.length ? "题目、写入资料和回答模型一致。" : "题目和写入资料一致。";
}

const wallWriteTime = (summary) => summary.ingestion_wall_time_ms;
const legacyWriteTime = (summary) => wallWriteTime(summary) == null ? summary.ingestion_time_ms : null;

async function drawComparison() {
  const version = ++comparisonVersion;
  const ids = [...compareIds];
  $("comparison").hidden = ids.length === 0;
  $("compare-jump").hidden = ids.length < 2;
  $("compare-count").textContent = ids.length ? `已选 ${ids.length} / ${compareLimit}` : "";
  $("compare-meta").replaceChildren();
  $("compare-charts").replaceChildren();
  if (ids.length < 2) {
    $("compare-message").textContent = "请再选择一轮评测。";
    return;
  }
  $("compare-message").textContent = "正在读取指标…";
  try {
    const runs = await Promise.all(ids.map((id) => api(`/runs/${id}/metrics`)));
    if (version !== comparisonVersion) return;
    $("compare-message").textContent = comparisonWarning(runs);
    for (const [i, run] of runs.entries()) {
      const item = document.createElement("li");
      const count = run.summary.total_queries;
      item.textContent = `${i + 1}. ${title(run)} · ${runType(run)} · ${count ?? "—"} 题` +
        ` · ${modelLabels(run)}` +
        (run.summary.scoring_failed_queries ? ` · 评分失败 ${run.summary.scoring_failed_queries} 题` : "");
      $("compare-meta").append(item);
    }
    const metrics = [
      ["平均检索延迟", (s) => s.avg_retrieve_time_ms, (value) => number(value, " ms")],
      ["检索 P95", (s) => s.p95_retrieve_time_ms, (value) => number(value, " ms")],
    ];
    if (knowledge) {
      const types = new Set(runs.map((run) => run.summary.score_type));
      for (const type of types) {
        metrics.unshift([knowledgeMetricLabels[type], (s) => s.score_type === type ? s.score : null, knowledgeScore, 1]);
      }
    } else {
      const label = runs.every((run) => retrievalOnly(run.selection.dataset)) ? "用例通过率"
        : runs.every((run) => run.selection.dataset === "beam") ? "平均得分" : "得分 / 准确率";
      metrics.unshift([label, (s) => s.score, scoreText, 1]);
    }
    if (runs.some((run) => run.summary.avg_context_tokens != null)) {
      metrics.push(["平均召回上下文 Token", (s) => s.avg_context_tokens, tokensText]);
    }
    if (!knowledge && runs.some((run) => retrievalOnly(run.selection.dataset))) {
      metrics.push(
        ["实际检索用例通过率", (s) => s.active_total ? s.active_passes / s.active_total : null, scoreText, 1],
        ["平均检索精确率", (s) => s.mean_precision, scoreText, 1],
        ["平均检索召回率", (s) => s.mean_recall, scoreText, 1],
      );
    }
    if (runs.some((run) => wallWriteTime(run.summary) != null)) {
      metrics.push(["写入实际耗时", wallWriteTime, seconds]);
    }
    if (runs.some((run) => legacyWriteTime(run.summary) != null)) {
      metrics.push(["累计写入耗时（旧口径）", legacyWriteTime, seconds]);
    }
    for (const [label, metric, format, max] of metrics) {
      const figure = document.createElement("figure");
      figure.className = "chart";
      const caption = document.createElement("figcaption");
      caption.textContent = label;
      const content = document.createElement("div");
      const rows = runs.map((run, i) => [`${i + 1}. ${title(run)}`, metric(run.summary)]);
      bars(content, rows, max ?? Math.max(1, ...rows.map(([, value]) => value ?? 0)), format);
      figure.append(caption, content);
      $("compare-charts").append(figure);
    }
  } catch (error) {
    if (version === comparisonVersion) $("compare-message").textContent = error.message;
  }
}

function drawHistory() {
  $("history-empty").hidden = jobs.length > 0;
  $("history").replaceChildren(...jobs.map((job) => {
    const row = document.createElement("li");
    row.className = "history-entry";
    const compare = document.createElement("input");
    compare.type = "checkbox";
    compare.checked = compareIds.has(job.id);
    compare.disabled = !["completed", "scoring_incomplete"].includes(job.status) ||
      (!compare.checked && compareIds.size >= compareLimit);
    compare.setAttribute("aria-label", `对比 ${title(job)}`);
    compare.addEventListener("change", () => {
      if (compare.checked) compareIds.add(job.id);
      else compareIds.delete(job.id);
      drawHistory();
      drawComparison();
    });
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-row";
    if (job.id === selected) button.setAttribute("aria-current", "true");
    const text = document.createElement("span");
    text.textContent = title(job);
    const note = document.createElement("small");
    note.textContent = `${runType(job)} · ${formatDate(job.started_at)}`;
    text.append(note);
    const status = document.createElement("span");
    status.className = "status";
    status.dataset.status = job.status;
    status.textContent = statuses[job.status];
    button.append(text, status);
    button.addEventListener("click", () => {
      selected = job.id;
      drawHistory();
      loadDetail(job.id).catch((error) => setError("run-error", error.message));
    });
    row.append(compare, button);
    return row;
  }));
}

function renderProgress(row, bar, counter, completed, total, unit) {
  $(row).hidden = !(total > 0);
  if ($(row).hidden) return;
  const done = completed ?? 0;
  $(bar).max = total;
  $(bar).value = done;
  $(counter).textContent = `${done} / ${total} ${unit}（${Math.floor(done * 100 / total)}%）`;
}

function renderRunProgress(job) {
  const { stage, completed, total, ingested_docs, total_docs } = job.progress;
  const active = isActive(job);
  $("stage").textContent = job.status === "cancelling" ? statuses.cancelling :
    active ? (stage === "evaluating" && retrievalOnly(job.selection.dataset) ? "检索与判定"
      : knowledge && stage === "ingesting" ? "写入知识库" : stages[stage] || "准备评测") : statuses[job.status];
  $("loading-progress").hidden = !(active && (!stage || stage === "loading"));
  renderProgress("ingestion-progress-row", "ingestion-progress", "ingestion-progress-count",
    ingested_docs, total_docs, "份文档");
  renderProgress("questions-progress-row", "progress", "progress-count", completed, total, "题");

  const finish = active ? Date.now() : Date.parse(job.ended_at);
  const elapsed = Math.max(0, Math.floor((finish - Date.parse(job.started_at)) / 1000));
  const minutes = Math.floor(elapsed / 60);
  $("elapsed").textContent = Number.isFinite(elapsed)
    ? `已用时 ${minutes ? `${minutes} 分 ` : ""}${elapsed % 60} 秒` : "";
}

async function renderQuestionTypes(job) {
  const finished = job.status === "completed" || job.status === "scoring_incomplete";
  const runId = finished && job.result_available ? job.id : null;
  $("question-types").hidden = !runId;
  if (runId === questionTypesRunId) return;
  questionTypesRunId = runId;
  const request = ++questionTypesRequest;
  const table = $("question-type-data");
  const status = $("question-types-status");
  const note = $("question-types-note");
  table.hidden = note.hidden = true;
  if (!runId) return;

  const dataset = job.selection.dataset;
  const beam = dataset === "beam";
  const retrieval = retrievalOnly(dataset);
  $("question-types-title").textContent = retrieval ? "按类别统计" : "按题型统计";
  $("question-types-label").textContent = retrieval ? "类别" : "题型";
  $("question-types-total").textContent = retrieval ? "用例数" : "题数";
  $("question-types-correct").textContent = retrieval ? "通过" : beam ? "达标" : "答对";
  $("question-types-incorrect").textContent = retrieval ? "未通过" : beam ? "未达标" : "答错";
  const metric = retrieval ? "通过率" : beam ? "平均得分" : "准确率";
  $("question-types-score").textContent = metric;
  status.textContent = "正在读取题型统计…";
  status.hidden = false;
  try {
    const rows = job.summary.question_types ?? (await api(`/runs/${runId}/question-types`)).rows;
    if (request !== questionTypesRequest) return;
    $("question-types-rows").replaceChildren(...rows.map((row) => {
      const tr = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = questionTypeLabels.get(row.label) ?? (row.label || "未分类");
      tr.append(name);
      for (const value of [row.total, row.correct, row.total - row.correct - row.scoring_failed,
        row.scoring_failed, scoreText(row.score)]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.append(cell);
      }
      return tr;
    }));
    table.hidden = !rows.length;
    status.hidden = Boolean(rows.length);
    status.textContent = "暂无题型统计。";
    const notes = [];
    if (rows.some((row) => row.scoring_failed)) notes.push(`${metric}只计入评分成功的${retrieval ? "用例" : "题目"}。`);
    if (dataset === "lifebench-v2" && rows.length) notes.push("一道题可计入多个题型。");
    note.textContent = notes.join("");
    note.hidden = !notes.length;
  } catch (error) {
    if (request !== questionTypesRequest) return;
    status.textContent = error.message;
  }
}

function renderContextTokens(job) {
  const average = job.summary.avg_context_tokens;
  $("context-token-metric").hidden = average == null;
  $("avg-context-tokens").textContent = tokensText(average);
  const details = $("context-tokens");
  const runId = average != null && job.result_available && !isActive(job) ? job.id : null;
  details.hidden = !runId;
  if (runId === contextTokenRunId) return;
  contextTokenRunId = runId;
  contextTokenOffset = 0;
  contextTokenRequest += 1;
  details.open = false;
  $("context-tokens-table").hidden = true;
  $("context-tokens-nav").hidden = true;
  $("context-tokens-rows").replaceChildren();
  $("context-tokens-status").textContent = "";
}

async function loadContextTokenPage(offset) {
  const id = contextTokenRunId;
  if (!id || id !== selected) return;
  const version = ++contextTokenRequest;
  const status = $("context-tokens-status");
  $("context-tokens-prev").disabled = true;
  $("context-tokens-next").disabled = true;
  status.textContent = "正在读取逐题指标…";
  try {
    const { rows, total } = await api(`/runs/${id}/context-tokens?offset=${offset}&limit=${contextTokenPageSize}`);
    if (id !== selected || version !== contextTokenRequest) return;
    $("context-tokens-rows").replaceChildren(...rows.map((row) => {
      const tr = document.createElement("tr");
      const question = document.createElement("td");
      question.textContent = row.query || row.query_id;
      if (row.query) {
        const ref = document.createElement("small");
        ref.textContent = row.query_id;
        question.append(ref);
      }
      const count = document.createElement("td");
      count.textContent = tokensText(row.context_tokens);
      tr.append(question, count);
      return tr;
    }));
    contextTokenOffset = offset;
    $("context-tokens-table").hidden = !rows.length;
    $("context-tokens-nav").hidden = total <= contextTokenPageSize;
    status.textContent = rows.length ? `${offset + 1}–${offset + rows.length} / ${total} 题` : "暂无逐题 Token 数据。";
    $("context-tokens-prev").disabled = offset === 0;
    $("context-tokens-next").disabled = offset + contextTokenPageSize >= total;
  } catch (error) {
    if (id !== selected || version !== contextTokenRequest) return;
    $("context-tokens-nav").hidden = true;
    status.textContent = error.message;
  }
}

async function loadDetail(id) {
  const job = await api(`/runs/${id}`);
  if (id !== selected) return;
  $("empty").hidden = true;
  $("run-detail").hidden = false;
  $("run-name").textContent = title(job);
  $("run-type").textContent = runType(job);
  $("run-started-at").dateTime = job.started_at;
  $("run-started-at").textContent = formatDate(job.started_at);
  const retrieval = retrievalOnly(job.selection.dataset);
  $("run-retrieval-mode").hidden = !retrieval;
  $("run-models").hidden = retrieval;
  if (!retrieval) {
    const answer = storedModelLabel(job, "answer");
    const shared = Boolean(answer) && answer === storedModelLabel(job, "judge")
      && job.models?.answer?.api_url === job.models?.judge?.api_url;
    $("run-answer-label").textContent = shared ? "回答与评分模型" : modelRoles.answer;
    $("run-answer-model").textContent = storedModelDetail(job, "answer");
    const hideJudge = shared || choiceOnly(job.selection.dataset);
    $("run-judge-label").hidden = hideJudge;
    $("run-judge-model").hidden = hideJudge;
    $("run-judge-model").textContent = hideJudge ? "" : storedModelDetail(job, "judge");
  }
  const target = job.target;
  $("run-target").hidden = !target?.api_url;
  $("run-target").textContent = target?.api_url ? [
    target.api_url,
    target.bank_id && `Bank ${target.bank_id}`,
    target.embedding_model_id && `Embedding ${target.embedding_model_id}`,
    target.vlm_model_id && `视觉模型 ${target.vlm_model_id}`,
  ].filter(Boolean).join(" · ") : "";
  $("status").textContent = statuses[job.status];
  $("status").dataset.status = job.status;
  $("status").hidden = false;
  $("cancel").hidden = !isActive(job);
  $("cancel").disabled = job.status === "cancelling";
  renderRunProgress(job);

  const s = job.summary;
  $("questions-progress-label").textContent = retrieval ? "检索用例进度" : "答题与评分进度";
  const metric = s.score_type ?? settings.score_types?.[job.selection.dataset];
  $("distribution-label").textContent = knowledge ? "证据命中分布" : retrieval ? "用例通过分布" : "答题分布";
  for (const [id, key] of [["knowledge-recall", "recall@10"], ["knowledge-precision", "precision@5"]]) {
    const value = s.metrics?.[key];
    $(id).hidden = !knowledge || value == null || key === metric;
    $(`${id}-value`).textContent = knowledgeScore(value);
  }
  for (const [id, value] of [
    ["active-passes", s.active_total == null ? null : `${s.active_passes} / ${s.active_total}`],
    ["precision", s.mean_precision == null ? null : scoreText(s.mean_precision)],
    ["recall", s.mean_recall == null ? null : scoreText(s.mean_recall)],
  ]) {
    $(`${id}-row`).hidden = !retrieval || value == null;
    $(id).textContent = value ?? "";
  }
  $("metrics").hidden = !s.total_queries;
  if (s.total_queries) {
    $("score-label").textContent = knowledge ? knowledgeMetricLabels[metric] : retrieval ? "用例通过率"
      : job.selection.dataset === "beam" ? "平均得分" : "准确率";
    $("correct-label").textContent = knowledge ? "命中证据的查询 / 总查询"
      : retrieval ? "通过 / 总用例数" : "答对 / 总题数";
    $("score").textContent = knowledge ? knowledgeScore(s.score) : scoreText(s.score);
    $("correct").textContent = s.scoring_failed_queries ? "—" : `${s.correct} / ${s.total_queries}`;
    $("ingestion-label").textContent = wallWriteTime(s) == null ? "累计写入耗时（旧口径）" : "写入实际耗时";
    $("ingestion").textContent = seconds(wallWriteTime(s) ?? legacyWriteTime(s));
    $("retrieval").textContent = number(s.avg_retrieve_time_ms, " ms");
    $("retrieval-p95").textContent = number(s.p95_retrieve_time_ms, " ms");
    $("ingested-docs").textContent = s.ingested_docs ?? "—";
    const failed = s.scoring_failed_queries ?? 0;
    bars($("distribution-bars"), [
      [knowledge ? "命中证据" : retrieval ? "通过" : "答对", s.correct], [knowledge ? "未命中" : "未达标", s.total_queries - s.correct - failed], ["评分失败", failed],
    ], s.total_queries, (value) => `${value} 题`);
  }
  $("distribution").hidden = !s.total_queries;
  $("distribution-note").hidden = job.selection.dataset !== "beam";
  if (!knowledge) {
    renderQuestionTypes(job);
    renderContextTokens(job);
  }
  $("download").hidden = !job.result_available;
  $("download").href = `${apiBase}/runs/${id}/result`;
  $("download").textContent = job.status === "completed" ? "下载结果 JSON" : "下载已保存的部分结果";
  if (job.status === "failed" && $("log").textContent !== job.log) $("logs").open = true;
  if ($("log").textContent !== job.log) $("log").textContent = job.log;
  const failedScoring = job.status === "scoring_incomplete";
  setError("run-error", failedScoring
    ? (retrieval
      ? `${s.scoring_failed_queries} 个用例因检索或来源错误无法评分。可下载逐题结果查看原因。`
      : `${s.scoring_failed_queries} 道题评分失败，未生成总分。可下载已保存的逐题结果。`)
    : job.status === "failed" ? "评测进程退出，请查看日志。" : "");
}

async function refresh() {
  const version = ++refreshVersion;
  const offset = historyOffset;
  const [page, capacity] = await Promise.all([
    api(`/runs?limit=${pageSize + 1}&offset=${offset}`),
    api("/runs/capacity"),
  ]);
  if (version !== refreshVersion) return;
  runCapacity = capacity;
  hasNextPage = page.length > pageSize;
  const updated = page.slice(0, pageSize);
  if (!selected && updated.length) selected = updated[0].id;
  if (JSON.stringify(jobs) !== JSON.stringify(updated)) {
    jobs = updated;
    drawHistory();
  }
  $("history-prev").hidden = offset === 0;
  $("history-next").hidden = !hasNextPage;
  updateForm();
  if (selected) await loadDetail(selected);
}

function changePage(offset) {
  historyOffset = offset;
  refresh().catch((error) => setError("form-error", error.message));
}

async function poll() {
  try {
    await refresh();
  } catch (error) {
    setError("form-error", error.message);
  } finally {
    window.setTimeout(poll, 2000);
  }
}

$("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("start").disabled) return;
  setError("form-error", "");
  const backend = $("backend").value;
  const connection = { ...connectionInput(), use_saved_key: $("connection-reuse").checked };
  submitting = true;
  updateForm();
  try {
    const job = await api("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: $("evaluation-name").value.trim(),
        connection,
        dataset: $("dataset").value,
        split: $("split").value,
        backend,
        query_limit: $("full").checked ? null : Number($("limit").value),
      }),
    });
    selected = job.id;
    historyOffset = 0;
    $("evaluation-name").value = "";
    settings.backends[backend] = job.backend_config;
    loadConnection();
    await refresh();
  } catch (error) {
    setError("form-error", error.message);
  } finally {
    submitting = false;
    updateForm();
  }
});

$("cancel").addEventListener("click", async () => {
  const runId = selected;
  $("cancel").disabled = true;
  try {
    await api(`/runs/${runId}/cancel`, { method: "POST" });
    await refresh();
  } catch (error) {
    if (runId === selected) setError("run-error", error.message);
  }
});

$("dataset").addEventListener("change", updateSplits);
$("split").addEventListener("change", () => {
  updateForm();
  updateQuestionCount();
});
$("backend").addEventListener("change", loadConnection);
$("connection-address").addEventListener("change", selectConnection);
for (const id of [
  "evaluation-name", "connection-url", "connection-bank", "connection-tenant",
  "connection-embedding-model", "connection-vlm-model",
]) {
  $(id).addEventListener("input", updateForm);
}
$("connection-key").addEventListener("input", () => {
  if ($("connection-key").value.trim()) $("connection-reuse").checked = false;
  updateForm();
});
$("connection-reuse").addEventListener("change", () => {
  if ($("connection-reuse").checked) $("connection-key").value = "";
  updateForm();
});
$("history-prev").addEventListener("click", () => changePage(Math.max(0, historyOffset - pageSize)));
$("history-next").addEventListener("click", () => changePage(historyOffset + pageSize));
$("context-tokens").addEventListener("toggle", (event) => {
  if (event.currentTarget.open) loadContextTokenPage(contextTokenOffset);
});
$("context-tokens-prev").addEventListener("click", () => loadContextTokenPage(contextTokenOffset - contextTokenPageSize));
$("context-tokens-next").addEventListener("click", () => loadContextTokenPage(contextTokenOffset + contextTokenPageSize));
$("compare-clear").addEventListener("click", () => {
  compareIds.clear();
  drawHistory();
  drawComparison();
});
$("full").addEventListener("change", updateQuestionCount);

async function init() {
  $(knowledge ? "knowledge-link" : "memory-link").setAttribute("aria-current", "page");
  if (knowledge) {
    document.title = "知识库检索评测 · Memory Lab";
    $("page-title").textContent = "知识库检索评测";
    $("page-description").textContent = "按标准证据给检索结果打分，无需配置回答或评分模型。";
    $("backend-label").textContent = "知识库后端";
    $("evaluation-name").placeholder = "例如：Hindsight PDF 基线";
    $("question-count-label").textContent = "查询数";
  }
  try {
    settings = await api("/config");
    options("dataset", Object.keys(settings.datasets).map((key) => [key, names[key]]));
    if (!knowledge) {
      for (const role of Object.keys(modelRoles)) createModelEditor(role);
      $("model-settings").hidden = false;
    }
    updateSplits();
    await poll();
  } catch (error) {
    setError("form-error", error.message);
  }
}
init();
