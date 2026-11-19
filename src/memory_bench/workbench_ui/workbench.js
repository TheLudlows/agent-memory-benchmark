const $ = (id) => document.getElementById(id);
const names = {
  locomo: "LoCoMo", beam: "BEAM", longmemeval: "LongMemEval",
  "lifebench-v2": "LifeBench 2.0 中文版", precisionmembench: "PrecisionMemBench",
};
const statuses = { running: "运行中", cancelling: "停止中", completed: "已完成", scoring_incomplete: "评分未完成", failed: "失败", cancelled: "已停止", interrupted: "已中断" };
const retrievalOnly = (dataset) => settings?.retrieval_datasets?.includes(dataset);
const modelLabels = (job) => retrievalOnly(job.selection.dataset)
  ? "纯检索评测"
  : `回答 ${storedModelDetail(job, "answer")} · 评分 ${storedModelDetail(job, "judge")}`;
const stages = {
  loading: "加载数据集", ingesting: "写入记忆", evaluating: "检索、回答与评分",
  ingesting_and_evaluating: "同时写入和答题", finalizing: "整理评测结果",
};
const isActive = (job) => job && ["running", "cancelling"].includes(job.status);
const formatDate = (value) => new Date(value).toLocaleString("zh-CN", { hour12: false });

const pageSize = 12;
const compareLimit = 4;
const compareIds = new Set();
let comparisonVersion = 0;
let settings;
let selected;
let jobs = [];
let historyOffset = 0;
let hasNextPage = false;
let activeJob = false;
let submitting = false;
let questionCountVersion = 0;

const modelRoles = { answer: "回答模型", judge: "评分模型" };
const providerLabels = { gemini: "Gemini", openai: "OpenAI 兼容", groq: "Groq" };
const modelEditors = {};
const savingModels = new Set();

async function api(path, options) {
  const response = await fetch(`/api${path}`, options);
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
  updateForm();
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
  $("connection-bank-row").hidden = backend !== "contextbase";
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
  $("connection-key").value = "";
  $("connection-reuse").checked = Boolean(connection?.has_key);
  updateForm();
}

function connectionInput() {
  return {
    api_url: $("connection-address").value || $("connection-url").value.trim(),
    bank_id: $("backend").value === "contextbase" ? $("connection-bank").value.trim() : "",
    api_key: $("connection-key").value.trim() || null,
  };
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
    "mem0-cloud": "使用 Mem0 Cloud 兼容 API。",
    "hindsight-http": "连接已启动的 Hindsight API。",
  };
  $("backend-help").textContent = descriptions[backend];
  $("connection-reuse-row").hidden = !saved?.has_key;
  $("connection-key").placeholder = saved?.has_key ? "填写新 Key 可更新" : config.key_optional ? "选填" : "请输入 API Key";
  const usingSaved = saved?.has_key && $("connection-reuse").checked;
  const splitError = settings.split_errors?.[$("dataset").value]?.[$("split").value];
  for (const editor of Object.values(modelEditors)) updateModelEditor(editor);
  const dataset = $("dataset").value;
  const retrieval = retrievalOnly(dataset);
  const missing = Object.keys(modelRoles).filter((role) => !settings.models[role].has_key);
  const modelError = !retrieval && missing.length
    ? `请在「全局模型设置」配置${missing.map((role) => modelRoles[role]).join("和")}。` : "";
  $("dataset-coverage").textContent = retrieval ? "纯检索评分，写入完整测试数据。"
    : dataset === "lifebench-v2" ? "按提问日期写入记录，之后的资料在该题作答时不可见。"
    : "限制题数时，仍会写入所选题目对应的会话历史。";
  const ready = $("evaluation-name").value.trim() && connection.api_url &&
    (config.key_optional || connection.api_key || usingSaved) &&
    (backend !== "contextbase" || connection.bank_id || saved?.has_bank_mapping);
  setError("config-error", splitError || modelError);
  for (const id of ["backend", "connection-address", "connection-bank", "connection-key", "connection-reuse"]) {
    $(id).disabled = submitting;
  }
  $("connection-url").disabled = submitting || Boolean(saved);
  $("start").disabled = submitting || savingModels.size > 0 || activeJob || !ready || Boolean(splitError || modelError);
  $("start").textContent = submitting ? "正在启动…" : activeJob ? "已有评测运行中" : "开始评测";
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
  const answerModels = runs.map((run) => storedModelLabel(run, "answer"));
  const judgeModels = runs.map((run) => storedModelLabel(run, "judge"));
  const endpoints = (role) => runs.map((run) => run.models?.[role]?.api_url);
  const selectionDiffers = ["dataset", "split"].some((key) => differs(runs.map((run) => run.selection[key])));
  const mismatches = [];
  if (selectionDiffers || differs(questions) || differs(runs.map((run) => run.summary.total_queries))) {
    mismatches.push("题目");
  }
  if (differs(documents)) mismatches.push("写入资料");
  if (differs(answerModels) || differs(endpoints("answer"))) mismatches.push("回答模型");
  if (differs(judgeModels) || differs(endpoints("judge"))) mismatches.push("评分模型");
  const missingEndpoint = ["answer", "judge"].some((role) => runs.some((run) =>
    storedModelLabel(run, role)?.startsWith("openai:") && !run.models?.[role]?.api_url));
  const checkModels = runs.some((run) => !retrievalOnly(run.selection.dataset));
  const fields = checkModels ? [questions, documents, answerModels, judgeModels] : [questions, documents];
  const missing = missingEndpoint || fields
    .some((values) => values.some((value) => !value));
  const unknown = "部分评测缺少题目、写入资料或模型信息，无法确认这些评测的口径。";
  if (mismatches.length) {
    return `${mismatches.join("、")}不同，分数不可直接比较。` + (missing ? unknown : "");
  }
  return missing ? unknown : checkModels ? "题目、写入资料、回答模型和评分模型一致。" : "题目和写入资料一致。";
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
      [runs.every((run) => retrievalOnly(run.selection.dataset)) ? "用例通过率"
        : runs.every((run) => run.selection.dataset === "beam") ? "平均得分" : "得分 / 准确率",
        (s) => s.score, scoreText, 1],
      ["平均检索延迟", (s) => s.avg_retrieve_time_ms, (value) => number(value, " ms")],
      ["检索 P95", (s) => s.p95_retrieve_time_ms, (value) => number(value, " ms")],
    ];
    if (runs.some((run) => retrievalOnly(run.selection.dataset))) {
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
      : stages[stage] || "准备评测") : statuses[job.status];
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

async function loadDetail(id) {
  const job = await api(`/runs/${id}`);
  if (id !== selected) return;
  $("empty").hidden = true;
  $("run-detail").hidden = false;
  $("run-name").textContent = title(job);
  $("run-meta").textContent = `${runType(job)} · ${formatDate(job.started_at)} · ${modelLabels(job)}`;
  const target = job.target;
  $("run-target").hidden = !target?.api_url;
  $("run-target").textContent = target?.api_url
    ? `${target.api_url}${target.bank_id ? ` · Bank ${target.bank_id}` : ""}` : "";
  $("status").textContent = statuses[job.status];
  $("status").dataset.status = job.status;
  $("status").hidden = false;
  $("cancel").hidden = !isActive(job);
  $("cancel").disabled = job.status === "cancelling";
  renderRunProgress(job);

  const s = job.summary;
  const retrieval = retrievalOnly(job.selection.dataset);
  $("questions-progress-label").textContent = retrieval ? "检索用例进度" : "答题与评分进度";
  $("distribution-label").textContent = retrieval ? "用例通过分布" : "答题分布";
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
    $("score-label").textContent = retrieval ? "用例通过率"
      : job.selection.dataset === "beam" ? "平均得分" : "准确率";
    $("correct-label").textContent = retrieval ? "通过 / 总用例数" : "答对 / 总题数";
    $("score").textContent = scoreText(s.score);
    $("correct").textContent = s.scoring_failed_queries ? "—" : `${s.correct} / ${s.total_queries}`;
    $("ingestion-label").textContent = wallWriteTime(s) == null ? "累计写入耗时（旧口径）" : "写入实际耗时";
    $("ingestion").textContent = seconds(wallWriteTime(s) ?? legacyWriteTime(s));
    $("retrieval").textContent = number(s.avg_retrieve_time_ms, " ms");
    $("retrieval-p95").textContent = number(s.p95_retrieve_time_ms, " ms");
    $("ingested-docs").textContent = s.ingested_docs ?? "—";
    const failed = s.scoring_failed_queries ?? 0;
    bars($("distribution-bars"), [
      [retrieval ? "通过" : "答对", s.correct], ["未达标", s.total_queries - s.correct - failed], ["评分失败", failed],
    ], s.total_queries, (value) => `${value} 题`);
  }
  $("distribution").hidden = !s.total_queries;
  $("distribution-note").hidden = job.selection.dataset !== "beam";
  $("download").hidden = !job.result_available;
  $("download").href = `/api/runs/${id}/result`;
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
  const offset = historyOffset;
  const [page, newest] = await Promise.all([
    api(`/runs?limit=${pageSize + 1}&offset=${offset}`),
    offset ? api("/runs?limit=1") : Promise.resolve(null),
  ]);
  if (offset !== historyOffset) return;
  activeJob = isActive((newest || page)[0]);
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
  $("cancel").disabled = true;
  try {
    await api(`/runs/${selected}/cancel`, { method: "POST" });
    await refresh();
  } catch (error) {
    setError("run-error", error.message);
  }
});

$("dataset").addEventListener("change", updateSplits);
$("split").addEventListener("change", () => {
  updateForm();
  updateQuestionCount();
});
$("backend").addEventListener("change", loadConnection);
$("connection-address").addEventListener("change", selectConnection);
for (const id of ["evaluation-name", "connection-url", "connection-bank"]) {
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
$("compare-clear").addEventListener("click", () => {
  compareIds.clear();
  drawHistory();
  drawComparison();
});
$("full").addEventListener("change", updateQuestionCount);

async function init() {
  try {
    settings = await api("/config");
    options("dataset", Object.keys(settings.datasets).map((key) => [key, names[key]]));
    options("backend", Object.entries(settings.backends).map(([key, value]) => [key, value.label]));
    options("split", settings.datasets[$("dataset").value].map((item) => [item, item]));
    for (const role of Object.keys(modelRoles)) createModelEditor(role);
    $("model-settings").hidden = false;
    loadConnection();
    updateQuestionCount();
    await poll();
  } catch (error) {
    setError("form-error", error.message);
  }
}
init();
