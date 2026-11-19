const $ = (id) => document.getElementById(id);
const names = { locomo: "LoCoMo", beam: "BEAM", longmemeval: "LongMemEval" };
const statuses = { running: "运行中", cancelling: "停止中", completed: "已完成", scoring_incomplete: "评分未完成", failed: "失败", cancelled: "已停止", interrupted: "已中断" };
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
let savingConnection = false;

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
}

function loadConnection() {
  const backend = $("backend").value;
  const config = settings.backends[backend];
  $("connection-panel").hidden = false;
  $("connection-url").value = config.api_url;
  $("connection-bank-row").hidden = backend !== "contextbase";
  $("connection-bank").placeholder = config.has_bank_mapping ? "使用已有 Bank 映射" : "已授权的测试 Bank";
  $("connection-bank").value = config.bank_id;
  $("connection-reset").hidden = !config.saved;
  $("connection-clear-default").hidden = !config.key_optional || !config.has_key;
  clearRunKey();
  setError("connection-message", "");
  updateForm();
}

function clearRunKey() {
  $("connection-key").value = "";
  $("connection-reuse").checked = false;
}

function connectionInput() {
  return {
    api_url: $("connection-url").value.trim(),
    bank_id: $("backend").value === "contextbase" ? $("connection-bank").value.trim() : "",
    api_key: $("connection-key").value.trim() || null,
  };
}

async function saveConnection(action = "save") {
  const reset = action === "reset";
  const clear = action === "clear";
  if (savingConnection) return;
  const backend = $("backend").value;
  const data = connectionInput();
  if (clear) {
    data.api_key = null;
    data.clear_key = true;
  }
  savingConnection = true;
  updateForm();
  const send = reset ? {} : { headers: {"Content-Type": "application/json"}, body: JSON.stringify(data) };
  try {
    settings.backends[backend] = await api(`/connections/${backend}${reset ? "/reset" : ""}`, { method: "POST", ...send });
    loadConnection();
    setError("connection-message", reset ? "已恢复初始配置。" :
      clear ? "已清除默认 Key。" : "已保存为默认连接。本轮仍可单独填写。");
  } catch (error) {
    setError("connection-message", error.message);
  } finally {
    savingConnection = false;
    updateForm();
  }
}

function updateForm() {
  if (!settings) return;
  const backend = $("backend").value;
  const config = settings.backends[backend];
  const connection = connectionInput();
  const descriptions = {
    "contextbase": "使用预先授权的测试 Bank。",
    "mem0-cloud": "使用 Mem0 Cloud 兼容 API。",
    "hindsight-http": "连接已启动的 Hindsight API。",
  };
  $("backend-help").textContent = descriptions[backend];
  const sameUrl = connection.api_url.replace(/\/+$/, "") === config.api_url;
  const canReuseKey = sameUrl && config.has_key;
  if (!canReuseKey) $("connection-reuse").checked = false;
  $("connection-reuse-row").hidden = !canReuseKey;
  $("connection-key").placeholder = config.key_optional ? "选填" : "请输入本轮 API Key";
  const usingSaved = $("connection-reuse").checked;
  const splitError = settings.split_errors?.[$("dataset").value]?.[$("split").value];
  const modelError = settings.missing_models.length ? `请在 .env 配置 ${settings.missing_models.join("、")}` : "";
  const ready = $("evaluation-name").value.trim() && connection.api_url &&
    (config.key_optional || connection.api_key || usingSaved) &&
    (backend !== "contextbase" || connection.bank_id || (sameUrl && config.has_bank_mapping));
  setError("config-error", splitError || modelError);
  for (const id of ["backend", "connection-url", "connection-bank", "connection-reuse", "connection-save", "connection-reset", "connection-clear-default"]) {
    $(id).disabled = savingConnection || submitting;
  }
  $("connection-key").disabled = savingConnection || submitting || usingSaved;
  for (const role of ["answer", "judge"]) {
    const { provider, model } = settings.models[role];
    const fallback = provider === "gemini" && role === "judge" && $("dataset").value === "beam"
      ? "gemini-3.5-flash" : "默认模型";
    $(`${role}-model`).textContent = `${provider} · ${model || fallback}`;
  }
  $("limit").disabled = $("full").checked;
  $("start").disabled = submitting || savingConnection || activeJob || !ready || Boolean(splitError || modelError);
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

function comparisonWarning(runs) {
  const first = runs[0];
  const questionMismatch = runs.some((run) =>
    run.selection.dataset !== first.selection.dataset ||
    run.selection.split !== first.selection.split ||
    run.summary.total_queries !== first.summary.total_queries ||
    (first.summary.question_fingerprint && run.summary.question_fingerprint &&
      run.summary.question_fingerprint !== first.summary.question_fingerprint));
  const modelMismatch = runs.some((run) => ["answer_llm", "judge_llm"].some((role) =>
    run.summary[role] !== first.summary[role]));
  if (questionMismatch || modelMismatch) return "所选评测的题目或模型不同，分数不可直接比较。";
  if (runs.some((run) => !run.summary.question_fingerprint))
    return "部分历史评测没有题目指纹，无法确认题目完全一致。";
  return "题目、回答模型和评分模型一致。";
}

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
        ` · 回答 ${run.summary.answer_llm || "—"} · 评分 ${run.summary.judge_llm || "—"}` +
        (run.summary.scoring_failed_queries ? ` · 评分失败 ${run.summary.scoring_failed_queries} 题` : "");
      $("compare-meta").append(item);
    }
    const metrics = [
      [runs.every((run) => run.selection.dataset === "beam") ? "平均得分" : "得分 / 准确率", "score", scoreText, 1],
      ["平均检索延迟", "avg_retrieve_time_ms", (value) => number(value, " ms")],
      ["检索 P95", "p95_retrieve_time_ms", (value) => number(value, " ms")],
      ["写入耗时", "ingestion_time_ms", seconds],
    ];
    for (const [label, key, format, max] of metrics) {
      const figure = document.createElement("figure");
      figure.className = "chart";
      const caption = document.createElement("figcaption");
      caption.textContent = label;
      const content = document.createElement("div");
      const rows = runs.map((run, i) => [`${i + 1}. ${title(run)}`, run.summary[key]]);
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
    active ? (stages[stage] || "准备评测") : statuses[job.status];
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
  const modelLabel = (role) => job.summary[`${role}_llm`] || job.models[role].model || job.models[role].provider;
  $("run-meta").textContent = `${runType(job)} · ${formatDate(job.started_at)} · 回答 ${modelLabel("answer")} · 评分 ${modelLabel("judge")}`;
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
  $("metrics").hidden = !s.total_queries;
  if (s.total_queries) {
    $("score-label").textContent = job.selection.dataset === "beam" ? "平均得分" : "准确率";
    $("score").textContent = scoreText(s.score);
    $("correct").textContent = s.scoring_failed_queries ? "—" : `${s.correct} / ${s.total_queries}`;
    $("ingestion").textContent = seconds(s.ingestion_time_ms);
    $("retrieval").textContent = number(s.avg_retrieve_time_ms, " ms");
    $("retrieval-p95").textContent = number(s.p95_retrieve_time_ms, " ms");
    $("ingested-docs").textContent = s.ingested_docs ?? "—";
    const failed = s.scoring_failed_queries ?? 0;
    bars($("distribution-bars"), [
      ["答对", s.correct], ["未达标", s.total_queries - s.correct - failed], ["评分失败", failed],
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
    ? `${s.scoring_failed_queries} 道题评分失败，未生成总分。可下载已保存的逐题结果。`
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
        backend: $("backend").value,
        query_limit: $("full").checked ? null : Number($("limit").value),
      }),
    });
    selected = job.id;
    historyOffset = 0;
    $("evaluation-name").value = "";
    clearRunKey();
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
$("split").addEventListener("change", updateForm);
$("backend").addEventListener("change", loadConnection);
for (const id of ["evaluation-name", "connection-url", "connection-bank", "connection-key", "connection-reuse"]) {
  $(id).addEventListener("input", () => {
    if (id === "connection-reuse" && $("connection-reuse").checked) $("connection-key").value = "";
    setError("connection-message", "");
    updateForm();
  });
}
$("history-prev").addEventListener("click", () => changePage(Math.max(0, historyOffset - pageSize)));
$("history-next").addEventListener("click", () => changePage(historyOffset + pageSize));
$("compare-clear").addEventListener("click", () => {
  compareIds.clear();
  drawHistory();
  drawComparison();
});
$("connection-save").addEventListener("click", () => saveConnection());
$("connection-reset").addEventListener("click", () => saveConnection("reset"));
$("connection-clear-default").addEventListener("click", () => saveConnection("clear"));
$("full").addEventListener("change", updateForm);

async function init() {
  try {
    settings = await api("/config");
    options("dataset", Object.keys(settings.datasets).map((key) => [key, names[key]]));
    options("backend", Object.entries(settings.backends).map(([key, value]) => [key, value.label]));
    loadConnection();
    updateSplits();
    await poll();
  } catch (error) {
    setError("form-error", error.message);
  }
}
init();
