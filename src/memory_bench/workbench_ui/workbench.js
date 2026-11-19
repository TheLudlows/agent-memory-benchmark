const $ = (id) => document.getElementById(id);
const names = { locomo: "LoCoMo", beam: "BEAM", longmemeval: "LongMemEval" };
const statuses = { running: "运行中", cancelling: "停止中", completed: "已完成", scoring_incomplete: "评分未完成", failed: "失败", cancelled: "已停止", interrupted: "已中断" };
const stages = { loading: "加载数据集", ingesting: "写入记忆", evaluating: "检索、回答与评分" };
const isActive = (job) => job && ["running", "cancelling"].includes(job.status);
const formatDate = (value) => new Date(value).toLocaleString("zh-CN", { hour12: false });

let settings;
let selected;
let jobs = [];
let submitting = false;
let connectionDirty = false;
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
  $("connection-panel").open = config.missing.length > 0;
  $("connection-url").value = config.api_url;
  $("connection-bank-row").hidden = backend !== "contextbase";
  $("connection-bank").placeholder = config.has_bank_mapping ? "使用已有 Bank 映射" : "已授权的测试 Bank";
  $("connection-bank").value = config.bank_id;
  $("connection-key").value = "";
  $("connection-key").placeholder = config.has_key ? "已配置，留空保留" : config.key_optional ? "选填" : "请输入 API Key";
  $("connection-clear-row").hidden = !config.key_optional || !config.has_key;
  $("connection-clear").checked = false;
  $("connection-reset").hidden = !config.saved;
  connectionDirty = false;
  setError("connection-message", "");
  updateForm();
}

async function saveConnection(reset = false) {
  if (savingConnection) return;
  const backend = $("backend").value;
  const apiKey = $("connection-key").value;
  const clearKey = !$("connection-clear-row").hidden && $("connection-clear").checked;
  const data = {
    api_url: $("connection-url").value,
    bank_id: backend === "contextbase" ? $("connection-bank").value : "",
    api_key: apiKey || null,
    clear_key: clearKey,
  };
  savingConnection = true;
  updateForm();
  const send = reset ? {} : { headers: {"Content-Type": "application/json"}, body: JSON.stringify(data) };
  try {
    settings.backends[backend] = await api(`/connections/${backend}${reset ? "/reset" : ""}`, { method: "POST", ...send });
    loadConnection();
    $("connection-panel").open = true;
    setError("connection-message", reset ? "已恢复默认配置。" : "已保存连接。");
  } catch (error) {
    setError("connection-message", error.message);
  } finally {
    savingConnection = false;
    updateForm();
  }
}

function updateForm() {
  if (!settings) return;
  const connection = settings.backends[$("backend").value];
  const descriptions = {
    "contextbase": "使用预先授权的测试 Bank。",
    "mem0-cloud": "使用 Mem0 托管 API。",
    "hindsight-http": "连接已启动的 Hindsight API。",
  };
  $("backend-help").textContent = descriptions[$("backend").value];
  const splitError = settings.split_errors?.[$("dataset").value]?.[$("split").value];
  const backendError = connectionDirty ? "请先保存连接配置。" :
    connection.missing.length ? `请配置后端的 ${connection.missing.join("、")}。` : "";
  const modelError = settings.missing_models.length ? `请在 .env 配置 ${settings.missing_models.join("、")}` : "";
  setError("config-error", splitError || backendError || modelError);
  $("connection-status").textContent = connectionDirty ? "未保存" : connection.missing.length ? "待配置" : "已配置";
  for (const id of ["backend", "connection-url", "connection-bank", "connection-clear", "connection-save", "connection-reset"]) {
    $(id).disabled = savingConnection;
  }
  $("connection-key").disabled = savingConnection || $("connection-clear").checked;
  for (const role of ["answer", "judge"]) {
    const { provider, model } = settings.models[role];
    const fallback = provider === "gemini" && role === "judge" && $("dataset").value === "beam"
      ? "gemini-3.5-flash" : "默认模型";
    $(`${role}-model`).textContent = `${provider} · ${model || fallback}`;
  }
  $("limit").disabled = $("full").checked;
  $("start").disabled = submitting || savingConnection || jobs.some(isActive) || Boolean(splitError || backendError || modelError);
  $("start").textContent = submitting ? "正在启动…" : jobs.some(isActive) ? "已有评测运行中" : "开始评测";
}

function title(job) {
  const { dataset, split, backend } = job.selection;
  return `${names[dataset]} · ${split} · ${settings.backends[backend].label}`;
}

function drawHistory() {
  $("history-empty").hidden = jobs.length > 0;
  $("history").replaceChildren(...jobs.map((job) => {
    const row = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-row";
    if (job.id === selected) button.setAttribute("aria-current", "true");
    const text = document.createElement("span");
    text.textContent = title(job);
    const note = document.createElement("small");
    note.textContent = `${formatDate(job.started_at)} · ${job.selection.query_limit ?? "全量"}${job.selection.query_limit ? " 题" : ""}`;
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
    row.append(button);
    return row;
  }));
}

async function loadDetail(id) {
  const job = await api(`/runs/${id}`);
  if (id !== selected) return;
  $("empty").hidden = true;
  $("run-detail").hidden = false;
  $("run-name").textContent = title(job);
  const modelLabel = (role) => job.summary[`${role}_llm`] || job.models[role].model || job.models[role].provider;
  $("run-meta").textContent = `${formatDate(job.started_at)} · 回答 ${modelLabel("answer")} · 评分 ${modelLabel("judge")}`;
  $("status").textContent = statuses[job.status];
  $("status").dataset.status = job.status;
  $("status").hidden = false;
  $("cancel").hidden = !isActive(job);
  $("cancel").disabled = job.status === "cancelling";
  const { stage, total, completed } = job.progress;
  $("stage").textContent = isActive(job) ? (stages[stage] || "准备评测") : statuses[job.status];
  $("progress-count").textContent = total ? `${completed} / ${total} 题` : "";
  $("progress").max = total || 1;
  if (total || !isActive(job)) $("progress").value = completed || 0;
  else $("progress").removeAttribute("value");
  $("progress").hidden = !isActive(job) && !total;

  const s = job.summary;
  $("metrics").hidden = !s.total_queries;
  if (s.total_queries) {
    $("score-label").textContent = job.selection.dataset === "beam" ? "平均得分" : "准确率";
    $("score").textContent = s.score == null ? "—" : `${(s.score * 100).toFixed(1)}%`;
    $("correct").textContent = s.scoring_failed_queries ? "—" : `${s.correct} / ${s.total_queries}`;
    $("ingestion").textContent = `${(s.ingestion_time_ms / 1000).toFixed(1)} s`;
    $("retrieval").textContent = s.avg_retrieve_time_ms == null ? "—" : `${s.avg_retrieve_time_ms.toFixed(1)} ms`;
  }
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
  const updated = await api("/runs");
  if (!selected && updated.length) selected = updated[0].id;
  if (JSON.stringify(jobs) !== JSON.stringify(updated)) {
    jobs = updated;
    drawHistory();
  }
  updateForm();
  if (selected) await loadDetail(selected);
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
  submitting = true;
  updateForm();
  try {
    const job = await api("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: $("dataset").value,
        split: $("split").value,
        backend: $("backend").value,
        query_limit: $("full").checked ? null : Number($("limit").value),
      }),
    });
    selected = job.id;
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
for (const id of ["connection-url", "connection-bank", "connection-key", "connection-clear"]) {
  $(id).addEventListener("input", () => {
    if (id === "connection-clear" && $("connection-clear").checked) $("connection-key").value = "";
    connectionDirty = true;
    setError("connection-message", "");
    updateForm();
  });
}
$("connection-save").addEventListener("click", () => saveConnection());
$("connection-reset").addEventListener("click", () => saveConnection(true));
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
