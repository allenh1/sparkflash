// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

"use strict";

// Every request that changes something carries this header. The server refuses a mutation
// without it, which keeps a cross-site form post from driving the cluster.
const ACTION_HEADER = "x-sparkflash-web";
const TABS = ["overview", "setup", "activity"];

const S = {
  state: null,
  stateError: null,
  series: null,
  jobs: [],
  windowSecs: 900,
  configDirty: false,
  configSignature: "",
  selectedJob: null,
  jobDetail: null,
  discoveryBusy: false,
  inspectBusy: false,
  inspectError: null,
  inspectedFor: null,
  autoInspected: false,
  jobSeen: new Map(),
  pick: null,
  manualHosts: [],
  guideOpen: {},
  guideHtml: {},
  models: null,
  autoDiscovered: false,
  firstState: true,
  tab: "overview",
  hover: null,
  detailBusy: false,
  colors: {},
};

// The five graphed series, in the fixed categorical order the CSS defines (--s1 .. --s5).
// A null value means rank 0 has nothing to report for that series; `none` says why, and the
// graphs leave a gap rather than drawing a zero.
const SERIES = [
  { key: "decode_tok_per_s", chart: "chart-decode", now: "chart-decode-now", spark: "spark-decode", label: "Decode", color: "--s1", unit: "tok/s", none: "No request has completed yet" },
  { key: "prefill_tok_per_s", chart: "chart-prefill", now: "chart-prefill-now", spark: "spark-prefill", label: "Prefill", color: "--s2", unit: "tok/s", none: "No request has completed yet" },
  { key: "acceptance_rate", chart: "chart-accept", now: "chart-accept-now", spark: "spark-accept", label: "Draft acceptance", color: "--s3", percent: true, none: "No draft tokens proposed yet" },
  { key: "cache_hit_rate", chart: "chart-cache", now: "chart-cache-now", spark: "spark-cache", label: "Prefix cache hit", color: "--s4", percent: true, none: "Rank 0 reports no prefix-cache lookups" },
  { key: "generated_tok_per_s", chart: "chart-throughput", now: "chart-throughput-now", spark: "spark-throughput", label: "Generation", color: "--s5", unit: "tok/s", none: "Waiting for a second sample" },
];

const hasValue = (point, key) => point.online && point[key] !== null && point[key] !== undefined;

const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat();
const SVG_NS = "http://www.w3.org/2000/svg";

/* Formatting */

function fmtInt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return nf.format(Math.round(Number(value)));
}

function fmtCompact(value) {
  const number = Number(value) || 0;
  if (number >= 1e9) return (number / 1e9).toFixed(2) + "B";
  if (number >= 1e6) return (number / 1e6).toFixed(2) + "M";
  if (number >= 1e4) return (number / 1e3).toFixed(0) + "k";
  if (number >= 1e3) return (number / 1e3).toFixed(1) + "k";
  return nf.format(Math.round(number));
}

function fmtRate(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  return number >= 100 ? nf.format(Math.round(number)) : number.toFixed(1);
}

function fmtPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return (Number(value) * 100).toFixed(1) + "%";
}

function fmtSeries(def, value) {
  if (def.percent) return fmtPercent(value);
  if (value === null || value === undefined) return "—";
  return fmtRate(value) + (def.unit ? " " + def.unit : "");
}

function fmtClock(ms, seconds = false) {
  const options = seconds
    ? { hour: "2-digit", minute: "2-digit", second: "2-digit" }
    : { hour: "2-digit", minute: "2-digit" };
  try {
    return new Date(ms).toLocaleTimeString([], options);
  } catch {
    return "";
  }
}

function fmtDuration(ms) {
  const seconds = Math.max(0, Math.round(ms / 1000));
  if (seconds < 60) return seconds + "s";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return minutes + "m " + (seconds % 60) + "s";
  return Math.floor(minutes / 60) + "h " + (minutes % 60) + "m";
}

function utcToday() {
  return new Date().toISOString().slice(0, 10);
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/* Network */

async function api(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = { ...(options.headers || {}) };
  if (method !== "GET" && method !== "HEAD") headers[ACTION_HEADER] = "1";
  if (options.body !== undefined) headers["content-type"] = "application/json";
  const response = await fetch(path, { ...options, method, headers, cache: "no-store" });
  const text = await response.text();
  let body = null;
  if (text) {
    try { body = JSON.parse(text); } catch { body = { error: text }; }
  }
  if (!response.ok) {
    throw new Error((body && body.error) || response.statusText || ("HTTP " + response.status));
  }
  return body;
}

/* Feedback */

function toast(message, error = false) {
  const container = $("toasts");
  const element = document.createElement("div");
  element.className = "toast" + (error ? " error" : "");
  element.textContent = message;
  container.appendChild(element);
  while (container.children.length > 4) container.firstElementChild.remove();
  setTimeout(() => {
    element.classList.add("leaving");
    setTimeout(() => element.remove(), 220);
  }, error ? 7000 : 4200);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  toast("Copied to clipboard");
}

function confirmAction({ title, body, ok, danger = false }) {
  const dialog = $("confirm");
  if (typeof dialog.showModal !== "function") return Promise.resolve(window.confirm(title + "\n\n" + body));
  $("confirm-title").textContent = title;
  $("confirm-body").textContent = body;
  const okButton = $("confirm-ok");
  okButton.textContent = ok;
  okButton.className = "btn " + (danger ? "btn-danger solid" : "btn-primary");
  return new Promise((resolve) => {
    dialog.returnValue = "";
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true });
    dialog.showModal();
    (danger ? dialog.querySelector('[value="cancel"]') : okButton).focus();
  });
}

/* Tabs */

function showTab(name) {
  if (!TABS.includes(name)) name = "overview";
  S.tab = name;
  for (const page of document.querySelectorAll("[data-page]")) {
    page.classList.toggle("hidden", page.dataset.page !== name);
  }
  for (const link of document.querySelectorAll("[data-tab]")) {
    link.setAttribute("aria-selected", String(link.dataset.tab === name));
  }
  if (name === "overview") {
    renderCharts();
    renderDaily();
    renderSparks();
  }
  if (name === "activity") {
    renderJobs();
    if (S.selectedJob !== null) refreshJobDetail(S.selectedJob, true);
  }
  renderGuide();
}

/* Configuration form state */

function markDirty() {
  S.configDirty = true;
  $("config-form").classList.add("dirty");
  syncContextChips();
}

function clearDirty() {
  S.configDirty = false;
  S.configSignature = "";
  $("config-form").classList.remove("dirty");
}

function requireSaved() {
  if (!S.configDirty) return true;
  toast("Save or discard the configuration changes first.", true);
  return false;
}

/* State */

async function refreshState() {
  try {
    S.state = await api("/api/state");
    S.stateError = null;
    S.jobs = S.state.jobs || [];
    if (S.firstState) {
      S.firstState = false;
      // A console that has never been set up opens on the setup steps.
      if (!location.hash && !S.state.config_exists) location.hash = "setup";
    }
    renderState();
    if (!S.autoDiscovered && !S.state.discovery) {
      S.autoDiscovered = true;
      discover(true);
    }
    maybeInspect();
  } catch (error) {
    S.stateError = error.message;
    renderStatus();
  }
}

function clusterStatus() {
  const latest = S.state?.latest;
  const online = !!(latest && latest.online);
  if (S.stateError) return { cls: "bad", text: "Console offline", title: S.stateError };
  if (online && latest.unhealthy) return { cls: "bad", text: "Stalled", title: latest.unhealthy };
  if (online && latest.ready) return { cls: "on", text: "Serving", title: "Rank 0 is ready" };
  if (online) return { cls: "drain", text: "Draining", title: "Rank 0 answers but is not accepting new requests" };
  return { cls: "off", text: "Offline", title: S.state?.last_error || "Rank 0 is not answering" };
}

function renderStatus() {
  const status = clusterStatus();
  const element = $("cluster-status");
  element.className = "status " + status.cls;
  element.title = status.title || "";
  $("cluster-status-text").textContent = status.text;
  if (S.stateError) {
    $("config-error-text").textContent = "Cannot reach the control plane: " + S.stateError;
    $("config-error").classList.remove("hidden");
  }
}

function renderState() {
  const state = S.state;
  if (!state) return;
  renderStatus();

  $("config-path").textContent = state.config_path;
  const exists = $("config-exists");
  exists.textContent = state.config_exists ? "saved" : "not saved yet";
  exists.className = "tag " + (state.config_exists ? "good" : "warn");
  if (state.config_error) {
    $("config-error-text").textContent = state.config_error;
    $("config-error").classList.remove("hidden");
  } else {
    $("config-error").classList.add("hidden");
  }
  $("footer-config").textContent = "Configuration: " + state.config_path;
  $("footer-advert").textContent = state.advertised
    ? "Advertised over mDNS as " + state.advertised
    : "mDNS advertisement unavailable";

  const latest = state.latest;
  const online = !!(latest && latest.online);
  const unhealthy = online && latest.unhealthy;
  $("health-banner").classList.toggle("hidden", !unhealthy);
  $("health-reason").textContent = unhealthy ? latest.unhealthy : "";

  const config = state.config;
  const cluster = state.cluster;
  $("overview-sub").textContent = online
    ? [
        cluster?.model || config.model.id,
        fmtInt(cluster?.max_seq_len || config.runtime.max_context_tokens) + "-token context",
        config.runtime.max_concurrency + " lanes",
        config.web.serve_url,
      ].join(" · ")
    : "Rank 0 is not answering at " + config.web.serve_url + ".";

  $("kpi-decode").textContent = online ? fmtRate(latest.decode_tok_per_s) : "—";
  $("kpi-prefill").textContent = online ? fmtRate(latest.prefill_tok_per_s) : "—";
  $("kpi-throughput").textContent = online ? fmtRate(latest.generated_tok_per_s) : "—";
  $("kpi-accept").textContent = online ? fmtPercent(latest.acceptance_rate) : "—";
  $("kpi-cache").textContent = online ? fmtPercent(latest.cache_hit_rate) : "—";
  $("kpi-accept-note").classList.toggle("hidden", !(online && latest.acceptance_rate === null));
  $("kpi-cache-note").classList.toggle("hidden", !(online && latest.cache_hit_rate === null));
  $("kpi-running").textContent = online ? fmtInt(latest.running) : "—";
  $("kpi-waiting").textContent = online ? fmtInt(latest.waiting) : "—";
  $("kpi-today").textContent = fmtCompact(state.totals.tokens_by_day[utcToday()] || 0);

  const running = S.jobs.filter((job) => job.status === "running");
  const clusterBusy = running.some((job) => job.kind === "launch" || job.kind === "stop");
  const converting = running.some((job) => job.kind.startsWith("convert-"));
  $("action-launch").disabled = online || clusterBusy || converting;
  $("action-stop").disabled = !online || clusterBusy;
  $("action-status").disabled = clusterBusy;
  for (const button of document.querySelectorAll("[data-convert-rank]")) {
    button.disabled = online || clusterBusy;
  }
  const count = $("tab-running");
  count.textContent = running.length;
  count.classList.toggle("hidden", running.length === 0);

  $("max-context").max = state.limits?.max_context_tokens || 1048576;
  $("max-concurrency").max = state.limits?.max_concurrency || 32;

  renderCluster(state);
  renderGuide();
  renderJobs();
  if (!S.configDirty) paintConfig(state.config, state.discovery);
}

/* Configuration form */

function paintConfig(config, report, force = false) {
  const statuses = {};
  for (const node of report?.nodes || []) {
    if (node.rank !== null && node.rank !== undefined) statuses[node.rank] = node;
  }
  const signature = JSON.stringify(config) + JSON.stringify(statuses);
  if (!force && signature === S.configSignature) return;
  S.configSignature = signature;

  populateSelect("model-weights", S.state?.supported?.weights || [config.model.weights], config.model.weights);
  populateSelect("engine-kind", S.state?.supported?.engines || [config.engine.kind], config.engine.kind);

  setValue("max-context", config.runtime.max_context_tokens);
  setValue("max-concurrency", config.runtime.max_concurrency);
  setValue("checkpoint", config.model.checkpoint);
  setValue("hub-cache", config.model.hub_cache);
  const checkpoint = S.state?.checkpoint;
  $("checkpoint").placeholder = checkpoint?.from_cache && checkpoint.path ? checkpoint.path : "from the Hugging Face cache";
  $("hub-cache").placeholder = checkpoint?.cache || "~/.cache/huggingface/hub";
  paintModels(config.model.id);
  setValue("tokenizer", config.model.tokenizer);
  setValue("converted-dir", config.model.converted_dir);
  setValue("cubins", config.model.cubins);
  setValue("repo-root", config.paths.repo_root);
  setValue("binary", config.paths.binary);
  setValue("serve-script", config.paths.serve_script);
  setValue("nccl-lib", config.paths.nccl_lib);
  setValue("log-root", config.paths.log_root);
  setValue("ssh-user", config.cluster.ssh_user);
  setValue("ssh-options", (config.cluster.ssh_options || []).join("\n"));
  setValue("http-bind", config.cluster.http_bind);
  setValue("base-port", config.cluster.base_port);
  setValue("nccl-ib-hca", config.network.nccl_ib_hca);
  setValue("nccl-socket-ifname", config.network.nccl_socket_ifname);
  setValue("nccl-gid", config.network.nccl_ib_gid_index ?? "");
  setValue("extra-env", formatEnv(config.network.env));
  setValue("web-bind", config.web.bind);
  setValue("serve-url", config.web.serve_url);
  setValue("allowed-hosts", (config.web.allowed_hosts || []).join("\n"));
  setValue("discovery-timeout", config.web.discovery_timeout_secs);
  setValue("ssh-timeout", config.web.ssh_timeout_secs);
  setValue("stats-path", config.stats.persist_path);
  setValue("stats-interval", config.stats.sample_interval_secs);
  setValue("stats-retention", config.stats.retention_secs);
  setValue("stats-flush", config.stats.flush_interval_secs);
  $("advertise").checked = !!config.web.advertise;
  renderNodeRows(config.cluster.nodes, statuses);
  syncContextChips();
}

function populateSelect(id, values, current) {
  const select = $(id);
  const all = Array.from(new Set([...(values || []), current].filter(Boolean)));
  const signature = all.join("|");
  if (select.dataset.signature !== signature) {
    select.innerHTML = all.map((value) => '<option value="' + esc(value) + '">' + esc(value) + "</option>").join("");
    select.dataset.signature = signature;
  }
  select.value = current;
}

/* Models in the Hugging Face cache */

async function loadModels(quiet = false) {
  const button = $("rescan-models");
  button.classList.add("busy");
  try {
    S.models = await api("/api/models");
  } catch (error) {
    S.models = { cache: null, models: [], error: error.message };
    if (!quiet) toast("Could not read the Hugging Face cache: " + error.message, true);
  } finally {
    button.classList.remove("busy");
  }
  paintModels();
}

function formatSize(bytes) {
  if (bytes >= 1e12) return (bytes / 1e12).toFixed(1) + " TB";
  if (bytes >= 1e9) return Math.round(bytes / 1e9) + " GB";
  return Math.max(1, Math.round(bytes / 1e6)) + " MB";
}

function modelLabel(model) {
  const notes = [formatSize(model.bytes)];
  if (!model.complete) notes.push("download incomplete");
  if (!model.supported) notes.push("unsupported " + (model.model_type || "model type"));
  return model.id + " — " + notes.join(", ");
}

// The cached models as options; the selected id stays listed even when it is not cached.
function paintModels(selected) {
  const select = $("model-id");
  const current = selected ?? (select.value || S.state?.config?.model?.id || "");
  const models = S.models?.models || [];
  const options = models.map((model) => ({
    value: model.id,
    text: modelLabel(model),
    disabled: !model.supported && model.id !== current,
  }));
  if (current && !models.some((model) => model.id === current)) {
    options.unshift({ value: current, text: current + (S.models ? " — not in the cache" : ""), disabled: false });
  }
  const signature = JSON.stringify(options);
  if (select.dataset.signature !== signature) {
    select.innerHTML = options
      .map((option) => '<option value="' + esc(option.value) + '"' + (option.disabled ? " disabled" : "") + ">" + esc(option.text) + "</option>")
      .join("");
    select.dataset.signature = signature;
  }
  select.value = current;
  paintModelSource();
}

// Where the selected model's weights come from, under the selector.
function paintModelSource() {
  const id = $("model-id").value;
  const override = $("checkpoint").value.trim();
  const model = (S.models?.models || []).find((entry) => entry.id === id);
  let text;
  let warn = false;
  if (override) {
    text = "Checkpoint set under Paths: " + override;
  } else if (model) {
    text = "Snapshot " + model.revision.slice(0, 12) + " · " + model.snapshot;
    if (!model.has_tokenizer) {
      text += " · no tokenizer.json";
      warn = true;
    }
    if (!model.supported) {
      text = "This build cannot load " + (model.model_type || "this model type") + ". " + text;
      warn = true;
    }
  } else if (!S.models) {
    text = "Reading the Hugging Face cache…";
  } else if (S.models.error) {
    text = S.models.error;
    warn = true;
  } else {
    text = (id || "This model") + " is not in the Hugging Face cache at " + S.models.cache + ". Download it on every node, or set Checkpoint under Paths.";
    warn = true;
  }
  const saved = S.state?.config?.model?.id;
  const note = saved && id && id !== saved ? "Changing the model needs a fresh conversion on every rank." : "";
  const element = $("model-source");
  element.innerHTML = esc(text) + (note ? '<span class="model-note">' + esc(note) + "</span>" : "");
  element.classList.toggle("warn", warn);
}

function setValue(id, value) {
  const element = $(id);
  if (!element || document.activeElement === element) return;
  element.value = value ?? "";
}

function syncContextChips() {
  const value = $("max-context").value;
  for (const chip of document.querySelectorAll("[data-context]")) {
    chip.classList.toggle("active", chip.dataset.context === value);
  }
}

function sshState(node) {
  if (!node || node.ssh_ok === null || node.ssh_ok === undefined) {
    return '<span class="ssh-state unknown">not probed</span>';
  }
  if (node.ssh_ok && node.local && node.configured) return '<span class="ssh-state ok">this node</span>';
  if (node.ssh_ok) return '<span class="ssh-state ok">passwordless</span>';
  return '<span class="ssh-state fail" title="' + esc(node.ssh_message) + '">failed</span>';
}

function renderNodeRows(nodes, statuses) {
  const body = $("node-rows");
  const online = !!(S.state?.latest && S.state.latest.online);
  body.innerHTML = (nodes || []).map((node, rank) =>
    '<tr data-rank="' + rank + '">' +
    '<td class="rank-id">R' + rank + "</td>" +
    '<td><input data-field="ssh" type="text" value="' + esc(node.ssh) + '" aria-label="Rank ' + rank + ' SSH host"></td>' +
    '<td><input data-field="peer" type="text" value="' + esc(node.peer) + '" aria-label="Rank ' + rank + ' RoCE peer"></td>' +
    '<td><label class="check"><input data-field="local" type="checkbox"' + (node.local ? " checked" : "") + ' aria-label="Rank ' + rank + ' is this node"></label></td>' +
    "<td>" + sshState(statuses[rank]) + "</td>" +
    '<td class="num"><button class="btn btn-ghost btn-xs" type="button" data-convert-rank="' + rank + '"' + (online ? " disabled" : "") +
    ' title="Convert this rank\'s quarter of the weights">Convert</button></td>' +
    "</tr>"
  ).join("");
  for (const button of body.querySelectorAll("[data-convert-rank]")) {
    button.addEventListener("click", async () => {
      if (!requireSaved()) return;
      const rank = Number(button.dataset.convertRank);
      const ok = await confirmAction({
        title: "Convert rank " + rank + "?",
        body: "This rewrites rank " + rank + "'s converted weights on " + (nodes[rank]?.ssh || "its node") +
          ". The cluster must stay stopped until it finishes.",
        ok: "Convert rank " + rank,
      });
      if (ok) postAction("/api/actions/convert", { ranks: [rank] });
    });
  }
}

function formatEnv(env) {
  return Object.keys(env || {}).sort().map((key) => key + "=" + env[key]).join("\n");
}

function parseEnv(text) {
  const env = {};
  for (const raw of String(text || "").split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const equals = line.indexOf("=");
    if (equals <= 0) continue;
    env[line.slice(0, equals).trim()] = line.slice(equals + 1).trim();
  }
  return env;
}

function lines(id) {
  return $(id).value.split("\n").map((line) => line.trim()).filter(Boolean);
}

function numberValue(id, fallback) {
  const raw = $(id).value.trim();
  const parsed = Number(raw);
  return raw !== "" && Number.isFinite(parsed) ? parsed : fallback;
}

function collectConfig() {
  const base = S.state?.config;
  if (!base) return null;
  const config = JSON.parse(JSON.stringify(base));
  config.model = {
    ...config.model,
    weights: $("model-weights").value,
    id: $("model-id").value.trim(),
    checkpoint: $("checkpoint").value.trim(),
    hub_cache: $("hub-cache").value.trim(),
    tokenizer: $("tokenizer").value.trim(),
    converted_dir: $("converted-dir").value.trim(),
    cubins: $("cubins").value.trim(),
  };
  config.engine = { ...config.engine, kind: $("engine-kind").value };
  config.runtime = {
    max_context_tokens: numberValue("max-context", config.runtime.max_context_tokens),
    max_concurrency: numberValue("max-concurrency", config.runtime.max_concurrency),
  };
  config.paths = {
    repo_root: $("repo-root").value.trim(),
    binary: $("binary").value.trim(),
    serve_script: $("serve-script").value.trim(),
    nccl_lib: $("nccl-lib").value.trim(),
    log_root: $("log-root").value.trim(),
  };
  config.cluster = {
    ...config.cluster,
    http_bind: $("http-bind").value.trim(),
    base_port: numberValue("base-port", config.cluster.base_port),
    ssh_user: $("ssh-user").value.trim(),
    ssh_options: lines("ssh-options"),
    nodes: [],
  };
  for (const row of document.querySelectorAll("#node-rows tr")) {
    const rank = Number(row.dataset.rank);
    config.cluster.nodes[rank] = {
      ssh: row.querySelector('[data-field="ssh"]').value.trim(),
      peer: row.querySelector('[data-field="peer"]').value.trim(),
      local: row.querySelector('[data-field="local"]').checked,
    };
  }
  const gid = $("nccl-gid").value.trim();
  config.network = {
    nccl_ib_hca: $("nccl-ib-hca").value.trim(),
    nccl_socket_ifname: $("nccl-socket-ifname").value.trim(),
    nccl_ib_gid_index: gid === "" ? null : Number(gid),
    env: parseEnv($("extra-env").value),
  };
  config.web = {
    ...config.web,
    bind: $("web-bind").value.trim(),
    serve_url: $("serve-url").value.trim(),
    allowed_hosts: lines("allowed-hosts"),
    discovery_timeout_secs: numberValue("discovery-timeout", config.web.discovery_timeout_secs),
    ssh_timeout_secs: numberValue("ssh-timeout", config.web.ssh_timeout_secs),
    advertise: $("advertise").checked,
  };
  config.stats = {
    ...config.stats,
    sample_interval_secs: numberValue("stats-interval", config.stats.sample_interval_secs),
    retention_secs: numberValue("stats-retention", config.stats.retention_secs),
    flush_interval_secs: numberValue("stats-flush", config.stats.flush_interval_secs),
    persist_path: $("stats-path").value.trim(),
  };
  return config;
}

async function saveConfig() {
  const config = collectConfig();
  if (!config) return;
  const button = $("save-config");
  button.disabled = true;
  try {
    await api("/api/config", { method: "PUT", body: JSON.stringify({ config }) });
    clearDirty();
    toast("Configuration saved to " + (S.state?.config_path || "the lead node"));
    await refreshState();
    paintConfig(S.state.config, S.state.discovery, true);
    loadModels(true);
  } catch (error) {
    toast("Not saved: " + error.message, true);
  } finally {
    button.disabled = false;
  }
}

function discardConfig() {
  clearDirty();
  if (S.state) paintConfig(S.state.config, S.state.discovery, true);
  toast("Changes discarded");
}

async function reloadConfig() {
  if (S.configDirty) {
    const ok = await confirmAction({
      title: "Reload the file?",
      body: "Your unsaved changes in this form will be replaced by what is on disk.",
      ok: "Reload",
      danger: true,
    });
    if (!ok) return;
  }
  const button = $("reload-config");
  button.disabled = true;
  try {
    const result = await api("/api/config/reload", { method: "POST" });
    clearDirty();
    toast("Reloaded " + result.path);
    await refreshState();
    paintConfig(result.config, S.state?.discovery, true);
    loadModels(true);
  } catch (error) {
    toast("Could not reload: " + error.message, true);
  } finally {
    button.disabled = false;
  }
}

/* Discovery */

async function discover(quiet = false) {
  if (S.discoveryBusy) return;
  S.discoveryBusy = true;
  renderGuide();
  try {
    const report = await api("/api/discover", { method: "POST" });
    if (S.state) {
      S.state.discovery = report;
      if (!S.configDirty && S.state.config) paintConfig(S.state.config, report, true);
    }
    if (report.mdns_error && !quiet) toast("Could not search the network: " + report.mdns_error, true);
  } catch (error) {
    if (!quiet) toast("Could not search the network: " + error.message, true);
  } finally {
    S.discoveryBusy = false;
    renderGuide();
  }
}

/* Guided setup */

// The steps, in order. Each one decides from the latest node check whether it is done, so the page
// can be closed and reopened at any point.
const GUIDE = ["find", "network", "software", "download", "copy", "convert", "launch"];
const GUIDE_TITLES = {
  find: "Find your Sparks",
  network: "Connect the fast network",
  software: "Install on the other Sparks",
  download: "Download the model",
  copy: "Copy the model to the other Sparks",
  convert: "Prepare the weights",
  launch: "Start the server",
};

// What each step needs finished first.
const GUIDE_NEEDS = {
  find: [],
  network: ["find"],
  software: ["find"],
  download: ["find"],
  copy: ["download"],
  convert: ["software", "copy"],
  launch: ["network", "convert"],
};

// Jobs whose result changes what the node check finds.
const isSetupJob = (kind) =>
  ["download", "sync-software", "sync-model", "launch"].includes(kind) || kind.startsWith("convert-");

function latestJob(match) {
  let found = null;
  for (const job of S.jobs) {
    if (match(job) && (!found || job.id > found.id)) found = job;
  }
  return found;
}

const runningJob = (match) => S.jobs.find((job) => job.status === "running" && match(job));
const ofKind = (kind) => (job) => job.kind === kind;
const isConvert = (job) => job.kind.startsWith("convert-");

// "spark-1.local." and "Spark-1" are the same host.
function bareHost(name) {
  return String(name || "").trim().replace(/\.$/, "").replace(/\.local$/i, "").toLowerCase();
}

function sshTarget(host) {
  const user = S.state?.config?.cluster?.ssh_user;
  const target = user ? user + "@" + host : host;
  return /^[A-Za-z0-9@._:-]+$/.test(target) ? target : "'" + target.replaceAll("'", "'\\''") + "'";
}

// The node check, when it was taken of the Sparks the configuration names now.
function freshSetup(state) {
  const setup = state?.setup;
  const nodes = state?.config?.cluster?.nodes || [];
  if (!setup || !Array.isArray(setup.nodes) || setup.nodes.length !== nodes.length) return null;
  return setup.nodes.every((node, rank) => node.ssh === nodes[rank].ssh) ? setup : null;
}

function planApplied(config, plan) {
  if (!plan) return false;
  const network = config.network;
  return network.nccl_ib_hca === plan.nccl_ib_hca &&
    network.nccl_socket_ifname === plan.nccl_socket_ifname &&
    (network.nccl_ib_gid_index ?? null) === (plan.nccl_ib_gid_index ?? null) &&
    plan.peers.every((peer, rank) => config.cluster.nodes[rank]?.peer === peer);
}

function guideModel(state) {
  const config = state.config;
  const setup = freshSetup(state);
  const nodes = setup?.nodes || [];
  const online = !!(state.latest && state.latest.online);
  const saved = !!state.config_exists && !state.config_error;
  const configured = (state.discovery?.nodes || []).filter((node) => node.configured);
  const sshOk = configured.length === 4 && configured.every((node) => node.ssh_ok === true);
  const all = (field) => !!setup && nodes.every((node) => node[field] === "ready");
  const errors = (setup?.findings || []).filter((finding) => finding.severity === "error");
  const found = saved && ((!!setup && nodes.every((node) => node.reachable)) || sshOk);
  const networkErrors = errors.filter((finding) => finding.rank === null || finding.rank === undefined || nodes[finding.rank]?.reachable !== false);
  const steps = {
    find: { done: found },
    network: { done: found && !!setup?.plan && networkErrors.length === 0 && planApplied(config, setup.plan) },
    software: { done: found && all("software"), running: runningJob(ofKind("sync-software")) },
    download: { done: found && nodes[0]?.model === "ready", running: runningJob(ofKind("download")) },
    copy: { done: found && all("model"), running: runningJob(ofKind("sync-model")) },
    convert: { done: found && all("converted"), running: runningJob(isConvert) },
    launch: { done: online && !!state.latest.ready, running: runningJob(ofKind("launch")) },
  };
  for (const key of GUIDE) {
    steps[key].waitsFor = GUIDE_NEEDS[key].find((need) => !steps[need].done) || null;
    steps[key].available = !steps[key].waitsFor;
  }
  const current = GUIDE.find((key) => !steps[key].done) || null;
  return { setup, nodes, online, steps, current };
}

/* Step 1: the four Sparks */

// The other Sparks this one could work with: what mDNS found, what the configuration names, and
// anything typed in by hand. This Spark itself is rank 0 and never a choice.
function sparkCandidates(state) {
  const me = bareHost(state.hostname);
  const seen = new Map();
  const add = (candidate) => {
    const key = bareHost(candidate.ssh);
    if (key && !seen.has(key)) seen.set(key, candidate);
  };
  for (const node of state.discovery?.nodes || []) {
    if (node.local || bareHost(node.hostname) === me || bareHost(node.ssh) === me) continue;
    // Before the first save the configured names are placeholders; only show one mDNS also found.
    if (node.configured && !state.config_exists && !node.hostname) continue;
    add({
      ssh: node.ssh,
      name: node.hostname || node.ssh,
      address: (node.addresses && node.addresses[0]) || "",
      ok: node.ssh_ok,
      message: node.ssh_message || "",
    });
  }
  for (const host of S.manualHosts) add({ ssh: host, name: host, address: "", ok: null, message: "" });
  return Array.from(seen.values());
}

function ensurePick(state, candidates) {
  if (S.pick) return;
  if (state.config_exists) {
    S.pick = state.config.cluster.nodes.filter((node) => !node.local).map((node) => node.ssh);
  } else if (state.discovery) {
    S.pick = candidates.filter((candidate) => candidate.ok === true).slice(0, 3).map((candidate) => candidate.ssh);
  }
}

function sshProblem(message) {
  if (/resolve/i.test(message)) return "Can't find this name on the network";
  if (/host key/i.test(message)) return "Doesn't know this Spark yet";
  if (/timed out|no route|refused|no answer/i.test(message)) return "Not answering";
  return "Needs this Spark's key";
}

function sshBadge(candidate) {
  if (candidate.ok === true) return '<span class="ssh-state ok">Connected</span>';
  if (candidate.ok === false) return '<span class="ssh-state fail" title="' + esc(candidate.message) + '">' + esc(sshProblem(candidate.message)) + "</span>";
  return '<span class="ssh-state unknown">' + (S.discoveryBusy ? "Checking…" : "Not checked yet") + "</span>";
}

function codeBlock(heading, text) {
  return '<div class="codeblock"><div class="codeblock-head"><span>' + heading + "</span>" +
    '<button class="btn btn-ghost btn-xs" type="button" data-copy="' + esc(text) + '"><svg><use href="#i-copy"/></svg>Copy</button></div>' +
    "<pre>" + esc(text) + "</pre></div>";
}

function findBody(state) {
  const candidates = sparkCandidates(state);
  ensurePick(state, candidates);
  const picked = candidates.filter((candidate) => (S.pick || []).includes(candidate.ssh));
  const parts = [];
  parts.push('<p class="guide-lead">This Spark, <strong>' + esc(state.hostname) + "</strong>, runs this page and is the first of the four. " +
    "Tick the three others. They need to be switched on and on the same network as this one.</p>");
  if (candidates.length === 0) {
    parts.push('<p class="guide-empty">' + (S.discoveryBusy || !state.discovery
      ? "Looking for Sparks on your network…"
      : "No other Sparks answered. Check that they are on and connected, then press Search again, or type a name below.") + "</p>");
  } else {
    parts.push('<div class="picks">' + candidates.map((candidate) => {
      const checked = picked.includes(candidate);
      const sub = candidate.address && candidate.address !== candidate.name ? candidate.address : (candidate.name !== candidate.ssh ? candidate.ssh : "");
      return '<label class="pick' + (checked ? " on" : "") + '"><input type="checkbox" data-pick="' + esc(candidate.ssh) + '"' + (checked ? " checked" : "") + ">" +
        '<span class="pick-name">' + esc(candidate.name) + (sub ? '<small class="mono">' + esc(sub) + "</small>" : "") + "</span>" +
        sshBadge(candidate) + "</label>";
    }).join("") + "</div>");
  }
  parts.push('<div class="manual"><input id="g-manual" type="text" placeholder="Not listed? Type its name, e.g. spark-3.local" aria-label="Add a Spark by name">' +
    '<button class="btn btn-sm" type="button" data-act="add-host">Add</button></div>');

  const failing = picked.filter((candidate) => candidate.ok === false);
  if (failing.length > 0) {
    const command = ["[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519"]
      .concat(failing.map((candidate) => "ssh-copy-id " + sshTarget(candidate.ssh)))
      .join("\n");
    parts.push('<p class="guide-note">This Spark has to be allowed to sign in to ' + esc(failing.map((candidate) => candidate.name).join(", ")) +
      ". Open a terminal on this Spark, paste the lines below, and answer each question (type <code>yes</code>, then that Spark's password). Then press Search again.</p>");
    parts.push(codeBlock("Run once in a terminal on this Spark", command));
  }

  const count = picked.length;
  const ready = count === 3 && failing.length === 0;
  const hint = count === 3
    ? (failing.length ? "Fix the sign-in above first." : "")
    : "Tick " + (count < 3 ? 3 - count + " more" : count - 3 + " fewer") + ".";
  parts.push('<div class="guide-actions">' +
    '<button class="btn" type="button" data-act="scan"' + (S.discoveryBusy ? " disabled" : "") + '><svg><use href="#i-search"/></svg>' + (S.discoveryBusy ? "Searching…" : "Search again") + "</button>" +
    (hint ? '<span class="guide-hint">' + esc(hint) + "</span>" : "") +
    '<button class="btn btn-primary" type="button" data-act="use-sparks"' + (ready ? "" : " disabled") + ">Use these Sparks</button></div>");
  return parts.join("");
}

async function useSparks() {
  if (!requireSaved()) return;
  const state = S.state;
  const candidates = sparkCandidates(state);
  const picked = (S.pick || []).filter((ssh) => candidates.some((candidate) => candidate.ssh === ssh));
  if (picked.length !== 3) return;
  const config = JSON.parse(JSON.stringify(state.config));
  const lead = state.config_exists ? config.cluster.nodes.find((node) => node.local) : null;
  // The fast-network addresses stay what they were until step 2 fills in the real ones.
  const peers = config.cluster.nodes.map((node) => node.peer);
  config.cluster.nodes = [{ ssh: lead?.ssh || state.hostname + ".local", peer: peers[0], local: true }]
    .concat(picked.map((ssh, index) => ({ ssh, peer: peers[index + 1], local: false })));
  try {
    await api("/api/config", { method: "PUT", body: JSON.stringify({ config }) });
    toast("Saved. Checking the four Sparks…");
    S.manualHosts = [];
    await refreshState();
    paintConfig(S.state.config, S.state.discovery, true);
    await Promise.all([discover(true), inspect(true)]);
  } catch (error) {
    toast("Not saved: " + error.message, true);
  }
}

/* Step 2: the fast network */

function railText(rail) {
  const bits = [rail.netdev, rail.ip + (rail.prefix ? "/" + rail.prefix : "")];
  if (rail.rate_gbps) bits.push(rail.rate_gbps + " Gb/s");
  return bits.join(" · ");
}

function findingsList(findings, config) {
  if (!findings.length) return "";
  return '<ul class="findings">' + findings.map((finding) => {
    const who = finding.rank === null || finding.rank === undefined ? "" : (config.cluster.nodes[finding.rank]?.ssh || "rank " + finding.rank) + ": ";
    return '<li class="' + esc(finding.severity) + '">' + esc(who + finding.message) + "</li>";
  }).join("") + "</ul>";
}

function waitingForCheck() {
  return '<p class="guide-empty">' + (S.inspectBusy ? "Checking the four Sparks…" : (S.inspectError ? esc(S.inspectError) : "Not checked yet.")) + "</p>" +
    (S.inspectBusy ? "" : '<div class="guide-actions"><button class="btn" type="button" data-act="check">Check the Sparks</button></div>');
}

function networkBody(state, model) {
  const setup = model.setup;
  if (!setup) return waitingForCheck();
  const config = state.config;
  const parts = [];
  parts.push('<table class="table compact"><thead><tr><th>Spark</th><th>Fast ports</th><th></th></tr></thead><tbody>' +
    setup.nodes.map((node) => {
      const problems = setup.findings.filter((finding) => finding.rank === node.rank);
      const worst = problems.some((finding) => finding.severity === "error") ? "fail" : (problems.length ? "warn" : "ok");
      const label = !node.reachable ? "Not answering" : worst === "fail" ? "Needs setup" : worst === "warn" ? "Works, see below" : "Ready";
      const rails = node.rails.length ? node.rails.map((rail) => '<span class="mono">' + esc(railText(rail)) + "</span>").join("<br>") : '<span class="dim">none connected</span>';
      return '<tr><td><span class="host-name">' + esc(node.hostname || node.ssh) + '</span><span class="host-meta">rank ' + node.rank + "</span></td>" +
        "<td>" + rails + '</td><td class="num"><span class="ssh-state ' + (worst === "warn" ? "warnish" : worst) + '">' + esc(label) + "</span></td></tr>";
    }).join("") + "</tbody></table>");
  parts.push(findingsList(setup.findings, config));
  for (const node of setup.nodes) {
    if (!node.network_fix) continue;
    parts.push(codeBlock("Run on " + esc(node.hostname || node.ssh) + ": sign in there, or first type <code>ssh " + esc(node.ssh) + "</code> in a terminal on this Spark", node.network_fix));
  }
  if (setup.nodes.some((node) => node.network_fix)) {
    parts.push('<p class="guide-note">The commands set each port\'s address and the jumbo frames the fast network needs, and keep them after a restart. Press Check again when they have run.</p>');
  }
  const plan = setup.plan;
  if (plan) {
    const applied = planApplied(config, plan);
    parts.push('<p class="guide-note">NCCL will use <code>' + esc(plan.nccl_ib_hca) + "</code> over <code>" + esc(plan.nccl_socket_ifname) + "</code>" +
      (plan.nccl_ib_gid_index === null || plan.nccl_ib_gid_index === undefined ? "" : ", GID index " + plan.nccl_ib_gid_index) +
      ", reaching the Sparks at " + esc(plan.peers.join(", ")) + "." + (applied ? " These settings are saved." : "") + "</p>");
    if (!applied) {
      parts.push('<div class="guide-actions"><span class="guide-hint"></span><button class="btn btn-primary" type="button" data-act="apply-network">Use these settings</button></div>');
    }
  }
  return parts.join("");
}

async function applyNetwork() {
  if (!requireSaved()) return;
  try {
    await api("/api/network/apply", { method: "POST" });
    toast("Network settings saved");
    await refreshState();
    paintConfig(S.state.config, S.state.discovery, true);
  } catch (error) {
    toast(error.message, true);
  }
}

/* Steps 3 to 7: jobs */

function progressBar(job, fallback) {
  const progress = job?.progress;
  const total = progress?.total || 0;
  const fraction = total > 0 ? Math.min(1, progress.done / total) : null;
  const text = progress?.detail || job?.last_line || fallback;
  const percent = fraction === null ? null : Math.floor(fraction * 100);
  return '<div class="progress' + (fraction === null ? " indeterminate" : "") + '" role="progressbar" aria-valuemin="0" aria-valuemax="100"' +
    (percent === null ? "" : ' aria-valuenow="' + percent + '"') + ">" +
    '<div class="progress-track"><div class="progress-fill" style="width:' + (fraction === null ? 100 : (fraction * 100).toFixed(1)) + '%"></div></div>' +
    '<div class="progress-text"><span>' + esc(text) + "</span>" + (percent === null ? "" : "<strong>" + percent + "%</strong>") + "</div></div>";
}

// The last run of a step, when it failed and nothing has run since.
function failureNote(job, what) {
  if (!job || job.status !== "failed") return "";
  return '<p class="note warn">' + esc(what) + " stopped with an error" + (job.last_line ? ": " + esc(job.last_line) : ".") +
    ' <a href="#activity" data-job-log="' + job.id + '">See the log</a></p>';
}

const CHECK_TEXT = {
  software: { ready: "Installed", missing: "Not installed", outdated: "Older version", partial: "Unfinished", unknown: "Unknown" },
  model: { ready: "Copied", missing: "Not copied", outdated: "Different", partial: "Partly copied", unknown: "Unknown" },
  converted: { ready: "Prepared", missing: "Not prepared", outdated: "Out of date", partial: "Unfinished", unknown: "Unknown" },
};

function nodeChips(nodes, field, extra) {
  return '<div class="chips-row">' + nodes.map((node) => {
    const value = node[field];
    // Not there yet is the normal state before a step runs, so only a mismatch is coloured.
    const cls = value === "ready" ? "ok" : value === "outdated" ? "warnish" : "unknown";
    const note = extra ? extra(node) : "";
    return '<div class="node-chip"><span class="host-name">' + esc(node.hostname || node.ssh) + "</span>" +
      '<span class="ssh-state ' + cls + '">' + esc(note || CHECK_TEXT[field][value] || value) + "</span></div>";
  }).join("") + "</div>";
}

function actionRow(button, hint) {
  return '<div class="guide-actions">' + (hint ? '<span class="guide-hint">' + hint + "</span>" : '<span class="guide-hint"></span>') + button + "</div>";
}

function softwareBody(state, model) {
  const setup = model.setup;
  if (!setup) return waitingForCheck();
  const lead = setup.nodes[0];
  if (lead.software !== "ready") {
    return '<p class="guide-note">This Spark does not have the server built yet. Open a terminal on this Spark and run the setup script; it builds everything and restarts this page.</p>' +
      codeBlock("Run in a terminal on this Spark", "cd " + state.config.paths.repo_root + "\n./scripts/setup.sh");
  }
  const others = setup.nodes.slice(1);
  const running = model.steps.software.running;
  const parts = [nodeChips(others, "software")];
  if (running) {
    parts.push(progressBar(running, "Connecting…"));
  } else {
    parts.push(failureNote(latestJob(ofKind("sync-software")), "The install"));
    const update = others.some((node) => node.software === "outdated");
    const label = model.steps.software.done ? "Install again" : update ? "Update" : "Install";
    parts.push(actionRow('<button class="btn btn-primary" type="button" data-act="install">' + label + "</button>",
      "Copies the server, its GPU kernels and the NCCL library. Takes about a minute."));
  }
  return parts.join("");
}

function downloadBody(state, model) {
  const setup = model.setup;
  if (!setup) return waitingForCheck();
  const config = state.config;
  const lead = setup.nodes[0];
  const parts = [];
  const explicit = (config.model.checkpoint || "").trim();
  parts.push('<p class="guide-lead">Model <strong>' + esc(config.model.id) + "</strong>" +
    ' <a href="#setup" data-act="advanced">change</a></p>');
  if (explicit) {
    parts.push('<p class="guide-note">The model is read from <code>' + esc(explicit) + "</code> (set under Advanced settings), so there is nothing to download." +
      (lead.model === "ready" ? "" : " That folder is missing or incomplete on this Spark.") + "</p>");
    return parts.join("");
  }
  const facts = [];
  if (lead.model === "ready") facts.push("On this Spark: " + fmtInt(lead.model_files) + " files, " + formatSize(lead.model_bytes));
  else if (lead.model_bytes > 0) facts.push("Downloaded so far: " + formatSize(lead.model_bytes));
  if (lead.free_bytes !== null && lead.free_bytes !== undefined) facts.push(formatSize(lead.free_bytes) + " free");
  if (facts.length) parts.push('<p class="guide-facts">' + esc(facts.join(" · ")) + "</p>");
  const running = model.steps.download.running;
  if (running) {
    parts.push(progressBar(running, "Asking Hugging Face for the file list…"));
  } else if (lead.model !== "ready") {
    parts.push(failureNote(latestJob(ofKind("download")), "The download"));
    const resume = lead.model_bytes > 0;
    parts.push(actionRow('<button class="btn btn-primary" type="button" data-act="download">' + (resume ? "Resume download" : "Download") + "</button>",
      "Hundreds of gigabytes take hours. If it stops, press " + (resume ? "Resume" : "Download") + " again and it carries on where it left off."));
  }
  return parts.join("");
}

function copyBody(state, model) {
  const setup = model.setup;
  if (!setup) return waitingForCheck();
  const lead = setup.nodes[0];
  const others = setup.nodes.slice(1);
  const running = model.steps.copy.running;
  const parts = [nodeChips(others, "model", (node) =>
    node.model === "partial" && lead.model_bytes ? formatSize(node.model_bytes) + " of " + formatSize(lead.model_bytes) : "")];
  if (running) {
    parts.push(progressBar(running, "Connecting…"));
  } else if (!model.steps.copy.done) {
    parts.push(failureNote(latestJob(ofKind("sync-model")), "The copy"));
    const fast = model.steps.network.done;
    parts.push(actionRow('<button class="btn btn-primary" type="button" data-act="copy-model">' + (others.some((node) => node.model_bytes > 0) ? "Finish copying" : "Copy") + "</button>",
      fast ? "Uses the fast network; files already there are skipped." : "Finish step 2 first to use the fast network; the regular one is much slower."));
  }
  return parts.join("");
}

function convertBody(state, model) {
  const setup = model.setup;
  if (!setup) return waitingForCheck();
  const running = S.jobs.filter((job) => job.status === "running" && isConvert(job));
  const parts = [nodeChips(setup.nodes, "converted", (node) => {
    const job = running.find((entry) => entry.kind === "convert-r" + node.rank);
    return job ? "Preparing, " + fmtDuration(Date.now() - job.started_ms) : "";
  })];
  if (running.length > 0) {
    const label = running.length === 4
      ? "All four Sparks are converting their part of the model at the same time."
      : running.length + (running.length === 1 ? " Spark is" : " Sparks are") + " converting weights…";
    parts.push(progressBar(null, label));
    return parts.join("");
  }
  for (const rank of [0, 1, 2, 3]) {
    parts.push(failureNote(latestJob(ofKind("convert-r" + rank)), "Preparing " + (setup.nodes[rank]?.hostname || "rank " + rank)));
  }
  if (model.online) {
    parts.push('<p class="guide-note">Stop the server (on the Overview page) before preparing the weights again.</p>');
  } else {
    const label = model.steps.convert.done ? "Prepare again" : "Prepare the weights";
    parts.push(actionRow('<button class="btn btn-primary" type="button" data-act="convert">' + label + "</button>",
      "Each Spark converts its quarter of the model at the same time. This takes a while."));
  }
  return parts.join("");
}

function apiUrl(state) {
  const bind = state.config.cluster.http_bind || "0.0.0.0:8000";
  const port = bind.slice(bind.lastIndexOf(":") + 1) || "8000";
  const host = location.hostname.includes(":") ? "[" + location.hostname + "]" : location.hostname;
  return "http://" + host + ":" + port + "/v1";
}

function launchBody(state, model) {
  const parts = [];
  const running = model.steps.launch.running;
  if (model.steps.launch.done) {
    const url = apiUrl(state);
    const name = state.cluster?.model || state.config.model.id;
    parts.push('<p class="guide-lead">The server is running. Point any OpenAI-compatible app at this address:</p>');
    parts.push(codeBlock("API address", url));
    parts.push(codeBlock("Try it from a terminal",
      "curl " + url + "/chat/completions -H 'Content-Type: application/json' \\\n  -d '{\"model\": \"" + name + "\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"));
    parts.push('<p class="guide-note">Speed, requests and memory are on the <a href="#overview">Overview</a> page.</p>');
    return parts.join("");
  }
  if (running || model.online) {
    parts.push(progressBar(running, "Starting the four Sparks and loading the model (~25 seconds)."));
    return parts.join("");
  }
  parts.push(failureNote(latestJob(ofKind("launch")), "Starting the server"));
  parts.push(actionRow('<button class="btn btn-primary" type="button" data-act="launch"><svg><use href="#i-play"/></svg>Start the server</button>',
    "Loads the model on all four Sparks. It stays running until you stop it on the Overview page."));
  return parts.join("");
}

const GUIDE_BODIES = {
  find: findBody,
  network: networkBody,
  software: softwareBody,
  download: downloadBody,
  copy: copyBody,
  convert: convertBody,
  launch: launchBody,
};

function guideSummary(key, state, model) {
  const step = model.steps[key];
  const setup = model.setup;
  if (!step.done) {
    if (!step.available) return "Waits for step " + (GUIDE.indexOf(step.waitsFor) + 1) + ".";
    return null;
  }
  switch (key) {
    case "find":
      return state.config.cluster.nodes.map((node) => node.ssh).join(", ");
    case "network": {
      const warnings = (setup?.findings || []).filter((finding) => finding.severity === "warning").length;
      const lanes = setup && setup.nodes.length ? Math.min(...setup.nodes.map((node) => node.rails.length)) : 0;
      return lanes + (lanes === 1 ? " lane" : " lanes") + " per Spark" + (warnings ? ", " + warnings + (warnings === 1 ? " warning" : " warnings") : "");
    }
    case "software":
      return "Installed on all four Sparks.";
    case "download":
      return setup ? formatSize(setup.nodes[0].model_bytes) + " on this Spark." : "Done.";
    case "copy":
      return "On all four Sparks.";
    case "convert":
      return "Prepared on all four Sparks.";
    case "launch":
      return "Serving at " + apiUrl(state);
    default:
      return null;
  }
}

function guideTag(key, model) {
  const step = model.steps[key];
  if (step.done) return ["Done", "good"];
  if (step.running || (key === "find" && S.discoveryBusy) || (key === "launch" && model.online)) return ["Working", "run"];
  if (!step.available) return ["Waiting", ""];
  return key === model.current ? ["Next", "next"] : ["To do", ""];
}

function renderGuide() {
  const state = S.state;
  if (!state || !state.config) return;
  const model = guideModel(state);
  GUIDE.forEach((key) => {
    const step = model.steps[key];
    const element = $("g-" + key);
    // Open: what is left to do, and the last step once the server runs (it has the address).
    const open = S.guideOpen[key] ?? ((step.available && !step.done) || (key === "launch" && step.done));
    element.classList.toggle("done", step.done);
    element.classList.toggle("current", key === model.current);
    element.classList.toggle("blocked", !step.available);
    element.classList.toggle("open", open);
    const [tag, cls] = guideTag(key, model);
    const tagElement = $("g-" + key + "-tag");
    tagElement.textContent = tag;
    tagElement.className = "tag" + (cls ? " " + cls : "");
    const summary = guideSummary(key, state, model);
    const summaryElement = $("g-" + key + "-summary");
    if (!summaryElement.dataset.base) summaryElement.dataset.base = summaryElement.textContent;
    summaryElement.textContent = summary || summaryElement.dataset.base;
    // A finished step folds away; its heading opens it again.
    const head = element.querySelector(".guide-head");
    if (step.done) {
      head.tabIndex = 0;
      head.setAttribute("role", "button");
      head.setAttribute("aria-expanded", String(open));
    } else {
      head.removeAttribute("tabindex");
      head.removeAttribute("role");
      head.removeAttribute("aria-expanded");
    }
    const body = $("g-" + key + "-body");
    const html = open ? GUIDE_BODIES[key](state, model) : "";
    // Never rebuild a step under the cursor of someone typing in it.
    const active = document.activeElement;
    const typing = active && active.type === "text" && body.contains(active);
    if (S.guideHtml[key] !== html && !typing) {
      S.guideHtml[key] = html;
      body.innerHTML = html;
    }
  });
  $("setup-banner").classList.toggle("hidden", !model.current || model.online || S.tab === "setup");
  if (model.current) {
    $("setup-banner-step").textContent = "Next is step " + (GUIDE.indexOf(model.current) + 1) + ", " + GUIDE_TITLES[model.current].toLowerCase() + ".";
  }
}

async function inspect(quiet = false) {
  if (S.inspectBusy) return;
  S.inspectBusy = true;
  S.inspectedFor = JSON.stringify(S.state?.config?.cluster?.nodes || null);
  const button = $("setup-check");
  button.classList.add("busy");
  renderGuide();
  try {
    const report = await api("/api/inspect", { method: "POST" });
    if (S.state) S.state.setup = report;
    S.inspectError = null;
  } catch (error) {
    S.inspectError = error.message;
    if (!quiet) toast("Could not check the Sparks: " + error.message, true);
  } finally {
    S.inspectBusy = false;
    button.classList.remove("busy");
    renderGuide();
  }
}

// Checks the Sparks again when something changed what they hold: a setup job finished, the
// configuration names other Sparks than the last check, or the page just opened on an old check.
function maybeInspect() {
  const state = S.state;
  let finished = false;
  for (const job of S.jobs) {
    if (S.jobSeen.get(job.id) === "running" && job.status !== "running" && isSetupJob(job.kind)) finished = true;
    S.jobSeen.set(job.id, job.status);
  }
  if (!state || !state.config_exists || state.config_error || S.inspectBusy) return;
  const nodes = JSON.stringify(state.config.cluster.nodes);
  const stale = !freshSetup(state) && S.inspectedFor !== nodes;
  const old = !S.autoInspected && (!state.setup || state.now_ms - state.setup.checked_ms > 120000);
  S.autoInspected = true;
  if (finished || stale || old) inspect(true);
}

function checkAgain() {
  discover(false);
  if (S.state?.config_exists) inspect(false);
}

async function convertAll() {
  const model = guideModel(S.state);
  const nodes = model.setup?.nodes || [];
  const pending = nodes.filter((node) => node.converted !== "ready").map((node) => node.rank);
  const ranks = pending.length ? pending : [0, 1, 2, 3];
  if (!pending.length) {
    const ok = await confirmAction({
      title: "Prepare the weights again?",
      body: "Every Spark rewrites its converted weights from the downloaded model. Only needed after changing the model.",
      ok: "Prepare again",
    });
    if (!ok) return;
  }
  postAction("/api/actions/convert", { ranks }, false);
}

function bindGuide() {
  const guide = $("guide");
  guide.addEventListener("click", (event) => {
    const head = event.target.closest(".guide-head");
    if (head) {
      const key = head.parentElement.dataset.step;
      const step = S.state && guideModel(S.state).steps[key];
      if (step && step.done) {
        S.guideOpen[key] = !head.parentElement.classList.contains("open");
        renderGuide();
      }
      return;
    }
    const copy = event.target.closest("[data-copy]");
    if (copy) {
      copyText(copy.dataset.copy);
      return;
    }
    const log = event.target.closest("[data-job-log]");
    if (log) {
      event.preventDefault();
      S.selectedJob = Number(log.dataset.jobLog);
      S.jobDetail = null;
      if (location.hash === "#activity") showTab("activity");
      else location.hash = "activity";
      return;
    }
    const action = event.target.closest("[data-act]");
    if (!action || action.disabled) return;
    switch (action.dataset.act) {
      case "scan":
        discover(false);
        break;
      case "add-host": {
        const input = $("g-manual");
        const host = input.value.trim();
        if (!/^[A-Za-z0-9][A-Za-z0-9._:-]*$/.test(host)) {
          toast("Type a host name or address, e.g. spark-3.local", true);
          return;
        }
        if (!S.manualHosts.includes(host)) S.manualHosts.push(host);
        S.pick = (S.pick || []).filter((ssh) => ssh !== host).concat(host);
        input.value = "";
        input.blur();
        renderGuide();
        break;
      }
      case "use-sparks":
        useSparks();
        break;
      case "check":
        inspect(false);
        break;
      case "apply-network":
        applyNetwork();
        break;
      case "install":
        if (requireSaved()) postAction("/api/actions/sync", { what: "software" }, false);
        break;
      case "download":
        if (requireSaved()) postAction("/api/actions/download", undefined, false);
        break;
      case "copy-model":
        if (requireSaved()) postAction("/api/actions/sync", { what: "model" }, false);
        break;
      case "convert":
        if (requireSaved()) convertAll();
        break;
      case "launch":
        if (requireSaved()) postAction("/api/actions/launch", undefined, false);
        break;
      case "advanced":
        event.preventDefault();
        $("advanced").open = true;
        $("advanced").scrollIntoView({ behavior: "smooth", block: "start" });
        break;
      default:
        break;
    }
  });
  guide.addEventListener("change", (event) => {
    const box = event.target.closest("[data-pick]");
    if (!box) return;
    const ssh = box.dataset.pick;
    const pick = (S.pick || []).filter((entry) => entry !== ssh);
    S.pick = box.checked ? pick.concat(ssh) : pick;
    renderGuide();
  });
  guide.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.id === "g-manual") {
      event.preventDefault();
      guide.querySelector('[data-act="add-host"]').click();
    } else if ((event.key === "Enter" || event.key === " ") && event.target.classList.contains("guide-head")) {
      event.preventDefault();
      event.target.click();
    }
  });
}

/* Actions */

// Starts a job. `follow` opens its log; the setup steps pass false and show progress in place.
async function postAction(path, body, follow = true) {
  try {
    const result = await api(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const jobs = (result && Array.isArray(result.jobs)) ? result.jobs : [];
    if (jobs.length > 0) {
      toast(jobs.length === 1 ? jobs[0].label + " started" : jobs.length + " jobs started");
      if (follow) {
        S.selectedJob = jobs[0].id;
        S.jobDetail = null;
        location.hash = "activity";
      }
    } else {
      toast("Request accepted");
    }
    await refreshState();
  } catch (error) {
    toast(error.message, true);
  }
}

/* Telemetry */

async function refreshSeries() {
  const charts = document.querySelectorAll(".chart");
  charts.forEach((chart) => chart.classList.add("loading"));
  try {
    S.series = await api("/api/series?window_secs=" + S.windowSecs);
    renderCharts();
    renderDaily();
    renderSparks();
  } catch (error) {
    $("telemetry-hint").textContent = "Telemetry unavailable: " + error.message;
  } finally {
    charts.forEach((chart) => chart.classList.remove("loading"));
  }
}

function svgEl(tag, attrs, parent) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs || {})) element.setAttribute(name, value);
  if (parent) parent.appendChild(element);
  return element;
}

function niceMax(value) {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const exponent = Math.pow(10, Math.floor(Math.log10(value)));
  for (const step of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) {
    if (step * exponent >= value) return step * exponent;
  }
  return 10 * exponent;
}

// Points with their x position in a window ending at the newest sample, split wherever rank 0
// was offline so a gap reads as a gap instead of a slope.
function chartFrame() {
  const points = S.series?.points || [];
  const t1 = points.length ? points[points.length - 1].t_ms : Date.now();
  const t0 = t1 - S.windowSecs * 1000;
  return { points, t0, t1 };
}

function renderCharts() {
  const series = S.series;
  if (!series || S.tab !== "overview") return;
  const frame = chartFrame();
  const latest = [...frame.points].reverse().find((point) => point.online);
  for (const def of SERIES) {
    drawTimeChart(def, frame);
    $(def.now).textContent = latest ? fmtSeries(def, latest[def.key]) : "—";
  }
  const interval = S.state?.config?.stats?.sample_interval_secs;
  $("telemetry-hint").textContent = series.last_error
    ? "Sampling rank 0 failed: " + series.last_error
    : "Sampled from rank 0" + (interval ? " every " + interval + " s" : "") + ", outside the inference path.";
  if (S.hover) paintHover();
}

function drawTimeChart(def, frame) {
  const svg = $(def.chart);
  const width = svg.clientWidth || 600;
  const height = svg.clientHeight || 150;
  const pad = { left: 44, right: 6, top: 8, bottom: 22 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.replaceChildren();

  const online = frame.points.filter((point) => point.online);
  const valued = online.filter((point) => hasValue(point, def.key));
  const peak = Math.max(0, ...valued.map((point) => Number(point[def.key]) || 0));
  const max = def.percent ? 1 : niceMax(Math.max(peak * 1.05, 10));
  const x = (t) => pad.left + ((t - frame.t0) / (frame.t1 - frame.t0 || 1)) * plotW;
  const y = (value) => pad.top + plotH - (Math.max(0, Math.min(max, Number(value) || 0)) / max) * plotH;

  for (const fraction of [0, 0.5, 1]) {
    const gy = Math.round(pad.top + plotH - fraction * plotH) + 0.5;
    svgEl("line", { class: fraction === 0 ? "baseline" : "gridline", x1: pad.left, x2: width - pad.right, y1: gy, y2: gy }, svg);
    const label = svgEl("text", { class: "axis-label", x: pad.left - 8, y: gy + 3.5, "text-anchor": "end" }, svg);
    label.textContent = def.percent ? Math.round(fraction * 100) + "%" : fmtCompact(max * fraction);
  }
  const ticks = width < 420 ? 3 : 4;
  for (let index = 0; index <= ticks; index += 1) {
    const t = frame.t0 + (index / ticks) * (frame.t1 - frame.t0);
    const anchor = index === 0 ? "start" : index === ticks ? "end" : "middle";
    const label = svgEl("text", { class: "axis-label", x: x(t), y: height - 4, "text-anchor": anchor }, svg);
    label.textContent = fmtClock(t);
  }

  if (valued.length === 0) {
    const text = svgEl("text", { class: "empty-chart", x: pad.left + plotW / 2, y: pad.top + plotH / 2 + 4, "text-anchor": "middle" }, svg);
    text.textContent = online.length === 0 ? "No samples from rank 0 in this window" : def.none;
    if (online.length === 0) return;
  }

  const color = S.colors[def.color];
  let line = "";
  let area = "";
  let segment = [];
  const flush = () => {
    if (segment.length === 0) return;
    const path = segment.map(([px, py], index) => (index === 0 ? "M" : "L") + px.toFixed(1) + " " + py.toFixed(1)).join(" ");
    line += path + " ";
    const base = (pad.top + plotH).toFixed(1);
    area += path + " L" + segment[segment.length - 1][0].toFixed(1) + " " + base + " L" + segment[0][0].toFixed(1) + " " + base + " Z ";
    segment = [];
  };
  for (const point of frame.points) {
    if (!hasValue(point, def.key)) { flush(); continue; }
    segment.push([x(point.t_ms), y(point[def.key])]);
  }
  flush();
  svgEl("path", { class: "series-area", d: area, fill: color }, svg);
  svgEl("path", { class: "series-line", d: line, stroke: color }, svg);
  if (valued.length === 1) {
    svgEl("circle", { cx: x(valued[0].t_ms), cy: y(valued[0][def.key]), r: 3, fill: color }, svg);
  }

  const crosshair = svgEl("line", { class: "crosshair", y1: pad.top, y2: pad.top + plotH, visibility: "hidden" }, svg);
  const dot = svgEl("circle", { class: "hover-dot", r: 4, fill: color, visibility: "hidden" }, svg);
  svg._chart = { online, x, y, crosshair, dot, pad, plotW, t0: frame.t0, t1: frame.t1 };
}

function nearestIndex(online, t) {
  let low = 0;
  let high = online.length - 1;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (online[mid].t_ms < t) low = mid + 1;
    else high = mid;
  }
  if (low > 0 && Math.abs(online[low - 1].t_ms - t) < Math.abs(online[low].t_ms - t)) return low - 1;
  return low;
}

// One crosshair across every chart, and one tooltip listing every series at that moment.
// S.hover is { def, t, anchor } while a chart is hovered or focused.
function paintHover() {
  const hover = S.hover;
  let tipPoint = null;
  for (const def of SERIES) {
    const chart = $(def.chart)._chart;
    if (!chart) continue;
    const point = hover && chart.online.length > 0 ? chart.online[nearestIndex(chart.online, hover.t)] : null;
    const visible = point ? "visible" : "hidden";
    chart.crosshair.setAttribute("visibility", visible);
    chart.dot.setAttribute("visibility", point && hasValue(point, def.key) ? "visible" : "hidden");
    if (!point) continue;
    const px = chart.x(point.t_ms);
    chart.crosshair.setAttribute("x1", px);
    chart.crosshair.setAttribute("x2", px);
    chart.dot.setAttribute("cx", px);
    chart.dot.setAttribute("cy", chart.y(point[def.key]));
    if (def === hover.def) tipPoint = point;
  }
  const tip = $("chart-tip");
  if (!tipPoint || !hover.anchor) {
    tip.classList.add("hidden");
    return;
  }
  tip.replaceChildren();
  const time = document.createElement("div");
  time.className = "tip-time";
  time.textContent = new Date(tipPoint.t_ms).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  tip.appendChild(time);
  for (const def of SERIES) {
    const row = document.createElement("div");
    row.className = "tip-row" + (def === hover.def ? " current" : "");
    const key = document.createElement("span");
    key.className = "tip-key";
    key.style.background = S.colors[def.color];
    const value = document.createElement("strong");
    value.textContent = fmtSeries(def, tipPoint[def.key]);
    const label = document.createElement("span");
    label.textContent = def.label;
    row.append(key, value, label);
    tip.appendChild(row);
  }
  placeTip(hover.anchor);
}

function placeTip(anchor) {
  const tip = $("chart-tip");
  tip.classList.remove("hidden");
  const rect = tip.getBoundingClientRect();
  let left = anchor.x + 14;
  if (left + rect.width > window.innerWidth - 8) left = anchor.x - rect.width - 14;
  tip.style.left = Math.max(8, left) + "px";
  tip.style.top = Math.max(8, Math.min(window.innerHeight - rect.height - 8, anchor.y - rect.height / 2)) + "px";
}

function clearHover() {
  S.hover = null;
  paintHover();
}

function bindChartHover() {
  for (const def of SERIES) {
    const svg = $(def.chart);
    svg.setAttribute("tabindex", "0");
    svg.addEventListener("pointermove", (event) => {
      const chart = svg._chart;
      if (!chart || chart.online.length === 0) return;
      const box = svg.getBoundingClientRect();
      const viewX = ((event.clientX - box.left) / box.width) * (svg.viewBox.baseVal.width || box.width);
      const t = chart.t0 + ((viewX - chart.pad.left) / chart.plotW) * (chart.t1 - chart.t0);
      S.hover = { def, t, anchor: { x: event.clientX, y: event.clientY } };
      paintHover();
    });
    svg.addEventListener("pointerleave", clearHover);
    svg.addEventListener("blur", clearHover);
    svg.addEventListener("focus", () => {
      const chart = svg._chart;
      if (!chart || chart.online.length === 0) return;
      focusPoint(def, svg, chart.online.length - 1);
    });
    svg.addEventListener("keydown", (event) => {
      const chart = svg._chart;
      if (!chart || chart.online.length === 0 || !S.hover) return;
      const index = nearestIndex(chart.online, S.hover.t);
      if (event.key === "ArrowLeft") focusPoint(def, svg, Math.max(0, index - 1));
      else if (event.key === "ArrowRight") focusPoint(def, svg, Math.min(chart.online.length - 1, index + 1));
      else return;
      event.preventDefault();
    });
  }
}

function focusPoint(def, svg, index) {
  const chart = svg._chart;
  const t = chart.online[index].t_ms;
  const box = svg.getBoundingClientRect();
  const scale = box.width / (svg.viewBox.baseVal.width || box.width);
  S.hover = { def, t, anchor: { x: box.left + chart.x(t) * scale, y: box.top + box.height / 2 } };
  paintHover();
}

function renderSparks() {
  const points = (S.series?.points || []).slice(-120);
  const specs = SERIES.map((def) => ({ id: def.spark, key: def.key, color: S.colors[def.color], percent: def.percent }));
  specs.push({ id: "spark-requests", key: "running", color: S.colors["--text-2"] });
  for (const spec of specs) {
    const svg = $(spec.id);
    const width = svg.clientWidth || 160;
    const height = svg.clientHeight || 26;
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.replaceChildren();
    const valued = points.filter((point) => hasValue(point, spec.key));
    if (valued.length < 2) continue;
    const t0 = points[0].t_ms;
    const t1 = points[points.length - 1].t_ms;
    const peak = Math.max(...valued.map((point) => Number(point[spec.key]) || 0));
    const max = spec.percent ? 1 : Math.max(peak, 1e-9) * 1.1;
    let d = "";
    let pen = "M";
    for (const point of points) {
      if (!hasValue(point, spec.key)) { pen = "M"; continue; }
      const px = ((point.t_ms - t0) / (t1 - t0 || 1)) * width;
      const py = height - 1 - ((Number(point[spec.key]) || 0) / max) * (height - 2);
      d += pen + px.toFixed(1) + " " + py.toFixed(1) + " ";
      pen = "L";
    }
    svgEl("path", { d, stroke: spec.color }, svg);
  }
}

function roundedTopBar(x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height);
  return "M" + x + " " + (y + height) +
    " V" + (y + r) + " Q" + x + " " + y + " " + (x + r) + " " + y +
    " H" + (x + width - r) + " Q" + (x + width) + " " + y + " " + (x + width) + " " + (y + r) +
    " V" + (y + height) + " Z";
}

function renderDaily() {
  const series = S.series;
  if (!series || S.tab !== "overview") return;
  const svg = $("chart-daily");
  const empty = $("daily-empty");
  const days = Object.entries(series.totals.tokens_by_day || {}).sort(([a], [b]) => a.localeCompare(b)).slice(-14);
  $("daily-total").textContent = fmtCompact(series.totals.prompt_tokens + series.totals.generation_tokens);
  $("kpi-today").textContent = fmtCompact(series.totals.tokens_by_day?.[utcToday()] || 0);
  svg.replaceChildren();
  if (days.length === 0) {
    svg.classList.add("hidden");
    empty.classList.remove("hidden");
    return;
  }
  svg.classList.remove("hidden");
  empty.classList.add("hidden");
  const width = svg.clientWidth || 480;
  const height = svg.clientHeight || 190;
  const pad = { left: 44, right: 4, top: 10, bottom: 24 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  const max = niceMax(Math.max(...days.map(([, value]) => Number(value) || 0), 1) * 1.05);
  for (const fraction of [0, 0.5, 1]) {
    const gy = Math.round(pad.top + plotH - fraction * plotH) + 0.5;
    svgEl("line", { class: fraction === 0 ? "baseline" : "gridline", x1: pad.left, x2: width - pad.right, y1: gy, y2: gy }, svg);
    const label = svgEl("text", { class: "axis-label", x: pad.left - 8, y: gy + 3.5, "text-anchor": "end" }, svg);
    label.textContent = fmtCompact(max * fraction);
  }
  const slot = plotW / Math.max(days.length, 7);
  const barWidth = Math.max(6, Math.min(36, slot - 6));
  const labelEvery = slot < 38 ? 2 : 1;
  days.forEach(([day, value], index) => {
    const bx = pad.left + slot * index + (slot - barWidth) / 2;
    const barHeight = Math.max(1, (Number(value) / max) * plotH);
    const group = svgEl("g", { class: "bar-group", tabindex: "0", "aria-label": day + ": " + fmtInt(value) + " tokens" }, svg);
    svgEl("rect", { class: "bar-hit", x: pad.left + slot * index, y: pad.top, width: slot, height: plotH }, group);
    svgEl("path", { class: "bar", d: roundedTopBar(bx, pad.top + plotH - barHeight, barWidth, barHeight, 3) }, group);
    if ((days.length - 1 - index) % labelEvery === 0) {
      const label = svgEl("text", { class: "axis-label", x: bx + barWidth / 2, y: height - 6, "text-anchor": "middle" }, svg);
      label.textContent = day.slice(5);
    }
    const show = (anchor) => showBarTip(day, value, anchor);
    group.addEventListener("pointermove", (event) => show({ x: event.clientX, y: event.clientY }));
    group.addEventListener("pointerleave", hideTip);
    group.addEventListener("focus", () => {
      const box = group.getBoundingClientRect();
      show({ x: box.right, y: box.top + box.height / 2 });
    });
    group.addEventListener("blur", hideTip);
  });
}

function showBarTip(day, value, anchor) {
  const tip = $("chart-tip");
  tip.replaceChildren();
  const time = document.createElement("div");
  time.className = "tip-time";
  time.textContent = day + (day === utcToday() ? " (today, UTC)" : " (UTC)");
  const row = document.createElement("div");
  row.className = "tip-row current";
  const key = document.createElement("span");
  key.className = "tip-key";
  key.style.background = S.colors["--s1"];
  const strong = document.createElement("strong");
  strong.textContent = fmtInt(value);
  const label = document.createElement("span");
  label.textContent = "tokens";
  row.append(key, strong, label);
  tip.append(time, row);
  placeTip(anchor);
}

function hideTip() {
  $("chart-tip").classList.add("hidden");
}

/* Ranks */

function renderCluster(state) {
  const body = $("cluster-nodes");
  const error = $("cluster-error");
  const online = !!(state.latest && state.latest.online);
  if (state.last_error && online) {
    error.textContent = "Last sample: " + state.last_error;
    error.classList.remove("hidden");
  } else {
    error.classList.add("hidden");
  }
  const cluster = state.cluster;
  const fabric = $("fabric-badge");
  fabric.textContent = cluster?.fabric_mode || "no snapshot";
  fabric.title = "Collective fabric";
  $("ranks-sub").textContent = cluster
    ? (online ? "Live from rank 0." : "Last snapshot before rank 0 went offline.")
    : "Reported by rank 0 while it serves.";
  if (!cluster || !cluster.nodes || cluster.nodes.length === 0) {
    const message = cluster
      ? "Rank 0 answers, but this server build does not report per-rank memory, clocks or rails."
      : "No cluster snapshot yet. Launch the cluster and this table fills in.";
    body.innerHTML = '<tr><td colspan="5" class="empty">' + message + "</td></tr>";
    return;
  }
  body.innerHTML = cluster.nodes.map((node) => {
    const total = Number(node.uma_total_gib) || 0;
    const free = Number(node.uma_free_gib) || 0;
    const used = total > 0 ? Math.min(1, Math.max(0, (total - free) / total)) : 0;
    const tight = total > 0 && free / total < 0.08;
    const rail = node.rail0_speed_mbps ? Math.round(node.rail0_speed_mbps / 1000) + " Gb/s" : "—";
    return "<tr>" +
      '<td class="rank-id">R' + esc(node.rank) + "</td>" +
      '<td><span class="host-name">' + esc(node.hostname || "rank " + node.rank) + "</span>" +
      '<span class="host-meta">driver ' + esc(node.driver_version || "unknown") + "</span></td>" +
      '<td><div class="mem" title="' + esc(fmtPercent(used)) + ' used">' +
      '<div class="mem-track"><div class="mem-fill' + (tight ? " tight" : "") + '" style="width:' + (used * 100).toFixed(1) + '%"></div></div>' +
      '<span class="mem-text">' + free.toFixed(1) + " of " + total.toFixed(0) + " GiB</span></div></td>" +
      '<td class="num">' + fmtInt(node.gpu_clock_mhz) + " MHz</td>" +
      '<td class="num">' + rail + "</td></tr>";
  }).join("");
}

/* Jobs */

function renderJobs() {
  const container = $("jobs");
  $("job-count").textContent = S.jobs.length;
  if (S.jobs.length === 0) {
    container.innerHTML = '<p class="empty">Nothing has run yet. Downloads, copies, conversions and launches show their output here.</p>';
    return;
  }
  if (S.selectedJob === null || !S.jobs.some((job) => job.id === S.selectedJob)) {
    S.selectedJob = S.jobs[0].id;
    S.jobDetail = null;
  }
  const signature = JSON.stringify([S.selectedJob, S.jobs]);
  if (container.dataset.signature !== signature) {
    container.dataset.signature = signature;
    container.innerHTML = S.jobs.map((job) => {
      const selected = job.id === S.selectedJob;
      const progress = job.status === "running" ? job.progress : null;
      const sub = job.status + " · " + (progress?.detail || job.last_line || job.command);
      const bar = progress && progress.total > 0
        ? '<span class="job-bar" aria-hidden="true"><span style="width:' + Math.min(100, (progress.done / progress.total) * 100).toFixed(1) + '%"></span></span>'
        : "";
      return '<div class="job ' + esc(job.status) + (selected ? " selected" : "") + '" data-job="' + job.id + '" tabindex="0" role="option" aria-selected="' + selected + '">' +
        '<span class="job-dot" aria-hidden="true"></span>' +
        '<span class="job-title">' + esc(job.label) + "</span>" +
        '<span class="job-time">' + esc(fmtClock(job.started_ms, true)) + "</span>" +
        '<span class="job-sub" title="' + esc(job.command) + '">' + esc(sub) + "</span>" + bar + "</div>";
    }).join("");
    for (const element of container.querySelectorAll("[data-job]")) {
      const id = Number(element.dataset.job);
      element.addEventListener("click", () => selectJob(id));
      element.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectJob(id);
        }
      });
    }
  }
  if (S.tab === "activity") {
    const summary = S.jobs.find((job) => job.id === S.selectedJob);
    const stale = !S.jobDetail || S.jobDetail.id !== S.selectedJob || S.jobDetail.status === "running" ||
      (summary && summary.status !== S.jobDetail.status);
    if (stale) refreshJobDetail(S.selectedJob);
  }
}

function selectJob(id) {
  if (S.selectedJob === id) return;
  S.selectedJob = id;
  S.jobDetail = null;
  renderJobs();
}

async function refreshJobDetail(id, force = false) {
  if (S.detailBusy && !force) return;
  S.detailBusy = true;
  try {
    const job = await api("/api/jobs/" + id);
    if (S.selectedJob !== id) return;
    S.jobDetail = job;
    $("log-title").textContent = job.label;
    const end = job.finished_ms || Date.now();
    const parts = ["#" + job.id, job.status, "started " + fmtClock(job.started_ms, true), fmtDuration(end - job.started_ms)];
    if (job.exit_code !== null && job.exit_code !== undefined) parts.push("exit " + job.exit_code);
    $("log-meta").textContent = parts.join(" · ");
    const copy = $("copy-command");
    copy.classList.remove("hidden");
    copy.dataset.command = job.command;
    const log = $("job-log");
    const pinned = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.textContent = (job.log || []).join("\n") || "Waiting for output…";
    if (pinned) log.scrollTop = log.scrollHeight;
  } catch (error) {
    console.error(error);
  } finally {
    S.detailBusy = false;
  }
}

/* Wiring */

function readColors() {
  const style = getComputedStyle(document.documentElement);
  for (const name of ["--s1", "--s2", "--s3", "--s4", "--s5", "--text-2"]) {
    S.colors[name] = style.getPropertyValue(name).trim();
  }
}

function bindEvents() {
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
  $("save-config").addEventListener("click", saveConfig);
  $("discard-config").addEventListener("click", discardConfig);
  $("reload-config").addEventListener("click", reloadConfig);
  $("setup-check").addEventListener("click", checkAgain);
  bindGuide();
  $("copy-command").addEventListener("click", (event) => copyText(event.currentTarget.dataset.command || ""));

  $("action-launch").addEventListener("click", async () => {
    if (!requireSaved()) return;
    const ok = await confirmAction({
      title: "Launch the cluster?",
      body: "Starts all four ranks with the saved configuration. Rank 0 answers on its serve URL once the weights are loaded.",
      ok: "Launch cluster",
    });
    if (ok) postAction("/api/actions/launch");
  });
  $("action-stop").addEventListener("click", async () => {
    if (!requireSaved()) return;
    const running = S.state?.latest?.running || 0;
    const ok = await confirmAction({
      title: "Stop the cluster?",
      body: (running > 0 ? running + " request(s) are in flight. " : "") +
        "Rank 0 drains in-flight requests, then all four ranks exit. Clients get errors until the next launch.",
      ok: "Stop cluster",
      danger: true,
    });
    if (ok) postAction("/api/actions/stop");
  });
  $("action-status").addEventListener("click", () => {
    if (requireSaved()) postAction("/api/actions/status");
  });

  $("rescan-models").addEventListener("click", () => loadModels());
  $("model-id").addEventListener("change", paintModelSource);
  $("checkpoint").addEventListener("input", paintModelSource);

  const form = $("config-form");
  form.addEventListener("input", markDirty);
  form.addEventListener("change", markDirty);
  form.addEventListener("submit", (event) => event.preventDefault());
  for (const chip of document.querySelectorAll("[data-context]")) {
    chip.addEventListener("click", () => {
      $("max-context").value = chip.dataset.context;
      markDirty();
    });
  }
  for (const button of document.querySelectorAll("[data-window]")) {
    button.addEventListener("click", () => {
      S.windowSecs = Number(button.dataset.window);
      for (const other of document.querySelectorAll("[data-window]")) {
        other.classList.toggle("active", other === button);
        other.setAttribute("aria-pressed", String(other === button));
      }
      clearHover();
      refreshSeries();
    });
  }
  window.addEventListener("beforeunload", (event) => {
    if (S.configDirty) event.preventDefault();
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      renderCharts();
      renderDaily();
      renderSparks();
    }, 120);
  });
  bindChartHover();
}

document.addEventListener("DOMContentLoaded", () => {
  readColors();
  bindEvents();
  showTab(location.hash.slice(1));
  refreshState();
  refreshSeries();
  loadModels(true);
  setInterval(refreshState, 2500);
  setInterval(() => {
    if (!document.hidden) refreshSeries();
  }, 5000);
});
