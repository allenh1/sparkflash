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
  models: null,
  autoDiscovered: false,
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
    renderState();
    if (!S.autoDiscovered && !S.state.discovery) {
      S.autoDiscovered = true;
      discover(true);
    }
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
  $("action-convert").disabled = online || converting || clusterBusy;
  for (const button of document.querySelectorAll("[data-convert-rank]")) {
    button.disabled = online || clusterBusy;
  }
  const count = $("tab-running");
  count.textContent = running.length;
  count.classList.toggle("hidden", running.length === 0);

  $("max-context").max = state.limits?.max_context_tokens || 1048576;
  $("max-concurrency").max = state.limits?.max_concurrency || 32;

  renderCluster(state);
  renderDiscovery(state.discovery);
  renderSteps(state);
  renderJobs();
  if (!S.configDirty) paintConfig(state.config, state.discovery);
}

function renderSteps(state) {
  const jobs = S.jobs;
  const convertDone = [0, 1, 2, 3].every((rank) =>
    jobs.some((job) => job.kind === "convert-r" + rank && job.status === "succeeded")
  );
  const convertActive = jobs.some((job) => job.kind.startsWith("convert-") && job.status === "running");
  const launchActive = jobs.some((job) => job.kind === "launch" && job.status === "running");
  const configured = (state.discovery?.nodes || []).filter((node) => node.configured);
  const sshDone = configured.length === 4 && configured.every((node) => node.ssh_ok === true);
  const online = !!(state.latest && state.latest.online);
  const configDone = !!state.config_exists && !state.config_error;
  const steps = [
    ["step-config", configDone, false],
    ["step-ssh", sshDone, S.discoveryBusy],
    ["step-convert", convertDone, convertActive],
    ["step-launch", online, launchActive],
  ];
  let done = 0;
  for (const [id, complete, active] of steps) {
    const element = $(id);
    element.classList.toggle("done", complete);
    element.classList.toggle("active", !complete && active);
    if (complete) done += 1;
  }
  const progress = $("setup-progress");
  progress.textContent = done + " / 4";
  progress.className = "tag" + (done === 4 ? " good" : "");
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
  const button = $("discover");
  button.disabled = true;
  button.classList.add("busy");
  button.querySelector("span").textContent = "Scanning…";
  if (S.state) renderSteps(S.state);
  try {
    const report = await api("/api/discover", { method: "POST" });
    if (S.state) {
      S.state.discovery = report;
      renderDiscovery(report);
      if (!S.configDirty && S.state.config) paintConfig(S.state.config, report, true);
    }
    if (report.mdns_error && !quiet) toast("mDNS: " + report.mdns_error, true);
    else if (!quiet) toast("Found " + report.discovered + " mDNS host(s) and probed the configured ranks.");
  } catch (error) {
    if (!quiet) toast("Discovery failed: " + error.message, true);
  } finally {
    S.discoveryBusy = false;
    button.disabled = false;
    button.classList.remove("busy");
    button.querySelector("span").textContent = "Rescan";
    if (S.state) renderSteps(S.state);
  }
}

function renderDiscovery(report) {
  const body = $("discovery-rows");
  const signature = JSON.stringify(report || null);
  if (body.dataset.signature === signature) return;
  body.dataset.signature = signature;
  if (!report || !report.nodes || report.nodes.length === 0) {
    body.innerHTML = '<tr><td colspan="4" class="empty">' + (report ? "Nothing answered." : "No scan yet.") + "</td></tr>";
    $("fix-wrap").classList.add("hidden");
    return;
  }
  body.innerHTML = report.nodes.map((node) => {
    const host = node.hostname || node.ssh || "unknown";
    const address = (node.addresses && node.addresses[0]) || node.peer || "—";
    const tags = (node.rank !== null && node.rank !== undefined ? ' <span class="tag">R' + node.rank + "</span>" : "") +
      (node.local ? ' <span class="tag">this node</span>' : "");
    let action;
    if (node.rank !== null && node.rank !== undefined) {
      const target = (node.addresses && node.addresses[0]) || node.hostname || node.ssh;
      action = '<button class="btn btn-ghost btn-xs" type="button" data-use-rank="' + node.rank + '" data-host="' + esc(target) +
        '" title="Put this address in rank ' + node.rank + "'s SSH host\">Use for R" + node.rank + "</button>";
    } else {
      action = '<button class="btn btn-ghost btn-xs" type="button" data-copy-host="' + esc(node.ssh) + '">Copy</button>';
    }
    return '<tr><td><span class="host-name">' + esc(host) + "</span>" + tags + "</td>" +
      "<td>" + esc(address) + "</td><td>" + sshState(node) + '</td><td class="num">' + action + "</td></tr>";
  }).join("");

  body.querySelectorAll("[data-use-rank]").forEach((button) => {
    button.addEventListener("click", () => useDiscovered(Number(button.dataset.useRank), button.dataset.host));
  });
  body.querySelectorAll("[data-copy-host]").forEach((button) => {
    button.addEventListener("click", () => copyText(button.dataset.copyHost));
  });

  const fix = report.fix_command || "";
  $("fix-wrap").classList.toggle("hidden", !fix);
  $("fix-command").textContent = fix;
}

function useDiscovered(rank, host) {
  const row = document.querySelector('#node-rows tr[data-rank="' + rank + '"]');
  if (!row) return;
  const ssh = row.querySelector('[data-field="ssh"]');
  if (host && ssh.value !== host) {
    ssh.value = host;
    markDirty();
    toast("Rank " + rank + " now uses " + host + ". Save the configuration to keep it.");
  } else {
    toast("Rank " + rank + " already uses " + host + ".");
  }
}

/* Actions */

async function postAction(path, body) {
  try {
    const result = await api(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const jobs = (result && Array.isArray(result.jobs)) ? result.jobs : [];
    if (jobs.length > 0) {
      toast(jobs.length === 1 ? jobs[0].label + " started" : jobs.length + " jobs started");
      S.selectedJob = jobs[0].id;
      S.jobDetail = null;
      location.hash = "activity";
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
    container.innerHTML = '<p class="empty">No jobs yet. Launch, stop or convert and the output streams here.</p>';
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
      const sub = job.status + " · " + (job.last_line || job.command);
      return '<div class="job ' + esc(job.status) + (selected ? " selected" : "") + '" data-job="' + job.id + '" tabindex="0" role="option" aria-selected="' + selected + '">' +
        '<span class="job-dot" aria-hidden="true"></span>' +
        '<span class="job-title">' + esc(job.label) + "</span>" +
        '<span class="job-time">' + esc(fmtClock(job.started_ms, true)) + "</span>" +
        '<span class="job-sub" title="' + esc(job.command) + '">' + esc(sub) + "</span></div>";
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
  $("discover").addEventListener("click", () => discover(false));
  $("copy-fix").addEventListener("click", () => copyText($("fix-command").textContent));
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
  $("action-convert").addEventListener("click", async () => {
    if (!requireSaved()) return;
    const ok = await confirmAction({
      title: "Convert all four ranks?",
      body: "Rewrites every node's converted weights from the checkpoint. It takes several minutes per rank and the cluster must stay stopped until all four finish.",
      ok: "Convert all ranks",
    });
    if (ok) postAction("/api/actions/convert", { ranks: [0, 1, 2, 3] });
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
