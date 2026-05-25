const state = {
  drivers: [],
  specialists: [],
  sessions: [],
  currentSessionId: null,
  currentRunId: null,
  runs: new Map(),
  messages: [],
  selectedArtifactId: null,
  activeTab: "artifacts",
  pollTimer: null,
};

const payloadPresets = {
  stockfish: {
    label: "Stockfish",
    task_type: "chess_eval",
    input_payload: {
      query: "Analyze this position with Stockfish.",
      fen: "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",
      depth: 12,
    },
  },
  sympy: {
    label: "SymPy",
    task_type: "symbolic_math",
    input_payload: {
      operation: "simplify",
      expression: "(x + 1)^2",
    },
  },
  blast: {
    label: "BLAST",
    task_type: "sequence_alignment",
    input_payload: {
      database: "/path/to/local/blast/db",
      query_fasta: ">query1\nACGTACGTACGT",
      task: "blastn-short",
      evalue: "1e-5",
      max_target_seqs: 10,
    },
  },
  timesfm: {
    label: "TimesFM",
    task_type: "demand_forecast",
    input_payload: {
      values: [120, 128, 131, 129, 136, 142, 145, 151],
      horizon: 4,
      period: "week",
      current_inventory: 600,
    },
  },
  terraform_plan: {
    label: "Terraform plan",
    task_type: "infrastructure_plan",
    input_payload: {
      operation: "plan",
      query: "Tell me what would change in the local Terraform fixture.",
    },
  },
  stockfish_config: {
    label: "Stockfish config",
    task_type: "chess_eval",
    input_payload: {
      operation: "configure",
      configuration_template: true,
      query: "Give me a Stockfish configuration.",
    },
  },
  sympy_config: {
    label: "SymPy config",
    task_type: "symbolic_math",
    input_payload: {
      operation: "configure",
      configuration_template: true,
      query: "Give me a SymPy configuration.",
    },
  },
  blast_config: {
    label: "BLAST config",
    task_type: "sequence_alignment",
    input_payload: {
      operation: "configure",
      configuration_template: true,
      query: "Give me a BLAST configuration.",
    },
  },
  terraform_config: {
    label: "Terraform config",
    task_type: "infrastructure_plan",
    input_payload: {
      operation: "configure",
      configuration_template: true,
      query: "Give me a Terraform plan-only configuration.",
    },
  },
  timesfm_config: {
    label: "TimesFM config",
    task_type: "demand_forecast",
    input_payload: {
      operation: "configure",
      configuration_template: true,
      query: "Give me a TimesFM demand forecast configuration.",
    },
  },
};

const el = {
  shell: document.getElementById("appShell"),
  sidebar: document.getElementById("sidebar"),
  messages: document.getElementById("messages"),
  working: document.getElementById("workingIndicator"),
  workingText: document.getElementById("workingText"),
  stepStrip: document.getElementById("stepStrip"),
  prompt: document.getElementById("promptInput"),
  form: document.getElementById("composerForm"),
  send: document.getElementById("sendButton"),
  driverSelect: document.getElementById("driverSelect"),
  driverMeta: document.getElementById("driverMeta"),
  costPill: document.getElementById("costPill"),
  sessionList: document.getElementById("sessionList"),
  specialistList: document.getElementById("specialistList"),
  payloadToggle: document.getElementById("payloadToggle"),
  payloadBox: document.getElementById("payloadBox"),
  payloadInput: document.getElementById("payloadInput"),
  payloadPreset: document.getElementById("payloadPreset"),
  applyPayloadPreset: document.getElementById("applyPayloadPreset"),
  payloadWarning: document.getElementById("payloadWarning"),
  drawer: document.getElementById("drawer"),
  runMeta: document.getElementById("runMeta"),
  artifactList: document.getElementById("artifactList"),
  artifactViewer: document.getElementById("artifactViewer"),
  timeline: document.getElementById("timeline"),
  rawList: document.getElementById("rawList"),
  toolState: document.getElementById("toolState"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function init() {
  bindEvents();
  renderPayloadPresets();
  applyInitialViewportState();
  await Promise.all([loadDrivers(), loadSpecialists(), loadSessions()]);
  if (!state.currentSessionId) {
    await newSession();
  }
  renderAll();
}

function renderPayloadPresets() {
  for (const [id, preset] of Object.entries(payloadPresets)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = preset.label;
    el.payloadPreset.append(option);
  }
}

function applyInitialViewportState() {
  el.shell.classList.remove("sidebar-collapsed", "drawer-collapsed");
  if (window.innerWidth <= 760) {
    el.shell.classList.add("sidebar-collapsed", "drawer-collapsed");
  } else if (window.innerWidth <= 1120) {
    el.shell.classList.add("drawer-collapsed");
  }
}

function bindEvents() {
  document.getElementById("collapseSidebar").addEventListener("click", () => {
    el.shell.classList.add("sidebar-collapsed");
  });
  document.getElementById("openSidebar").addEventListener("click", () => {
    el.shell.classList.toggle("sidebar-collapsed");
  });
  document.getElementById("toggleDrawer").addEventListener("click", () => {
    el.shell.classList.toggle("drawer-collapsed");
  });
  document.getElementById("closeDrawer").addEventListener("click", () => {
    el.shell.classList.add("drawer-collapsed");
  });
  document.getElementById("newSession").addEventListener("click", newSession);
  el.payloadToggle.addEventListener("click", () => {
    el.payloadBox.hidden = !el.payloadBox.hidden;
    updatePayloadWarning();
  });
  el.applyPayloadPreset.addEventListener("click", applySelectedPayloadPreset);
  el.payloadPreset.addEventListener("change", updatePayloadWarning);
  el.payloadInput.addEventListener("input", updatePayloadWarning);
  el.driverSelect.addEventListener("change", renderDriverMeta);
  document.getElementById("tabs").addEventListener("click", (event) => {
    const button = event.target.closest(".tab");
    if (!button) return;
    state.activeTab = button.dataset.tab;
    renderTabs();
  });
  el.form.addEventListener("submit", submitPrompt);
  el.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.form.requestSubmit();
    }
  });
  el.prompt.addEventListener("input", autosizePrompt);
  el.prompt.addEventListener("input", updatePayloadWarning);
}

async function loadDrivers() {
  const payload = await api("/api/drivers");
  state.drivers = payload.drivers || [];
  renderDriverSelect();
}

async function loadSpecialists() {
  const payload = await api("/api/specialists");
  state.specialists = payload.specialists || [];
  renderSpecialists();
}

async function loadSessions() {
  const payload = await api("/api/sessions");
  state.sessions = payload.sessions || [];
  if (!state.currentSessionId && state.sessions[0]) {
    state.currentSessionId = state.sessions[0].session_id;
  }
  renderSessions();
}

async function newSession() {
  const payload = await api("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ title: "New conversation" }),
  });
  state.currentSessionId = payload.session.session_id;
  state.sessions.unshift(payload.session);
  state.currentRunId = null;
  state.messages = [];
  state.selectedArtifactId = null;
  renderAll();
}

async function submitPrompt(event) {
  event.preventDefault();
  const prompt = el.prompt.value.trim();
  if (!prompt) return;
  const payload = parsePayloadInput();
  if (payload.error) {
    addLocalAssistant(`Payload JSON error: ${payload.error}`);
    return;
  }
  el.prompt.value = "";
  autosizePrompt();
  state.messages.push({ role: "user", text: prompt });
  state.messages.push({ role: "assistant", text: "Working...", runId: null });
  renderMessages();
  el.send.disabled = true;
  try {
    const response = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        driver: el.driverSelect.value,
        session_id: state.currentSessionId,
        payload: payload.value,
      }),
    });
    const run = response.run;
    state.currentRunId = run.run_id;
    state.runs.set(run.run_id, run);
    state.messages[state.messages.length - 1].runId = run.run_id;
    startPolling(run.run_id);
    await loadSessions();
    renderAll();
  } catch (error) {
    addLocalAssistant(`Submit failed: ${error.message}`);
  } finally {
    el.send.disabled = false;
  }
}

function parsePayloadInput() {
  const raw = el.payloadInput.value.trim();
  if (!raw) return { value: null };
  try {
    const value = JSON.parse(raw);
    if (!value || Array.isArray(value) || typeof value !== "object") {
      return { error: "payload must be a JSON object" };
    }
    if (value.task_type && value.input_payload && typeof value.input_payload === "object") {
      return { value };
    }
    if (value.task_type) {
      const { task_type: taskType, ...inputPayload } = value;
      return { value: { task_type: taskType, input_payload: inputPayload } };
    }
    const selectedPreset = payloadPresets[el.payloadPreset.value];
    if (!selectedPreset) {
      return {
        error:
          "raw payload objects need a selected specialist preset, or include task_type and input_payload",
      };
    }
    return {
      value: {
        task_type: selectedPreset.task_type,
        input_payload: value,
      },
    };
  } catch (error) {
    return { error: error.message };
  }
}

function applySelectedPayloadPreset() {
  const preset = payloadPresets[el.payloadPreset.value];
  if (!preset) return;
  el.payloadInput.value = JSON.stringify(
    {
      task_type: preset.task_type,
      input_payload: preset.input_payload,
    },
    null,
    2,
  );
  updatePayloadWarning();
}

function updatePayloadWarning() {
  const warnings = payloadWarnings();
  if (!warnings.length) {
    el.payloadWarning.hidden = true;
    el.payloadWarning.textContent = "";
    return;
  }
  el.payloadWarning.hidden = false;
  el.payloadWarning.textContent = warnings.join(" ");
}

function payloadWarnings() {
  const parsed = parsePayloadInput();
  if (!el.payloadInput.value.trim() || parsed.error || !parsed.value) return [];
  const prompt = el.prompt.value.trim();
  if (!prompt) return [];
  const envelope = parsed.value;
  const inputPayload = envelope.input_payload || {};
  const warnings = [];
  const promptTask = inferPromptTask(prompt);
  if (promptTask && promptTask !== envelope.task_type) {
    warnings.push(
      `Natural language looks like ${promptTask}, but payload uses ${envelope.task_type}; explicit payload will be used.`,
    );
  }
  const promptDepth = extractPromptDepth(prompt);
  if (
    promptDepth !== null &&
    inputPayload.depth !== undefined &&
    Number(inputPayload.depth) !== promptDepth
  ) {
    warnings.push(
      `Natural language depth ${promptDepth} differs from payload depth ${inputPayload.depth}; explicit payload will be used.`,
    );
  }
  const promptFen = extractPromptFen(prompt);
  if (promptFen && inputPayload.fen && promptFen !== String(inputPayload.fen).trim()) {
    warnings.push("Natural language FEN differs from payload FEN; explicit payload will be used.");
  }
  return warnings;
}

function inferPromptTask(prompt) {
  const lowered = prompt.toLowerCase();
  if (extractPromptFen(prompt) || lowered.includes("stockfish") || lowered.includes("chess")) {
    return "chess_eval";
  }
  if (
    lowered.includes("terraform") ||
    lowered.includes("tfstate") ||
    lowered.includes("apply") ||
    lowered.includes("destroy")
  ) {
    return "infrastructure_plan";
  }
  if (lowered.includes("sympy") || lowered.includes("simplify") || lowered.includes("factor")) {
    return "symbolic_math";
  }
  if (lowered.includes("blast") || lowered.includes("fasta")) {
    return "sequence_alignment";
  }
  if (lowered.includes("timesfm") || lowered.includes("forecast") || lowered.includes("demand")) {
    return "demand_forecast";
  }
  return "";
}

function extractPromptDepth(prompt) {
  const match = prompt.match(/\bdepth\s*(?:=|:)?\s*(\d+)\b/i);
  return match ? Number(match[1]) : null;
}

function extractPromptFen(prompt) {
  for (const segment of prompt.split(/["`,\n]/)) {
    const tokens = segment.trim().split(/\s+/).filter(Boolean);
    for (let index = 0; index <= tokens.length - 6; index += 1) {
      const candidate = tokens.slice(index, index + 6).join(" ");
      if (isValidFen(candidate)) return candidate;
    }
  }
  return "";
}

function isValidFen(fen) {
  const parts = fen.split(/\s+/);
  if (parts.length !== 6) return false;
  const [board, side, castling, enPassant, halfmove, fullmove] = parts;
  if (!["w", "b"].includes(side)) return false;
  if (castling !== "-" && !/^[KQkq]+$/.test(castling)) return false;
  if (enPassant !== "-" && !/^[a-h][36]$/.test(enPassant)) return false;
  if (!/^\d+$/.test(halfmove) || !/^\d+$/.test(fullmove)) return false;
  const ranks = board.split("/");
  if (ranks.length !== 8) return false;
  return ranks.every((rank) => {
    let total = 0;
    for (const char of rank) {
      if (/^[1-8]$/.test(char)) total += Number(char);
      else if (/^[pnbrqkPNBRQK]$/.test(char)) total += 1;
      else return false;
    }
    return total === 8;
  });
}

function startPolling(runId) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  const poll = async () => {
    try {
      const payload = await api(`/api/runs/${encodeURIComponent(runId)}`);
      const run = payload.run;
      state.runs.set(run.run_id, run);
      renderRun(run);
      if (run.status === "completed" || run.status === "failed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
    } catch (error) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      addLocalAssistant(`Polling failed: ${error.message}`);
    }
  };
  poll();
  state.pollTimer = setInterval(poll, 700);
}

function renderRun(run) {
  const assistant = state.messages.find((message) => message.runId === run.run_id);
  if (assistant) {
    assistant.text = run.final_answer || run.active_step || "Working...";
  }
  renderAll();
}

function renderAll() {
  renderDriverSelect();
  renderDriverMeta();
  renderSessions();
  renderSpecialists();
  renderMessages();
  renderTrace();
  renderWorking();
}

function renderDriverSelect() {
  const current = el.driverSelect.value;
  el.driverSelect.innerHTML = "";
  for (const driver of state.drivers) {
    const option = document.createElement("option");
    option.value = driver.provider;
    option.textContent = `${driver.label} - ${driver.model_id}`;
    option.disabled = !driver.enabled;
    el.driverSelect.append(option);
  }
  const available = state.drivers.find((driver) => driver.enabled);
  if (current && state.drivers.some((driver) => driver.provider === current && driver.enabled)) {
    el.driverSelect.value = current;
  } else if (available) {
    el.driverSelect.value = available.provider;
  }
}

function renderDriverMeta() {
  const driver = state.drivers.find((item) => item.provider === el.driverSelect.value);
  if (!driver) {
    el.driverMeta.textContent = "Driver not selected";
    return;
  }
  el.driverMeta.textContent = `${driver.provider} / ${driver.model_id} - ${driver.detail}`;
}

function renderSessions() {
  el.sessionList.innerHTML = "";
  if (!state.sessions.length) {
    el.sessionList.innerHTML = `<div class="empty">No sessions</div>`;
    return;
  }
  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.className = `session-item ${session.session_id === state.currentSessionId ? "active" : ""}`;
    button.innerHTML = `
      <div class="item-title">${escapeHtml(session.title)}</div>
      <div class="item-meta">${session.run_ids.length} runs</div>
    `;
    button.addEventListener("click", () => {
      state.currentSessionId = session.session_id;
      state.messages = [];
      state.currentRunId = session.run_ids[session.run_ids.length - 1] || null;
      renderAll();
    });
    el.sessionList.append(button);
  }
}

function renderSpecialists() {
  el.specialistList.innerHTML = "";
  for (const specialist of state.specialists) {
    const item = document.createElement("div");
    item.className = "specialist-item";
    item.innerHTML = `
      <div class="item-title">${escapeHtml(specialist.specialist_id)}</div>
      <div class="item-meta">${escapeHtml(specialist.description)}</div>
      <div class="item-meta">${escapeHtml(specialist.supported_task_types.join(", "))}</div>
      <div class="status-row">
        <span class="dot ${escapeHtml(specialist.status)}"></span>
        <span class="item-meta">${escapeHtml(specialist.status)} - ${escapeHtml(specialist.detail)}</span>
      </div>
      <div class="item-meta">trust ${formatNumber(specialist.effective_trust)} | cost ${formatNumber(specialist.cost_hint)} | latency ${formatNumber(specialist.latency_hint)}</div>
    `;
    el.specialistList.append(item);
  }
}

function renderMessages() {
  const shouldStickToBottom =
    el.messages.scrollHeight - el.messages.scrollTop - el.messages.clientHeight < 96 ||
    state.messages.length <= 2;
  el.messages.innerHTML = "";
  if (!state.messages.length) {
    const empty = document.createElement("div");
    empty.className = "message assistant";
    empty.innerHTML = `
      <div class="message-role">Assistant</div>
      <div class="bubble">Ready.</div>
    `;
    el.messages.append(empty);
    el.messages.scrollTop = 0;
    return;
  }
  for (const message of state.messages) {
    const div = document.createElement("div");
    div.className = `message ${message.role}`;
    div.innerHTML = `
      <div class="message-role">${message.role}</div>
      <div class="bubble">${escapeHtml(message.text)}</div>
    `;
    el.messages.append(div);
  }
  if (shouldStickToBottom) {
    el.messages.scrollTop = el.messages.scrollHeight;
  }
}

function renderWorking() {
  const run = currentRun();
  if (!run || (run.status !== "running" && run.status !== "queued")) {
    el.working.hidden = true;
    return;
  }
  el.working.hidden = false;
  const started = Date.parse(run.started_at || run.created_at);
  const seconds = Math.max(0, Math.round((Date.now() - started) / 1000));
  el.workingText.textContent = `Thought for ${seconds}s | ${run.active_step || run.status}`;
  el.stepStrip.innerHTML = "";
  for (const step of run.steps.slice(-6)) {
    const chip = document.createElement("div");
    chip.className = "step-chip";
    chip.textContent = `${step.label}`;
    el.stepStrip.append(chip);
  }
}

function renderTrace() {
  const run = currentRun();
  if (!run) {
    el.runMeta.textContent = "No run selected";
    el.costPill.textContent = "$0.00000000";
    renderArtifacts(null);
    renderTimeline(null);
    renderRaw(null);
    renderTool(null);
    return;
  }
  el.runMeta.textContent = `${run.status} | ${run.requested_driver} | ${run.task_type || "task"}`;
  const cost = run.cumulative_cost_usd;
  el.costPill.textContent = cost === null ? "cost n/a" : `$${Number(cost).toFixed(8)}`;
  renderArtifacts(run);
  renderTimeline(run);
  renderRaw(run);
  renderTool(run);
  renderTabs();
}

function renderTabs() {
  for (const button of document.querySelectorAll(".tab")) {
    button.classList.toggle("active", button.dataset.tab === state.activeTab);
  }
  for (const panel of document.querySelectorAll(".tab-panel")) {
    panel.classList.toggle("active", panel.id === `tab-${state.activeTab}`);
  }
}

function renderArtifacts(run) {
  el.artifactList.innerHTML = "";
  if (!run || !run.artifacts.length) {
    el.artifactViewer.innerHTML = "";
    el.artifactList.innerHTML = `<div class="empty">No artifacts</div>`;
    return;
  }
  const selectedStillExists = run.artifacts.some(
    (artifact) => artifact.artifact_id === state.selectedArtifactId,
  );
  if (!selectedStillExists) {
    state.selectedArtifactId = null;
    el.artifactViewer.innerHTML = "";
  }
  for (const artifact of run.artifacts) {
    const button = document.createElement("button");
    button.className = `artifact-item ${artifact.artifact_id === state.selectedArtifactId ? "active" : ""}`;
    button.innerHTML = `
      <div class="artifact-title">${escapeHtml(artifact.output_format)} | ${escapeHtml(artifact.producer_specialist_id)}</div>
      <div class="artifact-meta">${escapeHtml(artifact.artifact_id)}</div>
      <div class="artifact-meta">hash ${escapeHtml(artifact.output_hash.slice(0, 16))} | parent ${escapeHtml(artifact.parent_event_id)}</div>
    `;
    button.addEventListener("click", () => loadArtifact(artifact.artifact_id));
    el.artifactList.append(button);
  }
  if (!state.selectedArtifactId && run.artifacts[0]) {
    loadArtifact(run.artifacts[0].artifact_id);
  }
}

async function loadArtifact(artifactId) {
  state.selectedArtifactId = artifactId;
  renderTrace();
  try {
    const payload = await api(`/api/artifacts/${encodeURIComponent(artifactId)}`);
    renderArtifactContent(payload);
  } catch (error) {
    el.artifactViewer.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
  }
}

function renderArtifactContent(payload) {
  const artifact = payload.artifact;
  const fmt = artifact.output_format;
  let body = "";
  if (payload.is_binary && fmt === "png") {
    body = `<img class="preview-image" alt="artifact ${escapeHtml(artifact.artifact_id)}" src="data:image/png;base64,${payload.raw_base64}" />`;
  } else if (payload.text !== null && fmt === "json") {
    body = `<pre>${escapeHtml(prettyJson(payload.text))}</pre>`;
  } else if (payload.text !== null) {
    body = `<pre>${escapeHtml(payload.text)}</pre>`;
  } else {
    body = `<pre>${escapeHtml(payload.raw_base64)}</pre>`;
  }
  el.artifactViewer.innerHTML = `
    <div class="artifact-item">
      <div class="artifact-title">${escapeHtml(artifact.artifact_id)}</div>
      <div class="artifact-meta">format ${escapeHtml(fmt)} | bytes ${payload.byte_length}</div>
      <div class="artifact-meta">created ${escapeHtml(artifact.created_at)}</div>
    </div>
    <div style="height: 10px"></div>
    ${body}
    <div style="height: 10px"></div>
    <details>
      <summary>Raw bytes</summary>
      <pre>${escapeHtml(payload.raw_base64)}</pre>
    </details>
  `;
}

function renderTimeline(run) {
  el.timeline.innerHTML = "";
  if (!run || !run.provenance_events.length) {
    el.timeline.innerHTML = `<div class="empty">No provenance events</div>`;
    return;
  }
  for (const event of run.provenance_events) {
    const item = document.createElement("details");
    item.className = "event-item";
    item.innerHTML = `
      <summary>
        <span class="event-title">${escapeHtml(event.event_type)}</span>
        <span class="event-meta">${escapeHtml(event.actor)} | ${escapeHtml(event.event_id)}</span>
      </summary>
      <div class="event-meta">parents ${escapeHtml((event.parent_event_ids || []).join(", ") || "none")}</div>
      <pre>${escapeHtml(JSON.stringify(event, null, 2))}</pre>
    `;
    el.timeline.append(item);
  }
}

function renderRaw(run) {
  el.rawList.innerHTML = "";
  if (!run || !run.model_calls.length) {
    el.rawList.innerHTML = `<div class="empty">No model calls</div>`;
    return;
  }
  for (const call of run.model_calls) {
    const item = document.createElement("details");
    item.className = "model-call";
    item.innerHTML = `
      <summary>${escapeHtml(call.kind)} | ${escapeHtml(call.provider)} | ${escapeHtml(call.model_id)}</summary>
      <div class="item-meta">latency ${call.latency_ms ?? "n/a"} ms | cost ${call.estimated_cost_usd ?? "n/a"}</div>
      <div class="item-meta">Parsed</div>
      <pre>${escapeHtml(JSON.stringify(call.parsed, null, 2))}</pre>
      <div class="item-meta">Raw</div>
      <pre>${escapeHtml(call.raw_text || "")}</pre>
    `;
    el.rawList.append(item);
  }
}

function renderTool(run) {
  el.toolState.innerHTML = "";
  if (!run) {
    el.toolState.innerHTML = `<div class="empty">No run selected</div>`;
    return;
  }
  const steps = run.steps.map((step) => `${step.status}: ${step.label}`).join("\n");
  const actualRequest =
    run.proposed_payload && run.proposed_payload.actual_specialist_request
      ? run.proposed_payload.actual_specialist_request
      : run.proposed_payload && run.proposed_payload.specialist_request
        ? run.proposed_payload.specialist_request
        : {};
  const warnings =
    run.proposed_payload && Array.isArray(run.proposed_payload.warnings)
      ? run.proposed_payload.warnings
      : [];
  el.toolState.innerHTML = `
    <div class="tool-card">
      <div class="item-title">Proposal</div>
      <div class="item-meta">specialist ${escapeHtml(run.selected_specialist || "none")}</div>
      <div class="item-meta">task ${escapeHtml(run.task_type || "none")}</div>
      ${
        warnings.length
          ? `<div class="payload-warning">${escapeHtml(warnings.join(" "))}</div>`
          : ""
      }
      <pre>${escapeHtml(JSON.stringify(run.proposed_payload || {}, null, 2))}</pre>
    </div>
    <div class="tool-card">
      <div class="item-title">Exact SpecialistRequest payload</div>
      <pre>${escapeHtml(JSON.stringify(actualRequest, null, 2))}</pre>
    </div>
    <div class="tool-card">
      <div class="item-title">Validation</div>
      <pre>${escapeHtml(JSON.stringify(run.validation_result || {}, null, 2))}</pre>
    </div>
    <div class="tool-card">
      <div class="item-title">Execution</div>
      <pre>${escapeHtml(steps || "not started")}</pre>
    </div>
  `;
}

function currentRun() {
  if (!state.currentRunId) return null;
  return state.runs.get(state.currentRunId) || null;
}

function autosizePrompt() {
  el.prompt.style.height = "auto";
  el.prompt.style.height = `${Math.min(el.prompt.scrollHeight, 180)}px`;
}

function addLocalAssistant(text) {
  state.messages.push({ role: "assistant", text });
  renderMessages();
}

function prettyJson(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

function formatNumber(value) {
  const number = Number(value);
  if (Number.isNaN(number)) return "n/a";
  return number.toFixed(2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init().catch((error) => {
  addLocalAssistant(`Startup failed: ${error.message}`);
});
