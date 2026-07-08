const state = {
  presets: [],
  profiles: [],
  knobs: [],
  runtime: {},
  validation: null,
  flow: null,
  diagnostics: null,
  live: null,
  liveHealth: null,
  healthHistory: [],
  activity: [],
  modelMatrix: null,
  sources: null,
  bundle: null,
  bundleLibrary: null,
  bundleDetail: null,
  launchPlan: null,
  launchDiff: null,
  launchDecision: null,
  launchDecisionStale: false,
  launchCandidates: null,
  gates: null,
  logInsights: null,
  remediation: null,
  selectedStageId: null,
  validationReady: false,
  validationBlocking: false,
  activePhase: "configure",
  simpleMode: false,
  lastUpdated: {},
  observedRuntime: null,
  portsPinned: false,
  runtimeMirroredKey: null,
  launchDetached: false,
};

const CONSOLE_PHASES = [
  { id: "configure", label: "Configure" },
  { id: "preflight", label: "Preflight" },
  { id: "operate", label: "Operate" },
  { id: "debug", label: "Debug" },
];

const SIMPLE_SECTIONS = new Set(["decision", "preflight", "live", "logs"]);
const SIMPLE_MODE_KEY = "noesis_dev_console_simple";

const $ = (id) => document.getElementById(id);
const FORM_TARGETS = {
  pgie_profile: "profileSelect",
  size: "sizeSelect",
  tracking_mode: "trackingSelect",
  rtsp_port: "rtspPortInput",
  depth_enable_seconds: "depthSecondsInput",
  strict_baseline: "strictInput",
};
const COMMAND_SECTIONS = [
  { id: "launch", label: "Launch", statusId: "launchIdLabel", phase: "configure" },
  { id: "decision", label: "Decision", statusId: "decisionSummary", phase: "preflight" },
  { id: "candidates", label: "Candidates", statusId: "candidateSummary", phase: "preflight" },
  { id: "gates", label: "Gates", statusId: "gateSummary", phase: "configure" },
  { id: "preflight", label: "Preflight", statusId: "preflightCounts", phase: "preflight" },
  { id: "runtime", label: "Runtime", statusId: "pidLabel", phase: "preflight" },
  { id: "runbook", label: "Runbook", statusId: "runbookSummary", phase: "preflight" },
  { id: "plan", label: "Plan", statusId: "planSummary", phase: "preflight" },
  { id: "diff", label: "Diff", statusId: "diffSummary", phase: "preflight" },
  { id: "models", label: "Models", statusId: "matrixSummary", phase: "configure" },
  { id: "sources", label: "Sources", statusId: "sourceSummary", phase: "configure" },
  { id: "flow", label: "Flow", statusId: "flowHighlight", phase: "operate" },
  { id: "readiness", label: "Readiness", statusId: "readinessSummary", phase: "operate" },
  { id: "processes", label: "Processes", statusId: "processSummary", phase: "operate" },
  { id: "live", label: "Live", statusId: "liveSummary", phase: "operate" },
  { id: "bundles", label: "Bundles", statusId: "bundleLibrarySummary", phase: "debug" },
  { id: "health", label: "Health", statusId: "healthSummary", phase: "operate" },
  { id: "activity", label: "Activity", statusId: "activitySummary", phase: "debug" },
  { id: "logs", label: "Logs", statusId: "logInsightSummary", phase: "debug" },
];
let commandObserver = null;
const ACTION_STATUS_RANK = { blocked: 0, block: 0, attention: 1, warn: 1, running: 2, ready: 3, ok: 3, info: 4 };
const ACTION_TARGETS = {
  artifacts: { action: "plan", label: "Plan" },
  cuda: { action: "readiness", label: "Readiness" },
  depth_registration: { action: "plan", label: "Plan" },
  gpu: { action: "readiness", label: "Readiness" },
  model: { action: "models", label: "Models" },
  pgie_profile: { action: "candidates", label: "Candidates" },
  ports: { action: "free_ports", label: "Free Ports" },
  processes: { action: "processes", label: "Processes" },
  runbook: { action: "runbook", label: "Runbook" },
  runtime: { action: "runtime", label: "Runtime" },
  sources: { action: "sources", label: "Sources" },
  start: { action: "start", label: "Start" },
  strict_baseline: { action: "gates", label: "Gates" },
  validation: { action: "validate", label: "Validate" },
};

function touchFreshness(key) {
  state.lastUpdated[key] = Date.now();
  updateStatusStrip();
}

function freshnessLabel(key) {
  const ts = state.lastUpdated[key];
  if (!ts) return null;
  const ageSec = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (ageSec < 5) return "just now";
  if (ageSec < 60) return `${ageSec}s ago`;
  return `${Math.round(ageSec / 60)}m ago`;
}

function updateStatusStrip() {
  syncRuntimeIdentity();
  const runtime = state.runtime || {};
  const running = Boolean(runtime.running);
  const external = externalDs8Runtimes();
  const observed = activeObservedRuntime();
  const activePid = activeRuntimePid();
  const runtimeChip = $("statusChipRuntime");
  const runtimeText = $("statusRuntimeText");
  if (runtimeText) {
    if (running) {
      runtimeText.textContent = "running";
    } else if (external.length) {
      const ws = observed?.ws_port ? ` · ws ${observed.ws_port}` : "";
      runtimeText.textContent = `external (${external.length})${ws}`;
    } else {
      runtimeText.textContent = "stopped";
    }
  }
  if (runtimeChip) {
    runtimeChip.dataset.status = running ? "running" : (external.length ? "warn" : "idle");
  }

  const health = state.liveHealth || {};
  const healthText = $("statusHealthText");
  const healthChip = $("statusChipHealth");
  if (healthText) {
    healthText.textContent = health.status ? `health ${health.score ?? "--"}` : "health --";
  }
  if (healthChip) {
    healthChip.dataset.status = health.status === "healthy" ? "ok" : (health.status === "down" ? "block" : (health.status ? "warn" : "idle"));
  }

  const pidText = $("statusPidText");
  const pidChip = $("statusChipPid");
  if (pidText) pidText.textContent = activePid ? `pid ${activePid}` : "pid -";
  if (pidChip) {
    if (!activePid) {
      pidChip.dataset.status = "idle";
    } else if (running || observed?.managed_by_console) {
      pidChip.dataset.status = "running";
    } else {
      pidChip.dataset.status = "warn";
    }
  }

  const live = state.live || {};
  const wsText = $("statusWsText");
  const wsChip = $("statusChipWs");
  if (wsText) {
    wsText.textContent = live.connected ? (live.pong ? "ws pong" : "ws live") : "ws idle";
  }
  if (wsChip) {
    wsChip.dataset.status = live.connected ? "ok" : (live.error ? "block" : "idle");
  }

  const freshText = $("statusFreshText");
  if (freshText) {
    const runtimeAge = freshnessLabel("runtime");
    const diagAge = freshnessLabel("diagnostics");
    freshText.textContent = runtimeAge ? `runtime ${runtimeAge}` : (diagAge ? `diag ${diagAge}` : "sync pending");
  }
}

function setActivePhase(phaseId, { scroll = true } = {}) {
  if (state.simpleMode) return;
  if (!CONSOLE_PHASES.some((phase) => phase.id === phaseId)) return;
  state.activePhase = phaseId;
  document.querySelectorAll(".phase-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.phase === phaseId);
    tab.setAttribute("aria-selected", tab.dataset.phase === phaseId ? "true" : "false");
  });
  applyPhaseVisibility();
  filterCommandCenter();
  if (scroll) {
    $("mainGrid")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function applyPhaseVisibility() {
  document.body.classList.toggle("simple-mode", state.simpleMode);

  if (state.simpleMode) {
    document.querySelectorAll("#mainGrid [data-section]").forEach((panel) => {
      const section = panel.dataset.section || "";
      panel.classList.toggle("phase-hidden", !SIMPLE_SECTIONS.has(section));
    });
    document.querySelectorAll(".launch-section").forEach((section) => {
      section.open = true;
    });
    return;
  }

  const phase = state.activePhase;
  document.querySelectorAll("#mainGrid [data-phase]").forEach((panel) => {
    const phases = (panel.dataset.phase || "").split(/\s+/);
    const section = panel.dataset.section || "";
    const phaseOk = !phase || phases.includes(phase);
    panel.classList.toggle("phase-hidden", !phaseOk);
  });
  document.querySelectorAll("#mainGrid [data-section]").forEach((panel) => {
    if (!panel.dataset.phase) {
      panel.classList.remove("phase-hidden");
    }
  });
}

function relocateLaunchActions(simple) {
  const actions = $("launchActions");
  const mission = $("simpleMissionActions");
  const home = $("launchActionsHome");
  if (!actions || !mission || !home) return;
  if (simple) {
    mission.removeAttribute("hidden");
    mission.appendChild(actions);
  } else {
    home.appendChild(actions);
    mission.setAttribute("hidden", "");
  }
}

function updateSimpleRecipeLine() {
  const line = $("simpleRecipeLine");
  if (!line) return;
  if (!state.simpleMode) {
    line.hidden = true;
    return;
  }
  const preset = $("presetSelect")?.selectedOptions?.[0]?.textContent?.trim() || "custom";
  const model = $("profileSelect")?.value || "-";
  const size = $("sizeSelect")?.value || "auto";
  const track = $("trackingSelect")?.value || "baseline";
  const ws = $("wsPortInput")?.value || "-";
  const rest = $("restPortInput")?.value || "-";
  const rtsp = $("rtspPortInput")?.value || "-";
  line.textContent = `${preset} · ${model}:${size} · ${track} · ${ws}/${rest}/${rtsp}`;
  line.hidden = false;
}

function updateSimplePreflightState() {
  const split = document.querySelector('.split[data-section="preflight"]');
  if (!split) return;
  if (!state.simpleMode) {
    split.classList.remove("simple-clean", "simple-issues", "simple-hidden");
    return;
  }
  const counts = state.validation?.counts || state.validation?.validation?.counts || {};
  const blocks = counts.block || 0;
  const warns = counts.warn || 0;
  split.classList.toggle("simple-hidden", !state.validationReady);
  split.classList.toggle("simple-clean", Boolean(state.validationReady) && blocks === 0 && warns === 0);
  split.classList.toggle("simple-issues", Boolean(state.validationReady) && (blocks > 0 || warns > 0));
}

function updateSimpleLayout() {
  if (!state.simpleMode) return;
  updateSimpleRecipeLine();
  updateSimplePreflightState();
}

function setSimpleMode(enabled, { persist = true } = {}) {
  state.simpleMode = Boolean(enabled);
  const input = $("simpleModeInput");
  if (input) input.checked = state.simpleMode;
  if (persist) {
    try {
      localStorage.setItem(SIMPLE_MODE_KEY, state.simpleMode ? "1" : "0");
    } catch (_err) {
      /* ignore storage failures */
    }
  }
  relocateLaunchActions(state.simpleMode);
  applyPhaseVisibility();
  filterCommandCenter();
  updateSimpleLayout();
  if (state.simpleMode) {
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }
}

function phaseBadgeCount(phaseId) {
  let blocks = 0;
  let warns = 0;
  for (const section of COMMAND_SECTIONS.filter((item) => item.phase === phaseId)) {
    const status = $(section.statusId);
    const kind = classifyCommandStatus((status?.textContent || "").trim());
    if (kind === "block") blocks += 1;
    if (kind === "warn") warns += 1;
  }
  if (blocks) return { text: String(blocks), kind: "block" };
  if (warns) return { text: String(warns), kind: "warn" };
  return null;
}

function updatePhaseBadges() {
  for (const phase of CONSOLE_PHASES) {
    const tab = document.querySelector(`.phase-tab[data-phase="${phase.id}"]`);
    if (!tab) continue;
    const badge = phaseBadgeCount(phase.id);
    const existing = tab.querySelector(".phase-badge");
    if (existing) existing.remove();
    if (badge) {
      const node = document.createElement("span");
      node.className = `phase-badge ${badge.kind}`;
      node.textContent = badge.text;
      tab.appendChild(node);
    }
  }
}

function setupPhaseTabs() {
  const tabs = $("phaseTabs");
  if (!tabs) return;
  tabs.innerHTML = "";
  for (const phase of CONSOLE_PHASES) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `phase-tab ${phase.id === state.activePhase ? "active" : ""}`;
    button.dataset.phase = phase.id;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", phase.id === state.activePhase ? "true" : "false");
    button.textContent = phase.label;
    button.addEventListener("click", () => setActivePhase(phase.id));
    tabs.appendChild(button);
  }
  applyPhaseVisibility();
}

function openInspectorDrawer() {
  const drawer = $("inspectorDrawer");
  const backdrop = $("inspectorBackdrop");
  if (!drawer) return;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  if (backdrop) backdrop.hidden = false;
  state.inspectorOpen = true;
}

function closeInspectorDrawer() {
  const drawer = $("inspectorDrawer");
  const backdrop = $("inspectorBackdrop");
  if (!drawer) return;
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  if (backdrop) backdrop.hidden = true;
  state.inspectorOpen = false;
}

function openLogDrawer() {
  const drawer = $("logDrawer");
  if (!drawer) return;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  state.logDrawerOpen = true;
  tailLogs().catch(() => {});
}

function closeLogDrawer() {
  const drawer = $("logDrawer");
  if (!drawer) return;
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  state.logDrawerOpen = false;
}

function openShortcutsDialog() {
  const dialog = $("shortcutsDialog");
  if (!dialog) return;
  if (typeof dialog.showModal === "function") dialog.showModal();
}

function closeShortcutsDialog() {
  const dialog = $("shortcutsDialog");
  if (!dialog) return;
  if (typeof dialog.close === "function") dialog.close();
}

function isTypingTarget(target) {
  return Boolean(target?.closest("input, textarea, select, [contenteditable='true']"));
}

function highlightLogLine(line) {
  const text = String(line || "");
  const lower = text.toLowerCase();
  let cls = "";
  if (/\berror\b|traceback|exception|failed/.test(lower)) cls = "log-line-error";
  else if (/\bwarn(ing)?\b/.test(lower)) cls = "log-line-warn";
  else if (/\binfo\b/.test(lower)) cls = "log-line-info";
  return `<span class="${cls}">${escapeHtml(text)}</span>`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch (_err) {
    payload = { raw: text };
  }
  if (!response.ok) {
    const message = payload.detail?.error || payload.detail || payload.error || response.statusText;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return payload;
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("visible");
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.classList.remove("visible"), 2600);
}

function commandTarget(section) {
  const status = $(section.statusId);
  return status?.closest(".panel, .launch-panel, .hero-panel, .split") || status;
}

function classifyCommandStatus(text) {
  const value = String(text || "").trim().toLowerCase();
  if (!value || value === "--" || value === "-" || /^pid\s*-?$/.test(value) || /\b(not|pending|unknown|draft|stale)\b/.test(value)) return "idle";
  if (/\b[1-9]\d*\s+(block|blocked|blocking|error|failed|down)\b/.test(value)) return "block";
  if (/\b[1-9]\d*\s+(warn|busy|missing|unavailable|stale)\b/.test(value)) return "warn";
  const signal = value.replace(/\b0\s+(block|blocked|blocking|error|failed|down|warn|busy|missing|unavailable|stale)\w*\b/g, "");
  if (/\b(blocked|blocking|conflict|error|failed|down)\b/.test(signal)) return "block";
  if (/\b(warn|attention|busy|missing|unavailable|stale)\b/.test(signal)) return "warn";
  if (/\b(ready|healthy|connected|running|clean|free|ok|pong|saved|visible|evaluated|analyzed|sampled|activated)\b/.test(signal)) return "ok";
  if (/\b0\s+(block|warn|busy|missing|blocked)\b/.test(value)) return "ok";
  return "info";
}

function updateCommandCenter() {
  const rail = $("commandRail");
  if (!rail) return;
  for (const button of rail.querySelectorAll(".command-item")) {
    const status = $(button.dataset.statusId);
    const text = (status?.textContent || "unknown").trim();
    const kind = classifyCommandStatus(text);
    button.dataset.status = kind;
    button.classList.toggle("ok", kind === "ok");
    button.classList.toggle("warn", kind === "warn");
    button.classList.toggle("block", kind === "block");
    button.classList.toggle("idle", kind === "idle");
    button.classList.toggle("info", kind === "info");
    const summary = button.querySelector("strong");
    if (summary) summary.textContent = text || "unknown";
    button.title = `${button.dataset.label}: ${text || "unknown"}`;
  }
  filterCommandCenter();
  updatePhaseBadges();
  updateStatusStrip();
}

function jumpToCommandSection(sectionId) {
  const section = COMMAND_SECTIONS.find((item) => item.id === sectionId);
  if (section?.phase) setActivePhase(section.phase, { scroll: false });
  const target = section && commandTarget(section);
  if (!target) return;
  document.querySelectorAll(".command-item").forEach((item) => item.classList.toggle("active", item.dataset.sectionId === sectionId));
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function filterCommandCenter() {
  const input = $("commandFilter");
  const rail = $("commandRail");
  if (!input || !rail) return;
  if (state.simpleMode) {
    rail.hidden = true;
    return;
  }
  rail.hidden = false;
  const query = input.value.trim().toLowerCase();
  for (const button of rail.querySelectorAll(".command-item")) {
    const section = COMMAND_SECTIONS.find((item) => item.id === button.dataset.sectionId);
    const phaseMatch = !state.activePhase || section?.phase === state.activePhase;
    const haystack = `${button.dataset.label || ""} ${button.querySelector("strong")?.textContent || ""}`.toLowerCase();
    const textMatch = !query || haystack.includes(query);
    button.hidden = !(phaseMatch && textMatch);
  }
}

function setupCommandCenter() {
  const rail = $("commandRail");
  if (!rail) return;
  rail.innerHTML = "";
  for (const section of COMMAND_SECTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "command-item idle";
    button.dataset.sectionId = section.id;
    button.dataset.statusId = section.statusId;
    button.dataset.label = section.label;
    button.dataset.phase = section.phase || "";
    button.innerHTML = `<span>${escapeHtml(section.label)}</span><strong>unknown</strong>`;
    button.addEventListener("click", () => jumpToCommandSection(section.id));
    rail.appendChild(button);
  }

  const input = $("commandFilter");
  input?.addEventListener("input", filterCommandCenter);
  input?.addEventListener("keydown", (event) => {
    const visible = [...rail.querySelectorAll(".command-item:not([hidden])")];
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      event.preventDefault();
      const idx = visible.findIndex((item) => item.classList.contains("active"));
      const next = visible[(idx + 1 + visible.length) % visible.length];
      next?.focus();
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      event.preventDefault();
      const idx = visible.findIndex((item) => item.classList.contains("active"));
      const next = visible[(idx - 1 + visible.length) % visible.length];
      next?.focus();
    } else if (event.key === "Enter") {
      const firstVisible = visible[0];
      firstVisible?.click();
    } else if (event.key === "Escape") {
      input.value = "";
      filterCommandCenter();
      input.blur();
    }
  });
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      input?.focus();
      input?.select();
      return;
    }
    if (isTypingTarget(event.target)) return;
    if (event.key === "?") {
      event.preventDefault();
      openShortcutsDialog();
      return;
    }
    if (event.key >= "1" && event.key <= "4" && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const phase = CONSOLE_PHASES[Number(event.key) - 1];
      if (phase) {
        event.preventDefault();
        setActivePhase(phase.id);
      }
      return;
    }
    if (event.key.toLowerCase() === "v" && !event.shiftKey) {
      event.preventDefault();
      validateLaunch().catch((err) => toast(err.message));
      return;
    }
    if (event.key.toLowerCase() === "s" && event.shiftKey) {
      event.preventDefault();
      stopRuntime().catch((err) => toast(err.message));
      return;
    }
    if (event.key.toLowerCase() === "s" && !event.shiftKey) {
      event.preventDefault();
      startRuntime();
      return;
    }
    if (event.key.toLowerCase() === "l") {
      event.preventDefault();
      openLogDrawer();
      return;
    }
    if (event.key === "Escape") {
      if (state.inspectorOpen) closeInspectorDrawer();
      if (state.logDrawerOpen) closeLogDrawer();
      closeShortcutsDialog();
    }
  });

  commandObserver?.disconnect();
  commandObserver = new MutationObserver(updateCommandCenter);
  for (const section of COMMAND_SECTIONS) {
    const status = $(section.statusId);
    if (status) commandObserver.observe(status, { childList: true, characterData: true, subtree: true });
  }
  updateCommandCenter();
}

function actionStatusClass(status) {
  const value = String(status || "idle").toLowerCase();
  if (value === "blocked" || value === "block") return "blocked";
  if (value === "attention" || value === "warn") return "attention";
  if (value === "ready" || value === "ok") return "ready";
  if (value === "running") return "running";
  return "idle";
}

function actionTarget(target, fallbackLabel) {
  const key = String(target || "").toLowerCase();
  return ACTION_TARGETS[key] || { action: key || "decision", label: fallbackLabel || "Open" };
}

function componentAction(component) {
  const key = String(component?.key || "").toLowerCase();
  const byKey = {
    artifacts: "artifacts",
    external_runtime: "processes",
    gpu: "gpu",
    model: "model",
    ports: "ports",
    preflight: "validation",
    runtime: "runtime",
    sources: "sources",
  };
  return actionTarget(byKey[key] || key, component?.action || component?.label || "Open");
}

function addActionButton(actions, action, label, { primary = false, disabled = false } = {}) {
  if (!action || actions.some((item) => item.action === action)) return;
  actions.push({ action, label, primary, disabled });
}

function renderActionBoard(payload) {
  const board = $("actionBoard");
  if (!board) return;
  const statusNode = $("actionBoardStatus");
  const meter = $("actionBoardMeter");
  const title = $("actionBoardTitle");
  const copy = $("actionBoardCopy");
  const evidence = $("actionBoardEvidence");
  const actionsBox = $("actionBoardActions");
  const runtimeRunning = Boolean(state.runtime?.running);
  const decision = payload || null;
  const status = actionStatusClass(decision?.status || (runtimeRunning ? "running" : (state.launchDecisionStale ? "attention" : "idle")));
  const score = decision?.score;
  const primary = decision?.primary_action || {};

  board.dataset.status = status;
  statusNode.textContent = decision
    ? `${decision.status || "unknown"}${score !== undefined ? ` / ${score}` : ""}`
    : (state.launchDecisionStale ? "stale" : (runtimeRunning ? "running" : "pending"));
  meter.style.width = `${Math.max(0, Math.min(100, Number(score ?? (runtimeRunning ? 100 : 0))))}%`;
  title.textContent = decision
    ? (primary.title || decision.summary || "Launch decision")
    : (state.launchDecisionStale ? "Launch evidence stale" : (runtimeRunning ? "Console runtime active" : "Launch evidence pending"));
  copy.textContent = decision
    ? (primary.detail || decision.summary || "")
    : (runtimeRunning ? `pid ${state.runtime.pid || "-"} / ${state.runtime.launch_id || "active"}` : "No launch decision has been evaluated.");

  evidence.innerHTML = "";
  const components = (decision?.components || [])
    .slice()
    .sort((left, right) => {
      const leftRank = ACTION_STATUS_RANK[String(left.status || "info")] ?? 9;
      const rightRank = ACTION_STATUS_RANK[String(right.status || "info")] ?? 9;
      return leftRank - rightRank || String(left.label || "").localeCompare(String(right.label || ""));
    })
    .slice(0, 6);
  if (!components.length) {
    const item = document.createElement("div");
    item.className = "action-evidence-item idle";
    item.innerHTML = `<span>Evidence</span><strong>${escapeHtml(state.launchDecisionStale ? "stale" : "pending")}</strong>`;
    evidence.appendChild(item);
  } else {
    for (const component of components) {
      const item = document.createElement("div");
      item.className = `action-evidence-item ${actionStatusClass(component.status)}`;
      item.innerHTML = `
        <span>${escapeHtml(component.label || component.key || "Evidence")}</span>
        <strong>${escapeHtml(component.metric || component.status || "-")}</strong>
        <p>${escapeHtml(component.detail || "")}</p>
      `;
      evidence.appendChild(item);
    }
  }

  const buttons = [];
  if (decision?.start_allowed) {
    addActionButton(buttons, "start", "Start", { primary: true, disabled: runtimeRunning });
  } else if (runtimeRunning) {
    addActionButton(buttons, "stop", "Stop", { primary: true });
    addActionButton(buttons, "restart", "Restart", { disabled: state.validationBlocking || !state.validationReady });
    addActionButton(buttons, "runtime", "Runtime");
    addActionButton(buttons, "live", "Live");
  } else if (!decision) {
    addActionButton(buttons, "decision", "Evaluate", { primary: true });
  } else {
    const primaryTarget = actionTarget(primary.target, primary.title || "Open");
    addActionButton(buttons, primaryTarget.action, primaryTarget.label, { primary: true });
    for (const component of components.filter((item) => actionStatusClass(item.status) !== "ready")) {
      const next = componentAction(component);
      addActionButton(buttons, next.action, next.label);
    }
    for (const remediation of (decision?.top_actions || []).slice(0, 4)) {
      const next = actionTarget(remediation.target, remediation.title || "Open");
      addActionButton(buttons, next.action, next.label);
    }
    addActionButton(buttons, "validate", "Validate");
  }

  actionsBox.innerHTML = "";
  for (const item of buttons.slice(0, 5)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `action-button ${item.primary ? "primary" : ""}`;
    button.dataset.actionBoard = item.action;
    button.disabled = Boolean(item.disabled);
    button.textContent = item.label;
    actionsBox.appendChild(button);
  }
}

async function runActionBoardAction(action) {
  switch (action) {
    case "decision":
      await loadLaunchDecision(true);
      toast("Launch decision refreshed");
      break;
    case "free_ports":
      await useFreePorts();
      break;
    case "candidates":
      await loadLaunchCandidates(true);
      toast("Launch candidates compared");
      jumpToCommandSection("candidates");
      break;
    case "gates":
      jumpToCommandSection("gates");
      break;
    case "live":
      jumpToCommandSection("live");
      break;
    case "models":
      jumpToCommandSection("models");
      break;
    case "plan":
      await loadLaunchPlan(true);
      toast("Launch plan refreshed");
      jumpToCommandSection("plan");
      break;
    case "processes":
      jumpToCommandSection("processes");
      break;
    case "readiness":
      await loadDiagnostics();
      jumpToCommandSection("readiness");
      break;
    case "runbook":
      await loadRemediation();
      jumpToCommandSection("runbook");
      break;
    case "runtime":
      jumpToCommandSection("runtime");
      break;
    case "restart":
      await restartRuntime();
      break;
    case "sources":
      await loadSources(true);
      toast("Sources refreshed");
      jumpToCommandSection("sources");
      break;
    case "start":
      await startRuntime();
      break;
    case "stop":
      await stopRuntime();
      break;
    case "validate":
      await validateLaunch();
      break;
    default:
      jumpToCommandSection(action);
  }
}

function specFromForm() {
  return {
    preset_id: $("presetSelect").value,
    pipeline_config: $("pipelineInput").value.trim() || "config/infer.yaml",
    cameras_config: $("camerasInput").value.trim() || "config/cameras.yaml",
    pgie_profile: $("profileSelect").value,
    size: $("sizeSelect").value,
    tracking_mode: $("trackingSelect").value,
    ws_host: "127.0.0.1",
    ws_port: Number($("wsPortInput").value || 6008),
    rest_host: "127.0.0.1",
    rest_port: Number($("restPortInput").value || 8080),
    rtsp_port: Number($("rtspPortInput").value || 8554),
    enable_rest: true,
    log_level: $("logLevelSelect").value,
    depth_enable_seconds: Number($("depthSecondsInput").value || 0),
    strict_baseline: $("strictInput").checked,
    env_lines: $("envInput").value,
  };
}

function envLinesFromObject(env) {
  return Object.entries(env || {})
    .map(([key, value]) => `${key}=${value}`)
    .join("\n");
}

async function refreshDependentLaunchPanels({ includeDecision = true } = {}) {
  await renderFlow();
  await loadLaunchPlan().catch(() => {});
  await loadLaunchDiff().catch(() => {});
  await loadRemediation().catch(() => {});
  await loadGates().catch(() => {});
  await loadModelMatrix().catch(() => {});
  await loadSources().catch(() => {});
  if (includeDecision) await loadLaunchDecision().catch(() => {});
}

function applyLaunchSpecToFormFields(spec, { coreOnly = false } = {}) {
  if (!spec) return false;
  let changed = false;
  const apply = (id, value, { checkbox = false } = {}) => {
    if (value === undefined || value === null) return;
    const input = $(id);
    if (!input) return;
    const next = checkbox ? Boolean(value) : String(value);
    const current = checkbox ? input.checked : input.value;
    if (current === next || (!checkbox && String(current) === String(next))) return;
    if (checkbox) input.checked = next;
    else input.value = next;
    changed = true;
  };

  if ($("presetSelect") && $("presetSelect").value !== "") {
    $("presetSelect").value = "";
    changed = true;
  }

  apply("pipelineInput", spec.pipeline_config || "config/infer.yaml");
  apply("camerasInput", spec.cameras_config || "config/cameras.yaml");
  apply("profileSelect", spec.pgie_profile || "yolo11_seg");
  apply("sizeSelect", spec.size ?? "");
  apply("trackingSelect", spec.tracking_mode || "baseline");
  apply("wsPortInput", spec.ws_port ?? 6008);
  apply("restPortInput", spec.rest_port ?? 8080);
  apply("rtspPortInput", spec.rtsp_port ?? 8554);

  if (coreOnly) return changed;

  apply("logLevelSelect", spec.log_level || "WARNING");
  apply("depthSecondsInput", spec.depth_enable_seconds ?? 0);
  apply("strictInput", Boolean(spec.strict_baseline), { checkbox: true });
  if (spec.env && $("envInput")) {
    const nextEnv = envLinesFromObject(spec.env);
    if ($("envInput").value !== nextEnv) {
      $("envInput").value = nextEnv;
      changed = true;
    }
  }
  return changed;
}

function specFromObservedRuntime(observed = activeObservedRuntime()) {
  const runtime = state.runtime || {};
  if (runtime.running && runtime.spec) {
    return { ...runtime.spec, preset_id: "" };
  }
  if (!observed) return null;

  const process = findProcessByPid(observed.pid) || {};
  const launch = observed.launch || {};
  const env = process.env || {};
  return {
    preset_id: "",
    pipeline_config: launch.pipeline_config || env.NOESIS_DS8_PIPELINE_CONFIG || "config/infer.yaml",
    cameras_config: launch.cameras_config || env.NOESIS_CAMERAS_CONFIG || "config/cameras.yaml",
    pgie_profile: launch.pgie_profile || env.NOESIS_PGIE_PROFILE || "yolo11_seg",
    size: launch.size ?? env.NOESIS_PGIE_SIZE ?? "",
    tracking_mode: launch.tracking_mode || env.NOESIS_TRACKING_MODE || "baseline",
    ws_host: observed.ws_host || launch.ws_host || "127.0.0.1",
    ws_port: Number(observed.ws_port || launch.ws_port || 6008),
    rest_host: observed.rest_host || launch.rest_host || "127.0.0.1",
    rest_port: Number(observed.rest_port || launch.rest_port || 8080),
    rtsp_port: Number(observed.rtsp_port || launch.rtsp_port || 8554),
    enable_rest: launch.enable_rest !== false,
    log_level: launch.log_level || env.NOESIS_LOG_LEVEL || "WARNING",
    depth_enable_seconds: Number(launch.depth_enable_seconds || env.NOESIS_DEPTH_ENABLE_SECONDS || 0),
    strict_baseline: env.NOESIS_STRICT_BASELINE === "1",
    env: Object.fromEntries(
      Object.entries(env).filter(([key]) => key.startsWith("NOESIS_") || key === "CUDA_VISIBLE_DEVICES" || key === "NVIDIA_VISIBLE_DEVICES")
    ),
  };
}

async function applySpecToForm(spec) {
  state.launchDetached = true;
  state.runtimeMirroredKey = null;
  applyLaunchSpecToFormFields(spec);
  markValidationStale();
  await loadDiagnostics({ syncRuntime: false });
  await refreshDependentLaunchPanels();
}

function applyPreset(presetId) {
  state.launchDetached = true;
  state.runtimeMirroredKey = null;
  const preset = state.presets.find((item) => item.id === presetId);
  if (!preset) return;
  $("pipelineInput").value = preset.pipeline_config || "config/infer.yaml";
  $("camerasInput").value = preset.cameras_config || "config/cameras.yaml";
  $("rtspPortInput").value = preset.rtsp_port || 8554;
  $("depthSecondsInput").value = preset.depth_enable_seconds || 0;
  $("profileSelect").value = preset.pgie_profile || "yolo11_seg";
  $("sizeSelect").value = preset.size || "";
  $("trackingSelect").value = preset.tracking_mode || "baseline";
  $("heroTitle").textContent = preset.name;
  $("heroCopy").textContent = preset.notes || `${preset.preset_type} preset using ${preset.source_type} sources.`;
  markValidationStale();
  renderFlow();
  loadDiagnostics();
  loadLaunchPlan();
  loadLaunchDiff();
  loadRemediation();
  loadGates();
  loadModelMatrix();
  loadSources();
  loadLaunchDecision();
  $("decisionSummary").textContent = "stale";
  updateSimpleLayout();
}

function renderProfiles(items) {
  state.profiles = items || [];
  const select = $("savedProfileSelect");
  select.innerHTML = "";
  if (!state.profiles.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No saved profiles";
    select.appendChild(option);
  }
  for (const profile of state.profiles) {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = `${profile.name} · ${profile.pgie_profile || "-"} ${profile.size || ""}`;
    select.appendChild(option);
  }
  $("profileCount").textContent = `${state.profiles.length} saved`;
}

function coercePid(value) {
  if (value === undefined || value === null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function findProcessByPid(pid) {
  const needle = coercePid(pid);
  if (!needle) return null;
  return (state.diagnostics?.processes || []).find((item) => coercePid(item?.pid) === needle) || null;
}

function pickBestDs8Runtime(candidates) {
  const items = (candidates || []).filter(Boolean);
  if (!items.length) return null;
  const wsPort = Number($("wsPortInput")?.value || 0);
  if (wsPort) {
    const matched = items.find((process) => {
      const launch = process.launch || {};
      if (Number(launch.ws_port) === wsPort) return true;
      return (process.ports || []).some((port) => {
        const label = String(port.label || "").toLowerCase();
        return Number(port.port) === wsPort && label.includes("websocket");
      });
    });
    if (matched) return matched;
  }
  return [...items].sort((left, right) => {
    const portDelta = (right.ports || []).length - (left.ports || []).length;
    if (portDelta) return portDelta;
    return Number((right.stats || {}).elapsed_s || 0) - Number((left.stats || {}).elapsed_s || 0);
  })[0];
}

function externalDs8Runtimes() {
  const diagnostics = state.diagnostics || {};
  const runtimes = diagnostics.ds8_runtimes || (diagnostics.processes || []).filter((item) => item.looks_like_ds8);
  return runtimes.filter((item) => !item.managed_by_console);
}

function resolveObservedRuntime() {
  if (state.runtime?.running) {
    const managedPid = coercePid(state.runtime.pid);
    const managedProcess = findProcessByPid(managedPid);
    if (managedProcess) return runtimeTargetFromProcess(managedProcess);
    return runtimeTargetFromProcess({
      pid: managedPid,
      managed_by_console: true,
      looks_like_ds8: true,
      launch: {},
      ports: [],
      stats: { elapsed_s: state.runtime.uptime_s },
    });
  }

  const observed = state.diagnostics?.observed_runtime;
  if (observed) {
    const process = findProcessByPid(observed.pid);
    return runtimeTargetFromProcess(process || observed);
  }

  const external = pickBestDs8Runtime(externalDs8Runtimes());
  if (external) return runtimeTargetFromProcess(external);

  const ds8Processes = pickBestDs8Runtime((state.diagnostics?.processes || []).filter((item) => item.looks_like_ds8));
  if (ds8Processes) return runtimeTargetFromProcess(ds8Processes);

  return null;
}

function activeObservedRuntime() {
  return state.observedRuntime || resolveObservedRuntime();
}

function activeRuntimePid() {
  return coercePid(activeObservedRuntime()?.pid) ?? coercePid(state.runtime?.pid);
}

function syncRuntimeIdentity() {
  const observed = resolveObservedRuntime();
  if (observed) state.observedRuntime = observed;
  return observed;
}

function runtimeTargetFromProcess(process) {
  const launch = process?.launch || {};
  const port = (label) => (process?.ports || []).find((item) => item.label === label)?.port;
  return {
    pid: coercePid(process?.pid),
    managed_by_console: Boolean(process?.managed_by_console),
    looks_like_ds8: Boolean(process?.looks_like_ds8),
    launch,
    ports: process?.ports || [],
    stats: process?.stats || {},
    command: process?.command || "",
    ws_host: launch.ws_host || "127.0.0.1",
    ws_port: launch.ws_port || port("WebSocket"),
    rest_host: launch.rest_host || "127.0.0.1",
    rest_port: launch.rest_port || port("REST"),
    rtsp_port: launch.rtsp_port || port("RTSP mosaic"),
  };
}

function adoptObservedRuntime(observed, { silent = false } = {}) {
  if (!observed) return false;
  const changed = applyLaunchSpecToFormFields(specFromObservedRuntime(observed));
  state.observedRuntime = observed;
  if (changed && !silent) {
    toast(`Targeting ${observed.managed_by_console ? "console" : "external"} runtime pid ${observed.pid}`);
  }
  return changed;
}

async function syncFormWithActiveRuntime({ silent = true, force = false } = {}) {
  const observed = syncRuntimeIdentity();
  const activePid = activeRuntimePid();
  if (!activePid || !observed) {
    state.runtimeMirroredKey = null;
    return false;
  }

  if (state.launchDetached && !force) return false;

  const mirrorKey = String(activePid);
  const alreadyMirrored = state.runtimeMirroredKey === mirrorKey && !force;

  if (state.simpleMode) {
    if (state.runtime?.running) return false;
    if (state.portsPinned && !force) return false;
    if (observed.managed_by_console) return false;
    const changed = adoptObservedRuntime(observed, { silent });
    if (changed) {
      state.runtimeMirroredKey = mirrorKey;
      markValidationStale();
      updateSimpleLayout();
      await renderFlow();
    }
    return changed;
  }

  if (state.portsPinned && !force) return false;

  const spec = specFromObservedRuntime(observed);
  const changed = applyLaunchSpecToFormFields(spec) || force || !alreadyMirrored;
  state.observedRuntime = observed;
  state.runtimeMirroredKey = mirrorKey;

  if (!changed && alreadyMirrored) return false;

  if (!silent) {
    toast(`Mirroring ${observed.managed_by_console ? "console" : "external"} runtime pid ${activePid} in launch controls`);
  }
  $("launchIdLabel").textContent = observed.managed_by_console
    ? (state.runtime?.launch_id || spec?.launch_id || "running")
    : `ext:${activePid}`;
  markValidationStale();
  await refreshDependentLaunchPanels();
  return true;
}

function maybeSyncObservedRuntime(options = {}) {
  return syncFormWithActiveRuntime(options);
}

function hasLiveTarget() {
  const observed = activeObservedRuntime();
  if (observed?.ws_port) return true;
  return Boolean($("wsPortInput")?.value);
}

function shouldAutoProbeLive() {
  if (state.runtime?.running) return true;
  return Boolean(state.diagnostics?.observed_runtime?.ws_port);
}

function updateRuntimePill() {
  const runtime = state.runtime || {};
  const running = Boolean(runtime.running);
  const external = externalDs8Runtimes();
  const pill = $("runtimePill");
  if (running) {
    pill.textContent = "Console: running";
  } else if (external.length) {
    pill.textContent = `External DS8: running (${external.length})`;
  } else {
    pill.textContent = "Console: stopped";
  }
  pill.classList.toggle("running", running);
  pill.classList.toggle("external", !running && external.length > 0);
  pill.classList.toggle("stopped", !running && external.length === 0);
}

function renderRuntime(runtime) {
  state.runtime = runtime || {};
  syncRuntimeIdentity();
  const running = Boolean(runtime?.running);
  const observed = activeObservedRuntime();
  const activePid = activeRuntimePid();
  const external = !running && observed && !observed.managed_by_console;
  updateRuntimePill();
  if (external) {
    $("pidLabel").textContent = activePid ? `pid ${activePid}` : "pid -";
    $("stateMetric").textContent = "external";
    $("uptimeMetric").textContent = observed.stats?.elapsed_s ? `${Math.round(observed.stats.elapsed_s)}s` : "0s";
    $("launchMetric").textContent = observed.launch?.pgie_profile || observed.launch?.launch_id || "external";
    $("launchIdLabel").textContent = `ext:${activePid || "-"}`;
  } else {
    $("pidLabel").textContent = activePid ? `pid ${activePid}` : "pid -";
    $("stateMetric").textContent = running ? "running" : "stopped";
    $("uptimeMetric").textContent = running && runtime.uptime_s ? `${Math.round(runtime.uptime_s)}s` : "0s";
    $("launchMetric").textContent = runtime?.launch_id || "none";
    $("launchIdLabel").textContent = runtime?.launch_id || "draft";
  }
  const decisionBlocksStart = state.launchDecision ? !state.launchDecision.start_allowed : state.launchDecisionStale;
  $("startButton").disabled = running || state.validationBlocking || !state.validationReady || decisionBlocksStart;
  $("restartButton").disabled = !running || state.validationBlocking || !state.validationReady;
  $("stopButton").disabled = !running;
  renderActionBoard(state.launchDecision);
  touchFreshness("runtime");
  updateSimpleLayout();
}

function renderPresets(items) {
  state.presets = items || [];
  const select = $("presetSelect");
  select.innerHTML = "";
  const custom = document.createElement("option");
  custom.value = "";
  custom.textContent = "custom launch";
  select.appendChild(custom);
  for (const preset of state.presets) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = `${preset.name}`;
    select.appendChild(option);
  }
  if (state.presets.length) {
    select.value = "baseline-rtsp-yolo11-seg";
    applyPreset(select.value);
  }
}

function renderValidation(validation) {
  state.validation = validation;
  const list = $("validationList");
  const results = validation?.results || validation?.validation?.results || [];
  state.validationReady = Boolean(validation);
  state.validationBlocking = Boolean(validation?.blocking || validation?.validation?.blocking);
  renderRuntime(state.runtime);
  updateSimpleLayout();
  list.innerHTML = "";
  const counts = validation?.counts || validation?.validation?.counts || { block: 0, warn: 0, info: 0 };
  $("preflightCounts").textContent = `${counts.block || 0} block / ${counts.warn || 0} warn / ${counts.info || 0} info`;
  if (!results.length) {
    list.innerHTML = `
      <div class="panel-empty-cta">
        <strong>No validation run yet</strong>
        <p>Preview materializes the launch artifact. Validate runs full preflight checks.</p>
        <button type="button" class="button primary" id="validationEmptyValidate">Validate Launch</button>
      </div>`;
    list.querySelector("#validationEmptyValidate")?.addEventListener("click", () => validateLaunch().catch((err) => toast(err.message)));
    return;
  }
  for (const item of results) {
    const row = document.createElement("div");
    row.className = `validation-item ${item.severity || "info"}`;
    const hint = item.fix_hint ? `<p>${escapeHtml(item.fix_hint)}</p>` : "";
    row.innerHTML = `<strong>${escapeHtml(item.code)}</strong><p>${escapeHtml(item.message)}</p>${hint}`;
    list.appendChild(row);
  }
}

function renderFlowMap(flow) {
  state.flow = flow;
  const map = $("flowMap");
  map.innerHTML = "";
  const stages = flow?.stages || [];
  $("flowHighlight").textContent = flow?.highlights
    ? `${flow.highlights.pgie_profile} ${flow.highlights.size || ""} / ${flow.highlights.tracking_mode}`
    : "-";
  for (const stage of stages) {
    const node = document.createElement("div");
    node.className = `flow-node ${stage.lane || "main"} ${stage.kind || ""} ${stage.status === "disabled" ? "disabled" : ""}`;
    node.dataset.stageId = stage.id;
    if (state.selectedStageId === stage.id) node.classList.add("selected");
    node.innerHTML = `<h3>${escapeHtml(stage.label)}</h3><p>${escapeHtml(stage.status)} - ${escapeHtml(stage.detail || "")}</p>`;
    node.addEventListener("click", () => selectStage(stage.id));
    map.appendChild(node);
  }
  if (state.selectedStageId) renderInspector();
}

function selectStage(stageId) {
  state.selectedStageId = stageId;
  document.querySelectorAll(".flow-node").forEach((node) => {
    node.classList.toggle("selected", node.dataset.stageId === stageId);
  });
  renderInspector();
  openInspectorDrawer();
}

function renderInspector() {
  const stage = (state.flow?.stages || []).find((item) => item.id === state.selectedStageId);
  const box = $("nodeInspector");
  if (!stage) {
    box.innerHTML = "<strong>No stage selected</strong><p>Click any pipeline block to inspect it.</p>";
    $("inspectorKind").textContent = "select a node";
    return;
  }
  $("inspectorKind").textContent = `${stage.kind} / ${stage.status}`;
  const metrics = Object.entries(stage.metrics || {})
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `<div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
  box.innerHTML = `
    <strong>${escapeHtml(stage.label)}</strong>
    <p>${escapeHtml(stage.detail || "No additional detail")}</p>
    <div class="inspector-metrics">${metrics || "<div><span>lane</span><strong>" + escapeHtml(stage.lane || "main") + "</strong></div>"}</div>
  `;
  renderStageControls(stage, box);
}

function parseEnvOverlay() {
  const env = {};
  for (const rawLine of $("envInput").value.split(/\r?\n/)) {
    const match = rawLine.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$/);
    if (!match) continue;
    env[match[1]] = match[2].trim();
  }
  return env;
}

function removeEnvOverlayValue(key) {
  const nextLines = [];
  for (const rawLine of $("envInput").value.split(/\r?\n/)) {
    const match = rawLine.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=/);
    if (match && match[1] === key) continue;
    nextLines.push(rawLine);
  }
  $("envInput").value = nextLines.join("\n").replace(/\n{3,}/g, "\n\n").replace(/^\n+|\n+$/g, "");
}

function setEnvOverlayValue(key, value) {
  const nextValue = String(value ?? "").trim();
  if (nextValue === "") {
    removeEnvOverlayValue(key);
    return;
  }
  const lines = $("envInput").value.split(/\r?\n/);
  let replaced = false;
  const nextLines = lines.map((rawLine) => {
    const match = rawLine.match(/^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=)(.*)$/);
    if (match && match[2] === key) {
      replaced = true;
      return `${match[1]}${key}=${nextValue}`;
    }
    return rawLine;
  });
  if (!replaced) {
    while (nextLines.length && nextLines[nextLines.length - 1].trim() === "") nextLines.pop();
    nextLines.push(`${key}=${nextValue}`);
  }
  $("envInput").value = nextLines.join("\n").replace(/^\n+/, "");
}

function currentTargetValue(target, fallback = "") {
  if (target.startsWith("NOESIS_")) {
    const env = parseEnvOverlay();
    return Object.prototype.hasOwnProperty.call(env, target) ? env[target] : fallback;
  }
  const id = FORM_TARGETS[target];
  const input = id && $(id);
  if (!input) return fallback;
  if (input.type === "checkbox") return input.checked ? "1" : "0";
  return input.value;
}

function setTargetValue(target, value) {
  if (target.startsWith("NOESIS_")) {
    setEnvOverlayValue(target, value);
    return;
  }
  const id = FORM_TARGETS[target];
  const input = id && $(id);
  if (!input) return;
  if (input.type === "checkbox") {
    input.checked = String(value).trim().toLowerCase() === "1" || String(value).trim().toLowerCase() === "true" || value === true;
    return;
  }
  input.value = value;
}

function renderStageControls(stage, box) {
  const controls = stage.controls || [];
  if (!controls.length) return;
  const wrap = document.createElement("div");
  wrap.className = "stage-controls";
  const grid = document.createElement("div");
  grid.className = "stage-control-grid";
  for (const control of controls) {
    const label = document.createElement("label");
    label.className = "stage-control";
    const caption = document.createElement("span");
    caption.textContent = control.label || control.target;
    label.appendChild(caption);
    let input;
    if (control.type === "select" || control.type === "toggle") {
      input = document.createElement("select");
      const options = control.type === "toggle"
        ? [
            ["", "inherit"],
            ["1", "on"],
            ["0", "off"],
          ]
        : (control.options || []).map((item) => [String(item), String(item || "auto")]);
      for (const [value, text] of options) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = text;
        input.appendChild(option);
      }
    } else {
      input = document.createElement("input");
      input.type = "number";
      if (control.min !== undefined) input.min = control.min;
      if (control.max !== undefined) input.max = control.max;
      if (control.step !== undefined) input.step = control.step;
    }
    input.className = "stage-control-input";
    input.dataset.controlTarget = control.target;
    input.value = currentTargetValue(control.target, control.value ?? "");
    label.appendChild(input);
    grid.appendChild(label);
  }
  const actions = document.createElement("div");
  actions.className = "stage-control-actions";
  const apply = document.createElement("button");
  apply.className = "button subtle";
  apply.type = "button";
  apply.textContent = "Apply";
  apply.addEventListener("click", () => applyStageControls(stage, false).catch((err) => toast(err.message)));
  const applyValidate = document.createElement("button");
  applyValidate.className = "button primary";
  applyValidate.type = "button";
  applyValidate.textContent = "Apply + Validate";
  applyValidate.addEventListener("click", () => applyStageControls(stage, true).catch((err) => toast(err.message)));
  actions.appendChild(apply);
  actions.appendChild(applyValidate);
  wrap.appendChild(grid);
  wrap.appendChild(actions);
  box.appendChild(wrap);
}

async function applyStageControls(stage, validateAfter) {
  for (const input of document.querySelectorAll("#nodeInspector [data-control-target]")) {
    setTargetValue(input.dataset.controlTarget, input.value);
  }
  markValidationStale();
  await renderFlow();
  await loadDiagnostics();
  await loadGates().catch(() => {});
  await loadModelMatrix().catch(() => {});
  await loadSources().catch(() => {});
  await loadLaunchDecision().catch(() => {});
  toast(`${stage.label} controls applied`);
  if (validateAfter) await validateLaunch();
}

function renderKnobs() {
  const filter = $("knobFilter").value.trim().toLowerCase();
  const table = $("knobTable");
  table.innerHTML = "";
  const knobs = state.knobs.filter((item) => {
    const hay = `${item.key} ${item.category} ${item.description}`.toLowerCase();
    return !filter || hay.includes(filter);
  });
  $("knobMetric").textContent = `${state.knobs.length}`;
  for (const knob of knobs.slice(0, 220)) {
    const row = document.createElement("div");
    row.className = "knob-row";
    row.innerHTML = `
      <strong>${escapeHtml(knob.key)}</strong>
      <span>${escapeHtml(knob.category)}</span>
      <span>${escapeHtml(knob.description)}${knob.deprecated ? " Deprecated." : ""}</span>
    `;
    table.appendChild(row);
  }
}

function renderLogs(payload) {
  $("logPath").textContent = payload?.path || "runtime.log";
  const lines = payload?.lines || [];
  const output = $("logOutput");
  if (!output) return;
  if (!lines.length) {
    output.textContent = "No console runtime logs yet.";
    return;
  }
  output.innerHTML = lines.map((line) => highlightLogLine(line)).join("\n");
  touchFreshness("logs");
}

function renderLogInsights(payload) {
  state.logInsights = payload;
  const cards = $("logInsightCards");
  const recs = $("logRecommendations");
  const events = $("logEvents");
  cards.innerHTML = "";
  recs.innerHTML = "";
  events.innerHTML = "";
  if (!payload) {
    $("logInsightSummary").textContent = "not analyzed";
    return;
  }
  const counts = payload.counts || { block: 0, error: 0, warn: 0, info: 0 };
  $("logInsightSummary").textContent = payload.summary || payload.status || "analyzed";

  const statusCard = document.createElement("div");
  statusCard.className = `log-insight-card ${payload.status || "info"}`;
  statusCard.innerHTML = `
    <span>Status</span>
    <strong>${escapeHtml(payload.status || "unknown")}</strong>
    <p>${escapeHtml(payload.line_count || 0)} sampled line${payload.line_count === 1 ? "" : "s"}</p>
  `;
  cards.appendChild(statusCard);
  for (const [label, value, kind] of [
    ["Block", counts.block || 0, "blocked"],
    ["Error", counts.error || 0, "error"],
    ["Warn", counts.warn || 0, "attention"],
  ]) {
    const card = document.createElement("div");
    card.className = `log-insight-card ${Number(value) ? kind : "clean"}`;
    card.innerHTML = `<span>${label}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(payload.path || "console log")}</p>`;
    cards.appendChild(card);
  }

  const recommendations = payload.recommendations || [];
  if (!recommendations.length) {
    recs.innerHTML = '<div class="log-recommendation clean"><strong>No log action needed</strong><p>The sampled console log has no recognized warning or error signatures.</p></div>';
  } else {
    for (const item of recommendations.slice(0, 4)) {
      const row = document.createElement("div");
      row.className = `log-recommendation ${item.severity || "info"}`;
      row.innerHTML = `
        <strong>${escapeHtml(item.title || item.target || "Log finding")}</strong>
        <p>${escapeHtml(item.action || "")}</p>
        <span>${escapeHtml(item.evidence || "")}</span>
      `;
      recs.appendChild(row);
    }
  }

  const logEvents = payload.events || [];
  if (!logEvents.length) {
    events.innerHTML = '<div class="log-event clean"><strong>No warning/error events</strong><p>Raw tail remains available below.</p></div>';
  } else {
    for (const item of logEvents.slice(-10).reverse()) {
      const row = document.createElement("div");
      row.className = `log-event ${item.severity || "info"}`;
      const sig = (item.signatures || []).length ? `<span>${escapeHtml(item.signatures.join(", "))}</span>` : "";
      row.innerHTML = `<strong>Line ${escapeHtml(item.line)}</strong><p>${escapeHtml(item.message || "")}</p>${sig}`;
      events.appendChild(row);
    }
  }
}

function renderDiagnostics(payload) {
  state.diagnostics = payload;
  syncRuntimeIdentity();
  updateRuntimePill();
  const grid = $("readinessGrid");
  grid.innerHTML = "";
  const ports = payload?.ports || [];
  const artifacts = payload?.artifacts || { total: 0, missing: 0, ready: false, items: [] };
  const gpu = payload?.gpu || {};
  const busyPorts = ports.filter((item) => item.busy && item.label !== "Console");
  const ds8Runtimes = payload?.ds8_runtimes || (payload?.processes || []).filter((item) => item.looks_like_ds8);
  const externalDs8 = ds8Runtimes.filter((item) => !item.managed_by_console);
  $("readinessSummary").textContent = `${externalDs8.length} external DS8 / ${busyPorts.length} busy port / ${artifacts.missing || 0} missing artifact`;

  for (const port of ports) {
    const tile = document.createElement("div");
    tile.className = `readiness-tile ${port.busy ? "warn" : "ok"}`;
    const owner = port.owner?.users || port.owner?.raw || "";
    tile.innerHTML = `<span>${escapeHtml(port.label)}</span><strong>${port.busy ? "busy" : "free"}</strong><p>${escapeHtml(port.host)}:${port.port}${owner ? " - " + escapeHtml(owner) : ""}</p>`;
    grid.appendChild(tile);
  }

  const artifactTile = document.createElement("div");
  artifactTile.className = `readiness-tile ${artifacts.missing ? "warn" : "ok"}`;
  artifactTile.innerHTML = `<span>Artifacts</span><strong>${artifacts.ready ? "ready" : "attention"}</strong><p>${artifacts.total || 0} checked, ${artifacts.missing || 0} missing</p>`;
  grid.appendChild(artifactTile);

  const gpuTile = document.createElement("div");
  gpuTile.className = `readiness-tile ${gpu.available ? "ok" : "warn"}`;
  const gpuText = gpu.available && gpu.gpus?.length
    ? gpu.gpus.map((item) => `${item.name} ${item.utilization_gpu_pct}% / ${item.memory_used_mib}MiB`).join(", ")
    : (gpu.error || "unavailable");
  gpuTile.innerHTML = `<span>GPU</span><strong>${gpu.available ? "visible" : "unknown"}</strong><p>${escapeHtml(gpuText)}</p>`;
  grid.appendChild(gpuTile);

  const runtimeTile = document.createElement("div");
  runtimeTile.className = `readiness-tile ${externalDs8.length ? "warn" : "ok"}`;
  const managedCount = ds8Runtimes.filter((item) => item.managed_by_console).length;
  const runtimeText = externalDs8.length
    ? externalDs8.map((item) => `pid ${item.pid}`).join(", ")
    : (managedCount ? `${managedCount} console-managed` : "none observed");
  runtimeTile.innerHTML = `<span>DS8 Runtimes</span><strong>${externalDs8.length ? "external active" : "clear"}</strong><p>${escapeHtml(runtimeText)}</p>`;
  grid.appendChild(runtimeTile);

  for (const item of (artifacts.items || []).filter((entry) => !entry.exists).slice(0, 4)) {
    const tile = document.createElement("div");
    tile.className = "readiness-tile warn";
    tile.innerHTML = `<span>${escapeHtml(item.label)}</span><strong>missing</strong><p>${escapeHtml(item.path)}</p>`;
    grid.appendChild(tile);
  }
  renderProcesses(payload?.processes || []);
  if (shouldAutoProbeLive() && hasLiveTarget()) {
    probeLive().catch(() => {});
  }
  renderRuntime(state.runtime);
  touchFreshness("diagnostics");
}

function renderProcesses(processes) {
  const box = $("processList");
  const count = processes.length;
  const ds8Count = processes.filter((item) => item.looks_like_ds8).length;
  $("processSummary").textContent = count ? `${count} owner${count === 1 ? "" : "s"} / ${ds8Count} DS8` : "no owners";
  box.innerHTML = "";
  if (!count) {
    box.innerHTML = '<div class="process-card empty"><strong>No selected-port owners or DS8 runtimes observed</strong><p>Validate or scan diagnostics after choosing the DS8 ports you care about.</p></div>';
    return;
  }
  for (const process of processes) {
    const card = document.createElement("div");
    card.className = `process-card ${process.managed_by_console ? "managed" : "external"}`;
    const ports = (process.ports || []).map((item) => `${item.label}:${item.port}${item.selected ? " selected" : ""}`).join("  ");
    const stats = process.stats || {};
    const launch = process.launch || {};
    const launchSize = launch.size || (launch.pgie_profile ? "auto" : "");
    const launchRows = [
      ["PGIE", launch.pgie_profile],
      ["Size", launchSize],
      ["Tracking", launch.tracking_mode],
      ["Pipeline", launch.pipeline_config],
      ["Cameras", launch.cameras_config],
      ["WS", launch.ws_port],
      ["REST", launch.rest_port],
    ]
      .filter(([, value]) => value !== undefined && value !== null && value !== "")
      .slice(0, 7)
      .map(([key, value]) => `<span>${escapeHtml(key)} ${escapeHtml(value)}</span>`)
      .join("");
    const envRows = Object.entries(process.env || {})
      .slice(0, 6)
      .map(([key, value]) => `<span>${escapeHtml(key)}=${escapeHtml(value)}</span>`)
      .join("");
    const role = process.managed_by_console ? "console-managed" : "external";
    const kind = process.looks_like_ds8 ? "DS8 runtime" : "port owner";
    const adopt = process.looks_like_ds8
      ? `<div class="process-actions"><button type="button" class="button subtle" data-adopt-runtime="${escapeHtml(process.pid)}">Target this runtime</button></div>`
      : "";
    card.innerHTML = `
      <div class="process-title">
        <strong>PID ${escapeHtml(process.pid)} · ${escapeHtml(kind)}</strong>
        <span>${escapeHtml(role)}</span>
      </div>
      <p>${escapeHtml(ports || "no mapped ports")}</p>
      <code>${escapeHtml(process.command || process.stats?.comm || "command unavailable")}</code>
      <div class="process-meta">
        <span>cwd ${escapeHtml(process.cwd || "-")}</span>
        <span>age ${escapeHtml(stats.elapsed_s ?? "-")}s</span>
        <span>cpu ${escapeHtml(stats.cpu_pct ?? "-")}%</span>
        <span>rss ${escapeHtml(stats.rss_kib ?? "-")} KiB</span>
      </div>
      ${launchRows ? `<div class="process-launch">${launchRows}</div>` : ""}
      ${envRows ? `<div class="process-env">${envRows}</div>` : ""}
      ${adopt}
    `;
    box.appendChild(card);
  }
}

function renderBundle(payload) {
  state.bundle = payload;
  const box = $("bundleResult");
  if (!payload) {
    box.innerHTML = "";
    return;
  }
  const stateText = payload.blocking ? "saved with blocking findings" : "saved";
  box.innerHTML = `
    <strong>Support bundle ${escapeHtml(stateText)}</strong>
    <p>${escapeHtml(payload.bundle_path || "")}</p>
  `;
}

function renderBundleLibrary(payload) {
  state.bundleLibrary = payload;
  const list = $("bundleLibraryList");
  list.innerHTML = "";
  $("bundleLibraryRoot").textContent = payload?.root || "diagnostics/dev_console/bundles";
  const items = payload?.items || [];
  $("bundleLibrarySummary").textContent = items.length
    ? `${items.length} shown / ${payload.total || items.length} saved`
    : "no bundles";
  if (!items.length) {
    list.innerHTML = '<div class="bundle-card empty"><strong>No support bundles</strong><p>Save a bundle from Live Runtime to capture the current launch evidence.</p></div>';
    return;
  }
  for (const item of items) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `bundle-card ${item.blocking ? "blocked" : "ready"}`;
    card.dataset.bundleId = item.id;
    const counts = item.counts || {};
    const spec = item.spec || {};
    const ports = spec.ports || {};
    card.innerHTML = `
      <div class="bundle-card-title">
        <strong>${escapeHtml(item.created_at || item.id)}</strong>
        <span>${escapeHtml(counts.block || 0)} / ${escapeHtml(counts.warn || 0)} / ${escapeHtml(counts.info || 0)}</span>
      </div>
      <p>${escapeHtml(spec.pgie_profile || "-")}:${escapeHtml(spec.size || "auto")} ${escapeHtml(spec.tracking_mode || "")}</p>
      <code>${escapeHtml(ports.ws ?? "-")} / ${escapeHtml(ports.rest ?? "-")} / ${escapeHtml(ports.rtsp ?? "-")}</code>
    `;
    list.appendChild(card);
  }
}

function renderBundleDetail(payload) {
  state.bundleDetail = payload;
  const box = $("bundleDetail");
  if (!payload) {
    box.innerHTML = "<strong>No bundle selected</strong><p>Select a support bundle to inspect its validation, process, live, and artifact snapshot.</p>";
    return;
  }
  const summary = payload.summary || {};
  const counts = summary.counts || {};
  const artifacts = summary.artifacts || {};
  const runtime = summary.runtime || {};
  const decision = summary.decision || {};
  const spec = summary.spec || {};
  const markdown = payload.markdown || "";
  box.innerHTML = `
    <div class="bundle-detail-head">
      <strong>${escapeHtml(summary.created_at || summary.id || "Support bundle")}</strong>
      <span>${escapeHtml(summary.launch_id || "-")}</span>
    </div>
    <div class="bundle-detail-grid">
      <div><span>Validation</span><strong>${escapeHtml(counts.block || 0)} / ${escapeHtml(counts.warn || 0)} / ${escapeHtml(counts.info || 0)}</strong></div>
      <div><span>Artifacts</span><strong>${escapeHtml(artifacts.total || 0)} / ${escapeHtml(artifacts.missing || 0)}</strong></div>
      <div><span>Processes</span><strong>${escapeHtml(summary.process_count || 0)}</strong></div>
      <div><span>Live WS</span><strong>${summary.live_connected ? "connected" : "not connected"}</strong></div>
      <div><span>Runtime</span><strong>${runtime.running ? `pid ${escapeHtml(runtime.pid)}` : "stopped"}</strong></div>
      <div><span>Decision</span><strong>${escapeHtml(decision.status || "-")} ${decision.score !== undefined && decision.score !== null ? escapeHtml(decision.score) : ""}</strong></div>
    </div>
    <div class="bundle-detail-meta">
      <span>${escapeHtml(spec.pgie_profile || "-")}:${escapeHtml(spec.size || "auto")}</span>
      <span>${escapeHtml(spec.tracking_mode || "-")}</span>
      <span>${escapeHtml(summary.bundle_path || "")}</span>
    </div>
    <pre>${escapeHtml(markdown || "No summary markdown available.")}</pre>
  `;
}

function renderLaunchPlan(payload) {
  state.launchPlan = payload;
  const overview = $("planOverview");
  const gates = $("planGates");
  const changes = $("planChanges");
  overview.innerHTML = "";
  gates.innerHTML = "";
  changes.innerHTML = "";
  if (!payload) {
    $("planSummary").textContent = "not materialized";
    return;
  }
  const artifactText = `${payload.artifacts?.total || 0} artifacts / ${payload.artifacts?.missing || 0} missing`;
  $("planSummary").textContent = `${payload.change_count || 0} changed / ${artifactText}`;
  const ports = payload.ports || {};
  overview.innerHTML = `
    <div class="plan-command"><span>Command</span><code>${escapeHtml(payload.command || "")}</code></div>
    <div class="plan-facts">
      <div><span>Pipeline</span><strong>${escapeHtml(payload.materialized_pipeline_rel || payload.materialized_pipeline || "-")}</strong></div>
      <div><span>WebSocket</span><strong>${escapeHtml(ports.ws?.host || "-")}:${escapeHtml(ports.ws?.port ?? "-")}</strong></div>
      <div><span>REST</span><strong>${escapeHtml(ports.rest?.host || "-")}:${escapeHtml(ports.rest?.port ?? "-")}</strong></div>
      <div><span>RTSP</span><strong>${escapeHtml(ports.rtsp?.host || "-")}:${escapeHtml(ports.rtsp?.port ?? "-")}</strong></div>
      <div><span>Artifacts</span><strong>${escapeHtml(artifactText)}</strong></div>
    </div>
  `;
  for (const gate of payload.gates || []) {
    const row = document.createElement("div");
    row.className = "plan-gate";
    row.innerHTML = `<span>${escapeHtml(gate.label)}</span><strong>${escapeHtml(gate.value)}</strong><em>${escapeHtml(gate.source)}</em>`;
    gates.appendChild(row);
  }
  const changeItems = (payload.changes || []).filter((item) => item.changed).concat((payload.changes || []).filter((item) => !item.changed)).slice(0, 18);
  for (const change of changeItems) {
    const row = document.createElement("div");
    row.className = `plan-change ${change.changed ? "changed" : ""}`;
    row.innerHTML = `
      <strong>${escapeHtml(change.label)}</strong>
      <span>${escapeHtml(change.path)}</span>
      <p>${escapeHtml(change.before ?? "unset")} -> ${escapeHtml(change.after ?? "unset")}</p>
    `;
    changes.appendChild(row);
  }
}

function renderLaunchDiff(payload) {
  state.launchDiff = payload;
  const stats = $("diffStats");
  const launchFields = $("diffLaunchFields");
  const box = $("launchDiff");
  stats.innerHTML = "";
  launchFields.innerHTML = "";
  box.innerHTML = "";
  if (!payload) {
    $("diffSummary").textContent = "not compared";
    return;
  }
  const summary = payload.summary || { total: 0, high_impact: 0, env_overrides: 0, categories: 0 };
  $("diffSummary").textContent = `${summary.total || 0} changes / ${summary.high_impact || 0} high impact`;

  for (const item of [
    ["Total", summary.total || 0],
    ["High", summary.high_impact || 0],
    ["Env", summary.env_overrides || 0],
    ["Categories", summary.categories || 0],
  ]) {
    const tile = document.createElement("div");
    tile.className = `diff-stat ${item[0] === "High" && Number(item[1]) ? "high" : ""}`;
    tile.innerHTML = `<span>${escapeHtml(item[0])}</span><strong>${escapeHtml(item[1])}</strong>`;
    stats.appendChild(tile);
  }

  for (const field of (payload.launch || []).slice(0, 9)) {
    const chip = document.createElement("div");
    chip.className = "diff-launch-field";
    chip.innerHTML = `<span>${escapeHtml(field.path.replace(/^launch\\./, ""))}</span><strong>${escapeHtml(field.after_text ?? field.after ?? "-")}</strong>`;
    launchFields.appendChild(chip);
  }

  const highOnly = Boolean($("diffHighOnly")?.checked);
  const rows = (payload.changes || []).filter((item) => !highOnly || item.high_impact);
  if (!rows.length) {
    box.innerHTML = '<div class="diff-row empty"><strong>No matching differences</strong><p>Disable the high-impact filter to inspect low-level config deltas.</p></div>';
    return;
  }
  for (const change of rows.slice(0, 80)) {
    const row = document.createElement("div");
    row.className = `diff-row ${change.kind || "changed"} ${change.high_impact ? "high" : ""}`;
    row.innerHTML = `
      <div class="diff-title">
        <strong>${escapeHtml(change.path || "-")}</strong>
        <span>${escapeHtml(change.category || "")} / ${escapeHtml(change.kind || "changed")}</span>
      </div>
      <div class="diff-values">
        <code>${escapeHtml(change.before_text ?? "unset")}</code>
        <code>${escapeHtml(change.after_text ?? "unset")}</code>
      </div>
    `;
    box.appendChild(row);
  }
}

function renderLaunchCandidates(payload) {
  state.launchCandidates = payload;
  const deck = $("candidateDeck");
  deck.innerHTML = "";
  if (!payload) {
    $("candidateSummary").textContent = "not compared";
    deck.innerHTML = '<div class="candidate-card empty"><strong>No candidates compared</strong><p>Compare launch candidates after choosing a model, tracking mode, ports, and gates.</p></div>';
    return;
  }
  const summary = payload.summary || {};
  $("candidateSummary").textContent = `${summary.ready || 0} ready / ${summary.attention || 0} attention / ${summary.blocked || 0} blocked`;
  const items = payload.items || [];
  if (!items.length) {
    deck.innerHTML = '<div class="candidate-card empty"><strong>No candidates available</strong><p>The current launch selection did not produce alternate recipes.</p></div>';
    return;
  }
  for (const candidate of items.slice(0, 8)) {
    const card = document.createElement("div");
    const status = actionStatusClass(candidate.status);
    card.className = `candidate-card ${status} ${candidate.id === summary.best_id ? "best" : ""}`;
    const changes = (candidate.changes || [])
      .slice(0, 5)
      .map((item) => `<span>${escapeHtml(item.label || item.field)}: ${escapeHtml(item.after ?? "unset")}</span>`)
      .join("");
    const blockers = (candidate.blockers || [])
      .slice(0, 3)
      .map((item) => `<p>${escapeHtml(item.label || "Finding")}: ${escapeHtml(item.detail || item.status || "")}</p>`)
      .join("");
    const ports = candidate.ports || {};
    const model = candidate.model || {};
    card.innerHTML = `
      <div class="candidate-title">
        <div>
          <strong>${escapeHtml(candidate.title || candidate.id || "Candidate")}</strong>
          <p>${escapeHtml(candidate.intent || candidate.summary || "")}</p>
        </div>
        <span>${escapeHtml(candidate.status || "unknown")} / ${escapeHtml(candidate.score ?? 0)}</span>
      </div>
      <div class="candidate-facts">
        <div><span>Model</span><strong>${escapeHtml(model.pgie_profile || "-")}:${escapeHtml(model.size || "auto")}</strong></div>
        <div><span>Tracking</span><strong>${escapeHtml(model.tracking_mode || "-")}</strong></div>
        <div><span>Ports</span><strong>${escapeHtml(ports.ws ?? "-")} / ${escapeHtml(ports.rest ?? "-")} / ${escapeHtml(ports.rtsp ?? "-")}</strong></div>
      </div>
      <div class="candidate-changes">${changes || "<span>Current form</span>"}</div>
      <div class="candidate-findings">${blockers || `<p>${escapeHtml(candidate.summary || "No blocking finding")}</p>`}</div>
    `;
    const actions = document.createElement("div");
    actions.className = "candidate-actions";
    const apply = document.createElement("button");
    apply.className = "button subtle";
    apply.type = "button";
    apply.textContent = "Apply";
    apply.addEventListener("click", () => applyLaunchCandidate(candidate, false).catch((err) => toast(err.message)));
    const applyValidate = document.createElement("button");
    applyValidate.className = "button primary";
    applyValidate.type = "button";
    applyValidate.textContent = "Apply + Validate";
    applyValidate.addEventListener("click", () => applyLaunchCandidate(candidate, true).catch((err) => toast(err.message)));
    actions.appendChild(apply);
    actions.appendChild(applyValidate);
    card.appendChild(actions);
    deck.appendChild(card);
  }
}

function renderLaunchDecision(payload) {
  state.launchDecision = payload;
  state.launchDecisionStale = false;
  if (payload?.validation) {
    renderValidation(payload.validation);
  }
  if (!payload) {
    $("decisionSummary").textContent = "not evaluated";
    $("decisionScore").textContent = "--";
    $("decisionStatus").textContent = "unknown";
    document.querySelector(".decision-score")?.setAttribute("data-status", "unknown");
    $("decisionPrimary").innerHTML = `
      <strong>No decision yet</strong>
      <p>Evaluate the current launch selection to see the go/no-go verdict.</p>
      <div class="panel-empty-cta" style="margin-top:10px">
        <button type="button" class="button primary" id="decisionEmptyEvaluate">Evaluate Launch</button>
      </div>`;
    $("decisionPrimary").querySelector("#decisionEmptyEvaluate")?.addEventListener("click", () => loadLaunchDecision(true).catch((err) => toast(err.message)));
    $("decisionComponents").innerHTML = "";
    $("decisionActions").innerHTML = "";
    renderActionBoard(null);
    return;
  }
  const status = payload.status || "unknown";
  $("decisionSummary").textContent = payload.summary || status;
  $("decisionScore").textContent = `${payload.score ?? "--"}`;
  $("decisionStatus").textContent = status;
  document.querySelector(".decision-score")?.setAttribute("data-status", status);

  const primary = payload.primary_action || {};
  $("decisionPrimary").className = `decision-primary ${primary.severity || status}`;
  $("decisionPrimary").innerHTML = `
    <strong>${escapeHtml(primary.title || payload.summary || "Launch decision")}</strong>
    <p>${escapeHtml(primary.detail || "")}</p>
  `;

  const components = $("decisionComponents");
  components.innerHTML = "";
  for (const item of payload.components || []) {
    const tile = document.createElement("div");
    tile.className = `decision-component ${item.status || "ready"}`;
    tile.innerHTML = `
      <span>${escapeHtml(item.label)}</span>
      <strong>${escapeHtml(item.metric || item.status || "-")}</strong>
      <p>${escapeHtml(item.detail || "")}</p>
    `;
    components.appendChild(tile);
  }

  const actions = $("decisionActions");
  actions.innerHTML = "";
  const topActions = payload.top_actions || [];
  if (!topActions.length) {
    actions.innerHTML = '<div class="decision-action ready"><strong>No runbook action needed</strong><p>The selected launch is ready from the decision board evidence.</p></div>';
    return;
  }
  for (const action of topActions.slice(0, 4)) {
    const row = document.createElement("div");
    row.className = `decision-action ${action.severity || "info"}`;
    row.innerHTML = `<strong>${escapeHtml(action.title || "Action")}</strong><p>${escapeHtml(action.reason || action.action || "")}</p>`;
    actions.appendChild(row);
  }
  renderActionBoard(payload);
  updateSimpleLayout();
}

function selectedSegmentValue(segment) {
  const selected = segment.querySelector(".selected");
  return selected ? selected.dataset.value ?? "" : "";
}

function setSegmentValue(segment, value) {
  const nextValue = String(value ?? "");
  segment.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("selected", (button.dataset.value ?? "") === nextValue);
  });
}

function renderGateDeck(payload) {
  state.gates = payload;
  const deck = $("gateDeck");
  deck.innerHTML = "";
  const summary = payload?.summary || { total: 0, active: 0, explicit_overrides: 0 };
  $("gateSummary").textContent = payload
    ? `${summary.active || 0} active / ${summary.explicit_overrides || 0} override`
    : "not loaded";
  if (!payload || !(payload.groups || []).length) {
    deck.innerHTML = `
      <div class="gate-group empty panel-empty-cta">
        <strong>No gates loaded</strong>
        <p>Refresh the deck after selecting a launch profile and ports.</p>
        <button type="button" class="button primary" id="gateEmptyRefresh">Refresh Gates</button>
      </div>`;
    deck.querySelector("#gateEmptyRefresh")?.addEventListener("click", () => loadGates(true).catch((err) => toast(err.message)));
    return;
  }

  for (const group of payload.groups || []) {
    const card = document.createElement("div");
    card.className = "gate-group";
    const controls = document.createElement("div");
    controls.className = "gate-controls";
    card.innerHTML = `
      <div class="gate-group-title">
        <div>
          <strong>${escapeHtml(group.label)}</strong>
          <p>${escapeHtml(group.summary || "")}</p>
        </div>
        <span>${escapeHtml(group.active || 0)} on</span>
      </div>
    `;
    for (const control of group.controls || []) {
      const row = document.createElement("div");
      row.className = `gate-control ${control.active ? "active" : ""} ${control.explicit ? "explicit" : ""}`;
      const current = currentTargetValue(control.target, control.value ?? "");
      const selectedValue = control.inheritable && current === "" ? "" : (current || control.value || "");
      const source = control.source || (control.explicit ? "override" : "inherited");
      const effective = control.effective_display || control.effective_value || "-";
      row.innerHTML = `
        <div class="gate-copy">
          <strong>${escapeHtml(control.label)}</strong>
          <p>${escapeHtml(control.detail || "")}</p>
          <span>${escapeHtml(source)} · ${escapeHtml(effective)}</span>
        </div>
      `;
      const editor = document.createElement("div");
      editor.className = "gate-editor";
      if (control.kind === "boolean" && control.inheritable) {
        const segment = document.createElement("div");
        segment.className = "gate-segment";
        segment.dataset.gateTarget = control.target;
        segment.dataset.gateKind = control.kind;
        for (const option of control.options || []) {
          const button = document.createElement("button");
          button.type = "button";
          button.dataset.value = option.value ?? "";
          button.textContent = option.label || option.value || "inherit";
          button.addEventListener("click", () => setSegmentValue(segment, button.dataset.value ?? ""));
          segment.appendChild(button);
        }
        setSegmentValue(segment, selectedValue);
        editor.appendChild(segment);
      } else if (control.kind === "boolean") {
        const label = document.createElement("label");
        label.className = "gate-switch";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.dataset.gateTarget = control.target;
        input.dataset.gateKind = control.kind;
        input.checked = String(selectedValue || control.effective_value || "0") === "1";
        const span = document.createElement("span");
        label.appendChild(input);
        label.appendChild(span);
        editor.appendChild(label);
      } else if (control.kind === "select") {
        const select = document.createElement("select");
        select.dataset.gateTarget = control.target;
        select.dataset.gateKind = control.kind;
        for (const option of control.options || []) {
          const item = document.createElement("option");
          item.value = option.value;
          item.textContent = option.label || option.value;
          select.appendChild(item);
        }
        select.value = selectedValue || control.effective_value || "";
        editor.appendChild(select);
      } else {
        const input = document.createElement("input");
        input.type = "number";
        input.dataset.gateTarget = control.target;
        input.dataset.gateKind = control.kind || "number";
        if (control.min !== undefined) input.min = control.min;
        if (control.max !== undefined) input.max = control.max;
        if (control.step !== undefined) input.step = control.step;
        input.placeholder = control.inheritable ? `inherit ${control.inherited_value || ""}` : "";
        input.value = selectedValue;
        editor.appendChild(input);
      }
      const impact = document.createElement("p");
      impact.className = "gate-impact";
      impact.textContent = control.impact || "";
      row.appendChild(editor);
      row.appendChild(impact);
      controls.appendChild(row);
    }
    card.appendChild(controls);
    deck.appendChild(card);
  }
}

function readGateControlValue(node) {
  if (node.classList.contains("gate-segment")) return selectedSegmentValue(node);
  if (node.type === "checkbox") return node.checked ? "1" : "0";
  return node.value;
}

async function applyGateControls(validateAfter) {
  const nodes = document.querySelectorAll("#gateDeck [data-gate-target]");
  for (const node of nodes) {
    setTargetValue(node.dataset.gateTarget, readGateControlValue(node));
  }
  markValidationStale();
  await renderFlow();
  await loadDiagnostics();
  await loadLaunchPlan().catch(() => {});
  await loadLaunchDiff().catch(() => {});
  await loadRemediation().catch(() => {});
  await loadGates(true).catch(() => {});
  await loadModelMatrix().catch(() => {});
  await loadSources().catch(() => {});
  await loadLaunchDecision().catch(() => {});
  toast("Gate deck applied");
  if (validateAfter) await validateLaunch();
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!bytes) return "-";
  if (bytes > 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GiB`;
  if (bytes > 1024 * 1024) return `${Math.round(bytes / (1024 * 1024))} MiB`;
  if (bytes > 1024) return `${Math.round(bytes / 1024)} KiB`;
  return `${bytes} B`;
}

function renderModelMatrix(payload) {
  state.modelMatrix = payload;
  const box = $("modelMatrix");
  box.innerHTML = "";
  const summary = payload?.summary || { total: 0, ready: 0, warn: 0, blocked: 0 };
  $("matrixSummary").textContent = `${summary.ready || 0} ready / ${summary.warn || 0} warn / ${summary.blocked || 0} blocked`;
  const readyOnly = Boolean($("matrixReadyOnly")?.checked);
  const rows = (payload?.rows || []).filter((row) => !readyOnly || row.status === "ready");
  if (!rows.length) {
    box.innerHTML = '<div class="matrix-row empty"><strong>No rows</strong><p>No model combinations match the current filter.</p></div>';
    return;
  }
  for (const row of rows) {
    const card = document.createElement("div");
    card.className = `matrix-row ${row.status || "blocked"} ${row.active ? "active" : ""}`;
    const missing = row.artifacts?.missing_labels?.length
      ? `<p>${escapeHtml(row.artifacts.missing_labels.join(", "))}</p>`
      : "";
    const findings = (row.top_findings || [])
      .filter((item) => item.severity !== "info")
      .slice(0, 2)
      .map((item) => `<span>${escapeHtml(item.code)}</span>`)
      .join("");
    card.innerHTML = `
      <div class="matrix-title">
        <strong>${escapeHtml(row.pgie_profile)}:${escapeHtml(row.size)}</strong>
        <span>${escapeHtml(row.status)}${row.active ? " / active" : ""}${row.preferred ? " / default" : ""}</span>
      </div>
      <div class="matrix-badges">
        <span>${escapeHtml(row.kind || "model")}</span>
        <span>${row.mask_available ? "mask tensors" : "boxes"}</span>
        <span>${escapeHtml(row.network_type || "type -")}</span>
      </div>
      <div class="matrix-stats">
        <div><span>Artifacts</span><strong>${escapeHtml(row.artifacts?.total || 0)} / ${escapeHtml(row.artifacts?.missing || 0)}</strong></div>
        <div><span>Preflight</span><strong>${escapeHtml(row.counts?.block || 0)} / ${escapeHtml(row.counts?.warn || 0)}</strong></div>
        <div><span>Engine</span><strong>${escapeHtml(formatBytes(row.engine_size_bytes))}</strong></div>
      </div>
      <code>${escapeHtml(row.engine || row.config || row.error || "-")}</code>
      ${missing}
      ${findings ? `<div class="matrix-findings">${findings}</div>` : ""}
    `;
    const actions = document.createElement("div");
    actions.className = "matrix-actions";
    const apply = document.createElement("button");
    apply.className = "button subtle";
    apply.type = "button";
    apply.textContent = "Apply";
    apply.addEventListener("click", () => applyModelMatrixRow(row, false).catch((err) => toast(err.message)));
    const applyValidate = document.createElement("button");
    applyValidate.className = "button primary";
    applyValidate.type = "button";
    applyValidate.textContent = "Apply + Validate";
    applyValidate.addEventListener("click", () => applyModelMatrixRow(row, true).catch((err) => toast(err.message)));
    actions.appendChild(apply);
    actions.appendChild(applyValidate);
    card.appendChild(actions);
    box.appendChild(card);
  }
}

function renderSources(payload) {
  state.sources = payload;
  const summary = payload?.summary || { ready: 0, warn: 0, blocked: 0, total: 0, source_count: 0, camera_count: 0 };
  $("sourceSummary").textContent = `${summary.ready || 0} ready / ${summary.warn || 0} warn / ${summary.blocked || 0} blocked`;

  const alignmentBox = $("sourceAlignment");
  alignmentBox.innerHTML = "";
  const alignment = payload?.alignment || [];
  if (!alignment.length) {
    alignmentBox.innerHTML = `<div class="source-alignment-card ok"><strong>${escapeHtml(summary.source_count || 0)} sources match ${escapeHtml(summary.camera_count || 0)} cameras</strong><p>Batch sizing and camera count are aligned for this launch selection.</p></div>`;
  } else {
    for (const item of alignment) {
      const card = document.createElement("div");
      card.className = `source-alignment-card ${item.severity || "warn"}`;
      card.innerHTML = `<strong>${escapeHtml(item.message)}</strong><p>${escapeHtml(item.detail || item.code || "")}</p>`;
      alignmentBox.appendChild(card);
    }
  }

  const grid = $("sourceGrid");
  grid.innerHTML = "";
  const rows = payload?.rows || [];
  if (!rows.length) {
    grid.innerHTML = '<div class="source-card empty"><strong>No sources configured</strong><p>The selected pipeline has no source entries.</p></div>';
    return;
  }
  for (const row of rows) {
    const card = document.createElement("div");
    card.className = `source-card ${row.status || "blocked"}`;
    const probe = row.uri_probe || {};
    const probeText = probe.kind === "rtsp"
      ? (probe.ok === true ? `${probe.host}:${probe.port} ${probe.latency_ms} ms` : (probe.ok === null ? "probe skipped" : `${probe.host || "host"}:${probe.port || "-"} unavailable`))
      : (probe.kind === "file" ? (probe.ok ? "file present" : "file missing") : (probe.error || probe.kind || "-"));
    const dewarperText = row.dewarper?.enabled
      ? `${row.dewarper.exists ? "ready" : "missing"}${row.dewarper.matches_camera_model === false ? " / mismatch" : ""}`
      : "disabled";
    const findings = (row.findings || [])
      .slice(0, 3)
      .map((item) => `<span>${escapeHtml(item.code)}</span>`)
      .join("");
    card.innerHTML = `
      <div class="source-title">
        <strong>${escapeHtml(row.source_id)} · ${escapeHtml(row.camera_id)}</strong>
        <span>${escapeHtml(row.status || "unknown")}</span>
      </div>
      <code>${escapeHtml(row.uri_display || "-")}</code>
      <div class="source-stats">
        <div><span>Probe</span><strong>${escapeHtml(probeText)}</strong></div>
        <div><span>Intrinsics</span><strong>${row.camera?.intrinsics_ready ? "ready" : "missing"}</strong></div>
        <div><span>Dewarper</span><strong>${escapeHtml(dewarperText)}</strong></div>
      </div>
      <div class="source-meta">
        <span>${escapeHtml(row.camera?.model || "no model")}</span>
        <span>height ${escapeHtml(row.camera?.height_m ?? "-")} m</span>
        <span>latency ${escapeHtml(row.latency_ms ?? "-")} ms</span>
      </div>
      ${row.dewarper?.config_rel ? `<p>${escapeHtml(row.dewarper.config_rel)}</p>` : ""}
      ${findings ? `<div class="source-findings">${findings}</div>` : ""}
    `;
    grid.appendChild(card);
  }
}

function renderRemediation(payload) {
  state.remediation = payload;
  const list = $("runbookList");
  list.innerHTML = "";
  if (!payload) {
    $("runbookSummary").textContent = "not evaluated";
    return;
  }
  $("runbookSummary").textContent = payload.summary || payload.status || "evaluated";
  const actions = payload.actions || [];
  if (!actions.length) {
    list.innerHTML = '<div class="runbook-card info"><strong>Ready</strong><p>No runbook action is needed for this launch selection.</p></div>';
    return;
  }
  for (const action of actions) {
    const card = document.createElement("div");
    card.className = `runbook-card ${action.severity || "info"}`;
    const codes = (action.related_codes || []).length ? `<span>${escapeHtml(action.related_codes.join(", "))}</span>` : "";
    const command = action.command ? `<code>${escapeHtml(action.command)}</code>` : "";
    const target = actionTarget(action.target || "validation", action.title || "Open");
    const targetLabel = target.action === "plan" ? "Open Plan" : target.label;
    card.innerHTML = `
      <div class="runbook-title">
        <strong>${escapeHtml(action.title)}</strong>
        ${codes}
      </div>
      <p>${escapeHtml(action.reason || "")}</p>
      <p>${escapeHtml(action.action || "")}</p>
      ${command}
      <div class="runbook-actions">
        <button class="button subtle" type="button" data-runbook-action="${escapeHtml(target.action)}">${escapeHtml(targetLabel)}</button>
        <button class="button subtle" type="button" data-runbook-action="validate">Validate</button>
      </div>
    `;
    list.appendChild(card);
  }
}

function renderLive(payload) {
  state.live = payload;
  const box = $("liveStats");
  if (!payload || !payload.connected) {
    $("liveSummary").textContent = payload?.error ? "unavailable" : "not probed";
    box.innerHTML = `<strong>WebSocket unavailable</strong><p>${escapeHtml(payload?.error || "Probe the runtime WebSocket to read DS8 stats.")}</p>`;
    return;
  }
  const stats = payload.stats || {};
  const app = stats.application || {};
  const pipe = stats.pipeline || {};
  const stack = stats.stack || "ds8";
  const cameras = app.cameras_active ?? stats.camera_count ?? "-";
  $("liveSummary").textContent = `${stack} · ${cameras} cam${payload.pong ? " · pong" : ""}`;
  const types = Object.entries(payload.message_types || {}).map(([key, value]) => `${key}:${value}`).join(" ");
  box.innerHTML = `
    <div class="live-stat-grid">
      <div><span>Stack</span><strong>${escapeHtml(stats.stack || "ds8")}</strong></div>
      <div><span>Cameras</span><strong>${escapeHtml(app.cameras_active ?? stats.camera_count ?? "-")}</strong></div>
      <div><span>Prepared</span><strong>${escapeHtml(pipe.prepared)}</strong></div>
      <div><span>Activated</span><strong>${escapeHtml(pipe.activated)}</strong></div>
      <div><span>Depth</span><strong>${escapeHtml(pipe.depth_enabled)}</strong></div>
      <div><span>Zero Copy</span><strong>${escapeHtml(pipe.zero_copy_profile || "-")}</strong></div>
      <div><span>StableID</span><strong>${escapeHtml(pipe.stableid_backend_mode || "-")}</strong></div>
      <div><span>Trails</span><strong>${payload.trail_enabled === null ? "-" : escapeHtml(payload.trail_enabled)}</strong></div>
    </div>
    <p class="live-types">${escapeHtml(types || "No message types recorded")}</p>
  `;
  $("toggleTrailsButton").textContent = payload.trail_enabled === false ? "Enable Trails" : "Disable Trails";
  touchFreshness("live");
}

function renderLiveHealth(payload) {
  state.liveHealth = payload;
  if (!payload) {
    $("healthSummary").textContent = "not sampled";
    $("healthScore").textContent = "--";
    $("healthStatus").textContent = "unknown";
    $("healthIndicators").innerHTML = "";
    $("cameraList").innerHTML = "";
    $("healthHistory").innerHTML = "";
    return;
  }
  const score = payload.score ?? 0;
  $("healthSummary").textContent = payload.summary || payload.status || "sampled";
  $("healthScore").textContent = `${score}`;
  $("healthStatus").textContent = payload.status || "unknown";
  document.querySelector(".health-score")?.setAttribute("data-status", payload.status || "unknown");

  const indicators = $("healthIndicators");
  indicators.innerHTML = "";
  for (const item of payload.indicators || []) {
    const tile = document.createElement("div");
    tile.className = `health-indicator ${item.status || "info"}`;
    tile.innerHTML = `<span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong><p>${escapeHtml(item.detail || "")}</p>`;
    indicators.appendChild(tile);
  }

  const cameras = $("cameraList");
  cameras.innerHTML = "";
  for (const camera of payload.cameras || []) {
    const chip = document.createElement("span");
    chip.textContent = camera.id || "camera";
    cameras.appendChild(chip);
  }
  if (!(payload.cameras || []).length) {
    cameras.innerHTML = "<span>no cameras reported</span>";
  }

  state.healthHistory.unshift({
    sampled_at: payload.sampled_at,
    status: payload.status,
    score,
    summary: payload.summary,
  });
  state.healthHistory = state.healthHistory.slice(0, 8);
  renderHealthHistory();
  touchFreshness("health");
}

function renderHealthHistory() {
  const box = $("healthHistory");
  box.innerHTML = "";
  for (const sample of state.healthHistory) {
    const row = document.createElement("div");
    const when = sample.sampled_at ? new Date(sample.sampled_at * 1000).toLocaleTimeString() : "--";
    row.className = `health-history-row ${sample.status || ""}`;
    row.innerHTML = `<span>${escapeHtml(when)}</span><strong>${escapeHtml(sample.score)}</strong><p>${escapeHtml(sample.status || "unknown")}</p>`;
    box.appendChild(row);
  }
}

function activityPayloadSummary(payload) {
  const bits = [];
  if (payload?.profile?.id) bits.push(`profile ${payload.profile.id}`);
  if (payload?.spec?.pgie_profile) {
    const size = payload.spec.size ? `:${payload.spec.size}` : "";
    bits.push(`${payload.spec.pgie_profile}${size}`);
  }
  if (payload?.spec?.tracking_mode) bits.push(`tracking ${payload.spec.tracking_mode}`);
  if (payload?.spec?.ports) {
    const ports = payload.spec.ports;
    bits.push(`ports ${ports.ws}/${ports.rest}/${ports.rtsp}`);
  }
  if (payload?.validation?.counts) {
    const counts = payload.validation.counts;
    bits.push(`${counts.block || 0} block ${counts.warn || 0} warn`);
  }
  if (payload?.runtime?.pid) bits.push(`pid ${payload.runtime.pid}`);
  if (payload?.status !== undefined) bits.push(`status ${payload.status}`);
  if (payload?.score !== undefined) bits.push(`score ${payload.score}`);
  return bits.slice(0, 5).join(" · ");
}

function renderActivity(payload) {
  const items = payload?.items || [];
  state.activity = items;
  $("activitySummary").textContent = items.length
    ? `${items.length} shown / ${payload.total || items.length} recorded`
    : "no recorded actions";
  $("activityPath").textContent = payload?.path || "generated audit log";
  const list = $("activityList");
  list.innerHTML = "";
  if (!items.length) {
    list.innerHTML = '<div class="activity-item info"><strong>No activity yet</strong><p>Preview, validate, save a profile, or save a bundle to create the first audit entry.</p></div>';
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    const severity = item.severity || "info";
    const when = item.ts ? new Date(item.ts * 1000).toLocaleTimeString() : "--";
    const detail = item.detail ? `<p>${escapeHtml(item.detail)}</p>` : "";
    const summary = activityPayloadSummary(item.payload || {});
    row.className = `activity-item ${severity}`;
    row.innerHTML = `
      <div class="activity-title">
        <strong>${escapeHtml(item.title || item.type || "Activity")}</strong>
        <span>${escapeHtml(when)}</span>
      </div>
      ${detail}
      ${summary ? `<code>${escapeHtml(summary)}</code>` : ""}
    `;
    list.appendChild(row);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function refreshSummary() {
  const summary = await api("/api/summary");
  $("repoRoot").textContent = summary.repo_root || "workspace";
  renderRuntime(summary.runtime);
  renderPresets(summary.presets || []);
  $("knobMetric").textContent = `${summary.knob_count || 0}`;
}

async function loadKnobs() {
  const payload = await api("/api/knobs");
  state.knobs = payload.items || [];
  renderKnobs();
}

async function loadProfiles() {
  const payload = await api("/api/profiles");
  renderProfiles(payload.items || []);
  return payload;
}

async function loadActivity() {
  const payload = await api("/api/activity?limit=80");
  renderActivity(payload);
  return payload;
}

async function loadBundleLibrary() {
  const payload = await api("/api/support/bundles?limit=30");
  renderBundleLibrary(payload);
  return payload;
}

async function loadBundleDetail(bundleId) {
  const payload = await api(`/api/support/bundles/${encodeURIComponent(bundleId)}`);
  renderBundleDetail(payload);
  return payload;
}

async function loadLaunchDecision(recordActivity = false) {
  const spec = specFromForm();
  spec.probe_network = Boolean($("sourceProbeInput")?.checked);
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/launch/decision", { method: "POST", body: JSON.stringify(spec) });
  renderLaunchDecision(payload);
  if (recordActivity) await loadActivity().catch(() => {});
  return payload;
}

async function loadLaunchCandidates(recordActivity = false) {
  const spec = specFromForm();
  spec.probe_network = Boolean($("sourceProbeInput")?.checked);
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/launch/candidates", { method: "POST", body: JSON.stringify(spec) });
  renderLaunchCandidates(payload);
  if (recordActivity) await loadActivity().catch(() => {});
  return payload;
}

async function loadModelMatrix(recordActivity = false) {
  const spec = specFromForm();
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/model-matrix", { method: "POST", body: JSON.stringify(spec) });
  renderModelMatrix(payload);
  if (recordActivity) await loadActivity().catch(() => {});
  return payload;
}

async function loadSources(recordActivity = false) {
  const spec = specFromForm();
  spec.probe_network = Boolean($("sourceProbeInput")?.checked);
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/sources", { method: "POST", body: JSON.stringify(spec) });
  renderSources(payload);
  if (recordActivity) await loadActivity().catch(() => {});
  return payload;
}

async function loadGates(recordActivity = false) {
  const spec = specFromForm();
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/gates", { method: "POST", body: JSON.stringify(spec) });
  renderGateDeck(payload);
  if (recordActivity) await loadActivity().catch(() => {});
  return payload;
}

async function applyModelMatrixRow(row, validateAfter) {
  const spec = row.apply_spec || row;
  $("presetSelect").value = "";
  $("profileSelect").value = spec.pgie_profile || row.pgie_profile || "yolo11_seg";
  $("sizeSelect").value = spec.size || row.size || "";
  $("trackingSelect").value = spec.tracking_mode || $("trackingSelect").value || "baseline";
  markValidationStale();
  await renderFlow();
  await loadDiagnostics();
  await loadLaunchPlan().catch(() => {});
  await loadLaunchDiff().catch(() => {});
  await loadRemediation().catch(() => {});
  await loadGates().catch(() => {});
  await loadModelMatrix().catch(() => {});
  await loadSources().catch(() => {});
  await loadLaunchDecision().catch(() => {});
  toast(`Applied ${$("profileSelect").value}:${$("sizeSelect").value || "auto"}`);
  if (validateAfter) await validateLaunch();
}

async function applyLaunchCandidate(candidate, validateAfter) {
  const spec = candidate.apply_spec || {};
  await applySpecToForm(spec);
  toast(`Applied ${candidate.title || candidate.id || "candidate"}`);
  if (validateAfter) await validateLaunch();
}

async function saveCurrentProfile() {
  const name = $("profileNameInput").value.trim() || `${$("profileSelect").value} ${$("trackingSelect").value}`;
  const notes = $("profileNoteInput").value.trim();
  const payload = await api("/api/profiles/save", {
    method: "POST",
    body: JSON.stringify({ name, notes, spec: specFromForm() }),
  });
  await loadProfiles();
  $("savedProfileSelect").value = payload.summary?.id || "";
  await loadActivity().catch(() => {});
  toast(`Saved profile ${payload.summary?.name || name}`);
  return payload;
}

async function applySavedProfile() {
  const profileId = $("savedProfileSelect").value;
  if (!profileId) {
    toast("No saved profile selected");
    return null;
  }
  const payload = await api("/api/profiles/load", {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId }),
  });
  const spec = payload.profile?.spec || {};
  $("profileNameInput").value = payload.profile?.name || "";
  $("profileNoteInput").value = payload.profile?.notes || "";
  await applySpecToForm(spec);
  await loadActivity().catch(() => {});
  toast(`Applied profile ${payload.profile?.name || profileId}`);
  return payload;
}

async function deleteSavedProfile() {
  const profileId = $("savedProfileSelect").value;
  if (!profileId) {
    toast("No saved profile selected");
    return null;
  }
  const payload = await api("/api/profiles/delete", {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId }),
  });
  await loadProfiles();
  await loadActivity().catch(() => {});
  toast(`Removed profile ${profileId}`);
  return payload;
}

async function renderFlow() {
  try {
    const flow = await api("/api/pipeline/flow", { method: "POST", body: JSON.stringify(specFromForm()) });
    renderFlowMap(flow);
  } catch (err) {
    toast(`Flow unavailable: ${err.message}`);
  }
}

function markValidationStale() {
  const hadDecision = Boolean(state.launchDecision);
  state.launchDecision = null;
  state.launchDecisionStale = hadDecision || state.launchDecisionStale;
  renderValidation(null);
  if (state.launchPlan) $("planSummary").textContent = "stale";
  if (state.launchDiff) $("diffSummary").textContent = "stale";
  if (state.remediation) $("runbookSummary").textContent = "stale";
  if (state.launchDecisionStale) $("decisionSummary").textContent = "stale";
  if (state.launchCandidates) $("candidateSummary").textContent = "stale";
  if (state.gates) $("gateSummary").textContent = "stale";
  renderActionBoard(null);
}

async function loadLaunchPlan(recordActivity = false) {
  const spec = specFromForm();
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/launch/plan", { method: "POST", body: JSON.stringify(spec) });
  renderLaunchPlan(payload);
  if (recordActivity) await loadActivity().catch(() => {});
  return payload;
}

async function loadLaunchDiff(recordActivity = false) {
  const spec = specFromForm();
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/launch/diff", { method: "POST", body: JSON.stringify(spec) });
  renderLaunchDiff(payload);
  if (recordActivity) await loadActivity().catch(() => {});
  return payload;
}

async function loadRemediation() {
  const payload = await api("/api/remediation", { method: "POST", body: JSON.stringify(specFromForm()) });
  renderRemediation(payload);
  return payload;
}

async function loadDiagnostics({ syncRuntime = true } = {}) {
  try {
    const payload = await api("/api/diagnostics", { method: "POST", body: JSON.stringify(specFromForm()) });
    renderDiagnostics(payload);
    if (syncRuntime) await syncFormWithActiveRuntime({ silent: true });
    return payload;
  } catch (err) {
    toast(`Diagnostics unavailable: ${err.message}`);
    return null;
  }
}

async function adoptRuntimeByPid(pid) {
  const process = (state.diagnostics?.processes || []).find((item) => String(item.pid) === String(pid));
  if (!process) {
    toast("Runtime no longer observed");
    return;
  }
  state.portsPinned = false;
  state.launchDetached = false;
  state.runtimeMirroredKey = null;
  state.observedRuntime = runtimeTargetFromProcess(process);
  await syncFormWithActiveRuntime({ silent: false, force: true });
  await probeLive(true).catch((err) => toast(err.message));
}

async function validateLaunch() {
  const payload = await api("/api/launch/validate", { method: "POST", body: JSON.stringify(specFromForm()) });
  renderValidation(payload);
  touchFreshness("validation");
  await loadDiagnostics();
  await loadLaunchPlan();
  await loadLaunchDiff();
  await loadRemediation();
  await loadGates().catch(() => {});
  await loadModelMatrix().catch(() => {});
  await loadSources().catch(() => {});
  await loadLaunchDecision().catch(() => {});
  await loadActivity().catch(() => {});
  toast(payload.blocking ? "Launch blocked by preflight" : "Launch validation complete");
  return payload;
}

async function previewLaunch() {
  const payload = await api("/api/launch/preview", { method: "POST", body: JSON.stringify(specFromForm()) });
  renderValidation(payload.validation);
  $("heroCopy").textContent = `Materialized ${payload.materialized_pipeline}`;
  await loadLaunchPlan();
  await loadLaunchDiff();
  await renderFlow();
  await loadDiagnostics();
  await loadRemediation();
  await loadGates().catch(() => {});
  await loadModelMatrix().catch(() => {});
  await loadSources().catch(() => {});
  await loadLaunchDecision().catch(() => {});
  await loadActivity().catch(() => {});
  toast("Preview materialized");
}

async function startRuntime() {
  try {
    const payload = await api("/api/runtime/start", { method: "POST", body: JSON.stringify(specFromForm()) });
    renderRuntime(payload);
    if (!state.simpleMode) await syncFormWithActiveRuntime({ silent: true, force: true });
    toast("DS8 launch requested");
    await tailLogs();
    await loadActivity().catch(() => {});
    await loadLaunchDecision().catch(() => {});
  } catch (err) {
    toast(`Start blocked: ${err.message}`);
    const validation = state.runtime?.last_validation;
    if (validation) renderValidation(validation);
    await loadActivity().catch(() => {});
    await loadLaunchDecision().catch(() => {});
  }
}

async function restartRuntime() {
  try {
    const payload = await api("/api/runtime/restart", { method: "POST", body: JSON.stringify(specFromForm()) });
    renderRuntime(payload);
    if (!state.simpleMode) await syncFormWithActiveRuntime({ silent: true, force: true });
    toast("DS8 restart requested");
    await tailLogs();
    await loadActivity().catch(() => {});
    await loadLaunchDecision().catch(() => {});
  } catch (err) {
    toast(`Restart blocked: ${err.message}`);
    const validation = state.runtime?.last_validation;
    if (validation) renderValidation(validation);
    await loadActivity().catch(() => {});
    await loadLaunchDecision().catch(() => {});
  }
}

async function stopRuntime() {
  const payload = await api("/api/runtime/stop", { method: "POST", body: "{}" });
  state.runtimeMirroredKey = null;
  renderRuntime(payload);
  toast("Stop signal sent");
  await tailLogs();
  await loadActivity().catch(() => {});
  await loadLaunchDecision().catch(() => {});
}

async function refreshRuntime() {
  const payload = await api("/api/runtime/status");
  renderRuntime(payload);
  if (!state.simpleMode && activeRuntimePid()) {
    await syncFormWithActiveRuntime({ silent: true });
  }
}

async function tailLogs() {
  const payload = await api("/api/runtime/logs?lines=240");
  renderLogs(payload);
  await loadLogInsights().catch(() => {});
  return payload;
}

async function loadLogInsights() {
  const payload = await api("/api/runtime/log-insights?lines=600");
  renderLogInsights(payload);
  return payload;
}

async function depthBurst() {
  const payload = await api("/api/runtime/depth-refresh", {
    method: "POST",
    body: JSON.stringify({ spec: specFromForm(), seconds: 10 }),
  });
  await loadActivity().catch(() => {});
  toast(payload.ok ? "Depth burst requested" : `Depth burst failed: ${payload.error || payload.status}`);
}

async function runtimeHealth() {
  const payload = await api("/api/runtime/health", {
    method: "POST",
    body: JSON.stringify(specFromForm()),
  });
  await loadActivity().catch(() => {});
  toast(payload.ok ? "Runtime health endpoint responded" : `Runtime health unavailable: ${payload.error || payload.status}`);
}

async function probeLive(recordActivity = false) {
  const spec = specFromForm();
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/live/probe", { method: "POST", body: JSON.stringify(spec) });
  renderLive(payload);
  await loadLiveHealth(recordActivity);
  if (recordActivity) await loadActivity().catch(() => {});
  if (recordActivity) {
    toast(payload.connected ? "Live WS probe complete" : `Live WS unavailable: ${payload.error || "unknown"}`);
  }
  return payload;
}

async function loadLiveHealth(recordActivity = false) {
  const spec = specFromForm();
  if (recordActivity) spec.record_activity = true;
  const payload = await api("/api/live/health", { method: "POST", body: JSON.stringify(spec) });
  renderLiveHealth(payload);
  if (recordActivity) await loadActivity().catch(() => {});
  return payload;
}

async function liveControl(action, enabled = null) {
  const payload = await api("/api/live/control", {
    method: "POST",
    body: JSON.stringify({ spec: specFromForm(), action, enabled }),
  });
  toast(payload.ok ? `${action} sent` : `Control failed: ${payload.error || "unknown"}`);
  await probeLive();
  await loadActivity().catch(() => {});
}

async function saveBundle() {
  const payload = await api("/api/support/bundle", {
    method: "POST",
    body: JSON.stringify(specFromForm()),
  });
  renderBundle(payload);
  toast(payload.ok ? "Support bundle saved" : "Support bundle failed");
  await loadDiagnostics();
  await loadBundleLibrary().catch(() => {});
  await loadActivity().catch(() => {});
}

async function useFreePorts() {
  const payload = await loadDiagnostics();
  if (!payload?.suggested_ports) return;
  $("wsPortInput").value = payload.suggested_ports.ws_port;
  $("restPortInput").value = payload.suggested_ports.rest_port;
  $("rtspPortInput").value = payload.suggested_ports.rtsp_port;
  renderValidation(null);
  await renderFlow();
  await loadDiagnostics();
  await loadLaunchPlan();
  await loadLaunchDiff();
  await loadRemediation();
  await loadGates().catch(() => {});
  await loadModelMatrix().catch(() => {});
  await loadSources().catch(() => {});
  await loadLaunchDecision().catch(() => {});
  toast(`Ports set to ${payload.suggested_ports.ws_port} / ${payload.suggested_ports.rest_port} / ${payload.suggested_ports.rtsp_port}`);
}

function bindEvents() {
  $("presetSelect").addEventListener("change", (event) => applyPreset(event.target.value));
  $("simpleModeInput")?.addEventListener("change", (event) => {
    setSimpleMode(Boolean(event.target.checked));
  });
  $("refreshAllButton")?.addEventListener("click", () => refreshAll().catch((err) => toast(err.message)));
  $("shortcutsButton")?.addEventListener("click", () => openShortcutsDialog());
  $("shortcutsClose")?.addEventListener("click", () => closeShortcutsDialog());
  $("inspectorDrawerClose")?.addEventListener("click", () => closeInspectorDrawer());
  $("inspectorBackdrop")?.addEventListener("click", () => closeInspectorDrawer());
  $("openLogDrawerButton")?.addEventListener("click", () => openLogDrawer());
  $("openLogDrawerButton2")?.addEventListener("click", () => openLogDrawer());
  $("logDrawerClose")?.addEventListener("click", () => closeLogDrawer());
  $("logDrawerRefresh")?.addEventListener("click", () => tailLogs().then(() => toast("Log tail refreshed")).catch((err) => toast(err.message)));
  $("profileSelect").addEventListener("change", () => { markValidationStale(); updateSimpleLayout(); renderFlow(); loadDiagnostics(); loadGates(); loadModelMatrix(); loadSources(); });
  $("sizeSelect").addEventListener("change", () => { markValidationStale(); updateSimpleLayout(); renderFlow(); loadDiagnostics(); loadGates(); loadModelMatrix(); loadSources(); });
  $("trackingSelect").addEventListener("change", () => { markValidationStale(); updateSimpleLayout(); renderFlow(); loadDiagnostics(); loadGates(); loadModelMatrix(); loadSources(); });
  ["pipelineInput", "camerasInput", "wsPortInput", "restPortInput", "rtspPortInput", "depthSecondsInput", "logLevelSelect", "strictInput", "envInput"].forEach((id) => {
    const eventName = id === "envInput" || id.endsWith("Input") ? "input" : "change";
    $(id).addEventListener(eventName, () => {
      markValidationStale();
      updateSimpleLayout();
      if (id === "wsPortInput" || id === "restPortInput" || id === "rtspPortInput" || id === "pipelineInput" || id === "camerasInput") {
        loadDiagnostics();
      }
      if (id === "pipelineInput" || id === "camerasInput") {
        loadSources();
      }
      if (id === "depthSecondsInput" || id === "envInput") {
        renderFlow();
      }
      if (id === "strictInput") {
        loadGates();
        loadModelMatrix();
      }
      if (id === "depthSecondsInput" || id === "rtspPortInput") {
        loadGates();
      }
    });
  });
  $("previewButton").addEventListener("click", () => previewLaunch().catch((err) => toast(err.message)));
  $("validateButton").addEventListener("click", () => validateLaunch().catch((err) => toast(err.message)));
  $("startButton").addEventListener("click", () => startRuntime());
  $("restartButton").addEventListener("click", () => restartRuntime());
  $("stopButton").addEventListener("click", () => stopRuntime().catch((err) => toast(err.message)));
  $("actionBoardActions").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action-board]");
    if (!button || button.disabled) return;
    runActionBoardAction(button.dataset.actionBoard).catch((err) => toast(err.message));
  });
  $("refreshButton").addEventListener("click", () => init().catch((err) => toast(err.message)));
  $("planButton").addEventListener("click", () => loadLaunchPlan(true).then(() => toast("Launch plan refreshed")).catch((err) => toast(err.message)));
  $("runbookList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-runbook-action]");
    if (!button || button.disabled) return;
    runActionBoardAction(button.dataset.runbookAction).catch((err) => toast(err.message));
  });
  $("diffButton").addEventListener("click", () => loadLaunchDiff(true).then(() => toast("Launch diff refreshed")).catch((err) => toast(err.message)));
  $("diffHighOnly").addEventListener("change", () => renderLaunchDiff(state.launchDiff));
  $("decisionButton").addEventListener("click", () => loadLaunchDecision(true).then(() => toast("Launch decision refreshed")).catch((err) => toast(err.message)));
  $("decisionValidateButton").addEventListener("click", () => validateLaunch().then(() => loadLaunchDecision(true)).then(() => toast("Decision and validation refreshed")).catch((err) => toast(err.message)));
  $("candidateButton").addEventListener("click", () => loadLaunchCandidates(true).then(() => toast("Launch candidates compared")).catch((err) => toast(err.message)));
  $("matrixButton").addEventListener("click", () => loadModelMatrix(true).then(() => toast("Model matrix refreshed")).catch((err) => toast(err.message)));
  $("matrixReadyOnly").addEventListener("change", () => renderModelMatrix(state.modelMatrix));
  $("sourceButton").addEventListener("click", () => loadSources(true).then(() => toast("Sources refreshed")).catch((err) => toast(err.message)));
  $("sourceProbeInput").addEventListener("change", () => loadSources().then(() => loadLaunchDecision()).catch((err) => toast(err.message)));
  $("gateButton").addEventListener("click", () => loadGates(true).then(() => toast("Gate deck refreshed")).catch((err) => toast(err.message)));
  $("gateApplyButton").addEventListener("click", () => applyGateControls(false).catch((err) => toast(err.message)));
  $("gateValidateButton").addEventListener("click", () => applyGateControls(true).catch((err) => toast(err.message)));
  $("activityRefreshButton").addEventListener("click", () => loadActivity().then(() => toast("Activity refreshed")).catch((err) => toast(err.message)));
  $("saveProfileButton").addEventListener("click", () => saveCurrentProfile().catch((err) => toast(err.message)));
  $("applyProfileButton").addEventListener("click", () => applySavedProfile().catch((err) => toast(err.message)));
  $("deleteProfileButton").addEventListener("click", () => deleteSavedProfile().catch((err) => toast(err.message)));
  $("freePortsButton").addEventListener("click", () => useFreePorts().catch((err) => toast(err.message)));
  $("depthButton").addEventListener("click", () => depthBurst().catch((err) => toast(err.message)));
  $("runtimeHealthButton").addEventListener("click", () => runtimeHealth().catch((err) => toast(err.message)));
  $("logsButton").addEventListener("click", () => tailLogs().then(() => openLogDrawer()).catch((err) => toast(err.message)));
  $("logTailButton").addEventListener("click", () => tailLogs().then(() => toast("Log tail refreshed")).catch((err) => toast(err.message)));
  $("logInsightButton").addEventListener("click", () => loadLogInsights().then(() => toast("Log analysis refreshed")).catch((err) => toast(err.message)));
  $("probeLiveButton").addEventListener("click", () => probeLive(true).catch((err) => toast(err.message)));
  $("pingLiveButton").addEventListener("click", () => liveControl("ping").catch((err) => toast(err.message)));
  $("clearStatsButton").addEventListener("click", () => liveControl("clear_stats").catch((err) => toast(err.message)));
  $("toggleTrailsButton").addEventListener("click", () => {
    const next = !(state.live?.trail_enabled === true);
    liveControl("trail_visualization_enabled", next).catch((err) => toast(err.message));
  });
  $("bundleButton").addEventListener("click", () => saveBundle().catch((err) => toast(err.message)));
  $("bundleLibraryButton").addEventListener("click", () => loadBundleLibrary().then(() => toast("Bundle library refreshed")).catch((err) => toast(err.message)));
  $("bundleLibraryList").addEventListener("click", (event) => {
    const card = event.target.closest("[data-bundle-id]");
    if (!card) return;
    document.querySelectorAll("#bundleLibraryList [data-bundle-id]").forEach((item) => item.classList.toggle("selected", item === card));
    loadBundleDetail(card.dataset.bundleId).catch((err) => toast(err.message));
  });
  $("knobFilter").addEventListener("input", renderKnobs);
  $("processList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-adopt-runtime]");
    if (!button) return;
    adoptRuntimeByPid(button.dataset.adoptRuntime).catch((err) => toast(err.message));
  });
  for (const id of ["profileSelect", "sizeSelect", "trackingSelect", "pipelineInput", "camerasInput", "logLevelSelect", "depthSecondsInput", "strictInput", "envInput", "presetSelect"]) {
    $(id)?.addEventListener("input", () => {
      if (!state.simpleMode) state.launchDetached = true;
    });
    $(id)?.addEventListener("change", () => {
      if (!state.simpleMode) state.launchDetached = true;
    });
  }
  for (const id of ["wsPortInput", "restPortInput", "rtspPortInput"]) {
    $(id)?.addEventListener("input", () => {
      state.portsPinned = true;
      if (!state.simpleMode) state.launchDetached = true;
    });
  }
}

async function refreshAll() {
  await init();
  toast("Console refreshed");
}

async function init() {
  await refreshSummary();
  await loadKnobs();
  await loadProfiles();
  await loadActivity();
  await loadBundleLibrary().catch(() => {});
  renderValidation(null);
  renderLaunchCandidates(null);
  await renderFlow();
  await loadDiagnostics();
  await loadLaunchPlan().catch(() => {});
  await loadLaunchDiff().catch(() => {});
  await loadRemediation().catch(() => {});
  await loadGates().catch(() => {});
  await loadModelMatrix().catch(() => {});
  await loadSources().catch(() => {});
  await loadLaunchDecision().catch(() => {});
  if (shouldAutoProbeLive() && hasLiveTarget()) {
    probeLive().catch(() => {});
    loadLiveHealth().catch(() => {});
  }
  await refreshRuntime();
  await tailLogs();
  updateStatusStrip();
}

setupPhaseTabs();
setupCommandCenter();
bindEvents();
try {
  if (localStorage.getItem(SIMPLE_MODE_KEY) === "1") setSimpleMode(true, { persist: false });
} catch (_err) {
  /* ignore storage failures */
}
init().catch((err) => toast(err.message));
setInterval(() => {
  refreshRuntime().catch(() => {});
  updateStatusStrip();
}, 4000);
setInterval(() => loadDiagnostics().catch(() => {}), 12000);
setInterval(() => {
  if (hasLiveTarget()) loadLiveHealth().catch(() => {});
}, 10000);
