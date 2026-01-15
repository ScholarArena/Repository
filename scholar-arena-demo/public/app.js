const steps = [
  { id: "input", label: "Input validation" },
  { id: "parse", label: "Parse paper context" },
  { id: "invert", label: "Behavior inversion" },
  { id: "foundry", label: "Offline foundry" },
  { id: "reground", label: "Evidence re-grounding" },
  { id: "train", label: "Evidence-gated policy" },
  { id: "arena", label: "Online arena" },
];

const stepListEl = document.getElementById("stepList");
const logListEl = document.getElementById("logList");
const threadListEl = document.getElementById("threadList");
const primitiveListEl = document.getElementById("primitiveList");
const skillListEl = document.getElementById("skillList");
const artifactBodyEl = document.getElementById("artifactBody");
const timelineListEl = document.getElementById("timelineList");
const paperDirInput = document.getElementById("paperDir");
const pdfPathInput = document.getElementById("pdfPath");
const useDemoBtn = document.getElementById("useDemo");
const runArenaBtn = document.getElementById("runArena");

const agentBodies = {
  Reviewer: document.getElementById("agent-Reviewer"),
  Author: document.getElementById("agent-Author"),
  Meta: document.getElementById("agent-Meta"),
};

let eventSource = null;
let threadState = {};
const stepNodes = new Map();
const agentHistory = { Reviewer: [], Author: [], Meta: [] };

function createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function renderSteps() {
  stepListEl.innerHTML = "";
  steps.forEach((step, index) => {
    const item = createEl("div", "step");
    item.dataset.status = "pending";

    const icon = createEl("div", "step-icon", `${index + 1}`);
    const content = createEl("div", "step-content");
    const title = createEl("div", "step-title", step.label);
    const detail = createEl("div", "step-detail", "Waiting...");

    content.appendChild(title);
    content.appendChild(detail);
    item.appendChild(icon);
    item.appendChild(content);
    stepListEl.appendChild(item);

    stepNodes.set(step.id, { item, detail });
  });
}

function setStepStatus(id, status, detail) {
  const node = stepNodes.get(id);
  if (!node) return;
  node.item.dataset.status = status;
  node.detail.textContent = detail || "";
}

function appendLog(level, message, ts) {
  const item = createEl("div", `log-item ${level}`);
  const time = ts ? ` ${new Date(ts).toLocaleTimeString()}` : "";
  item.textContent = `${level.toUpperCase()}${time} | ${message}`;
  logListEl.appendChild(item);
  logListEl.scrollTop = logListEl.scrollHeight;
}

function appendLLMCall(data) {
  const toolNames = (data.toolCalls || []).map((tool) => tool.name).join(", ");
  const summary = data.summary ? ` | ${data.summary}` : "";
  const tools = toolNames ? ` | tools: ${toolNames}` : "";
  const mode = data.mode ? ` | ${data.mode}` : "";
  const latency = data.latencyMs ? ` | ${data.latencyMs}ms` : "";
  const model = data.modelId ? `${data.model} (${data.modelId})` : data.model;
  appendLog("llm", `${model} :: ${data.role} :: ${data.task}${summary}${tools}${mode}${latency}`, data.ts);
}

function updateArtifacts(data) {
  const { markdownFile, images, pdfPath } = data;
  const lines = [
    `Markdown: ${markdownFile}`,
    `Images: ${images.join(", ") || "(none)"}`,
    `PDF: ${pdfPath}`,
  ];
  artifactBodyEl.textContent = lines.join("\n");
}

function updateLibrary(library) {
  primitiveListEl.innerHTML = "";
  skillListEl.innerHTML = "";

  (library.primitives || []).forEach((prim) => {
    const li = createEl("li", null, `${prim.name} - ${prim.description}`);
    primitiveListEl.appendChild(li);
  });

  (library.skills || []).forEach((skill) => {
    const li = createEl("li", null, `${skill.name} (${skill.intent})`);
    skillListEl.appendChild(li);
  });
}

function renderThreads() {
  threadListEl.innerHTML = "";
  Object.values(threadState).forEach((thread) => {
    const card = createEl("div", "thread-card");
    const updatedAt = thread.lastUpdated ? Date.parse(thread.lastUpdated) : 0;
    if (updatedAt && Date.now() - updatedAt < 1400) {
      card.classList.add("active");
    }

    const head = createEl("div", "thread-head");
    head.appendChild(createEl("span", null, `Thread ${thread.threadId}`));
    head.appendChild(createEl("span", "thread-phase", `Phase: ${thread.phase}`));

    const body = createEl("div", "thread-body");
    body.appendChild(
      createEl("div", null, `Issue: ${thread.issueType} | Severity: ${thread.severity}`)
    );
    body.appendChild(
      createEl(
        "div",
        null,
        `Observations: ${thread.observations.map((o) => o.id).join(", ") || "-"}`
      )
    );
    body.appendChild(
      createEl(
        "div",
        null,
        `Requests: ${thread.requests.map((r) => r.id).join(", ") || "-"}`
      )
    );
    body.appendChild(
      createEl(
        "div",
        null,
        `Commitments: ${thread.commitments.map((c) => c.id).join(", ") || "-"}`
      )
    );
    if (thread.lastMove) {
      body.appendChild(
        createEl(
          "div",
          null,
          `Last move: ${thread.lastMove.role} -> ${thread.lastMove.intent}`
        )
      );
    }

    card.appendChild(head);
    card.appendChild(body);
    threadListEl.appendChild(card);
  });
}

function renderAgent(role) {
  const body = agentBodies[role];
  if (!body) return;
  body.innerHTML = "";
  const history = agentHistory[role] || [];
  if (history.length === 0) {
    body.textContent = "Waiting...";
    return;
  }
  history.forEach((entry) => {
    body.appendChild(createEl("div", "agent-line", entry));
  });
}

function updateAgentOutput(role, payload) {
  if (!agentBodies[role]) return;
  const note = payload.threadId ? `[${payload.threadId}] ` : "";
  const intent = payload.intent ? payload.intent : "Update";
  const status = payload.observationStatus ? ` (${payload.observationStatus})` : "";
  const text = payload.text || "";
  const line = `${note}${intent}${status}: ${text}`.trim();

  const list = agentHistory[role] || [];
  list.unshift(line);
  agentHistory[role] = list.slice(0, 3);
  renderAgent(role);
}

function appendTimeline(event) {
  const item = createEl("div", "timeline-item");
  const meta = createEl("div", "timeline-meta");
  const role = createEl("div", "timeline-role", event.role);
  const thread = createEl("div", "timeline-thread", event.threadId);
  meta.appendChild(role);
  meta.appendChild(thread);

  const body = createEl("div", "timeline-body");
  const title = createEl("div", "timeline-title", `${event.intent} - ${event.actionType}`);
  const time = event.ts ? new Date(event.ts).toLocaleTimeString() : "";
  const detail = createEl(
    "div",
    "timeline-detail",
    `Phase: ${event.phase} | Evidence: ${event.observationStatus} ${time ? `| ${time}` : ""}`
  );
  const text = createEl("div", "timeline-text", event.actionText);

  body.appendChild(title);
  body.appendChild(detail);
  body.appendChild(text);
  item.appendChild(meta);
  item.appendChild(body);

  timelineListEl.appendChild(item);
  timelineListEl.scrollTop = timelineListEl.scrollHeight;
}

function resetAgents() {
  Object.keys(agentHistory).forEach((role) => {
    agentHistory[role] = [];
    renderAgent(role);
  });
}

function resetUI() {
  renderSteps();
  logListEl.innerHTML = "";
  threadListEl.innerHTML = "";
  timelineListEl.innerHTML = "";
  primitiveListEl.innerHTML = "";
  skillListEl.innerHTML = "";
  artifactBodyEl.textContent = "Awaiting input...";
  threadState = {};
  resetAgents();
}

function startRun() {
  if (eventSource) eventSource.close();
  resetUI();

  const paperDir = paperDirInput.value.trim() || "demo_paper";
  const pdfPath = pdfPathInput.value.trim();

  const params = new URLSearchParams({
    paperDir,
    pdfPath,
  });

  eventSource = new EventSource(`/api/run?${params.toString()}`);

  eventSource.addEventListener("step", (event) => {
    const data = JSON.parse(event.data);
    setStepStatus(data.id, data.status, data.detail);
  });

  eventSource.addEventListener("log", (event) => {
    const data = JSON.parse(event.data);
    appendLog(data.level, data.message, data.ts);
  });

  eventSource.addEventListener("llm_call", (event) => {
    const data = JSON.parse(event.data);
    appendLLMCall(data);
  });

  eventSource.addEventListener("agent_result", (event) => {
    const data = JSON.parse(event.data);
    updateAgentOutput(data.role, data);
  });

  eventSource.addEventListener("artifact", (event) => {
    const data = JSON.parse(event.data);
    updateArtifacts(data);
  });

  eventSource.addEventListener("library", (event) => {
    const data = JSON.parse(event.data);
    updateLibrary(data);
  });

  eventSource.addEventListener("thread", (event) => {
    const data = JSON.parse(event.data);
    threadState[data.threadId] = { ...data.state, lastMove: data.move };
    renderThreads();
  });

  eventSource.addEventListener("thread_event", (event) => {
    const data = JSON.parse(event.data);
    appendTimeline(data);
  });

  eventSource.addEventListener("done", () => {
    appendLog("info", "Pipeline completed.");
    eventSource.close();
  });

  eventSource.onerror = () => {
    appendLog("error", "Connection lost. Check server status.");
    eventSource.close();
  };
}

useDemoBtn.addEventListener("click", () => {
  paperDirInput.value = "demo_paper";
  pdfPathInput.value = "demo_paper/paper.pdf";
});

runArenaBtn.addEventListener("click", startRun);

renderSteps();
resetAgents();
