const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const url = require("url");

const PORT = process.env.PORT ? Number(process.env.PORT) : 3030;
const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, "public");
const MODEL_LABEL = process.env.CLOSED_LLM_MODEL || "ChatGPT-5.2 Thinking";
const DMXAPI_MODEL = process.env.DMXAPI_MODEL || "gpt-5.2";
const DMXAPI_BASE_URL = process.env.DMXAPI_BASE_URL || "https://www.dmxapi.cn/v1";
const DMXAPI_API_KEY = process.env.DMXAPI_API_KEY || "";
const DMXAPI_REASONING_EFFORT = process.env.DMXAPI_REASONING_EFFORT || "none";
const DMXAPI_TEXT_VERBOSITY = process.env.DMXAPI_TEXT_VERBOSITY || "low";
const LLM_MODE = process.env.LLM_MODE || "auto";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json; charset=utf-8",
  ".ico": "image/x-icon",
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function sendEvent(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function sendLog(res, level, message) {
  sendEvent(res, "log", {
    level,
    message,
    ts: new Date().toISOString(),
  });
}

function serveStatic(req, res) {
  const parsedUrl = url.parse(req.url);
  let pathname = parsedUrl.pathname || "/";
  if (pathname === "/") pathname = "/index.html";
  const filePath = path.join(PUBLIC_DIR, pathname);

  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { "Content-Type": MIME[ext] || "text/plain" });
    res.end(data);
  });
}

function normalizeSpan(text) {
  return text.replace(/\s+/g, " ").trim();
}

function compactText(text, max = 96) {
  if (!text) return "";
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(0, max - 3))}...`;
}

function buildUrl(base, suffix) {
  const trimmed = base.replace(/\/$/, "");
  return `${trimmed}${suffix}`;
}

function postJson(endpoint, headers, body, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const target = new URL(endpoint);
    const payload = JSON.stringify(body);
    const options = {
      method: "POST",
      hostname: target.hostname,
      port: target.port || 443,
      path: target.pathname + target.search,
      headers: {
        ...headers,
        "Content-Length": Buffer.byteLength(payload),
      },
    };

    const req = https.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => {
        data += chunk;
      });
      res.on("end", () => {
        let json = null;
        try {
          json = JSON.parse(data);
        } catch (err) {
          json = null;
        }
        resolve({
          status: res.statusCode || 0,
          json,
          text: data,
        });
      });
    });

    req.on("error", (err) => reject(err));
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error("DMXAPI request timeout"));
    });

    req.write(payload);
    req.end();
  });
}

function extractResponseText(payload) {
  if (!payload) return "";
  if (typeof payload.output_text === "string") return payload.output_text;

  if (Array.isArray(payload.output)) {
    for (const item of payload.output) {
      if (!item || !Array.isArray(item.content)) continue;
      for (const part of item.content) {
        if (!part) continue;
        if (typeof part.text === "string") return part.text;
        if (typeof part.output_text === "string") return part.output_text;
      }
    }
  }

  if (Array.isArray(payload.choices)) {
    for (const choice of payload.choices) {
      if (choice && choice.message && typeof choice.message.content === "string") {
        return choice.message.content;
      }
      if (choice && typeof choice.text === "string") return choice.text;
    }
  }

  return "";
}

function safeJsonParse(text) {
  if (!text) return null;
  const trimmed = text.trim();

  if (trimmed.startsWith("```")) {
    const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (fencedMatch) {
      return safeJsonParse(fencedMatch[1]);
    }
  }

  try {
    return JSON.parse(trimmed);
  } catch (err) {
    const match = trimmed.match(/\{[\s\S]*\}/);
    if (!match) return null;
    try {
      return JSON.parse(match[0]);
    } catch (innerErr) {
      return null;
    }
  }
}

function findMarkdownFile(dirPath) {
  const items = fs.readdirSync(dirPath);
  const md = items.find((name) => name.toLowerCase().endsWith(".md"));
  if (!md) return null;
  return path.join(dirPath, md);
}

function listImages(dirPath) {
  const imagesDir = path.join(dirPath, "images");
  if (!fs.existsSync(imagesDir)) return [];
  const items = fs.readdirSync(imagesDir);
  return items.filter((name) => /\.(png|jpg|jpeg|svg|webp)$/i.test(name));
}

function parseMarkdownToContext(markdown) {
  const blocks = markdown
    .split(/\n\s*\n/)
    .map((b) => b.trim())
    .filter(Boolean);
  let idCounter = 1;
  return blocks.map((block) => {
    let type = "paragraph";
    if (block.startsWith("#")) type = "heading";
    if (block.startsWith("![")) type = "figure";
    return {
      id: `C${idCounter++}`,
      type,
      text: normalizeSpan(block),
    };
  });
}

function loadDialogue(dirPath) {
  const filePath = path.join(dirPath, "dialogue.json");
  if (!fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw);
}

function heuristicThreadId(utterance) {
  const text = utterance.toLowerCase();
  if (text.includes("metric")) return "T1";
  if (text.includes("figure 1") || text.includes("ablation")) return "T2";
  return "T0";
}

function heuristicIntent(role, utterance) {
  const text = utterance.toLowerCase();
  if (role === "Reviewer") return "RequestEvidence";
  if (role === "Author") {
    if (text.includes("define") || text.includes("figure") || text.includes("evidence")) {
      return "ProvideEvidence";
    }
    return "CommitUpdate";
  }
  if (role === "Meta") return "ScheduleThreads";
  return "RequestEvidence";
}

function resolveMode(apiKey) {
  if (LLM_MODE === "heuristic") return "heuristic";
  if (LLM_MODE === "live") return apiKey ? "live" : "heuristic";
  return apiKey ? "live" : "heuristic";
}

class ClosedLLMCore {
  constructor(config) {
    this.modelLabel = config.modelLabel;
    this.modelId = config.modelId;
    this.baseUrl = config.baseUrl;
    this.apiKey = config.apiKey;
    this.reasoningEffort = config.reasoningEffort;
    this.textVerbosity = config.textVerbosity;
    this.emit = config.emit;
    this.callId = 1;
    this.mode = resolveMode(this.apiKey);
  }

  _emit(role, task, summary, toolCalls = [], extra = {}) {
    const payload = {
      id: `LLM${this.callId++}`,
      model: this.modelLabel,
      modelId: this.modelId,
      role,
      task,
      summary,
      toolCalls,
      mode: this.mode,
      ts: new Date().toISOString(),
      ...extra,
    };
    this.emit(payload);
  }

  async callResponses(task, prompt) {
    if (this.mode !== "live") {
      return { ok: false, status: 0, text: "", json: null, latencyMs: 0, mode: this.mode };
    }

    const body = {
      model: this.modelId,
      input: prompt,
    };

    if (this.reasoningEffort) {
      body.reasoning = { effort: this.reasoningEffort };
    }

    if (this.textVerbosity) {
      body.text = { verbosity: this.textVerbosity };
    }

    const headers = {
      Authorization: this.apiKey,
      "Content-Type": "application/json",
    };

    const endpoint = buildUrl(this.baseUrl, "/responses");
    const start = Date.now();

    try {
      const response = await postJson(endpoint, headers, body);
      const latencyMs = Date.now() - start;
      const outputText = extractResponseText(response.json) || response.text || "";
      const ok = response.status >= 200 && response.status < 300;
      return {
        ok,
        status: response.status,
        text: outputText,
        json: response.json,
        latencyMs,
        mode: this.mode,
      };
    } catch (err) {
      return {
        ok: false,
        status: 0,
        text: "",
        json: null,
        latencyMs: Date.now() - start,
        mode: this.mode,
        error: err.message || "DMXAPI request failed",
      };
    }
  }

  async callJson(task, prompt, fallback) {
    if (this.mode !== "live") {
      return { data: fallback, mode: this.mode, latencyMs: 0 };
    }

    const result = await this.callResponses(task, prompt);
    if (!result.ok) {
      return {
        data: fallback,
        mode: "fallback",
        latencyMs: result.latencyMs,
        error: result.error || `HTTP ${result.status}`,
      };
    }

    const parsed = safeJsonParse(result.text);
    if (!parsed) {
      return {
        data: fallback,
        mode: "fallback",
        latencyMs: result.latencyMs,
        error: "JSON parse failed",
      };
    }

    return { data: parsed, mode: "live", latencyMs: result.latencyMs };
  }

  async detectThread(utterance) {
    const fallback = heuristicThreadId(utterance);
    const prompt = [
      "Return ONLY valid JSON.",
      "Task: classify an utterance into a thread.",
      "Rules: T1=metric definition; T2=figure/ablation; T0=other.",
      `Utterance: ${utterance}`,
      "JSON schema: {\"threadId\":\"T1|T2|T0\"}",
    ].join("\n");

    const result = await this.callJson("thread_detection", prompt, { threadId: fallback });
    const threadId = result.data && result.data.threadId ? result.data.threadId : fallback;
    this._emit("System", "thread_detection", `Thread=${threadId} | mode=${result.mode}`, [], {
      latencyMs: result.latencyMs,
    });
    return threadId;
  }

  async inferIntent(role, utterance) {
    const fallback = heuristicIntent(role, utterance);
    const prompt = [
      "Return ONLY valid JSON.",
      "Task: classify intent for a peer-review utterance.",
      "Allowed intents: RequestEvidence, ProvideEvidence, CommitUpdate, ScheduleThreads.",
      `Role: ${role}`,
      `Utterance: ${utterance}`,
      "JSON schema: {\"intent\":\"...\"}",
    ].join("\n");

    const result = await this.callJson("intent_classification", prompt, { intent: fallback });
    const intent = result.data && result.data.intent ? result.data.intent : fallback;
    this._emit(role, "intent_classification", `Intent=${intent} | mode=${result.mode}`, [], {
      latencyMs: result.latencyMs,
    });
    return intent;
  }

  async canonicalizeSpan(text) {
    const fallback = normalizeSpan(text);
    const prompt = [
      "Return ONLY valid JSON.",
      "Task: canonicalize a span into a compact, single-line summary.",
      `Span: ${text}`,
      "JSON schema: {\"span\":\"...\"}",
    ].join("\n");

    const result = await this.callJson("canonicalize_span", prompt, { span: fallback });
    const span = result.data && result.data.span ? normalizeSpan(result.data.span) : fallback;
    this._emit("System", "canonicalize_span", `Span=${compactText(span, 72)} | mode=${result.mode}`, [], {
      latencyMs: result.latencyMs,
    });
    return span;
  }

  async mapNeed(instance) {
    if (instance.intent !== "RequestEvidence") {
      this._emit("Foundry", "evidence_need", "No evidence request", [], { latencyMs: 0 });
      return null;
    }

    let fallback = { type: "general_evidence", query: "General evidence request" };
    if (instance.issue === "T1") {
      fallback = { type: "metric_definition", query: "Metric M definition" };
    } else if (instance.issue === "T2") {
      fallback = { type: "figure_evidence", query: "Figure 1 ablation" };
    }

    const prompt = [
      "Return ONLY valid JSON.",
      "Task: map an evidence request to a need type and query.",
      "Possible types: metric_definition, figure_evidence, general_evidence.",
      `Issue: ${instance.issue}`,
      `Intent: ${instance.intent}`,
      `Span: ${instance.x}`,
      "JSON schema: {\"type\":\"...\", \"query\":\"...\"}",
    ].join("\n");

    const result = await this.callJson("evidence_need", prompt, fallback);
    const need = result.data && result.data.type ? result.data : fallback;
    this._emit("Foundry", "evidence_need", `Need=${need.type} | mode=${result.mode}`, [], {
      latencyMs: result.latencyMs,
    });
    return need;
  }

  async synthesizeSkill(need) {
    const fallback = {
      name: need.type === "figure_evidence" ? "LocateFigureEvidence" : need.type === "metric_definition"
        ? "LocateMetricDefinition"
        : "LocateGeneralEvidence",
      dag: need.type === "figure_evidence"
        ? ["figure_lookup", "span_extract"]
        : ["keyword_retrieval", "span_extract"],
      description: "Locate supporting evidence in the paper context.",
    };

    const prompt = [
      "Return ONLY valid JSON.",
      "Task: synthesize a deterministic skill spec for evidence lookup.",
      "Allowed primitives: keyword_retrieval, figure_lookup, span_extract.",
      `Need type: ${need.type}`,
      `Need query: ${need.query}`,
      "JSON schema: {\"name\":\"...\", \"dag\":[\"...\"], \"description\":\"...\"}",
    ].join("\n");

    const result = await this.callJson("skill_synthesis", prompt, fallback);
    const spec = result.data && result.data.name ? result.data : fallback;
    const toolCalls = [{ name: "skill_spec", arguments: spec }];

    this._emit("Foundry", "skill_synthesis", `Skill=${spec.name} | mode=${result.mode}`, toolCalls, {
      latencyMs: result.latencyMs,
    });

    return { spec, tests: { status: "pass" } };
  }

  async expandQuery(query, skillName) {
    const fallbackTokens = new Set();
    query.split(/\s+/).forEach((item) => fallbackTokens.add(item));
    if (skillName === "LocateMetricDefinition") {
      ["Metric", "definition", "normalized", "overlap"].forEach((item) => fallbackTokens.add(item));
    }
    if (skillName === "LocateFigureEvidence") {
      ["Figure", "ablation", "results", "caption"].forEach((item) => fallbackTokens.add(item));
    }
    if (skillName === "LocateGeneralEvidence") {
      ["method", "experiment", "result", "evidence"].forEach((item) => fallbackTokens.add(item));
    }
    const fallback = { keywords: Array.from(fallbackTokens).filter(Boolean) };

    const prompt = [
      "Return ONLY valid JSON.",
      "Task: expand a query into short keywords for retrieval.",
      `Skill: ${skillName}`,
      `Query: ${query}`,
      "JSON schema: {\"keywords\":[\"...\"]}",
    ].join("\n");

    const result = await this.callJson("query_expansion", prompt, fallback);
    const keywords = Array.isArray(result.data && result.data.keywords)
      ? result.data.keywords.filter(Boolean)
      : fallback.keywords;

    this._emit("System", "query_expansion", `Keywords=${keywords.slice(0, 6).join(", ")} | mode=${result.mode}`, [], {
      latencyMs: result.latencyMs,
    });

    return keywords;
  }

  async planMove(role, state, library) {
    const fallbackIntent = role === "Author" ? "ProvideEvidence" : state.phase === "Negotiation" ? "CloseIssue" : "RequestEvidence";
    const fallbackSkill = state.issueType === "figure_evidence"
      ? "LocateFigureEvidence"
      : state.issueType === "metric_definition"
        ? "LocateMetricDefinition"
        : "LocateGeneralEvidence";

    const fallback = {
      intent: fallbackIntent,
      skill: fallbackSkill,
      arguments: { query: state.query },
    };

    const skills = (library.skills || []).map((skill) => skill.name).join(", ");
    const prompt = [
      "Return ONLY valid JSON.",
      "Task: plan a move (intent + skill) for the role.",
      "Allowed intents: RequestEvidence, ProvideEvidence, CloseIssue, CommitUpdate.",
      `Role: ${role}`,
      `Phase: ${state.phase}`,
      `Issue type: ${state.issueType}`,
      `Available skills: ${skills || "none"}`,
      `Query: ${state.query}`,
      "JSON schema: {\"intent\":\"...\", \"skill\":\"...\", \"arguments\":{\"query\":\"...\"}}",
    ].join("\n");

    const result = await this.callJson("plan_move", prompt, fallback);
    const plan = {
      role,
      intent: result.data && result.data.intent ? result.data.intent : fallback.intent,
      skill_call: {
        name: result.data && result.data.skill ? result.data.skill : fallback.skill,
        arguments: (result.data && result.data.arguments) || fallback.arguments,
      },
    };

    this._emit(role, "plan_move", `Intent=${plan.intent}, Skill=${plan.skill_call.name} | mode=${result.mode}`, [], {
      latencyMs: result.latencyMs,
    });

    return plan;
  }

  async composeAction(role, intent, observation) {
    const fallback = (() => {
      if (role === "Reviewer" && intent === "CloseIssue") {
        return {
          type: "CloseIssue",
          text: observation.status === "ok"
            ? `Issue closed. Evidence confirmed in ${observation.id}.`
            : "Issue remains open due to missing evidence.",
          claims: observation.status === "ok" ? [{ text: "Evidence confirmed", cites: [observation.id] }] : [],
        };
      }

      if (role === "Reviewer") {
        return {
          type: "RequestEvidence",
          text: observation.status === "ok"
            ? `Please cite ${observation.id} to ground this response.`
            : "Evidence is missing. Please clarify or add it in the revision.",
          claims: [],
        };
      }

      if (role === "Author") {
        if (observation.status === "ok") {
          return {
            type: "ProvideEvidence",
            text: `We confirm the evidence in ${observation.id} and will cite it explicitly.`,
            claims: [{ text: "Evidence confirmed", cites: [observation.id] }],
          };
        }
        return {
          type: "CommitUpdate",
          text: "We will add the missing evidence in the revision and cite it clearly.",
          claims: [],
        };
      }

      return { type: "Unknown", text: "", claims: [] };
    })();

    const prompt = [
      "Return ONLY valid JSON.",
      "Task: compose the agent action given intent and evidence.",
      "Allowed action types: RequestEvidence, ProvideEvidence, CommitUpdate, CloseIssue.",
      `Role: ${role}`,
      `Intent: ${intent}`,
      `Observation status: ${observation.status}`,
      `Observation id: ${observation.id}`,
      `Observation payload: ${compactText(observation.payload || "", 140)}`,
      "JSON schema: {\"type\":\"...\", \"text\":\"...\", \"claims\":[{\"text\":\"...\", \"cites\":[\"O1\"]}]}",
    ].join("\n");

    const result = await this.callJson("compose_action", prompt, fallback);
    const action = result.data && result.data.type ? result.data : fallback;

    this._emit(role, "compose_action", `Action=${action.type} | mode=${result.mode}`, [], {
      latencyMs: result.latencyMs,
    });

    return action;
  }

  async warmPolicy(groundedCount, total) {
    const summary = `Grounded ${groundedCount}/${total} instances.`;
    this._emit("System", "policy_warmup", summary, [], { latencyMs: 0 });
    return summary;
  }

  async scheduleThreads(threads) {
    const fallback = Array.from(threads.values())
      .sort((a, b) => {
        const order = { high: 0, medium: 1, low: 2 };
        return order[a.severity] - order[b.severity];
      })
      .map((item) => item.threadId);

    const threadSummary = Array.from(threads.values()).map((item) => {
      return `${item.threadId}(${item.severity},${item.phase})`;
    });

    const prompt = [
      "Return ONLY valid JSON.",
      "Task: order thread ids by priority.",
      "Rule: prioritize high severity, then open/evidence pending, then negotiation, avoid closed.",
      `Threads: ${threadSummary.join(", ")}`,
      "JSON schema: {\"order\":[\"...\"]}",
    ].join("\n");

    const result = await this.callJson("schedule_threads", prompt, { order: fallback });
    const ordered = Array.isArray(result.data && result.data.order) ? result.data.order : fallback;

    this._emit("Meta", "schedule_threads", `Threads=${ordered.join(", ")} | mode=${result.mode}`, [], {
      latencyMs: result.latencyMs,
    });

    return ordered;
  }
}

async function invertBehavior(dialogue, llm) {
  const instances = [];
  for (let index = 0; index < dialogue.length; index += 1) {
    const turn = dialogue[index];
    const threadId = await llm.detectThread(turn.utterance);
    const intent = await llm.inferIntent(turn.role, turn.utterance);
    const canonical = await llm.canonicalizeSpan(turn.utterance);
    instances.push({
      role: turn.role,
      issue: threadId,
      intent,
      x: canonical.slice(0, 160),
      rho: [],
      kappa: index + 1,
    });
  }
  return instances;
}

async function mineNeeds(instances, llm) {
  const needs = [];
  for (const inst of instances) {
    const need = await llm.mapNeed(inst);
    if (need) needs.push(need);
  }
  return needs;
}

async function buildLibrary(needs, llm) {
  const primitives = [
    {
      name: "keyword_retrieval",
      description: "Find context segments that match query keywords.",
    },
    {
      name: "figure_lookup",
      description: "Locate figure references and captions in context.",
    },
    {
      name: "span_extract",
      description: "Extract a concise span around matched evidence.",
    },
  ];

  const skills = [];
  const seen = new Set();
  for (const need of needs) {
    const result = await llm.synthesizeSkill(need);
    if (!result || !result.spec || seen.has(result.spec.name)) continue;
    skills.push({
      name: result.spec.name,
      intent: need.type,
      dag: result.spec.dag,
      description: result.spec.description,
    });
    seen.add(result.spec.name);
  }

  return { primitives, skills };
}

function findSegmentsByKeyword(context, keywords) {
  const lower = keywords.map((k) => k.toLowerCase());
  return context.filter((seg) =>
    lower.some((k) => seg.text.toLowerCase().includes(k))
  );
}

let observationCounter = 1;
async function executeSkill(skillCall, context, llm) {
  const name = skillCall && skillCall.name;
  if (!name) {
    return {
      id: `O${observationCounter++}`,
      type: "evidence",
      payload: "Skill call missing.",
      prov: [],
      status: "fail",
    };
  }

  const query = (skillCall.arguments && skillCall.arguments.query) || name;
  const keywords = await llm.expandQuery(query, name);
  const matches = findSegmentsByKeyword(context, keywords);

  if (matches.length === 0) {
    return {
      id: `O${observationCounter++}`,
      type: "evidence",
      payload: "No matching evidence found.",
      prov: [],
      status: "missing",
    };
  }

  const payload = matches[0].text.slice(0, 220);
  return {
    id: `O${observationCounter++}`,
    type: "evidence",
    payload,
    prov: matches.map((m) => m.id),
    status: "ok",
  };
}

function enforceGating(action, observation) {
  if (!action || observation.status !== "ok") {
    return { ...action, claims: [] };
  }

  const claims = Array.isArray(action.claims) ? action.claims : [];
  const filteredClaims = claims.filter((claim) =>
    Array.isArray(claim.cites) && claim.cites.includes(observation.id)
  );

  return { ...action, claims: filteredClaims };
}

function initThreads(instances) {
  const threads = new Map();
  for (const inst of instances) {
    if (!threads.has(inst.issue)) {
      let issueType = "general_evidence";
      let query = "General evidence request";
      let severity = "low";

      if (inst.issue === "T1") {
        issueType = "metric_definition";
        query = "Metric M definition";
        severity = "medium";
      } else if (inst.issue === "T2") {
        issueType = "figure_evidence";
        query = "Figure 1 ablation";
        severity = "high";
      }

      threads.set(inst.issue, {
        threadId: inst.issue,
        issueType,
        query,
        phase: "Open",
        severity,
        observations: [],
        requests: [],
        commitments: [],
        lastUpdated: null,
      });
    }
  }
  return threads;
}

function updateLedger(state, role, intent, observation, action) {
  if (role === "Reviewer" && intent === "RequestEvidence") {
    state.requests.push({ id: `R${state.requests.length + 1}`, text: action.text });
    state.phase = "EvidencePending";
  }

  if (role === "Author") {
    if (observation.status === "ok") {
      state.observations.push(observation);
      state.phase = "Negotiation";
    } else {
      state.commitments.push({ id: `K${state.commitments.length + 1}`, text: action.text });
      state.phase = "Negotiation";
    }
  }

  if (role === "Reviewer" && intent === "CloseIssue") {
    if (observation.status === "ok") state.phase = "Closed";
  }

  state.lastUpdated = new Date().toISOString();
}

async function runPipeline({ paperDir, pdfPath }, res) {
  observationCounter = 1;
  const llm = new ClosedLLMCore({
    modelLabel: MODEL_LABEL,
    modelId: DMXAPI_MODEL,
    baseUrl: DMXAPI_BASE_URL,
    apiKey: DMXAPI_API_KEY,
    reasoningEffort: DMXAPI_REASONING_EFFORT,
    textVerbosity: DMXAPI_TEXT_VERBOSITY,
    emit: (payload) => sendEvent(res, "llm_call", payload),
  });
  let moveCounter = 1;

  sendEvent(res, "step", { id: "input", status: "running", detail: "Validating input paths" });
  sendLog(res, "info", `Paper directory: ${paperDir}`);
  sendLog(res, "info", `LLM core: ${MODEL_LABEL} (${DMXAPI_MODEL})`);
  sendLog(res, "info", `LLM mode: ${llm.mode} | Endpoint: ${DMXAPI_BASE_URL}`);

  const markdownFile = findMarkdownFile(paperDir);
  const images = listImages(paperDir);
  const pdfResolved = pdfPath && fs.existsSync(pdfPath) ? pdfPath : path.join(paperDir, "paper.pdf");

  if (!markdownFile || images.length === 0) {
    sendEvent(res, "step", { id: "input", status: "error", detail: "Missing markdown or images" });
    sendLog(res, "error", "Input directory must contain a .md file and an images/ folder.");
    return;
  }

  if (!DMXAPI_API_KEY) {
    sendLog(res, "warn", "DMXAPI_API_KEY not set. Falling back to heuristic mode.");
  }

  sendEvent(res, "artifact", {
    markdownFile,
    images,
    pdfPath: fs.existsSync(pdfResolved) ? pdfResolved : "(missing)",
  });

  sendEvent(res, "step", { id: "input", status: "done", detail: "Inputs loaded" });
  await sleep(250);

  sendEvent(res, "step", { id: "parse", status: "running", detail: "Parsing markdown into context" });
  const markdown = fs.readFileSync(markdownFile, "utf-8");
  const context = parseMarkdownToContext(markdown);
  sendLog(res, "info", `Parsed ${context.length} context segments.`);
  sendEvent(res, "step", { id: "parse", status: "done", detail: "Context ready" });
  await sleep(250);

  sendEvent(res, "step", { id: "invert", status: "running", detail: "Inverting dialogue into behaviors" });
  const dialogue = loadDialogue(paperDir) || [];
  if (dialogue.length === 0) sendLog(res, "warn", "No dialogue.json found, using empty log.");
  const instances = await invertBehavior(dialogue, llm);
  sendLog(res, "info", `Inverted ${instances.length} turns into behavior instances.`);
  sendEvent(res, "step", { id: "invert", status: "done", detail: "Behavior instances created" });
  await sleep(250);

  sendEvent(res, "step", { id: "foundry", status: "running", detail: "Compiling skills with LLM assistance" });
  sendLog(res, "info", "Offline foundry incomplete, LLM assists skill synthesis.");
  const needs = await mineNeeds(instances, llm);
  sendLog(res, "info", `Mined ${needs.length} evidence needs.`);
  const library = await buildLibrary(needs, llm);
  sendEvent(res, "library", library);
  sendEvent(res, "step", { id: "foundry", status: "done", detail: "Library compiled" });
  await sleep(250);

  sendEvent(res, "step", { id: "reground", status: "running", detail: "Re-grounding evidence" });
  const grounded = [];
  for (const inst of instances) {
    if (inst.intent !== "RequestEvidence") {
      grounded.push(inst);
      continue;
    }
    const call = inst.issue === "T1"
      ? { name: "LocateMetricDefinition" }
      : inst.issue === "T2"
        ? { name: "LocateFigureEvidence" }
        : { name: "LocateGeneralEvidence" };
    const obs = await executeSkill(call, context, llm);
    inst.rho = obs.status === "ok" ? obs.prov : [];
    grounded.push(inst);
  }
  sendLog(res, "info", `Grounded ${grounded.length} instances.`);
  sendEvent(res, "step", { id: "reground", status: "done", detail: "Evidence rewritten" });
  await sleep(250);

  sendEvent(res, "step", { id: "train", status: "running", detail: "Warming policy with evidence gating" });
  const groundedCount = grounded.filter((g) => g.rho && g.rho.length > 0).length;
  const policySummary = await llm.warmPolicy(groundedCount, grounded.length);
  sendLog(res, "info", `Policy warmup: ${policySummary}`);
  sendEvent(res, "step", { id: "train", status: "done", detail: "Policy ready" });
  await sleep(250);

  sendEvent(res, "step", { id: "arena", status: "running", detail: "Running online arena" });
  const threads = initThreads(grounded);
  const scheduled = await llm.scheduleThreads(threads);
  sendLog(res, "info", `Meta scheduled ${scheduled.length} threads.`);
  sendEvent(res, "agent_result", {
    role: "Meta",
    intent: "ScheduleThreads",
    text: `Scheduled threads: ${scheduled.join(", ") || "none"}.`,
    model: MODEL_LABEL,
  });

  for (const threadId of scheduled) {
    const thread = threads.get(threadId);
    if (!thread) continue;

    const rounds = ["Reviewer", "Author", "Reviewer"];
    for (const role of rounds) {
      const plan = await llm.planMove(role, thread, library);
      sendLog(res, "info", `${role} plan: ${plan.intent} via ${plan.skill_call.name}`);
      const observation = await executeSkill(plan.skill_call, context, llm);
      let action = await llm.composeAction(role, plan.intent, observation);
      action = enforceGating(action, observation);
      updateLedger(thread, role, plan.intent, observation, action);

      sendEvent(res, "thread", {
        threadId: thread.threadId,
        state: thread,
        move: {
          role,
          intent: plan.intent,
          observation,
          action,
        },
      });

      sendEvent(res, "thread_event", {
        id: moveCounter++,
        threadId: thread.threadId,
        role,
        intent: plan.intent,
        phase: thread.phase,
        observationStatus: observation.status,
        actionType: action.type,
        actionText: action.text,
        ts: new Date().toISOString(),
      });

      sendEvent(res, "agent_result", {
        role,
        intent: plan.intent,
        threadId: thread.threadId,
        observationStatus: observation.status,
        text: action.text,
        model: MODEL_LABEL,
      });

      await sleep(350);
    }
  }

  sendEvent(res, "step", { id: "arena", status: "done", detail: "Arena complete" });
  sendEvent(res, "done", { summary: "Run finished" });
}

const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  if (parsedUrl.pathname === "/api/run") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    const paperDirParam = parsedUrl.query.paperDir || "demo_paper";
    const pdfPathParam = parsedUrl.query.pdfPath || "";
    const paperDir = path.isAbsolute(paperDirParam)
      ? paperDirParam
      : path.join(ROOT, paperDirParam);
    const pdfPath = pdfPathParam
      ? (path.isAbsolute(pdfPathParam) ? pdfPathParam : path.join(ROOT, pdfPathParam))
      : "";

    runPipeline({ paperDir, pdfPath }, res)
      .then(() => res.end())
      .catch((err) => {
        sendLog(res, "error", err.message || "Pipeline error");
        sendEvent(res, "step", { id: "arena", status: "error", detail: "Pipeline failed" });
        res.end();
      });
    return;
  }

  serveStatic(req, res);
});

server.listen(PORT, () => {
  console.log(`ScholarArena demo running at http://localhost:${PORT}`);
});
