import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from urllib import request as url_request

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from utils import parse_json_from_text


def call_chat(messages, model, api_key, base_url, max_retries=3, sleep_seconds=0.5):
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0.2}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(1, max_retries + 1):
        try:
            req = url_request.Request(endpoint, data=payload, headers=headers, method="POST")
            with url_request.urlopen(req, timeout=360) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(sleep_seconds)


def extract_message_text(resp: Dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return ""


class PolicyClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        mode: str = "llm",
        log_raw: bool = False,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.mode = mode
        self.log_raw = log_raw
        self.raw_logs: List[Dict[str, Any]] = []

    def plan_move(
        self,
        role: str,
        state: Dict[str, Any],
        intents: List[str],
        skills: List[str],
    ) -> Dict[str, Any]:
        if self.mode != "llm":
            return self._stub_plan(role, state, intents, skills)
        prompt = [
            "You are planning the next Move in ScholarArena.",
            "Return JSON only.",
            "Schema: {\"intent\": str, \"skill_call\": {\"name\": str, \"arguments\": object}}",
            f"Role: {role}",
            f"Issue: {state.get('issue_text') or state.get('tag')}",
            f"Phase: {state.get('phase')} | Budget: {state.get('budget')} | Severity: {state.get('severity')}",
            f"Requests: {state.get('requests')}",
            f"Commitments: {state.get('commitments')}",
            f"Available intents: {intents}",
            f"Available skills: {skills}",
            "Pick the best intent and a single skill call that can fetch evidence.",
        ]
        data = call_chat([
            {"role": "system", "content": "You output strict JSON with no extra text."},
            {"role": "user", "content": "\n".join(prompt)},
        ], self.model, self.api_key, self.base_url)
        text = extract_message_text(data)
        parsed = parse_json_from_text(text)
        if self.log_raw:
            self.raw_logs.append({"stage": "plan", "response": text})
        if not parsed:
            return self._stub_plan(role, state, intents, skills)
        return normalize_plan(parsed, intents, skills)

    def compose_action(
        self,
        role: str,
        state: Dict[str, Any],
        observation: Dict[str, Any],
        intent: str,
        allowed_obs_ids: List[str],
    ) -> Dict[str, Any]:
        if self.mode != "llm":
            return self._stub_action(role, state, observation, intent, allowed_obs_ids)
        status = observation.get("status")
        obs_id = observation.get("id")
        payload = observation.get("payload")
        prompt = [
            "You are composing a structured action in ScholarArena.",
            "Return JSON only.",
            "Schema: {\"action_type\": str, \"text\": str, \"claims\": [{\"text\": str, \"cites\": [str]}], \"requests\": [{\"text\": str}], \"commitments\": [{\"text\": str}]}",
            f"Role: {role}",
            f"Intent: {intent}",
            f"Phase: {state.get('phase')} | Requests: {state.get('requests')}",
            f"Observation status: {status}",
            f"Observation id: {obs_id}",
            f"Observation payload: {truncate_text(payload)}",
            f"Allowed citations: {allowed_obs_ids}",
            "If status is not ok, do not include factual claims.",
        ]
        data = call_chat([
            {"role": "system", "content": "You output strict JSON with no extra text."},
            {"role": "user", "content": "\n".join(prompt)},
        ], self.model, self.api_key, self.base_url)
        text = extract_message_text(data)
        parsed = parse_json_from_text(text)
        if self.log_raw:
            self.raw_logs.append({"stage": "action", "response": text})
        if not parsed:
            return self._stub_action(role, state, observation, intent, allowed_obs_ids)
        return normalize_action(parsed)

    def schedule_threads(self, thread_summaries: List[Dict[str, Any]]) -> List[str]:
        if self.mode != "llm":
            return self._stub_schedule(thread_summaries)
        prompt = [
            "You are the Meta role scheduling threads in ScholarArena.",
            "Return JSON only.",
            "Schema: {\"order\": [str]}",
            f"Threads: {thread_summaries}",
            "Order thread ids by priority. Prefer Open/EvidencePending and higher severity.",
        ]
        data = call_chat([
            {"role": "system", "content": "You output strict JSON with no extra text."},
            {"role": "user", "content": "\n".join(prompt)},
        ], self.model, self.api_key, self.base_url)
        text = extract_message_text(data)
        parsed = parse_json_from_text(text)
        if self.log_raw:
            self.raw_logs.append({"stage": "schedule", "response": text})
        if not parsed:
            return self._stub_schedule(thread_summaries)
        order = parsed.get("order") or []
        if isinstance(order, list) and order:
            return [str(item) for item in order]
        return self._stub_schedule(thread_summaries)

    def _stub_plan(self, role, state, intents, skills):
        intent = intents[0] if intents else "Request_Evidence"
        issue_text = state.get("issue_text") or state.get("tag") or "evidence"
        skill_name = skills[0] if skills else "Extract_Span"
        return {
            "intent": intent,
            "skill_call": {"name": skill_name, "arguments": {"query": issue_text}},
        }

    def _stub_action(self, role, state, observation, intent, allowed_obs_ids):
        if observation.get("status") != "ok":
            return {
                "action_type": "clarification",
                "text": "Evidence is missing; please clarify or provide additional context.",
                "claims": [],
                "requests": [{"text": "Clarify the missing evidence for this issue."}],
                "commitments": [],
            }
        obs_id = observation.get("id")
        return {
            "action_type": "evidence_response",
            "text": f"Evidence found in {obs_id}.",
            "claims": [{"text": "Evidence located", "cites": [obs_id]}],
            "requests": [],
            "commitments": [],
        }

    def _stub_schedule(self, summaries):
        def rank(item):
            phase = item.get("phase")
            severity = item.get("severity")
            sev_score = {"high": 3, "medium": 2, "low": 1}.get(severity, 1)
            phase_score = {"Open": 3, "EvidencePending": 2, "Negotiation": 1}.get(phase, 0)
            return (-phase_score, -sev_score)
        ordered = sorted(summaries, key=rank)
        return [item.get("issue_id") for item in ordered if item.get("issue_id")]


def normalize_plan(plan: Dict[str, Any], intents: List[str], skills: List[str]) -> Dict[str, Any]:
    intent = plan.get("intent")
    if not intent or intent not in intents:
        intent = intents[0] if intents else "Request_Evidence"
    skill_call = plan.get("skill_call") or {}
    name = skill_call.get("name") or ""
    if name not in skills:
        name = skills[0] if skills else "Extract_Span"
    args = skill_call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    return {"intent": intent, "skill_call": {"name": name, "arguments": args}}


def normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    action_type = action.get("action_type") or "response"
    text = action.get("text") or ""
    claims = action.get("claims") or []
    requests = action.get("requests") or []
    commitments = action.get("commitments") or []
    if not isinstance(claims, list):
        claims = []
    if not isinstance(requests, list):
        requests = []
    if not isinstance(commitments, list):
        commitments = []
    return {
        "action_type": action_type,
        "text": text,
        "claims": claims,
        "requests": requests,
        "commitments": commitments,
    }


def truncate_text(payload: Any, max_chars: int = 280) -> str:
    if payload is None:
        return ""
    text = json.dumps(payload, ensure_ascii=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
