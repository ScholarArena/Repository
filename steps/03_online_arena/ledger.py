from datetime import datetime
from typing import Any, Dict, List, Optional

PHASES = {"Open", "EvidencePending", "Negotiation", "Closed"}


def init_thread_state(seed: Dict[str, Any]) -> Dict[str, Any]:
    issue_id = seed.get("issue_id") or seed.get("thread_id")
    if not issue_id:
        raise ValueError("thread seed missing issue_id")
    state = {
        "issue_id": issue_id,
        "forum_id": seed.get("forum_id"),
        "tag": seed.get("issue_tag") or seed.get("issue_type"),
        "budget": int(seed.get("budget") or 6),
        "phase": seed.get("phase") or "Open",
        "severity": seed.get("severity") or "medium",
        "ledger": {
            "observations": list(seed.get("observations") or []),
            "requests": list(seed.get("requests") or []),
            "commitments": list(seed.get("commitments") or []),
        },
        "issue_text": seed.get("issue_text") or seed.get("request") or "",
        "context_id": seed.get("context_id"),
        "context_path": seed.get("context_path"),
        "oracle_moves": seed.get("oracle_moves") or [],
        "oracle_index": int(seed.get("oracle_index") or 0),
        "last_move": None,
        "updated_at": None,
    }
    if state["phase"] not in PHASES:
        state["phase"] = "Open"
    return state


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    ledger = state.get("ledger") or {}
    return {
        "issue_id": state.get("issue_id"),
        "tag": state.get("tag"),
        "phase": state.get("phase"),
        "severity": state.get("severity"),
        "budget": state.get("budget"),
        "requests": ledger.get("requests", []),
        "commitments": ledger.get("commitments", []),
        "observations": [obs.get("id") for obs in ledger.get("observations", []) if obs],
        "issue_text": state.get("issue_text"),
    }


def next_role(state: Dict[str, Any]) -> str:
    phase = state.get("phase")
    if phase in {"Open", "EvidencePending"}:
        return "Reviewer"
    if phase == "Negotiation":
        last = (state.get("last_move") or {}).get("role")
        return "Author" if last != "Author" else "Reviewer"
    return "Reviewer"


def update_state(
    state: Dict[str, Any],
    role: str,
    intent: str,
    observation: Dict[str, Any],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    ledger = state.get("ledger") or {}
    requests = list(ledger.get("requests") or [])
    commitments = list(ledger.get("commitments") or [])
    observations = list(ledger.get("observations") or [])

    if observation and observation.get("status") == "ok":
        observations.append(observation)

    for req in action.get("requests", []) or []:
        if req:
            requests.append(req)

    for com in action.get("commitments", []) or []:
        if com:
            commitments.append(com)

    action_type = action.get("action_type") or ""
    if role == "Reviewer" and action_type in {"request", "clarification"}:
        state["phase"] = "EvidencePending"
    elif role == "Author":
        state["phase"] = "Negotiation"
        if observation.get("status") == "ok":
            requests = []
    if action_type == "close" or state.get("budget", 0) <= 1:
        state["phase"] = "Closed"

    state["budget"] = max(0, int(state.get("budget") or 0) - 1)
    state["ledger"] = {
        "observations": observations,
        "requests": requests,
        "commitments": commitments,
    }
    state["last_move"] = {
        "role": role,
        "intent": intent,
        "observation_status": observation.get("status"),
        "action_type": action.get("action_type"),
    }
    state["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return state


def allowed_obs_ids(state: Dict[str, Any], current_obs: Dict[str, Any]) -> List[str]:
    ids = []
    ledger = state.get("ledger") or {}
    for obs in ledger.get("observations") or []:
        obs_id = obs.get("id")
        if obs_id and obs_id not in ids:
            ids.append(obs_id)
    if current_obs and current_obs.get("status") == "ok":
        obs_id = current_obs.get("id")
        if obs_id and obs_id not in ids:
            ids.append(obs_id)
    return ids
