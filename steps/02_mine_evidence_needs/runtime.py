import copy


def make_observation(obs_type, payload, prov, status):
    return {
        "type": obs_type,
        "payload": payload or {},
        "prov": list(prov or []),
        "status": status,
    }


def ok(payload=None, prov=None, obs_type="evidence", **kwargs):
    if "type" in kwargs and obs_type == "evidence":
        obs_type = kwargs["type"]
    if prov is None and "prov" in kwargs:
        prov = kwargs["prov"]
    if payload is None and "payload" in kwargs:
        payload = kwargs["payload"]
    return make_observation(obs_type, payload, prov, "ok")


def missing(reason="", obs_type="evidence", payload=None, prov=None, **kwargs):
    if "type" in kwargs and obs_type == "evidence":
        obs_type = kwargs["type"]
    if payload is None:
        payload = {"reason": reason} if reason else {}
    if prov is None:
        prov = []
    return make_observation(obs_type, payload, prov, "missing")


def fail(error="", obs_type="evidence", payload=None, prov=None, **kwargs):
    if "type" in kwargs and obs_type == "evidence":
        obs_type = kwargs["type"]
    if payload is None:
        payload = {"error": error} if error else {}
    if prov is None:
        prov = []
    return make_observation(obs_type, payload, prov, "fail")


def normalize_observation(obj, default_type="evidence"):
    if obj is None:
        raise ValueError("Observation is None")
    if isinstance(obj, dict):
        data = dict(obj)
    else:
        data = {
            "type": getattr(obj, "type", default_type),
            "payload": getattr(obj, "payload", {}),
            "prov": getattr(obj, "prov", []),
            "status": getattr(obj, "status", "ok"),
        }
    data.setdefault("type", default_type)
    data.setdefault("payload", {})
    data.setdefault("prov", [])
    data.setdefault("status", "ok")
    data["prov"] = [int(item) for item in data.get("prov", []) if item is not None]
    status = data.get("status")
    if status not in {"ok", "missing", "fail"}:
        data["status"] = "fail"
        data["payload"] = {"error": f"invalid_status:{status}"}
    return data


def controlled_llm_stub(prompt=None, evidence=None, **kwargs):
    if evidence is None:
        evidence = kwargs.get("evidence")
    if evidence is None:
        evidence = kwargs.get("context")
    if evidence is None:
        evidence = kwargs.get("segments")
    if evidence is None:
        evidence = prompt
    if evidence is None:
        raise ValueError("controlled_llm requires evidence or prompt")
    evidence_copy = copy.deepcopy(evidence)
    if isinstance(evidence_copy, list):
        summary = " | ".join(str(item) for item in evidence_copy)
    else:
        summary = str(evidence_copy)
    return {"summary": summary}
