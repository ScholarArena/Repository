import math
import runtime


def _normalize_params(params):
    return params if isinstance(params, dict) else {}


def _require_primitive(primitives, name):
    if not primitives or name not in primitives:
        return None, runtime.fail(error=f"primitive_missing:{name}")
    return primitives[name], None


def _merge_prov(*observations):
    prov = []
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        for item in obs.get("prov", []) or []:
            if item not in prov:
                prov.append(item)
    return prov


def _extract_match_texts(obs):
    payload = (obs or {}).get("payload") or {}
    matches = payload.get("matches") or []
    texts = []
    for match in matches:
        text = match.get("text")
        if text:
            texts.append(str(text))
    return texts


def _truncate_texts(texts, max_chars):
    if not max_chars or max_chars <= 0:
        return texts
    output = []
    total = 0
    for text in texts:
        if total >= max_chars:
            break
        remaining = max_chars - total
        snippet = text[:remaining]
        output.append(snippet)
        total += len(snippet)
    return output


def _normalize_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        cite_key = str(params.get("cite_key") or params.get("citation") or "").strip()
        if not cite_key:
            return runtime.missing(reason="missing_cite_key", obs_type="verification_result")
        prim, err = _require_primitive(primitives, "Resolve_Citation")
        if err:
            return err
        obs = prim(context, {"cite_key": cite_key, "max_matches": params.get("max_matches")})
        payload = {
            "cite_key": cite_key,
            "supported": obs.get("status") == "ok",
            "matches": (obs.get("payload") or {}).get("matches", []),
        }
        if obs.get("status") == "ok":
            return runtime.ok(payload=payload, prov=obs.get("prov", []), obs_type="verification_result")
        return runtime.missing(reason="citation_not_found", payload=payload, prov=obs.get("prov", []), obs_type="verification_result")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="verification_result")
