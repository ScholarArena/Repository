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
        eq_id = params.get("eq_id")
        terms = _normalize_list(params.get("assumption_terms"))
        if not eq_id:
            return runtime.missing(reason="missing_eq_id", obs_type="verification_result")
        prim_eq, err = _require_primitive(primitives, "Extract_Equation")
        if err:
            return err
        obs_eq = prim_eq(context, {"eq_id": eq_id})
        if obs_eq.get("status") != "ok":
            return runtime.missing(reason="equation_not_found", payload={"eq_id": eq_id}, prov=obs_eq.get("prov", []), obs_type="verification_result")
        prim_span, err = _require_primitive(primitives, "Extract_Span")
        if err:
            return err
        missing_terms = []
        prov = obs_eq.get("prov", [])
        for term in terms:
            obs = prim_span(context, {"query": term})
            if obs.get("status") == "ok":
                prov = _merge_prov({"prov": prov}, obs)
            else:
                missing_terms.append(term)
        supported = not missing_terms if terms else True
        payload = {"eq_id": eq_id, "assumption_terms": terms, "missing_terms": missing_terms, "supported": supported}
        return runtime.ok(payload=payload, prov=prov, obs_type="verification_result")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="verification_result")
