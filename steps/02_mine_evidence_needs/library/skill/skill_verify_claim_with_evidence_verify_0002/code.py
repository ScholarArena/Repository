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
        queries = _normalize_list(params.get("queries") or params.get("query"))
        if not queries:
            return runtime.missing(reason="missing_queries", obs_type="verification_result")
        prim, err = _require_primitive(primitives, "Extract_Span")
        if err:
            return err
        all_matches = []
        matched_queries = []
        prov = []
        for query in queries:
            obs = prim(context, {"query": query, "regex": params.get("regex"), "case_sensitive": params.get("case_sensitive")})
            if obs.get("status") == "ok":
                matched_queries.append(query)
                all_matches.extend((obs.get("payload") or {}).get("matches", []))
                prov = _merge_prov({"prov": prov}, obs)
        payload = {
            "queries": queries,
            "matched_queries": matched_queries,
            "matches": all_matches,
            "supported": bool(matched_queries),
        }
        if matched_queries:
            return runtime.ok(payload=payload, prov=prov, obs_type="verification_result")
        return runtime.missing(reason="no_evidence", payload=payload, prov=prov, obs_type="verification_result")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="verification_result")
