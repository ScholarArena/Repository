import re
import runtime

def _get_segments(context):
    segments = []
    if isinstance(context, dict):
        raw = context.get("segments")
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                if "id" not in item or "text" not in item:
                    continue
                try:
                    seg_id = int(item.get("id"))
                except Exception:
                    continue
                segments.append({"id": seg_id, "text": str(item.get("text"))})
    return segments

def _normalize_params(params):
    return params if isinstance(params, dict) else {}

def _make_matches(segments, matcher, max_matches):
    matches = []
    for seg in segments:
        text = seg.get("text", "")
        match_info = matcher(text)
        if match_info:
            matches.append({
                "segment_id": seg.get("id"),
                "text": text,
                "match": match_info,
            })
            if len(matches) >= max_matches:
                break
    return matches

def _payload_with_matches(matches, extra=None):
    payload = {"matches": matches}
    if extra:
        payload.update(extra)
    return payload

def _collect_prov(matches):
    prov = []
    for item in matches:
        seg_id = item.get("segment_id")
        if seg_id is not None and seg_id not in prov:
            prov.append(seg_id)
    return prov


def _extract_candidates(text):
    candidates = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9\-+]{1,}\b", text):
        if token in {"We", "Our", "Table", "Figure", "Section", "Results", "Baseline", "Baselines"}:
            continue
        if token not in candidates:
            candidates.append(token)
    return candidates


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        segments = _get_segments(context)
        params = _normalize_params(params)
        keywords = ["baseline", "baselines", "compare", "compared", "comparison", "against"]
        matches = []
        candidates = []
        for seg in segments:
            text = seg.get("text", "")
            lower = text.lower()
            if not any(k in lower for k in keywords):
                continue
            seg_candidates = _extract_candidates(text)
            if seg_candidates:
                matches.append({"segment_id": seg.get("id"), "text": text, "match": "baseline"})
                for item in seg_candidates:
                    if item not in candidates:
                        candidates.append(item)
        prov = _collect_prov(matches)
        payload = {"candidates": candidates, "matches": matches}
        if candidates:
            return runtime.ok(payload=payload, prov=prov, obs_type="text")
        return runtime.missing(reason="no_candidates", payload=payload, prov=prov, obs_type="text")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="text")
