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


def _extract_items(text):
    items = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-") or line.startswith("*") or line.startswith("+"):
            items.append(line.lstrip("-*+ "))
            continue
        if re.match(r"^\d+\.", line):
            items.append(re.sub(r"^\d+\.\s*", "", line))
    return items


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        segments = _get_segments(context)
        params = _normalize_params(params)
        query = str(params.get("query") or "").strip().lower()
        matches = []
        all_items = []
        for seg in segments:
            items = _extract_items(seg.get("text", ""))
            if query:
                items = [item for item in items if query in item.lower()]
            if items:
                matches.append({"segment_id": seg.get("id"), "items": items})
                all_items.extend(items)
        prov = [m.get("segment_id") for m in matches if m.get("segment_id") is not None]
        payload = {"items": all_items, "query": query, "matches": matches}
        if all_items:
            return runtime.ok(payload=payload, prov=prov, obs_type="text")
        return runtime.missing(reason="no_match", payload=payload, prov=prov, obs_type="text")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="text")
