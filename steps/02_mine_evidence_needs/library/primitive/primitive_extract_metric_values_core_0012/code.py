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


def _extract_numbers(text):
    values = []
    for match in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?%?", text, flags=re.IGNORECASE):
        values.append(match)
    return values


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        segments = _get_segments(context)
        params = _normalize_params(params)
        metric = str(params.get("metric") or "").strip()
        if not metric:
            return runtime.missing(error="missing_metric")
        metric_lower = metric.lower()
        matches = []
        values = []
        for seg in segments:
            text = seg.get("text", "")
            if metric_lower not in text.lower():
                continue
            nums = _extract_numbers(text)
            if nums:
                matches.append({"segment_id": seg.get("id"), "text": text, "match": metric})
                values.extend(nums)
        prov = _collect_prov(matches)
        payload = {"metric": metric, "values": values, "matches": matches}
        if values:
            return runtime.ok(payload=payload, prov=prov, obs_type="text")
        return runtime.missing(reason="no_values", payload=payload, prov=prov, obs_type="text")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="text")
