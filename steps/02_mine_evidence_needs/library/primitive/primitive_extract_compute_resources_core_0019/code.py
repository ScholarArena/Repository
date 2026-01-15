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


def _extract_resources(text):
    resources = []
    for count, model in re.findall(r"(\d+)\s*x?\s*(A100|V100|RTX\s*\d+|GPU|TPU)", text, flags=re.IGNORECASE):
        resources.append(f"{count}x{model}")
    for count, unit in re.findall(r"(\d+)\s*(hours?|days?)", text, flags=re.IGNORECASE):
        resources.append(f"{count} {unit}")
    return resources


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        segments = _get_segments(context)
        keywords = ["gpu", "tpu", "cpu", "a100", "v100", "rtx", "hours", "days"]
        matches = []
        resources = []
        for seg in segments:
            text = seg.get("text", "")
            lower = text.lower()
            if not any(k in lower for k in keywords):
                continue
            matches.append({"segment_id": seg.get("id"), "text": text, "match": "resources"})
            resources.extend(_extract_resources(text))
        prov = _collect_prov(matches)
        payload = {"resources": resources, "matches": matches}
        if matches:
            return runtime.ok(payload=payload, prov=prov, obs_type="text")
        return runtime.missing(reason="no_match", payload=payload, prov=prov, obs_type="text")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="text")
