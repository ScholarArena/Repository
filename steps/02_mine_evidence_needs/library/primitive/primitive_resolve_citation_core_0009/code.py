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


def _parse_cite_digits(value):
    digits = re.findall(r"\d+", str(value or ""))
    return digits


def _match_numeric_citation(text, digits):
    if not digits:
        return False
    for block in re.findall(r"\[[^\]]+\]", text):
        nums = re.findall(r"\d+", block)
        for d in digits:
            if d in nums:
                return True
    return False


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        segments = _get_segments(context)
        params = _normalize_params(params)
        cite_key = str(params.get("cite_key") or "").strip()
        if not cite_key:
            return runtime.missing(error="missing_cite_key")
        digits = _parse_cite_digits(cite_key)
        key_lower = cite_key.lower()
        def matcher(text):
            lower = text.lower()
            if digits and _match_numeric_citation(text, digits):
                return cite_key
            if "10." in cite_key or "doi" in key_lower:
                return cite_key if key_lower in lower else ""
            return cite_key if key_lower in lower else ""
        matches = _make_matches(segments, matcher, int(params.get("max_matches") or 5))
        prov = _collect_prov(matches)
        payload = _payload_with_matches(matches, {"cite_key": cite_key})
        if matches:
            return runtime.ok(payload=payload, prov=prov, obs_type="text")
        return runtime.missing(reason="no_match", payload=payload, prov=prov, obs_type="text")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="text")
