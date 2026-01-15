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


def _extract_hparams(text):
    hparams = {}
    patterns = {
        "learning_rate": r"(?:learning rate|lr)\s*(?:=|:)?\s*([0-9\.eE-]+)",
        "batch_size": r"(?:batch size|batch)\s*(?:=|:)?\s*(\d+)",
        "epochs": r"(?:epochs?|training steps?)\s*(?:=|:)?\s*(\d+)",
        "weight_decay": r"weight decay\s*(?:=|:)?\s*([0-9\.eE-]+)",
        "dropout": r"dropout\s*(?:=|:)?\s*([0-9\.eE-]+)",
        "momentum": r"momentum\s*(?:=|:)?\s*([0-9\.eE-]+)",
    }
    lower = text.lower()
    for key, pattern in patterns.items():
        match = re.search(pattern, lower)
        if match:
            hparams[key] = match.group(1)
    return hparams


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        segments = _get_segments(context)
        matches = []
        hparams = {}
        for seg in segments:
            text = seg.get("text", "")
            seg_hparams = _extract_hparams(text)
            if seg_hparams:
                matches.append({"segment_id": seg.get("id"), "text": text, "match": "hyperparams"})
                hparams.update(seg_hparams)
        prov = _collect_prov(matches)
        payload = {"hyperparameters": hparams, "matches": matches}
        if hparams:
            return runtime.ok(payload=payload, prov=prov, obs_type="text")
        return runtime.missing(reason="no_hyperparameters", payload=payload, prov=prov, obs_type="text")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="text")
