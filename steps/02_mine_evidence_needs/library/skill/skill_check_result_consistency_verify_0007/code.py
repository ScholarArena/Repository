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


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _match_expected(values, expected, tolerance):
    expected_val = _to_float(expected)
    if expected_val is None:
        return str(expected) in [str(v) for v in values]
    for val in values:
        fval = _to_float(val)
        if fval is None:
            continue
        if abs(fval - expected_val) <= tolerance:
            return True
    return False


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        expected = params.get("expected") or {}
        tolerance = float(params.get("tolerance") or 0.0)
        if not isinstance(expected, dict) or not expected:
            return runtime.missing(reason="missing_expected", obs_type="verification_result")
        prim, err = _require_primitive(primitives, "Extract_Metric_Values")
        if err:
            return err
        report = {}
        prov = []
        any_found = False
        for metric, expected_value in expected.items():
            obs = prim(context, {"metric": metric})
            values = (obs.get("payload") or {}).get("values", [])
            found = obs.get("status") == "ok"
            any_found = any_found or found
            consistent = False
            if found:
                consistent = _match_expected(values, expected_value, tolerance)
                prov = _merge_prov({"prov": prov}, obs)
            report[metric] = {
                "found": found,
                "expected": expected_value,
                "values": values,
                "consistent": consistent,
            }
        supported = all(item.get("consistent") for item in report.values())
        if not any_found:
            return runtime.missing(reason="no_metrics", payload={"report": report, "supported": supported}, prov=prov, obs_type="verification_result")
        return runtime.ok(payload={"report": report, "supported": supported}, prov=prov, obs_type="verification_result")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="verification_result")
