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
        requirements = set(_normalize_list(params.get("requirements")))
        metrics = _normalize_list(params.get("metrics"))
        found = {}
        prov = []
        for name, primitive_name, primitive_params in [
            ("dataset", "Extract_Dataset_Info", {}),
            ("splits", "Extract_Split_Info", {}),
            ("hyperparameters", "Extract_Hyperparameters", {}),
            ("training", "Extract_Training_Details", {}),
        ]:
            prim, err = _require_primitive(primitives, primitive_name)
            if err:
                return err
            obs = prim(context, primitive_params)
            found[name] = obs.get("status") == "ok"
            prov = _merge_prov({"prov": prov}, obs)
        if metrics:
            prim, err = _require_primitive(primitives, "Extract_Metric_Values")
            if err:
                return err
            metric_found = False
            for metric in metrics:
                obs = prim(context, {"metric": metric})
                if obs.get("status") == "ok":
                    metric_found = True
                    prov = _merge_prov({"prov": prov}, obs)
            found["metrics"] = metric_found
        else:
            found["metrics"] = False if "metrics" in requirements else True
        supported = True
        for requirement in requirements:
            if not found.get(requirement, False):
                supported = False
                break
        if not any(found.values()):
            return runtime.missing(reason="no_setup_evidence", payload={"found": found, "supported": supported}, prov=prov, obs_type="verification_result")
        payload = {"found": found, "supported": supported}
        return runtime.ok(payload=payload, prov=prov, obs_type="verification_result")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="verification_result")
