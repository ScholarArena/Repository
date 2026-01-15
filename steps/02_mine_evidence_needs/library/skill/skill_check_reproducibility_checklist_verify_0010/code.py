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
        verify_remote = bool(params.get("verify_remote"))
        prim_links, err = _require_primitive(primitives, "Extract_Code_Data_Links")
        if err:
            return err
        prim_hp, err = _require_primitive(primitives, "Extract_Hyperparameters")
        if err:
            return err
        prim_compute, err = _require_primitive(primitives, "Extract_Compute_Resources")
        if err:
            return err
        prim_train, err = _require_primitive(primitives, "Extract_Training_Details")
        if err:
            return err
        obs_links = prim_links(context, {})
        urls = (obs_links.get("payload") or {}).get("urls", [])
        code_urls = _normalize_list(params.get("code_urls")) or urls
        data_urls = _normalize_list(params.get("data_urls")) or urls
        found = {
            "code": bool(code_urls),
            "data": bool(data_urls),
        }
        if verify_remote:
            prim_code, err = _require_primitive(primitives, "Check_Code_Availability")
            if err:
                return err
            prim_data, err = _require_primitive(primitives, "Check_Data_Availability")
            if err:
                return err
            obs_code = prim_code(context, {"urls": code_urls, "verify_remote": True, "snapshot": params.get("code_snapshot")})
            obs_data = prim_data(context, {"urls": data_urls, "verify_remote": True, "snapshot": params.get("data_snapshot")})
            found["code"] = (obs_code.get("payload") or {}).get("available", False)
            found["data"] = (obs_data.get("payload") or {}).get("available", False)
        obs_hp = prim_hp(context, {})
        obs_compute = prim_compute(context, {})
        obs_train = prim_train(context, {})
        found["hyperparameters"] = obs_hp.get("status") == "ok"
        found["compute"] = obs_compute.get("status") == "ok"
        found["training"] = obs_train.get("status") == "ok"
        supported = True
        for requirement in requirements:
            if not found.get(requirement, False):
                supported = False
                break
        prov = _merge_prov(obs_links, obs_hp, obs_compute, obs_train)
        payload = {"found": found, "supported": supported}
        if not any(found.values()):
            return runtime.missing(reason="no_repro_evidence", payload=payload, prov=prov, obs_type="verification_result")
        return runtime.ok(payload=payload, prov=prov, obs_type="verification_result")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="verification_result")
