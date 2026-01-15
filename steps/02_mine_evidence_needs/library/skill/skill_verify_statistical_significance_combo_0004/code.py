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


def _normalize_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _strip_doi_prefix(doi):
    doi = str(doi or "").strip()
    if "doi.org/" in doi:
        return doi.split("doi.org/")[-1]
    return doi


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        prim_test, err = _require_primitive(primitives, "Run_T_Test")
        if err:
            return err
        prim_check, err = _require_primitive(primitives, "Check_Significance_Claim")
        if err:
            return err
        obs_test = prim_test(context, {
            "group_a": params.get("group_a"),
            "group_b": params.get("group_b"),
            "paired": params.get("paired"),
            "equal_var": params.get("equal_var"),
            "round": params.get("round"),
        })
        if obs_test.get("status") != "ok":
            return runtime.missing(reason="t_test_failed", payload={}, prov=[], obs_type="significance_check")
        p_value = (obs_test.get("payload") or {}).get("p_value")
        obs_check = prim_check(context, {
            "claim": params.get("claim"),
            "claim_type": params.get("claim_type"),
            "p_value": p_value,
            "alpha": params.get("alpha"),
        })
        supported = (obs_check.get("payload") or {}).get("supported")
        payload = {
            "t_stat": (obs_test.get("payload") or {}).get("t_stat"),
            "p_value": p_value,
            "supported": bool(supported),
        }
        return runtime.ok(payload=payload, prov=[], obs_type="significance_check")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="significance_check")
