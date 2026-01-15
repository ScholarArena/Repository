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


def _diffs(a, b):
    if not isinstance(a, list) or not isinstance(b, list):
        return []
    if len(a) != len(b):
        return []
    out = []
    for x, y in zip(a, b):
        try:
            out.append(float(x) - float(y))
        except Exception:
            continue
    return out


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        prim_effect, err = _require_primitive(primitives, "Compute_Effect_Size")
        if err:
            return err
        prim_ci, err = _require_primitive(primitives, "Bootstrap_CI")
        if err:
            return err
        obs_effect = prim_effect(context, {
            "group_a": params.get("group_a"),
            "group_b": params.get("group_b"),
            "round": params.get("round"),
        })
        if obs_effect.get("status") != "ok":
            return runtime.missing(reason="effect_size_failed", payload={}, prov=[], obs_type="effect_size_ci")
        values = params.get("values")
        if values is None:
            values = _diffs(params.get("group_a"), params.get("group_b"))
        obs_ci = prim_ci(context, {
            "values": values,
            "n_boot": params.get("n_boot"),
            "alpha": params.get("alpha"),
            "seed": params.get("seed"),
            "stat": params.get("stat"),
            "round": params.get("round"),
        })
        if obs_ci.get("status") != "ok":
            return runtime.missing(reason="ci_failed", payload={}, prov=[], obs_type="effect_size_ci")
        payload = {
            "effect_size": (obs_effect.get("payload") or {}).get("effect_size"),
            "ci_lower": (obs_ci.get("payload") or {}).get("lower"),
            "ci_upper": (obs_ci.get("payload") or {}).get("upper"),
        }
        return runtime.ok(payload=payload, prov=[], obs_type="effect_size_ci")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="effect_size_ci")
