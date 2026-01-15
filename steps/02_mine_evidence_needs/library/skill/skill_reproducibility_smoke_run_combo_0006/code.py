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
        prim_code, err = _require_primitive(primitives, "Check_Code_Availability")
        if err:
            return err
        prim_data, err = _require_primitive(primitives, "Check_Data_Availability")
        if err:
            return err
        prim_run, err = _require_primitive(primitives, "Sandbox_Run")
        if err:
            return err
        obs_code = prim_code(context, {
            "urls": params.get("code_urls"),
            "paths": params.get("code_paths"),
            "verify_remote": params.get("verify_remote"),
            "snapshot": params.get("code_snapshot"),
        })
        obs_data = prim_data(context, {
            "urls": params.get("data_urls"),
            "paths": params.get("data_paths"),
            "verify_remote": params.get("verify_remote"),
            "snapshot": params.get("data_snapshot"),
        })
        obs_run = prim_run(context, {
            "command": params.get("command"),
            "python_code": params.get("python_code"),
            "timeout": params.get("timeout"),
            "max_output_chars": params.get("max_output_chars"),
        })
        code_ok = (obs_code.get("payload") or {}).get("available", False)
        data_ok = (obs_data.get("payload") or {}).get("available", False)
        run_ok = obs_run.get("status") == "ok"
        payload = {
            "code_available": bool(code_ok),
            "data_available": bool(data_ok),
            "run_ok": bool(run_ok),
            "supported": bool(run_ok and (code_ok or data_ok)),
        }
        return runtime.ok(payload=payload, prov=[], obs_type="repro_smoke")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="repro_smoke")
