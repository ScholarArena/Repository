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


def _normalize_reference(ref):
    if isinstance(ref, dict):
        doi = ref.get("doi")
        query = ref.get("query") or ref.get("title")
        return _strip_doi_prefix(doi), query
    return "", str(ref)


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        references = params.get("references") or []
        references = references if isinstance(references, list) else [references]
        if not references:
            return runtime.missing(reason="missing_references", obs_type="external_exists")
        prim, err = _require_primitive(primitives, "Verify_Reference_Existence")
        if err:
            return err
        results = []
        exists_all = True
        for ref in references:
            doi, query = _normalize_reference(ref)
            obs = prim(context, {
                "doi": doi,
                "query": query,
                "cache_dir": params.get("cache_dir"),
                "cache_only": params.get("cache_only"),
                "timeout": params.get("timeout"),
                "mailto": params.get("mailto"),
                "snapshot": params.get("snapshot"),
            })
            payload = obs.get("payload") or {}
            exists = bool(payload.get("exists"))
            results.append({"doi": doi, "query": query, "exists": exists})
            if not exists:
                exists_all = False
        payload = {"results": results, "exists_all": exists_all}
        return runtime.ok(payload=payload, prov=[], obs_type="external_exists")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="external_exists")
