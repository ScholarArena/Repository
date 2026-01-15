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
        openalex_id = params.get("openalex_id")
        doi = params.get("doi")
        if not openalex_id and not doi:
            return runtime.missing(reason="missing_identifier", obs_type="external_citation")
        prim_refs, err = _require_primitive(primitives, "Fetch_Reference_List")
        if err:
            return err
        prim_cited, err = _require_primitive(primitives, "Citation_Graph_Neighbors")
        if err:
            return err
        ref_obs = prim_refs(context, {
            "openalex_id": openalex_id,
            "doi": doi,
            "max_refs": params.get("max_refs"),
            "cache_dir": params.get("cache_dir"),
            "cache_only": params.get("cache_only"),
            "timeout": params.get("timeout"),
            "mailto": params.get("mailto"),
            "snapshot": params.get("references_snapshot"),
        })
        cited_obs = prim_cited(context, {
            "openalex_id": openalex_id,
            "doi": doi,
            "direction": "cited_by",
            "per_page": params.get("per_page"),
            "cache_dir": params.get("cache_dir"),
            "cache_only": params.get("cache_only"),
            "timeout": params.get("timeout"),
            "mailto": params.get("mailto"),
            "snapshot": params.get("cited_by_snapshot"),
        })
        references = (ref_obs.get("payload") or {}).get("references", [])
        cited_by = (cited_obs.get("payload") or {}).get("neighbors", [])
        payload = {
            "reference_count": len(references),
            "references": references,
            "cited_by_count": len(cited_by),
            "cited_by": cited_by,
            "supported": bool(references or cited_by),
        }
        if references or cited_by:
            return runtime.ok(payload=payload, prov=[], obs_type="external_citation")
        return runtime.missing(reason="no_neighbors", payload=payload, prov=[], obs_type="external_citation")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="external_citation")
