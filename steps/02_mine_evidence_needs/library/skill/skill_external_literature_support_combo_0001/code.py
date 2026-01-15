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
        query = str(params.get("query") or "").strip()
        if not query:
            return runtime.missing(reason="missing_query", obs_type="external_support")
        prim_search, err = _require_primitive(primitives, "External_Search_OpenAlex")
        if err:
            return err
        prim_abs, err = _require_primitive(primitives, "Fetch_Abstract")
        if err:
            return err
        prim_meta, err = _require_primitive(primitives, "Resolve_DOI_Metadata")
        if err:
            return err
        search_params = {
            "query": query,
            "per_page": params.get("per_page"),
            "filter": params.get("filter"),
            "sort": params.get("sort"),
            "cache_dir": params.get("cache_dir"),
            "cache_only": params.get("cache_only"),
            "timeout": params.get("timeout"),
            "mailto": params.get("mailto"),
            "snapshot": params.get("search_snapshot"),
        }
        obs_search = prim_search(context, search_params)
        if obs_search.get("status") != "ok":
            return runtime.missing(reason="search_failed", payload={"query": query}, prov=[], obs_type="external_support")
        results = (obs_search.get("payload") or {}).get("results", [])
        max_results = int(params.get("max_results") or len(results))
        include_abs = bool(params.get("include_abstracts", True))
        include_meta = bool(params.get("include_metadata", False))
        works = []
        for item in results[:max_results]:
            entry = dict(item or {})
            openalex_id = entry.get("id")
            doi = _strip_doi_prefix(entry.get("doi"))
            if include_abs and openalex_id:
                obs_abs = prim_abs(context, {
                    "openalex_id": openalex_id,
                    "cache_dir": params.get("cache_dir"),
                    "cache_only": params.get("cache_only"),
                    "timeout": params.get("timeout"),
                    "mailto": params.get("mailto"),
                    "snapshot": params.get("abstract_snapshot"),
                })
                if obs_abs.get("status") == "ok":
                    entry["abstract"] = (obs_abs.get("payload") or {}).get("abstract")
            if include_meta and doi:
                obs_meta = prim_meta(context, {
                    "doi": doi,
                    "cache_dir": params.get("cache_dir"),
                    "cache_only": params.get("cache_only"),
                    "timeout": params.get("timeout"),
                    "mailto": params.get("mailto"),
                    "snapshot": params.get("metadata_snapshot"),
                })
                if obs_meta.get("status") == "ok":
                    entry["metadata"] = obs_meta.get("payload")
            works.append(entry)
        payload = {
            "query": query,
            "count": (obs_search.get("payload") or {}).get("count", 0),
            "results": works,
            "supported": bool(works),
        }
        if works:
            return runtime.ok(payload=payload, prov=[], obs_type="external_support")
        return runtime.missing(reason="no_results", payload=payload, prov=[], obs_type="external_support")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="external_support")
