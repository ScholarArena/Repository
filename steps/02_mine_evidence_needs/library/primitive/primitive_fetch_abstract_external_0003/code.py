import json
import hashlib
import os
import time
from pathlib import Path
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib import error as urlerror

import runtime

OPENALEX_BASE = "https://api.openalex.org"


def _normalize_params(params):
    return params if isinstance(params, dict) else {}


def _get_cache_dir(params):
    cache_dir = params.get("cache_dir") or os.environ.get("OPENALEX_CACHE_DIR") or os.environ.get("OPENALEX_SNAPSHOT_DIR")
    if not cache_dir:
        cache_dir = Path(__file__).parent / "_cache"
    return Path(cache_dir)


def _ensure_cache_dir(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _build_url(endpoint, query_params=None):
    base = OPENALEX_BASE.rstrip("/")
    endpoint = endpoint.lstrip("/")
    if query_params:
        clean = {}
        for key, value in query_params.items():
            if value is None or value == "":
                continue
            clean[key] = value
        query = urlparse.urlencode(sorted(clean.items()), doseq=True)
        return f"{base}/{endpoint}?{query}"
    return f"{base}/{endpoint}"


def _snapshot_path(cache_dir, url):
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _load_snapshot(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "payload" in data:
        return data.get("payload"), data
    return data, {"payload": data}


def _save_snapshot(path, url, payload):
    snapshot = {
        "url": url,
        "fetched_at": int(time.time()),
        "payload": payload,
    }
    path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot, sha


def _fetch_json(url, timeout):
    req = urlrequest.Request(url, headers={"User-Agent": "ScholarArena/1.0"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def _get_payload(params, url):
    params = _normalize_params(params)
    timeout = int(params.get("timeout") or 10)
    cache_only = bool(params.get("cache_only"))
    cache_dir = _get_cache_dir(params)
    try:
        _ensure_cache_dir(cache_dir)
    except Exception as exc:
        return None, None, "cache_unavailable", str(exc)
    path = _snapshot_path(cache_dir, url)
    if path.exists():
        payload, snapshot = _load_snapshot(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return payload, {"snapshot_path": str(path), "snapshot_sha256": sha, "url": url}, None, None
    snapshot_param = params.get("snapshot")
    if isinstance(snapshot_param, str):
        try:
            snapshot_param = json.loads(snapshot_param)
        except Exception:
            snapshot_param = None
    if snapshot_param is not None:
        if isinstance(snapshot_param, dict) and "payload" in snapshot_param:
            payload = snapshot_param.get("payload")
        else:
            payload = snapshot_param
        _, sha = _save_snapshot(path, url, payload)
        return payload, {"snapshot_path": str(path), "snapshot_sha256": sha, "url": url}, None, None
    if cache_only:
        return None, None, "cache_miss", "cache_only"
    try:
        payload = _fetch_json(url, timeout)
    except urlerror.HTTPError as exc:
        if getattr(exc, "code", None) == 404:
            return None, None, "not_found", "http_404"
        return None, None, "http_error", str(exc)
    except Exception as exc:
        return None, None, "fetch_error", str(exc)
    _, sha = _save_snapshot(path, url, payload)
    return payload, {"snapshot_path": str(path), "snapshot_sha256": sha, "url": url}, None, None


def _source_payload(source_meta):
    if not source_meta:
        return []
    return [
        {
            "source": "openalex",
            "url": source_meta.get("url"),
            "snapshot_path": source_meta.get("snapshot_path"),
            "snapshot_sha256": source_meta.get("snapshot_sha256"),
        }
    ]


def _abstract_from_inverted(index):
    if not isinstance(index, dict):
        return ""
    positions = {}
    for term, locs in index.items():
        for pos in locs:
            positions[int(pos)] = term
    if not positions:
        return ""
    max_pos = max(positions.keys())
    words = []
    for i in range(max_pos + 1):
        words.append(positions.get(i, ""))
    return " ".join(word for word in words if word)


def _resolve_endpoint(params):
    openalex_id = params.get("openalex_id")
    doi = params.get("doi")
    if openalex_id:
        return f"works/{str(openalex_id).strip()}"
    if doi:
        doi = str(doi).strip()
        return "works/" + urlparse.quote("https://doi.org/" + doi)
    return ""


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        endpoint = _resolve_endpoint(params)
        if not endpoint:
            return runtime.missing(error="missing_identifier")
        mailto = params.get("mailto") or os.environ.get("OPENALEX_MAILTO")
        url = _build_url(endpoint, {"mailto": mailto})
        payload, source_meta, err_code, err_text = _get_payload(params, url)
        if err_code == "cache_miss":
            return runtime.missing(reason="cache_miss", payload={}, prov=[])
        if err_code == "not_found":
            return runtime.missing(reason="not_found", payload={}, prov=[])
        if err_code:
            return runtime.fail(error=f"{err_code}:{err_text}")
        abstract = ""
        if isinstance(payload, dict):
            abstract = payload.get("abstract") or ""
            if not abstract:
                abstract = _abstract_from_inverted(payload.get("abstract_inverted_index"))
        out_payload = {
            "openalex_id": params.get("openalex_id"),
            "doi": params.get("doi"),
            "abstract": abstract,
            "sources": _source_payload(source_meta),
        }
        if abstract:
            return runtime.ok(payload=out_payload, prov=[], obs_type="external_abstract")
        return runtime.missing(reason="no_abstract", payload=out_payload, prov=[], obs_type="external_abstract")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="external_abstract")
