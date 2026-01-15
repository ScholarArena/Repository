import json
import hashlib
import os
import sys
import time
import subprocess
from pathlib import Path
from urllib import request as urlrequest
from urllib import error as urlerror

import runtime


def _normalize_params(params):
    return params if isinstance(params, dict) else {}


def _get_cache_dir(params):
    cache_dir = params.get("cache_dir") or os.environ.get("AVAILABILITY_CACHE_DIR")
    if not cache_dir:
        cache_dir = Path(__file__).parent / "_cache"
    return Path(cache_dir)


def _ensure_cache_dir(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _snapshot_path(cache_dir, url):
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _load_snapshot(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_snapshot(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_map(params):
    snapshot = params.get("snapshot")
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except Exception:
            snapshot = None
    if isinstance(snapshot, list):
        return {item.get("url"): item for item in snapshot if isinstance(item, dict) and item.get("url")}
    if isinstance(snapshot, dict):
        return snapshot
    return {}


def _check_url(url, params):
    params = _normalize_params(params)
    cache_only = bool(params.get("cache_only"))
    timeout = int(params.get("timeout") or 10)
    cache_dir = _get_cache_dir(params)
    try:
        _ensure_cache_dir(cache_dir)
    except Exception as exc:
        return {"url": url, "ok": False, "error": "cache_unavailable", "detail": str(exc)}
    path = _snapshot_path(cache_dir, url)
    if path.exists():
        data = _load_snapshot(path)
        data["snapshot_path"] = str(path)
        data["snapshot_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return data
    snap_map = _snapshot_map(params)
    if url in snap_map:
        data = dict(snap_map[url])
        data.setdefault("url", url)
        data["snapshot_sha256"] = _save_snapshot(path, data)
        data["snapshot_path"] = str(path)
        return data
    if cache_only:
        return {"url": url, "ok": False, "error": "cache_miss"}
    try:
        req = urlrequest.Request(url, method="HEAD")
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
        ok = 200 <= int(status) < 400
        data = {"url": url, "ok": ok, "status": int(status), "fetched_at": int(time.time())}
    except urlerror.HTTPError as exc:
        data = {"url": url, "ok": False, "status": int(exc.code), "error": "http_error", "fetched_at": int(time.time())}
    except Exception as exc:
        data = {"url": url, "ok": False, "error": "fetch_error", "detail": str(exc), "fetched_at": int(time.time())}
    data["snapshot_sha256"] = _save_snapshot(path, data)
    data["snapshot_path"] = str(path)
    return data


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        urls = params.get("urls") or []
        paths = params.get("paths") or []
        verify_remote = bool(params.get("verify_remote"))
        checked_urls = []
        checked_paths = []
        available = False
        for path in paths:
            exists = Path(str(path)).exists()
            checked_paths.append({"path": str(path), "exists": exists})
            if exists:
                available = True
        if verify_remote:
            for url in urls:
                status = _check_url(str(url), params)
                checked_urls.append(status)
                if status.get("ok"):
                    available = True
        else:
            available = available or bool(urls)
        payload = {
            "available": bool(available),
            "checked_urls": checked_urls,
            "checked_paths": checked_paths,
        }
        return runtime.ok(payload=payload, prov=[], obs_type="data_availability")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="data_availability")
