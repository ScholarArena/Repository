import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from utils import iter_jsonl


RUNTIME_DIR = os.path.join(MODULE_DIR, "..", "02_mine_evidence_needs")

if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)

try:
    import runtime
except Exception as exc:
    raise RuntimeError(f"Failed to import runtime from {RUNTIME_DIR}: {exc}")


@dataclass
class Artifact:
    name: str
    kind: str
    code_path: str
    spec_path: str
    module: Optional[Any] = None
    execute: Optional[Callable[..., Dict[str, Any]]] = None

    def load(self) -> None:
        if self.execute is not None:
            return
        spec = importlib.util.spec_from_file_location(
            f"arena_{self.kind}_{self.name}", self.code_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load artifact: {self.code_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        if not hasattr(module, "execute"):
            raise RuntimeError(f"Artifact missing execute(): {self.code_path}")
        self.execute = module.execute


class Library:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.artifacts: Dict[str, Artifact] = {}
        self._primitives: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for record in iter_jsonl(self.index_path):
            name = record.get("name")
            kind = record.get("kind")
            code_path = record.get("code_path")
            spec_path = record.get("spec_path")
            if not name or not kind or not code_path:
                continue
            artifact = Artifact(
                name=name,
                kind=kind,
                code_path=code_path,
                spec_path=spec_path or "",
            )
            if name in self.artifacts:
                raise RuntimeError(f"Duplicate artifact name: {name}")
            self.artifacts[name] = artifact
        self._loaded = True

    def _build_primitives(self) -> Dict[str, Callable[..., Dict[str, Any]]]:
        if self._primitives:
            return self._primitives
        self.load()
        primitives: Dict[str, Callable[..., Dict[str, Any]]] = {}
        for name, artifact in self.artifacts.items():
            if artifact.kind != "primitive":
                continue
            artifact.load()
            def _wrap(ctx, params, _exec=artifact.execute):
                obs = _exec(ctx, params, primitives=primitives, controlled_llm=runtime.controlled_llm_stub)
                return runtime.normalize_observation(obs)
            primitives[name] = _wrap
        self._primitives = primitives
        return primitives

    def available_skills(self) -> Dict[str, Artifact]:
        self.load()
        return {name: art for name, art in self.artifacts.items() if art.kind == "skill"}

    def available_primitives(self) -> Dict[str, Artifact]:
        self.load()
        return {name: art for name, art in self.artifacts.items() if art.kind == "primitive"}

    def execute(self, call: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        self.load()
        name = (call or {}).get("name")
        params = (call or {}).get("arguments") or {}
        if not name:
            return self._make_fail_obs("missing_skill_name")
        artifact = self.artifacts.get(name)
        if not artifact:
            return self._make_fail_obs(f"unknown_skill:{name}")
        artifact.load()
        primitives = self._build_primitives()
        obs = artifact.execute(context, params, primitives=primitives, controlled_llm=runtime.controlled_llm_stub)
        obs = runtime.normalize_observation(obs)
        obs["id"] = make_observation_id(obs)
        obs["skill_name"] = name
        return obs

    def _make_fail_obs(self, error: str) -> Dict[str, Any]:
        obs = runtime.fail(error=error)
        obs = runtime.normalize_observation(obs)
        obs["id"] = make_observation_id(obs)
        obs["skill_name"] = None
        return obs


def make_observation_id(obs: Dict[str, Any]) -> str:
    obs_type = obs.get("type", "evidence")
    prov = obs.get("prov") or []
    payload = {"type": obs_type, "prov": prov}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"O{digest[:12]}"
