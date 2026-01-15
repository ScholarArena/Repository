import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
runtime_root = None
for parent in [ROOT] + list(ROOT.parents):
    if (parent / "runtime.py").exists():
        runtime_root = parent
        break
if runtime_root is None:
    raise RuntimeError("runtime.py not found")
sys.path.insert(0, str(runtime_root))

from runtime import normalize_observation, controlled_llm_stub

CODE_PATH = Path(__file__).parent / "code.py"
TESTS_PATH = Path(__file__).parent / "tests.json"

spec = importlib.util.spec_from_file_location("artifact", CODE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
execute = getattr(module, "execute")

def make_stub_primitives(stubs):
    primitives = {}
    if not stubs:
        return primitives
    for name, obs in stubs.items():
        obs_copy = json.loads(json.dumps(obs))
        def _fn(*args, _obs=obs_copy, **kwargs):
            return _obs
        primitives[name] = _fn
    return primitives

def assert_payload_contains(payload, expected):
    for key, value in expected.items():
        if key not in payload:
            raise AssertionError(f"payload missing key: {key}")
        actual = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value not in str(actual):
                raise AssertionError(f"payload key {key} does not contain '{value}'")
        else:
            if actual != value:
                raise AssertionError(f"payload key {key} expected {value} got {actual}")

def run():
    tests = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
    for test in tests:
        context = test.get("context") or {}
        params = test.get("params") or {}
        stubs = test.get("primitive_stubs")
        primitives = make_stub_primitives(stubs)
        obs = execute(context, params, primitives=primitives, controlled_llm=controlled_llm_stub)
        obs = normalize_observation(obs)
        expected = test.get("expected") or {}
        if expected.get("status") and obs.get("status") != expected.get("status"):
            raise AssertionError(f"status expected {expected.get('status')} got {obs.get('status')}")
        if expected.get("type") and obs.get("type") != expected.get("type"):
            raise AssertionError(f"type expected {expected.get('type')} got {obs.get('type')}")
        if expected.get("prov_contains"):
            for item in expected.get("prov_contains"):
                if item not in obs.get("prov", []):
                    raise AssertionError(f"prov missing {item}")
        if expected.get("prov_len") is not None:
            if len(obs.get("prov", [])) != expected.get("prov_len"):
                raise AssertionError("prov length mismatch")
        if expected.get("payload_keys"):
            for key in expected.get("payload_keys"):
                if key not in obs.get("payload", {}):
                    raise AssertionError(f"payload missing key: {key}")
        if expected.get("payload_contains"):
            assert_payload_contains(obs.get("payload", {}), expected.get("payload_contains"))
    print("ok")

if __name__ == "__main__":
    run()
