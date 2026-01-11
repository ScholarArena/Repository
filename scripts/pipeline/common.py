import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("PYTHON", sys.executable)


def run(cmd, cwd=ROOT):
    cmd = [str(part) for part in cmd]
    print("+ " + " ".join(shlex.quote(part) for part in cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, check=True)


def require_api_key():
    if not (os.environ.get("DMX_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("Missing DMX_API_KEY or OPENAI_API_KEY in environment")
