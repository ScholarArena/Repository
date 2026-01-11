import json
from pathlib import Path

def test_manifest_has_required_fields():
    data = json.loads((Path(__file__).resolve().parents[1] / 'primitive.json').read_text())
    assert 'primitive_id' in data
    assert 'signature' in data
    assert 'evidence_schema' in data
