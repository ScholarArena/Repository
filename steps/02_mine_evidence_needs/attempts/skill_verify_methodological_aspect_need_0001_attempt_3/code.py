import json
import runtime

def execute(context, params, primitives=None, controlled_llm=None):
    try:
        if not context or 'segments' not in context:
            return runtime.fail(error="Missing or invalid context")
        if not params or 'aspect' not in params:
            return runtime.fail(error="Missing or invalid params")
        if primitives is None or 'Extract_Aspect_Segments' not in primitives:
            return runtime.fail(error="Primitive Extract_Aspect_Segments not available")
        
        obs1 = primitives['Extract_Aspect_Segments'](context['segments'], params['aspect'])
        if obs1.get('status') != 'ok':
            return runtime.fail(error=f"Primitive failed: {obs1.get('error', 'unknown')}")
        
        if 'payload' not in obs1 or 'text' not in obs1['payload']:
            return runtime.fail(error="Primitive output missing text in payload")
        text = obs1['payload']['text']
        prov = obs1.get('prov', [])
        
        if controlled_llm is None:
            return runtime.fail(error="Controlled LLM not provided")
        
        aspect = params['aspect']
        prompt = f"Given the extracted text: {text}\n\nDetermine if there is a clear justification, explanation, or definition for the aspect '{aspect}'. Output a JSON object with 'evidence_found' (boolean) and 'reason' (string summarizing the evidence or lack thereof). Only use information from the provided text."
        
        response = controlled_llm(prompt)
        
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return runtime.fail(error="Controlled LLM response is not valid JSON")
        
        if 'evidence_found' not in parsed or 'reason' not in parsed:
            return runtime.fail(error="Controlled LLM response missing required fields")
        
        evidence_found = parsed['evidence_found']
        summary = parsed['reason']
        relevant_segments = prov
        
        payload = {
            "evidence_found": evidence_found,
            "summary": summary,
            "relevant_segments": relevant_segments
        }
        
        if evidence_found:
            return runtime.ok(payload=payload, prov=prov)
        else:
            return runtime.missing(payload=payload, prov=prov)
    except Exception as e:
        return runtime.fail(error=str(e))