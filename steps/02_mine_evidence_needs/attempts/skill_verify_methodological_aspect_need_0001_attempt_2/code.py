import json
from typing import Dict, Any, List, Optional

def execute(context: Dict[str, Any], params: Dict[str, Any], primitives: Optional[Dict[str, Any]] = None, controlled_llm: Optional[Any] = None) -> Dict[str, Any]:
    try:
        # Import runtime functions
        from runtime import ok, missing, fail
        
        # Validate inputs
        if not context or 'segments' not in context:
            return fail(error="Missing or invalid context with 'segments' key")
        
        if not params or 'aspect' not in params:
            return fail(error="Missing or invalid params with 'aspect' key")
        
        if not primitives or 'Extract_Aspect_Segments' not in primitives:
            return fail(error="Required primitive 'Extract_Aspect_Segments' not available")
        
        if not controlled_llm:
            return fail(error="Controlled LLM not available")
        
        # Step 1: Extract aspect segments using primitive
        segments = context['segments']
        aspect = params['aspect']
        
        primitive_result = primitives['Extract_Aspect_Segments'](segments, aspect)
        
        if primitive_result.get('status') != 'ok':
            # Propagate primitive status
            return {
                'type': 'verification_result',
                'payload': {
                    'evidence_found': False,
                    'summary': f"Primitive failed: {primitive_result.get('error', 'unknown error')}",
                    'relevant_segments': []
                },
                'prov': primitive_result.get('prov', []),
                'status': primitive_result.get('status', 'fail')
            }
        
        extracted_text = primitive_result.get('payload', {}).get('text', '')
        prov = primitive_result.get('prov', [])
        
        if not extracted_text:
            return missing(
                payload={
                    'evidence_found': False,
                    'summary': "No relevant text extracted for the specified aspect.",
                    'relevant_segments': prov
                },
                prov=prov
            )
        
        # Step 2: Assess justification using controlled LLM
        prompt = f"Given the extracted text, determine if there is a clear justification, explanation, or definition for the aspect '{aspect}'. Output a JSON object with 'evidence_found' (boolean) and 'reason' (string summarizing the evidence or lack thereof). Only use information from the provided text.\n\nExtracted text: {extracted_text}"
        
        try:
            llm_response = controlled_llm(prompt)
        except Exception as e:
            return fail(error=f"Controlled LLM call failed: {str(e)}", prov=prov)
        
        # Parse LLM response
        try:
            assessment = json.loads(llm_response)
            if not isinstance(assessment, dict) or 'evidence_found' not in assessment or 'reason' not in assessment:
                return fail(error="Invalid LLM response format", prov=prov)
            
            evidence_found = bool(assessment['evidence_found'])
            summary = str(assessment['reason'])
            
            if evidence_found:
                return ok(
                    payload={
                        'evidence_found': True,
                        'summary': summary,
                        'relevant_segments': prov
                    },
                    prov=prov
                )
            else:
                return missing(
                    payload={
                        'evidence_found': False,
                        'summary': summary,
                        'relevant_segments': prov
                    },
                    prov=prov
                )
                
        except json.JSONDecodeError:
            return fail(error="LLM response is not valid JSON", prov=prov)
        
    except Exception as e:
        return fail(error=f"Unexpected error: {str(e)}")
