import runtime
import json

def execute(context, params, primitives=None, controlled_llm=None):
    try:

        if not isinstance(context, dict) or 'segments' not in context:
            return runtime.fail(error="Invalid or missing context.segments")
        if not isinstance(params, dict) or 'aspect' not in params:
            return runtime.fail(error="Invalid or missing params.aspect")


        if primitives is None or not isinstance(primitives, dict) or 'Extract_Aspect_Segments' not in primitives:
            return runtime.fail(error="Primitive Extract_Aspect_Segments not available")


        primitive_func = primitives['Extract_Aspect_Segments']
        if not callable(primitive_func):
            return runtime.fail(error="Primitive Extract_Aspect_Segments is not callable")

        primitive_result = primitive_func(context['segments'], params['aspect'])
        if not isinstance(primitive_result, dict):
            return runtime.fail(error="Primitive did not return a dict")

        if primitive_result.get('status') != 'ok':
            error_msg = primitive_result.get('error', 'Primitive failed')
            return runtime.fail(error=error_msg)

        extracted_text = primitive_result.get('payload', {}).get('text', '')
        prov = primitive_result.get('prov', [])
        if not isinstance(prov, list):
            prov = []


        if controlled_llm is None or not callable(controlled_llm):
            return runtime.fail(error="Controlled LLM not provided or not callable")


        aspect = params['aspect']
        prompt = f"Given the extracted text, determine if there is a clear justification, explanation, or definition for the aspect '{aspect}'. Output a JSON object with 'evidence_found' (boolean) and 'reason' (string summarizing the evidence or lack thereof). Only use information from the provided text.\n\nExtracted text: {extracted_text}"


        llm_response = controlled_llm(prompt)
        if isinstance(llm_response, str):
            try:
                llm_data = json.loads(llm_response)
            except json.JSONDecodeError:
                return runtime.fail(error="Controlled LLM returned invalid JSON")
        elif isinstance(llm_response, dict):
            llm_data = llm_response
        else:
            return runtime.fail(error="Controlled LLM returned non-dict, non-string response")

        evidence_found = llm_data.get('evidence_found', False)
        reason = llm_data.get('reason', '')

        payload = {
            'evidence_found': evidence_found,
            'summary': reason,
            'relevant_segments': prov
        }

        if evidence_found:
            return runtime.ok(payload, prov)
        else:
            return runtime.missing(payload, prov)

    except Exception as e:
        return runtime.fail(error=str(e))
