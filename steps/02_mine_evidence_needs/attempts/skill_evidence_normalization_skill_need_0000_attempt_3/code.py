import json

def execute(context, params, primitives=None, controlled_llm=None):
    from runtime import ok, missing, fail
    
    try:
        # Validate inputs
        if not context or not isinstance(context, dict):
            return missing()
        if 'segments' not in context:
            return missing()
        if not isinstance(context['segments'], list):
            return missing()
        
        if not params or not isinstance(params, dict):
            return missing()
        if 'target_type' not in params:
            return missing()
        
        # Check primitives
        if primitives is None:
            return fail(error="Primitives not provided")
        
        if 'FormatPrompt' not in primitives:
            return fail(error="Missing FormatPrompt primitive")
        if 'ParseObservation' not in primitives:
            return fail(error="Missing ParseObservation primitive")
        
        # Step 1: Generate prompt
        try:
            prompt_text = primitives['FormatPrompt'](
                segments=context['segments'],
                target_type=params['target_type']
            )
        except Exception as e:
            return fail(error=f"FormatPrompt failed: {str(e)}")
        
        # Step 2: Call controlled LLM
        if controlled_llm is None:
            return fail(error="Controlled LLM not provided")
        
        try:
            llm_response = controlled_llm(prompt=prompt_text)
        except Exception as e:
            return fail(error=f"LLM call failed: {str(e)}")
        
        # Step 3: Parse response
        segment_ids = [seg['id'] for seg in context['segments']]
        
        try:
            observation_output = primitives['ParseObservation'](
                response=llm_response,
                target_type=params['target_type'],
                segment_ids=segment_ids
            )
        except Exception as e:
            return fail(error=f"ParseObservation failed: {str(e)}")
        
        # Check if evidence was found
        if not observation_output.get('observation', '').strip():
            return missing()
        
        # Return normalized evidence
        return ok(
            type="evidence_normalized",
            payload={
                "observation": observation_output['observation'],
                "target": params['target_type']
            },
            prov=observation_output.get('prov', [])
        )
        
    except Exception as e:
        return fail(error=f"Unexpected error: {str(e)}")