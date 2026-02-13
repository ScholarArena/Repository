import sys

def execute(context, params, primitives=None, controlled_llm=None):
    try:

        from runtime import ok, missing, fail


        if not isinstance(context, dict):
            return fail(error="Context must be a dictionary")
        if 'segments' not in context:
            return fail(error="Context missing 'segments' key")
        segments = context['segments']
        if not isinstance(segments, list):
            return fail(error="'segments' must be a list")
        for seg in segments:
            if not isinstance(seg, dict) or 'id' not in seg or 'text' not in seg:
                return fail(error="Each segment must have 'id' (int) and 'text' (str)")
            if not isinstance(seg['id'], int) or not isinstance(seg['text'], str):
                return fail(error="Segment 'id' must be int and 'text' must be str")


        if not isinstance(params, dict):
            return fail(error="Params must be a dictionary")
        if 'target_type' not in params:
            return fail(error="Params missing 'target_type' key")
        target_type = params['target_type']
        if not isinstance(target_type, str):
            return fail(error="'target_type' must be a string")


        if primitives is None or 'FormatPrompt' not in primitives:
            return fail(error="Required primitive 'FormatPrompt' not available")
        format_prompt_func = primitives['FormatPrompt']
        if not callable(format_prompt_func):
            return fail(error="'FormatPrompt' primitive is not callable")
        prompt_text = format_prompt_func(segments=segments, target_type=target_type)
        if not isinstance(prompt_text, str):
            return fail(error="FormatPrompt did not return a string prompt")


        if controlled_llm is None:
            return fail(error="Controlled LLM not provided")
        if not callable(controlled_llm):
            return fail(error="Controlled LLM is not callable")
        llm_response = controlled_llm(prompt=prompt_text)
        if llm_response is None:
            return fail(error="LLM call returned no response")

        if not isinstance(llm_response, str):
            llm_response = str(llm_response)


        if 'ParseObservation' not in primitives:
            return fail(error="Required primitive 'ParseObservation' not available")
        parse_observation_func = primitives['ParseObservation']
        if not callable(parse_observation_func):
            return fail(error="'ParseObservation' primitive is not callable")
        segment_ids = [seg['id'] for seg in segments]
        observation_output = parse_observation_func(
            response=llm_response,
            target_type=target_type,
            segment_ids=segment_ids
        )
        if not isinstance(observation_output, dict):
            return fail(error="ParseObservation did not return a dictionary")


        observation = observation_output.get('observation')
        prov = observation_output.get('prov')
        status = observation_output.get('status')


        if prov is not None:
            if not isinstance(prov, list):
                return fail(error="'prov' must be a list")
            for seg_id in prov:
                if not isinstance(seg_id, int):
                    return fail(error="'prov' must contain integers")
        else:
            prov = []


        if observation is None:
            observation = ''
        if not isinstance(observation, str):
            observation = str(observation)


        payload = {
            'observation': observation,
            'target': target_type
        }


        if status == 'ok':
            return ok(type='evidence_normalized', payload=payload, prov=prov)
        elif status == 'missing':

            return missing(type='evidence_normalized', payload=payload, prov=prov)
        else:

            return fail(error=f"ParseObservation returned unexpected status: {status}")

    except Exception as e:

        return fail(error=f"Unexpected error: {e}")
