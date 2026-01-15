import json

def execute(context, params, primitives=None, controlled_llm=None):
    try:
        # Check required primitives
        if primitives is None:
            return {
                "type": "evidence_normalized",
                "payload": {"observation": "", "target": params.get('target_type', '')},
                "prov": [],
                "status": "fail",
                "error": "primitives not provided"
            }
        
        required_primitives = ["FormatPrompt", "ParseObservation"]
        for primitive in required_primitives:
            if primitive not in primitives:
                return {
                    "type": "evidence_normalized",
                    "payload": {"observation": "", "target": params.get('target_type', '')},
                    "prov": [],
                    "status": "fail",
                    "error": f"missing primitive: {primitive}"
                }
        
        # Check controlled LLM
        if controlled_llm is None:
            return {
                "type": "evidence_normalized",
                "payload": {"observation": "", "target": params.get('target_type', '')},
                "prov": [],
                "status": "fail",
                "error": "controlled_llm not provided"
            }
        
        # Validate inputs
        if not isinstance(context, dict) or 'segments' not in context:
            return {
                "type": "evidence_normalized",
                "payload": {"observation": "", "target": params.get('target_type', '')},
                "prov": [],
                "status": "missing"
            }
        
        segments = context['segments']
        if not isinstance(segments, list):
            return {
                "type": "evidence_normalized",
                "payload": {"observation": "", "target": params.get('target_type', '')},
                "prov": [],
                "status": "missing"
            }
        
        target_type = params.get('target_type', '')
        if not target_type:
            return {
                "type": "evidence_normalized",
                "payload": {"observation": "", "target": ""},
                "prov": [],
                "status": "missing"
            }
        
        # Step 1: Generate prompt
        prompt_text = primitives["FormatPrompt"]({
            "segments": segments,
            "target_type": target_type
        })
        
        # Step 2: Call LLM
        llm_response = controlled_llm({"prompt": prompt_text})
        
        # Step 3: Parse response
        segment_ids = [seg['id'] for seg in segments if 'id' in seg]
        observation_output = primitives["ParseObservation"]({
            "response": llm_response,
            "target_type": target_type,
            "segment_ids": segment_ids
        })
        
        # Check if evidence was found
        if not observation_output.get('observation'):
            return {
                "type": "evidence_normalized",
                "payload": {"observation": "", "target": target_type},
                "prov": [],
                "status": "missing"
            }
        
        # Return successful observation
        return {
            "type": "evidence_normalized",
            "payload": {
                "observation": observation_output.get('observation', ''),
                "target": target_type
            },
            "prov": observation_output.get('prov', []),
            "status": "ok"
        }
        
    except Exception as e:
        return {
            "type": "evidence_normalized",
            "payload": {"observation": "", "target": params.get('target_type', '')},
            "prov": [],
            "status": "fail",
            "error": str(e)
        }
