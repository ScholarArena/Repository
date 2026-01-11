import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

from foundry.utils import now_iso, read_json_or_jsonl


DEFAULT_INTENTS = [
    {
        "name": "Challenge_Claim",
        "definition": "Dispute validity/novelty/correctness of a claim or result.",
    },
    {
        "name": "Request_Evidence",
        "definition": "Ask for additional evidence, experiments, or justification.",
    },
    {
        "name": "Request_Clarification",
        "definition": "Ask to clarify ambiguous statements or definitions.",
    },
    {
        "name": "Defend_Claim",
        "definition": "Justify or defend an existing claim or method.",
    },
    {
        "name": "Concede_and_Patch",
        "definition": "Accept a critique and propose a fix or revision.",
    },
    {
        "name": "Suggest_Experiment",
        "definition": "Recommend additional experiments or analyses.",
    },
    {
        "name": "Improve_Presentation",
        "definition": "Focus on writing, clarity, formatting, or presentation issues.",
    },
    {
        "name": "Summarize",
        "definition": "Summarize the work or synthesize key contributions.",
    },
    {
        "name": "Establish_Context",
        "definition": "Set context, background, or baseline understanding.",
    },
    {
        "name": "Decide",
        "definition": "Make or recommend an accept/reject or meta-level decision.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Label raw strategic intents with a canonical ontology.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output intent map JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--base-url", default="https://www.dmxapi.cn/v1/", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--per-intent", type=int, default=2, help="Examples per raw intent")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for failed requests")
    parser.add_argument("--max-intents", type=int, default=0, help="Optional cap on number of raw intents")
    return parser.parse_args()


def call_chat(messages, model, api_key, base_url, max_retries):
    if not api_key:
        raise ValueError("Missing API key. Provide --api-key or set DMX_API_KEY/OPENAI_API_KEY.")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0.2}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    req = url_request.Request(endpoint, data=payload, headers=headers, method="POST")

    last_exc = None
    for _ in range(max_retries):
        try:
            with url_request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(body)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except (url_error.HTTPError, url_error.URLError, json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            time.sleep(1.5)
    raise last_exc


def build_prompt(canonical, raw_items):
    return [
        {
            "role": "system",
            "content": (
                "You map noisy intent labels from peer-review discourse to a canonical intent ontology. "
                "Choose exactly one canonical intent for each raw_intent. "
                "Return JSON with key 'items' as a list of {raw_intent, mapped_intent, rationale}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"canonical_intents": canonical, "raw_intents": raw_items},
                ensure_ascii=True,
            ),
        },
    ]


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("DMX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    records = read_json_or_jsonl(args.input_path)

    intents = defaultdict(lambda: {"count": 0, "examples": []})
    for rec in records:
        intent = rec.get("strategic_intent")
        if not intent:
            continue
        entry = intents[intent]
        entry["count"] += 1
        if len(entry["examples"]) < args.per_intent:
            action = rec.get("action")
            if action:
                entry["examples"].append(action)

    raw_items = [
        {
            "name": name,
            "count": data["count"],
            "examples": data["examples"],
        }
        for name, data in intents.items()
    ]
    raw_items.sort(key=lambda x: (-x["count"], x["name"]))
    if args.max_intents and len(raw_items) > args.max_intents:
        raw_items = raw_items[: args.max_intents]

    messages = build_prompt(DEFAULT_INTENTS, raw_items)
    content = call_chat(messages, args.model, api_key, args.base_url, args.max_retries)

    output = {
        "generated_at": now_iso(),
        "canonical_intents": DEFAULT_INTENTS,
        "items": [],
        "raw_response": content,
    }
    try:
        parsed = json.loads(content)
        for item in parsed.get("items", []):
            raw_intent = item.get("raw_intent") or item.get("name")
            mapped_intent = item.get("mapped_intent") or item.get("canonical_intent")
            if not raw_intent or not mapped_intent:
                continue
            output["items"].append(
                {
                    "raw_intent": raw_intent,
                    "mapped_intent": mapped_intent,
                    "rationale": item.get("rationale"),
                }
            )
    except Exception:
        output["parse_error"] = True

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(
        json.dumps(output, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
