import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import normalize_tool_calls, read_json_or_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Label issue clusters with a teacher LLM.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output labels JSONL")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument(
        "--base-url",
        default="https://www.dmxapi.cn/v1/",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--per-cluster", type=int, default=20, help="Samples per cluster")
    parser.add_argument("--max-clusters", type=int, default=400, help="Max clusters to label")
    parser.add_argument("--min-count", type=int, default=10, help="Min cluster size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for failed requests")
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


def format_sample(rec):
    calls = []
    for call in normalize_tool_calls(rec.get("latent_tool_calls")):
        if isinstance(call, dict):
            calls.append(
                {
                    "tool_category": call.get("tool_category"),
                    "operation": call.get("operation"),
                    "target_type": call.get("target_type"),
                }
            )
        else:
            calls.append(str(call))
    return {
        "role": rec.get("role"),
        "intent": rec.get("strategic_intent"),
        "grounding_ref": rec.get("grounding_ref"),
        "action": rec.get("action"),
        "tool_calls": calls,
    }


def build_prompt(cluster_id, samples):
    return [
        {
            "role": "system",
            "content": (
                "You label issue clusters for scientific peer review. "
                "Infer a domain-agnostic issue type from the samples (no paper-specific names). "
                "Return valid JSON only with keys: issue_type, definition, evidence_types, intent_patterns. "
                "Constraints: issue_type is 2-5 words using underscores, definition is one sentence, "
                "evidence_types is a list (1-4 items) chosen from "
                "[Citation, Statistical, Figure, Table, Document, Text, Symbolic, Logical, Implementation, Unknown], "
                "intent_patterns is a list (2-4 items) summarizing common strategic intents or interaction patterns "
                "seen in the samples (use raw intent labels if present). "
                "If the cluster is mixed or unclear, set issue_type to Mixed_or_Unclear and evidence_types to [Unknown]."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "cluster_id": cluster_id,
                    "samples": samples,
                },
                ensure_ascii=True,
            ),
        },
    ]


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("DMX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    records = read_json_or_jsonl(args.input_path)
    random.seed(args.seed)

    clusters = defaultdict(list)
    for rec in records:
        cluster_id = rec.get("issue_cluster_id") or rec.get("cluster_id") or "unclustered"
        clusters[cluster_id].append(rec)

    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    selected = [c for c in sorted_clusters if len(c[1]) >= args.min_count][: args.max_clusters]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for cluster_id, items in selected:
            sample = items if len(items) <= args.per_cluster else random.sample(items, args.per_cluster)
            samples = [format_sample(rec) for rec in sample]
            messages = build_prompt(cluster_id, samples)
            content = call_chat(messages, args.model, api_key, args.base_url, args.max_retries)
            label = {
                "cluster_id": cluster_id,
                "count": len(items),
                "raw_response": content,
            }
            try:
                parsed = json.loads(content)
                label.update(parsed)
            except Exception:
                label["parse_error"] = True
            f.write(json.dumps(label, ensure_ascii=True) + "\n")
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
