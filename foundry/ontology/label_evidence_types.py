import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import normalize_tool_calls, now_iso, read_json_or_jsonl


ALLOWED_EVIDENCE_TYPES = [
    "Citation",
    "Statistical",
    "Figure",
    "Table",
    "Document",
    "Text",
    "Symbolic",
    "Logical",
    "Implementation",
    "Unknown",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Map tool categories to evidence types with a teacher LLM.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input issues JSONL")
    parser.add_argument("--out", dest="output_path", required=True, help="Output evidence map JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--base-url", default="https://www.dmxapi.cn/v1/", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum occurrences to include a category")
    parser.add_argument("--max-categories", type=int, default=0, help="Optional cap on categories")
    parser.add_argument("--ops-per-category", type=int, default=3, help="Top operations per category")
    parser.add_argument("--pair-min-count", type=int, default=3, help="Minimum occurrences to include a category::operation pair")
    parser.add_argument(
        "--category-fallback",
        action="store_true",
        help="Ask LLM for category-only mapping when no operations are selected",
    )
    parser.add_argument("--chunk-size", type=int, default=40, help="Categories per LLM request")
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


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def build_prompt(items):
    return [
        {
            "role": "system",
            "content": (
                "You map tool categories and operations from peer-review discourse to evidence types. "
                "Return valid JSON only with key: items. "
                "Each item has {key, tool_category, operation, evidence_type, rationale}. "
                "key must be echoed exactly as provided. "
                "operation may be '*' to indicate a category-level fallback. "
                "evidence_type must be one of: " + ", ".join(ALLOWED_EVIDENCE_TYPES) + ". "
                "Use best-fit evidence types: "
                "Statistical for quantitative/experiment/metric analysis; "
                "Citation for prior-work checks; "
                "Document/Text for span extraction or narrative validation; "
                "Figure/Table for visual/tabular evidence; "
                "Symbolic/Logical for math/proof/logic checks; "
                "Implementation for code/algorithm/implementation evidence. "
                "Avoid Unknown unless truly unclear."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"tool_categories": items}, ensure_ascii=True),
        },
    ]


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("DMX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    records = read_json_or_jsonl(args.input_path)

    groups = defaultdict(lambda: {"count": 0, "operations": Counter(), "targets": Counter()})
    pairs = defaultdict(lambda: {"count": 0, "targets": Counter()})
    for rec in records:
        for call in normalize_tool_calls(rec.get("latent_tool_calls")):
            if not isinstance(call, dict):
                continue
            tool_category = call.get("tool_category") or "unknown"
            groups[tool_category]["count"] += 1
            operation = call.get("operation")
            if operation:
                groups[tool_category]["operations"][operation] += 1
                pair_key = (tool_category, operation)
                pairs[pair_key]["count"] += 1
            target_type = call.get("target_type")
            if target_type:
                groups[tool_category]["targets"][target_type] += 1
                if operation:
                    pairs[pair_key]["targets"][target_type] += 1

    items = []
    for tool_category, info in groups.items():
        if info["count"] < args.min_count:
            continue
        ops = info["operations"].most_common()
        ops_added = 0
        for operation, count in ops:
            if count < args.pair_min_count:
                continue
            key = f"{tool_category}::{operation}"
            pair_info = pairs.get((tool_category, operation), {"targets": Counter()})
            items.append(
                {
                    "key": key,
                    "tool_category": tool_category,
                    "operation": operation,
                    "count": count,
                    "sample_targets": [tt for tt, _ in pair_info["targets"].most_common(args.ops_per_category)],
                }
            )
            ops_added += 1
            if ops_added >= args.ops_per_category:
                break
        if ops_added == 0 and args.category_fallback:
            items.append(
                {
                    "key": tool_category,
                    "tool_category": tool_category,
                    "operation": "*",
                    "count": info["count"],
                    "sample_targets": [tt for tt, _ in info["targets"].most_common(args.ops_per_category)],
                }
            )

    items.sort(key=lambda x: (-x["count"], x["tool_category"], x.get("operation") or ""))
    if args.max_categories and len(items) > args.max_categories:
        items = items[: args.max_categories]

    chunks = [items[i : i + args.chunk_size] for i in range(0, len(items), args.chunk_size)]

    output_items = []
    mapping = {}
    raw_responses = []

    for idx, chunk in enumerate(chunks, start=1):
        messages = build_prompt(chunk)
        content = call_chat(messages, args.model, api_key, args.base_url, args.max_retries)
        raw_responses.append({"chunk": idx, "response": content})

        parsed = None
        try:
            payload = extract_json(content)
            parsed = json.loads(payload)
        except Exception:
            raw_responses[-1]["parse_error"] = True
            continue

        for item in parsed.get("items", []):
            key = item.get("key")
            tool_category = item.get("tool_category")
            operation = item.get("operation")
            evidence_type = item.get("evidence_type") or item.get("mapped_evidence")
            if not evidence_type:
                continue
            if not key:
                if tool_category and operation and operation != "*":
                    key = f"{tool_category}::{operation}"
                elif tool_category:
                    key = tool_category
            if not key:
                continue
            mapping[key] = evidence_type
            output_items.append(
                {
                    "key": key,
                    "tool_category": tool_category,
                    "operation": operation,
                    "evidence_type": evidence_type,
                    "rationale": item.get("rationale"),
                }
            )

        time.sleep(args.sleep)

    for item in items:
        key = item.get("key")
        if key and key not in mapping:
            mapping[key] = "Unknown"
            output_items.append(
                {
                    "key": key,
                    "tool_category": item.get("tool_category"),
                    "operation": item.get("operation"),
                    "evidence_type": "Unknown",
                    "rationale": "fallback",
                }
            )

    output = {
        "generated_at": now_iso(),
        "evidence_types": ALLOWED_EVIDENCE_TYPES,
        "items": output_items,
        "mapping": mapping,
        "raw_responses": raw_responses,
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(
        json.dumps(output, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
