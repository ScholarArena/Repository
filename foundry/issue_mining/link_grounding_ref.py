import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, write_jsonl


TEXT_EXTS = {".txt", ".md", ".tex"}
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
NO_REF_PATTERNS = [
    (re.compile(r"^\s*n/?a\b", re.IGNORECASE), "na"),
    (re.compile(r"\bno\s+ref(?:erence)?\b", re.IGNORECASE), "no_ref"),
    (re.compile(r"\bauthor(?:'s)?\s+response\b", re.IGNORECASE), "author_response"),
    (re.compile(r"\bauthor\s+reply\b", re.IGNORECASE), "author_response"),
    (re.compile(r"\brebuttal\b", re.IGNORECASE), "rebuttal"),
    (re.compile(r"\brevision note\b", re.IGNORECASE), "revision_note"),
    (re.compile(r"\breviewer[_\s-]*comments?\b", re.IGNORECASE), "reviewer_comments"),
    (re.compile(r"\bcomment[_\s-]*id\b", re.IGNORECASE), "reviewer_comments"),
    (re.compile(r"\b(entire|whole|full)\s+paper\b", re.IGNORECASE), "global_scope"),
    (re.compile(r"\bpaper length\b", re.IGNORECASE), "global_scope"),
    (re.compile(r"\bappendix\s+length\b", re.IGNORECASE), "global_scope"),
    (re.compile(r"\bthroughout\b", re.IGNORECASE), "global_scope"),
    (re.compile(r"^\s*figures?\s*$", re.IGNORECASE), "generic_figure"),
    (re.compile(r"^\s*tables?\s*$", re.IGNORECASE), "generic_table"),
    (re.compile(r"^\s*appendix\s*$", re.IGNORECASE), "generic_appendix"),
    (re.compile(r"^\s*appendices\s*$", re.IGNORECASE), "generic_appendix"),
]
PAGE_ONLY_RE = re.compile(r"\bpages?\b", re.IGNORECASE)
STRUCTURED_REF_RE = re.compile(r"\b(sec|section|fig|figure|table|appendix|abstract|related work)\b", re.IGNORECASE)
NO_REF_TOOL_CATEGORIES = {
    "Cognitive_Synthesis",
    "Textual_Analysis",
    "Conceptual_Clarification",
    "Terminology_Clarification",
    "Textual_Correction",
    "Literature_Expansion",
}
NO_REF_INTENTS = {
    "Establish_Baseline_Understanding",
    "Improve_Presentation",
    "Demonstrate_Responsiveness",
    "Concede_Minor_Point_To_Win_Major",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve grounding_ref to paper spans.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL")
    parser.add_argument("--papers", dest="papers_dir", required=True, help="Directory with paper text files")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL")
    parser.add_argument("--log-every", type=int, default=5000, help="Progress log interval (issues)")
    parser.add_argument("--quiet", action="store_true", help="Disable progress and summary logs")
    parser.add_argument("--sample-failures", type=int, default=20, help="Number of unresolved samples to print")
    parser.add_argument("--no-snippet", action="store_true", help="Disable snippet extraction to reduce output size")
    return parser.parse_args()


def split_refs(grounding_ref):
    if not grounding_ref:
        return []
    return [part.strip() for part in grounding_ref.split(",") if part.strip()]


def find_paper_bundle(papers_dir, forum_id):
    papers_dir = Path(papers_dir)
    auto_dir = papers_dir / forum_id / "auto"
    preferred = auto_dir / f"{forum_id}.md"
    if preferred.exists():
        images_dir = auto_dir / "images"
        return preferred, images_dir if images_dir.exists() else None
    if auto_dir.exists():
        for path in sorted(auto_dir.glob("*.md")):
            images_dir = path.parent / "images"
            return path, images_dir if images_dir.exists() else None
    subdir = papers_dir / forum_id
    if subdir.exists() and subdir.is_dir():
        for path in sorted(subdir.glob("*.md")):
            images_dir = path.parent / "images"
            return path, images_dir if images_dir.exists() else None
    direct = [papers_dir / f"{forum_id}{ext}" for ext in TEXT_EXTS]
    for path in direct:
        if path.exists():
            images_dir = path.parent / "images"
            return path, images_dir if images_dir.exists() else None
    return None, None


def load_text(path):
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def extract_image_refs(text, doc_path, images_dir):
    refs = []
    if not text:
        return refs
    doc_parent = Path(doc_path).parent if doc_path else None
    for match in IMAGE_RE.finditer(text):
        rel_path = match.group(1)
        resolved = None
        if doc_parent:
            resolved = str((doc_parent / rel_path).resolve())
        elif images_dir:
            resolved = str((images_dir / Path(rel_path).name).resolve())
        refs.append(
            {
                "path": rel_path,
                "resolved_path": resolved,
                "index": match.start(),
            }
        )
    return refs


def make_snippet(text, start, end, window=80, include=True):
    if not include:
        return None
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ").strip()
    return snippet[:200]


def find_nearest_image(image_refs, start, window=600):
    for ref in image_refs:
        if start <= ref["index"] <= start + window:
            return ref
    return None


def extract_section_number(ref):
    match = re.search(r"(?:sec(?:tion)?\.?\s*)(\d+(?:\.\d+)*)", ref, re.IGNORECASE)
    return match.group(1) if match else None


def extract_appendix_letter(ref):
    match = re.search(r"appendix\s*([a-z])", ref, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_number(ref, label):
    match = re.search(rf"{label}\.?\s*(\d+)", ref, re.IGNORECASE)
    return match.group(1) if match else None


def regex_matches(text, regex, ref, match_type, image_refs, include_snippet):
    matches = []
    for match in regex.finditer(text):
        start, end = match.start(), match.end()
        entry = {
            "ref": ref,
            "match_type": match_type,
            "start": start,
            "end": end,
            "snippet": make_snippet(text, start, end, include=include_snippet),
        }
        if match_type == "figure":
            image = find_nearest_image(image_refs, start)
            if image:
                entry["image_ref"] = image
        matches.append(entry)
    return matches


def match_ref(text, ref, image_refs, include_snippet):
    if not text or not ref:
        return []
    lower = ref.lower()
    if "sec" in lower or "section" in lower:
        sec_num = extract_section_number(ref)
        if sec_num:
            regex = re.compile(rf"^#+\s*{re.escape(sec_num)}(\s|\.|$)", re.IGNORECASE | re.MULTILINE)
            matches = regex_matches(text, regex, ref, "section", image_refs, include_snippet)
            if matches:
                return matches
    if "abstract" in lower:
        regex = re.compile(r"^#+\s*abstract\b", re.IGNORECASE | re.MULTILINE)
        matches = regex_matches(text, regex, ref, "section", image_refs, include_snippet)
        if matches:
            return matches
    if "related work" in lower:
        regex = re.compile(r"^#+\s*related work(s)?\b", re.IGNORECASE | re.MULTILINE)
        matches = regex_matches(text, regex, ref, "section", image_refs, include_snippet)
        if matches:
            return matches
    if "appendix" in lower:
        appendix = extract_appendix_letter(ref)
        if appendix:
            regex = re.compile(rf"^#+\s*appendix\s*{appendix}\b", re.IGNORECASE | re.MULTILINE)
            matches = regex_matches(text, regex, ref, "appendix", image_refs, include_snippet)
            if matches:
                return matches
    if "fig" in lower or "figure" in lower:
        fig_num = extract_number(ref, "fig(?:ure)?")
        if fig_num:
            regex = re.compile(rf"(fig(?:ure)?\.?\s*{fig_num})", re.IGNORECASE)
            matches = regex_matches(text, regex, ref, "figure", image_refs, include_snippet)
            if matches:
                return matches
    if "table" in lower:
        table_num = extract_number(ref, "table")
        if table_num:
            regex = re.compile(rf"(table\.?\s*{table_num})", re.IGNORECASE)
            matches = regex_matches(text, regex, ref, "table", image_refs, include_snippet)
            if matches:
                return matches
    lower_text = text.lower()
    idx = lower_text.find(lower)
    if idx != -1:
        return [
            {
                "ref": ref,
                "match_type": "substring",
                "start": idx,
                "end": idx + len(ref),
                "snippet": make_snippet(text, idx, idx + len(ref), include=include_snippet),
            }
        ]
    return []


def match_refs(text, refs, image_refs, include_snippet):
    matches = []
    if not text:
        return matches
    for ref in refs:
        matches.extend(match_ref(text, ref, image_refs, include_snippet))
    return matches


def classify_ref(ref):
    if not ref:
        return "unknown"
    lower = ref.lower()
    if "fig" in lower or "figure" in lower:
        return "figure"
    if "table" in lower:
        return "table"
    if "appendix" in lower:
        return "appendix"
    if "abstract" in lower:
        return "abstract"
    if "related work" in lower:
        return "related_work"
    if "sec" in lower or "section" in lower:
        return "section"
    return "other"


def is_structured_specific(ref):
    if not ref:
        return False
    lower = ref.lower()
    if "abstract" in lower or "related work" in lower:
        return True
    if extract_section_number(ref):
        return True
    if extract_appendix_letter(ref):
        return True
    if extract_number(ref, "fig(?:ure)?"):
        return True
    if extract_number(ref, "table"):
        return True
    return False


def match_no_ref_reason(ref):
    if not ref:
        return "no_ref"
    for pattern, reason in NO_REF_PATTERNS:
        if pattern.search(ref):
            return reason
    if PAGE_ONLY_RE.search(ref) and not STRUCTURED_REF_RE.search(ref):
        return "page_only"
    return None


def not_required_reason(grounding_ref, refs, strategic_intent, tool_calls):
    if refs:
        reasons = []
        for ref in refs:
            if is_structured_specific(ref):
                return None
            reason = match_no_ref_reason(ref)
            if not reason:
                return None
            reasons.append(reason)
        if reasons:
            return reasons[0]
    tool_categories = {call.get("tool_category") for call in tool_calls or [] if call.get("tool_category")}
    if strategic_intent in NO_REF_INTENTS or tool_categories.intersection(NO_REF_TOOL_CATEGORIES):
        return "no_ref_intent_or_tool"
    if grounding_ref:
        reason = match_no_ref_reason(grounding_ref)
        if reason:
            return reason
    return None


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    cache = {}
    total_issues = len(records)
    resolved_issues = 0
    not_required_issues = 0
    required_unresolved = 0
    missing_docs = 0
    ref_type_counts = Counter()
    ref_type_matched = Counter()
    figure_with_image = 0
    samples = []
    out = []
    for idx_issue, rec in enumerate(records, start=1):
        forum_id = rec.get("forum_id") or "UNKNOWN"
        grounding_ref = rec.get("grounding_ref")
        refs = split_refs(grounding_ref)
        if forum_id not in cache:
            paper_path, images_dir = find_paper_bundle(args.papers_dir, forum_id)
            text = load_text(paper_path) if paper_path else None
            image_refs = extract_image_refs(text, paper_path, images_dir)
            cache[forum_id] = {
                "path": paper_path,
                "images_dir": images_dir,
                "text": text,
                "image_refs": image_refs,
            }
        doc = cache[forum_id]
        text = doc["text"]
        tool_calls = rec.get("latent_tool_calls") or []
        intent = rec.get("strategic_intent")
        not_required = not_required_reason(grounding_ref, refs, intent, tool_calls)
        matches = match_refs(text, refs, doc["image_refs"], include_snippet=not args.no_snippet)
        if not doc["path"]:
            missing_docs += 1
        status = "resolved" if matches else "unresolved"
        if not_required:
            status = "not_required"
            not_required_issues += 1
        elif matches:
            resolved_issues += 1
        else:
            required_unresolved += 1
        matched_refs = {match["ref"] for match in matches}
        image_matched_refs = {match["ref"] for match in matches if match.get("image_ref")}
        for ref in refs:
            ref_type = classify_ref(ref)
            ref_type_counts[ref_type] += 1
            if ref in matched_refs:
                ref_type_matched[ref_type] += 1
            if ref_type == "figure" and ref in image_matched_refs:
                figure_with_image += 1
        if status == "unresolved" and len(samples) < args.sample_failures:
            samples.append(
                {
                    "issue_id": rec.get("issue_id"),
                    "forum_id": forum_id,
                    "grounding_ref": grounding_ref,
                    "doc_path": str(doc["path"]) if doc["path"] else None,
                }
            )
        rec["paper_span"] = {
            "raw": grounding_ref,
            "refs": refs,
            "doc_path": str(doc["path"]) if doc["path"] else None,
            "doc_kind": "markdown" if doc["path"] and doc["path"].suffix == ".md" else "text",
            "images_dir": str(doc["images_dir"]) if doc["images_dir"] else None,
            "matches": matches,
            "resolved": bool(matches),
            "status": status,
            "reason": not_required,
        }
        out.append(rec)
        if not args.quiet and args.log_every > 0 and idx_issue % args.log_every == 0:
            print(
                f"[link] {idx_issue}/{total_issues} issues | resolved={resolved_issues}",
                file=sys.stderr,
            )
    write_jsonl(args.output_path, out)
    if not args.quiet:
        effective_total = max(total_issues - not_required_issues, 0)
        resolved_rate = (resolved_issues / effective_total) if effective_total else 0
        print(
            f"[summary] issues={total_issues} resolved={resolved_issues} "
            f"resolved_rate={resolved_rate:.3f} not_required={not_required_issues} "
            f"required_unresolved={required_unresolved} missing_docs={missing_docs}",
            file=sys.stderr,
        )
        for ref_type in sorted(ref_type_counts.keys()):
            total = ref_type_counts[ref_type]
            matched = ref_type_matched.get(ref_type, 0)
            rate = matched / total if total else 0
            extra = ""
            if ref_type == "figure":
                extra = f" image_matched={figure_with_image}"
            print(
                f"[summary] refs.{ref_type}: total={total} matched={matched} rate={rate:.3f}{extra}",
                file=sys.stderr,
            )
        if samples:
            print("[summary] unresolved_samples:", file=sys.stderr)
            for sample in samples:
                print(f"  - {sample}", file=sys.stderr)


if __name__ == "__main__":
    main()
