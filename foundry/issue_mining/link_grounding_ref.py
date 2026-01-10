import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, write_jsonl


TEXT_EXTS = {".txt", ".md", ".tex"}
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


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


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    cache = {}
    total_issues = len(records)
    resolved_issues = 0
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
        matches = match_refs(text, refs, doc["image_refs"], include_snippet=not args.no_snippet)
        if not doc["path"]:
            missing_docs += 1
        if matches:
            resolved_issues += 1
        matched_refs = {match["ref"] for match in matches}
        image_matched_refs = {match["ref"] for match in matches if match.get("image_ref")}
        for ref in refs:
            ref_type = classify_ref(ref)
            ref_type_counts[ref_type] += 1
            if ref in matched_refs:
                ref_type_matched[ref_type] += 1
            if ref_type == "figure" and ref in image_matched_refs:
                figure_with_image += 1
        if not matches and len(samples) < args.sample_failures:
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
        }
        out.append(rec)
        if not args.quiet and args.log_every > 0 and idx_issue % args.log_every == 0:
            print(
                f"[link] {idx_issue}/{total_issues} issues | resolved={resolved_issues}",
                file=sys.stderr,
            )
    write_jsonl(args.output_path, out)
    if not args.quiet:
        resolved_rate = (resolved_issues / total_issues) if total_issues else 0
        print(
            f"[summary] issues={total_issues} resolved={resolved_issues} "
            f"resolved_rate={resolved_rate:.3f} missing_docs={missing_docs}",
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
