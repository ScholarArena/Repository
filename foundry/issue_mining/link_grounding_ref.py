import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foundry.utils import read_json_or_jsonl, write_jsonl


TEXT_EXTS = {".txt", ".md", ".tex"}


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve grounding_ref to paper spans.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL")
    parser.add_argument("--papers", dest="papers_dir", required=True, help="Directory with paper text files")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL")
    return parser.parse_args()


def split_refs(grounding_ref):
    if not grounding_ref:
        return []
    return [part.strip() for part in grounding_ref.split(",") if part.strip()]


def find_paper_file(papers_dir, forum_id):
    papers_dir = Path(papers_dir)
    direct = [papers_dir / f"{forum_id}{ext}" for ext in TEXT_EXTS]
    for path in direct:
        if path.exists():
            return path
    subdir = papers_dir / forum_id
    if subdir.exists() and subdir.is_dir():
        for ext in TEXT_EXTS:
            for path in subdir.glob(f"*{ext}"):
                return path
    return None


def load_text(path):
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def match_refs(text, refs):
    matches = []
    if not text:
        return matches
    for ref in refs:
        idx = text.find(ref)
        if idx != -1:
            matches.append({"ref": ref, "index": idx})
    return matches


def main():
    args = parse_args()
    records = read_json_or_jsonl(args.input_path)
    out = []
    for rec in records:
        forum_id = rec.get("forum_id") or "UNKNOWN"
        grounding_ref = rec.get("grounding_ref")
        refs = split_refs(grounding_ref)
        paper_path = find_paper_file(args.papers_dir, forum_id)
        text = load_text(paper_path) if paper_path else None
        matches = match_refs(text, refs)
        rec["paper_span"] = {
            "raw": grounding_ref,
            "refs": refs,
            "doc_path": str(paper_path) if paper_path else None,
            "matches": matches,
            "resolved": bool(matches),
        }
        out.append(rec)
    write_jsonl(args.output_path, out)


if __name__ == "__main__":
    main()
