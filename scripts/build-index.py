#!/usr/bin/env python3
"""Generate structured chapter indexes (index/zh.json, index/en.json) from SUMMARY.md
and per-chapter metadata. Output is committed to the repo so agents can read it
without rebuilding."""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE = "https://haozhe-xing.github.io/fun-rec-mdbook"

LINK_RE = re.compile(r"^\s*\* \[(.+?)\]\((.+?)\)\s*$")


def parse_summary(lang: str):
    """Return [(part_no, part_title, [(chapter_title, path), ...]), ...] in SUMMARY order."""
    summary = REPO / "src" / lang / "SUMMARY.md"
    parts, current_part, part_no = [], None, 0
    for line in summary.read_text(encoding="utf-8").splitlines():
        m = LINK_RE.match(line)
        if not m:
            continue
        title, path = m.group(1), m.group(2)
        if path == "README.md":  # book home, skip
            continue
        indent = len(line) - len(line.lstrip())
        if path.endswith("README.md"):  # part cover
            part_no += 1
            current_part = {"part": part_no, "title": title, "chapters": []}
            parts.append(current_part)
        elif indent > 0 and current_part is not None:
            current_part["chapters"].append((title, path))
        else:  # top-level standalone chapter (appendix etc.)
            parts.append({"part": part_no, "title": title, "chapters": [(title, path)]})
    return parts


def chapter_number(part_no: int, idx: int) -> str:
    return f"{part_no}.{idx}"


def md_path_is_cover(md: Path) -> bool:
    return md.name == "README.md"


def extract_meta(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    meta = {"difficulty": None, "read_minutes": None}
    m = re.search(r"🎯\s*(Beginner|Intermediate|Advanced)", text)
    if m:
        meta["difficulty"] = m.group(1)
    m = re.search(r"⏱️\s*~?(\d+)\s*min", text)
    if m:
        meta["read_minutes"] = int(m.group(1))
    # title: first markdown h1
    m = re.search(r"^# (.+)$", text, re.M)
    meta["title"] = m.group(1).strip() if m else md_path.stem
    # objectives: bullet list after "读完本章，你将能够" / "读完本部分" / "After reading"
    meta["objectives"] = []
    om = re.search(
        r"(?:读完本章，你将能够|读完本 [Pp]art[^\n]*|After reading (?:this )?(?:chapter|part)[^\n]*)[：:]\s*\n+((?:\s*[-•]\s+.+\n?)+)",
        text,
    )
    if om:
        meta["objectives"] = [
            re.sub(r"\*\*|\$[^$]*\$", "", b.lstrip("-• \t")).strip()
            for b in om.group(1).strip().splitlines()
            if b.strip()
        ][:8]
    # key terms: bold tokens, cleaned and deduped by frequency
    bolds = re.findall(r"\*\*(.{2,40}?)\*\*", text)
    stop = {"Analysis:", "Approach:", "Key points:", "Note:", "Data Point:",
            "Before You Continue:", "Key Insight:", "Mental Model:", "Hint",
            "Common Mistakes", "Key Takeaways", "Practice Problems"}
    freq = {}
    for b in bolds:
        b = b.strip()
        if b in stop or b.endswith((":", "：")):
            continue
        if any(ch in b for ch in "，。；、？！"):  # sentence fragments, not terms
            continue
        if not re.match(r"^[A-Za-z0-9一-鿿 /&\-+.()（）·]+$", b):
            continue
        if len(b) > 30:
            continue
        freq[b] = freq.get(b, 0) + 1
    meta["key_terms"] = [t for t, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:12]
    return meta


def build(lang: str):
    parts = parse_summary(lang)
    out = {"site": SITE, "language": lang, "parts": []}
    n_ch = 0
    for p in parts:
        pobj = {"part": p["part"], "title": p["title"], "path": f"src/{lang}/", "chapters": []}
        for idx, (title, rel) in enumerate(p["chapters"], 1):
            md = REPO / "src" / lang / rel
            if not md.exists():
                print(f"WARN missing: {md}", file=sys.stderr)
                continue
            m = extract_meta(md)
            is_cover = md_path_is_cover(md)
            pobj["chapters"].append({
                "number": None if is_cover else chapter_number(p["part"], idx),
                "title": title,
                "source": f"src/{lang}/{rel}",
                "url": f"{SITE}/{lang}/{rel.replace('.md', '.html')}",
                "difficulty": m["difficulty"],
                "read_minutes": m["read_minutes"],
                "objectives": m["objectives"],
                "key_terms": m["key_terms"],
            })
            n_ch += 1
        out["parts"].append(pobj)
    return out, n_ch


def main():
    for lang in ("zh", "en"):
        data, n = build(lang)
        outdir = REPO / "index"
        outdir.mkdir(exist_ok=True)
        (outdir / f"{lang}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"{lang}: {len(data['parts'])} parts, {n} chapters -> index/{lang}.json")


if __name__ == "__main__":
    main()
