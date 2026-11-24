#!/usr/bin/env python3
"""T54: merge the thesis chapters into one Word report.

    python scripts/build_report.py                      # -> thesis/build/report.docx
    python scripts/build_report.py --out other.docx

Front matter (title page, declaration, certificate, acknowledgement, abstract) comes from
thesis/00_front_matter.md; the chapters are thesis/01_*.md ... 08_*.md in order; references are
rendered from thesis/references.bib. The page setup follows the common UPES report format (A4, Times
New Roman 12 pt, 1.5 line spacing, 1.5" left margin for binding). When the official UPES template
arrives, pass it with --reference-doc and the same styles are reused.

Needs pandoc (>= 3) and python-docx.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "thesis"


def make_reference_doc(path: Path) -> None:
    """A pandoc reference .docx restyled to the usual UPES report layout."""
    from docx import Document
    from docx.enum.text import WD_LINE_SPACING
    from docx.shared import Inches, Pt, RGBColor

    with open(path, "wb") as f:
        f.write(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                               check=True, capture_output=True).stdout)
    doc = Document(str(path))
    for sec in doc.sections:
        sec.page_height, sec.page_width = Inches(11.69), Inches(8.27)          # A4
        sec.left_margin, sec.right_margin = Inches(1.5), Inches(1.0)
        sec.top_margin, sec.bottom_margin = Inches(1.0), Inches(1.0)
    sizes = {"Title": 20, "Subtitle": 13, "Heading 1": 16, "Heading 2": 14, "Heading 3": 12, "Heading 4": 12}
    for st in doc.styles:
        if st.type != 1:                                                           # paragraph styles only
            continue
        st.font.name = "Times New Roman"
        st.font.color.rgb = RGBColor(0, 0, 0)
        if st.name in sizes:
            st.font.size = Pt(sizes[st.name])
            st.font.bold = True
        elif st.name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text", "Author", "Date", "Abstract"):
            st.font.size = Pt(12)
        pf = st.paragraph_format
        if st.name in ("Body Text", "First Paragraph", "Normal"):
            pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            pf.space_after = Pt(6)
    doc.save(str(path))


def chapter_text(md: Path) -> str:
    """Chapter file -> markdown for the merged report: '# Chapter 3 — Data' becomes '# 3. Data'."""
    text = md.read_text()
    return re.sub(r"^# Chapter (\d+) — ", r"# \1. ", text, count=1, flags=re.M)


PAGE_BREAK = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n'


def page_breaks(md: str) -> str:
    """Start every top-level section (declaration, abstract, each chapter, references) on a new page."""
    return re.sub(r"^(# )", lambda m: PAGE_BREAK + m.group(1), md, flags=re.M)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=THESIS / "build" / "report.docx")
    ap.add_argument("--reference-doc", type=Path, default=None, help="the official UPES .docx template, when available")
    args = ap.parse_args()
    if shutil.which("pandoc") is None:
        sys.exit("pandoc not found: brew install pandoc")

    chapters = sorted(THESIS.glob("0[1-9]_*.md"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref = args.reference_doc or tmp / "reference.docx"
        if args.reference_doc is None:
            make_reference_doc(ref)
        parts = [(THESIS / "00_front_matter.md").read_text()]
        parts += [chapter_text(c) for c in chapters]
        parts.append("# References {.unnumbered}\n\n::: {#refs}\n:::\n")
        merged = tmp / "report.md"
        merged.write_text(page_breaks("\n\n".join(parts)))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["pandoc", str(merged), "--from", "markdown+tex_math_dollars+pipe_tables",
               "--citeproc", "--bibliography", str(THESIS / "references.bib"),
               "--toc", "--toc-depth=2", "--reference-doc", str(ref),
               "--resource-path", f"{THESIS}:{ROOT}:{ROOT / 'notes'}", "-o", str(args.out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr)
        warn = [ln for ln in r.stderr.splitlines() if ln.strip()]
        print("\n".join(warn[:20]))
    print(f"wrote {args.out} ({len(chapters)} chapters)")


if __name__ == "__main__":
    main()
