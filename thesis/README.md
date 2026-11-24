# Thesis drafts

Markdown drafts of the report, one file per chapter, so they can be reviewed in PRs like code.
Merged report (front matter, abstract, all chapters, references, A4 / Times 12 / 1.5 spacing):

```bash
python scripts/build_report.py                                   # -> thesis/build/report.docx
python scripts/build_report.py --reference-doc upes_template.docx  # once the official template arrives
```

In Word, right-click the table of contents → Update Field after opening. Plain pandoc, if needed:

```bash
pandoc thesis/0*.md --citeproc --bibliography thesis/references.bib -o thesis.docx   # Word
pandoc thesis/0*.md --citeproc --bibliography thesis/references.bib -o thesis.tex    # LaTeX
```

| File | Chapter | Owner | Status |
|---|---|---|---|
| `00_front_matter.md` | Title page, declaration, certificate, acknowledgement, abstract | A | Draft; fill guide name and SAP IDs |
| `01_introduction.md` | 1 Introduction | A (T54) | Own wording; final numbers (T28 locked, 2026-11-23) |
| `02_related_work.md` | 2 Related work | A (T54) | Draft; citations checked 2026-10-27 |
| `03_data.md` | 3 Data | A (T50) | Draft, all numbers from the frozen release |
| `04_router.md` | 4 Features and router | B (T51) | Own wording; Track B + D-15 sections; final numbers |
| `05_results.md` | 5 Cost model, thresholds, results | C (T52) | Locked numbers (tau* 8.2187, 41.3% M1, 28.4% M2); AI declaration |
| `06_system.md` | 6 Local inference and demo | D (T53) | Draft with T47 numbers from `notes/demo_perf.md` (soak frontier stats include 429s; offline answers 24/24). Bundle/figures reproducible with scikit-learn==1.8.0; author declaration; needs Kartiken's own wording |
| `07_discussion.md`, `08_conclusion.md` | 7–8 | A (T54) | Own wording; final numbers (2026-11-23) |

Rules: every number cites the script or note it comes from; nothing is reported from the test split
except through `scripts/08_threshold_report.py`; each author rewrites drafts in their own words and
follows UPES's rules on declaring AI assistance.
