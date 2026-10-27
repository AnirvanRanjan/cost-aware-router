# Thesis drafts

Markdown drafts of the report, one file per chapter, so they can be reviewed in PRs like code.
Convert when the format is decided:

```bash
pandoc thesis/0*.md --citeproc --bibliography thesis/references.bib -o thesis.docx   # Word
pandoc thesis/0*.md --citeproc --bibliography thesis/references.bib -o thesis.tex    # LaTeX
```

| File | Chapter | Owner | Status |
|---|---|---|---|
| `01_introduction.md` | 1 Introduction | A (T54) | Draft; [TBD] numbers wait for T28 |
| `02_related_work.md` | 2 Related work | A (T54) | Draft; citations checked 2026-10-27, three [check] notes |
| `03_data.md` | 3 Data | A (T50) | Draft, all numbers from the frozen release |
| `04_router.md` | 4 Features and router | B (T51) | — |
| `05_results.md` | 5 Cost model, thresholds, results | C (T52) | — |
| `06_system.md` | 6 Local inference and demo | D (T53) | — |
| `07_discussion.md`, `08_conclusion.md` | 7–8 | A (T54) | — |

Rules: every number cites the script or note it comes from; nothing is reported from the test split
except through `scripts/08_threshold_report.py`; each author rewrites drafts in their own words and
follows UPES's rules on declaring AI assistance.
