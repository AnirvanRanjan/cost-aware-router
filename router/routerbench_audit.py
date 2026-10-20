"""RouterBench audit: answers the five Week-1 questions straight from the data file.

Used by scripts/audit_routerbench.py. It deliberately does NOT assume RouterBench's layout:
it detects model columns from their suffixes and reports what it finds, so a wrong guess about
the schema shows up as a visible "not found" line instead of a silent wrong number.

Expected layout (from the paper, to be confirmed): one row per prompt, with `sample_id`,
`prompt`, `eval_name`, and for each model a score column `<model>`, plus
`<model>|model_response` and `<model>|total_cost`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COST_SUFFIX = "|total_cost"
RESPONSE_SUFFIX = "|model_response"

# Track B sizes, used to sanity-check whether IDs look like original dataset indices.
ORIGINAL_SIZES = {"gsm8k": 1319, "mmlu": 14042}


def family(name: str) -> str:
    """Map a RouterBench eval_name onto a dataset family."""
    n = str(name).lower()
    rules = [("mmlu", "mmlu"), ("grade-school-math", "gsm8k"), ("gsm", "gsm8k"), ("mbpp", "mbpp"),
             ("mt-bench", "mt-bench"), ("mtbench", "mt-bench"), ("hellaswag", "hellaswag"),
             ("winogrande", "winogrande"), ("arc", "arc-challenge"), ("rag", "rag")]
    for key, fam in rules:
        if key in n:
            return fam
    return n


def _text(x) -> str:
    if isinstance(x, (list, tuple, np.ndarray)):
        return "\n".join(map(str, x))
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


@dataclass
class Audit:
    shape: tuple
    columns: list
    models: list
    score_cols: dict
    eval_col: str | None
    id_col: str | None
    prompt_col: str | None
    sections: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def detect(df: pd.DataFrame) -> Audit:
    cols = list(map(str, df.columns))
    models = sorted(c[: -len(COST_SUFFIX)] for c in cols if c.endswith(COST_SUFFIX))
    score_cols = {m: m for m in models if m in cols}
    pick = lambda *names: next((c for c in names if c in cols), None)
    a = Audit(df.shape, cols, models, score_cols,
              eval_col=pick("eval_name", "dataset", "source"),
              id_col=pick("sample_id", "id", "qid"),
              prompt_col=pick("prompt", "question", "input"))
    if not models:
        a.warnings.append(f"No '*{COST_SUFFIX}' columns found — the layout differs from the paper. "
                          "Read the column list below and adjust `detect()`.")
    missing = [m for m in models if m not in score_cols]
    if missing:
        a.warnings.append(f"Models with a cost column but no score column: {missing}")
    for name, col in (("eval_name", a.eval_col), ("sample_id", a.id_col), ("prompt", a.prompt_col)):
        if col is None:
            a.warnings.append(f"No {name} column found.")
    return a


def run(df: pd.DataFrame) -> Audit:
    a = detect(df)
    fam = (df[a.eval_col].map(family) if a.eval_col else
           df[a.id_col].astype(str).str.split(".").str[0].map(family) if a.id_col else
           pd.Series("unknown", index=df.index))
    a.sections["families"] = fam

    # Q1 — datasets and sizes
    rows = []
    for f, idx in fam.groupby(fam).groups.items():
        sub = df.loc[idx]
        n_sub = sub[a.eval_col].nunique() if a.eval_col else "—"
        rows.append([f, len(idx), n_sub])
    rows.sort(key=lambda r: -r[1])
    a.sections["q1"] = _md_table(rows, ["family", "items", "distinct eval_names"])

    # Q2 — score type per family
    rows = []
    for f, idx in fam.groupby(fam).groups.items():
        vals = pd.concat([pd.to_numeric(df.loc[idx, c], errors="coerce") for c in a.score_cols.values()]) \
            if a.score_cols else pd.Series(dtype=float)
        vals = vals.dropna()
        if vals.empty:
            rows.append([f, "no numeric scores", "—", "—", "—"]); continue
        binary = vals.isin([0, 1]).mean()
        kind = "binary 0/1" if binary == 1 else ("mostly binary" if binary > 0.95 else "continuous")
        rows.append([f, kind, f"{vals.min():.3g}", f"{vals.max():.3g}", vals.nunique()])
    a.sections["q2"] = _md_table(sorted(rows), ["family", "score type", "min", "max", "distinct values"])

    # Q3 — token counts vs cost
    token_cols = [c for c in a.columns if "token" in c.lower()]
    rows = []
    for m in a.models:
        cost = pd.to_numeric(df[m + COST_SUFFIX], errors="coerce")
        resp = df[m + RESPONSE_SUFFIX] if (m + RESPONSE_SUFFIX) in df.columns else None
        coverage = f"{resp.map(lambda x: len(_text(x)) > 0).mean():.0%}" if resp is not None else "absent"
        rows.append([m, f"{cost.notna().mean():.0%}", f"${cost.median():.2e}" if cost.notna().any() else "—", coverage])
    a.sections["q3_tokens"] = token_cols
    a.sections["q3"] = _md_table(rows, ["model", "cost present", "median cost / item", "response text present"])

    # Q4 — can GSM8K / MMLU items be traced to original IDs?
    rows = []
    if a.id_col:
        for f in ("gsm8k", "mmlu"):
            ids = df.loc[fam == f, a.id_col].astype(str)
            if ids.empty:
                rows.append([f, 0, "—", "—", "—"]); continue
            nums = pd.to_numeric(ids.str.extract(r"(\d+)\s*$")[0], errors="coerce")
            rows.append([f, len(ids), "; ".join(ids.head(3)), f"{nums.notna().mean():.0%}",
                         f"{int(nums.max())}" if nums.notna().any() else "—"])
    a.sections["q4"] = _md_table(rows, ["family", "items", "example ids", "ids ending in an index", "max index"]) \
        if rows else "No id column, so items cannot be traced."

    # Q5 — duplicates
    if a.prompt_col:
        norm = df[a.prompt_col].map(lambda x: _norm(_text(x)))
        dup_mask = norm.duplicated(keep=False)
        within = sum(int(norm[fam == f].duplicated().sum()) for f in fam.unique())
        cross = norm[dup_mask].groupby(norm[dup_mask]).apply(lambda s: fam[s.index].nunique() > 1).sum()
        a.sections["q5"] = {
            "exact_dup_rows": int(dup_mask.sum()),
            "within_family_extra_copies": within,
            "prompts_shared_across_families": int(cross),
            "dup_ids": int(df[a.id_col].duplicated().sum()) if a.id_col else None,
        }

    # Bonus — headroom preview for candidate S/L pairs (DESIGN §4 Phase 3 kill-check)
    if a.score_cols:
        means = pd.DataFrame({m: pd.to_numeric(df[c], errors="coerce") for m, c in a.score_cols.items()})
        table = means.groupby(fam).mean()
        a.sections["model_means"] = table
        small = next((m for m in a.models if "mistral-7b" in m.lower()), None)
        large = next((m for m in a.models if "gpt-4" in m.lower()), None)
        if small and large:
            hr = (table[large] - table[small]).round(3)
            a.sections["headroom"] = (small, large, hr)
        else:
            a.warnings.append("Could not find a Mistral-7B and a GPT-4 column for the headroom preview.")
    return a


def to_markdown(a: Audit, source: str) -> str:
    L = ["# RouterBench audit", "",
         f"_Generated by `scripts/audit_routerbench.py` from `{source}`. Re-run to refresh; "
         "edit only the **Interpretation** lines by hand._", ""]
    if a.warnings:
        L += ["## ⚠ Warnings", ""] + [f"- {w}" for w in a.warnings] + [""]
    L += ["## Schema", "", f"- Shape: **{a.shape[0]:,} rows × {a.shape[1]} columns**",
          f"- Detected id / prompt / dataset columns: `{a.id_col}` / `{a.prompt_col}` / `{a.eval_col}`",
          f"- Models ({len(a.models)}): " + ", ".join(f"`{m}`" for m in a.models), "",
          "<details><summary>All columns</summary>", "", "\n".join(f"- `{c}`" for c in a.columns), "",
          "</details>", ""]
    L += ["## Q1 · Which datasets, and how many items each?", "", a.sections["q1"], "",
          "**Interpretation:** _TODO_", ""]
    L += ["## Q2 · Is the per-item score 0/1 or continuous?", "", a.sections["q2"], "",
          "**Interpretation:** _TODO_", ""]
    tok = a.sections["q3_tokens"]
    L += ["## Q3 · Are token counts stored, or only cost?", "",
          f"- Columns mentioning tokens: {', '.join(f'`{c}`' for c in tok) if tok else '**none**'}", "",
          a.sections["q3"], "",
          "If tokens are absent, don't back them out of `total_cost`: one number can't be split into input "
          "and output. Count input tokens from the prompt text and output tokens from each model's response "
          "text instead, then re-price at 2026 rates.", "", "**Interpretation:** _TODO_", ""]
    L += ["## Q4 · Can GSM8K and MMLU items be traced to their original IDs?", "", a.sections["q4"], "",
          f"Original sizes for reference: GSM8K test = {ORIGINAL_SIZES['gsm8k']:,}, "
          f"MMLU test = {ORIGINAL_SIZES['mmlu']:,}.", "", "**Interpretation:** _TODO_", ""]
    if "q5" in a.sections:
        q = a.sections["q5"]
        L += ["## Q5 · Are there duplicate prompts?", "",
              f"- Rows whose normalised prompt appears more than once: **{q['exact_dup_rows']:,}**",
              f"- Extra copies within a dataset: **{q['within_family_extra_copies']:,}**",
              f"- Prompts shared across datasets: **{q['prompts_shared_across_families']:,}**",
              f"- Duplicate sample ids: **{q['dup_ids']}**", "",
              "This counts exact duplicates only (lower-cased, whitespace collapsed). Near-duplicates are "
              "`dedup.py`'s job in week 2.", "", "**Interpretation:** _TODO_", ""]
    if "model_means" in a.sections:
        t = a.sections["model_means"]
        rows = [[f] + [f"{v:.3f}" for v in t.loc[f]] for f in t.index]
        L += ["## Bonus · Mean score per model and dataset", "", _md_table(rows, ["family"] + list(t.columns)), ""]
    if "headroom" in a.sections:
        s, l, hr = a.sections["headroom"]
        rows = [[f, f"{v:+.3f}", "✅ keep" if v >= 0.05 else "⛔ below 0.05"] for f, v in hr.items()]
        L += [f"## Bonus · Headroom preview: `{l}` − `{s}`", "",
              "The Phase-3 kill-check, early. Datasets below 0.05 have no routing headroom for this pair.", "",
              _md_table(rows, ["family", "Q_L − Q_S", "verdict"]), ""]
    return "\n".join(L)
