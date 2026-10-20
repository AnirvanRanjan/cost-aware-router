"""The audit runs on a fake frame laid out the way the RouterBench paper describes."""
import numpy as np
import pandas as pd

from router.routerbench_audit import detect, family, run, to_markdown

MODELS = ["mistralai/mistral-7b-chat", "gpt-4-1106-preview", "claude-v2"]


def fake(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ev, n in [("grade-school-math", n_per), ("mmlu-anatomy", n_per), ("mmlu-virology", n_per), ("mtbench", 10)]:
        for i in range(n):
            r = {"sample_id": f"{ev}.dev.{i}", "eval_name": ev, "prompt": f"{ev} question {i}"}
            for m in MODELS:
                strong = "gpt-4" in m
                if ev == "mtbench":
                    r[m] = float(rng.uniform(0.3, 1.0))
                else:
                    r[m] = float(rng.random() < (0.9 if strong else 0.5))
                r[m + "|model_response"] = "answer"
                r[m + "|total_cost"] = 1e-4 if strong else 1e-6
            rows.append(r)
    rows.append({**rows[0], "sample_id": "grade-school-math.dev.999"})  # one duplicate prompt
    return pd.DataFrame(rows)


def test_family_mapping():
    assert family("grade-school-math") == "gsm8k"
    assert family("mmlu-abstract-algebra") == "mmlu"
    assert family("mtbench") == "mt-bench"


def test_detects_models_and_columns():
    a = detect(fake())
    assert a.models == sorted(MODELS)
    assert (a.id_col, a.prompt_col, a.eval_col) == ("sample_id", "prompt", "eval_name")
    assert not a.warnings


def test_answers_all_five_questions():
    a = run(fake())
    md = to_markdown(a, "fake.pkl")
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        assert f"## {q}" in md
    assert "| gsm8k | 41 | 1 |" in md          # Q1 counts
    assert "| mmlu | 80 | 2 |" in md
    assert "binary 0/1" in md and "continuous" in md   # Q2 tells them apart
    assert a.sections["q5"]["exact_dup_rows"] == 2      # the planted duplicate
    assert "Headroom preview" in md


def test_unknown_layout_warns_instead_of_crashing():
    a = run(pd.DataFrame({"text": ["a", "b"], "label": [0, 1]}))
    assert any("layout differs" in w for w in a.warnings)
    assert "## Q1" in to_markdown(a, "weird.pkl")
