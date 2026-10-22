"""scripts/08_threshold_report.py and plot_pareto.py run end to end on a small fake pair."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from router import schema

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake(tmp_path, n=1200, seed=0):
    rng = np.random.default_rng(seed)
    qids = [f"q{i}" for i in range(n)]
    split = rng.choice(["train", "cal", "test"], size=n, p=[0.5, 0.25, 0.25])
    corpus = pd.DataFrame({"qid": qids, "source": rng.choice(["mmlu", "gsm8k"], n), "subject": None,
                           "group_id": qids, "split": split, "prompt": "What is 2+2?", "reference": None,
                           "meta": None})
    s_S = (rng.uniform(size=n) < 0.6).astype(float)
    s_L = np.maximum(s_S, (rng.uniform(size=n) < 0.6).astype(float))
    rows = []
    for m, s in (("mistralai/mixtral-8x7b-chat", s_S), ("gpt-4-1106-preview", s_L)):
        rows.append(pd.DataFrame({"qid": qids, "model": m, "score": s, "grader": "x", "grader_version": "1",
                                  "mean_n_in": 100.0, "mean_n_out": rng.uniform(10, 300, n)}))
    scores = pd.concat(rows, ignore_index=True)
    d = np.clip((s_L - s_S) * 0.6 + rng.normal(0, 0.3, n), -1, 1)
    preds = pd.DataFrame({"qid": qids, "router": "toy", "seed": 0, "s_hat_S": np.clip(0.5 - d / 2, 0, 1),
                          "s_hat_L": np.clip(0.5 + d / 2, 0, 1), "delta_hat": d, "c_hat_L": 0.002})
    paths = {k: tmp_path / f"{k}.parquet" for k in ("corpus", "scores", "preds")}
    schema.write(corpus, schema.CORPUS, paths["corpus"])
    schema.write(scores, schema.SCORES, paths["scores"])
    schema.write(preds, schema.PREDICTIONS, paths["preds"])
    return paths


def test_report_end_to_end(tmp_path):
    rep = _load("08_threshold_report.py")
    p = _fake(tmp_path)
    f, _ = rep.load_frame(p["preds"], p["corpus"], p["scores"], "mistralai/mixtral-8x7b-chat", "gpt-4-1106-preview")
    out = rep.report_one(f, "v1", 0.95, 0.05, B=50, seed=0)
    for m in rep.METHODS:
        s = out["selections"][m]
        assert 0 <= s["test"]["M1"]["alpha"] <= 1
        assert s["test"]["M2"]["savings"] <= s["test"]["M1"]["savings"] + 1e-12   # M2 charges for S
    assert out["test_curve"]["envelope_cpt95"] <= out["test_curve"]["cpt95"] + 1e-12


def test_plot_pareto_writes_a_figure(tmp_path, monkeypatch):
    plot = _load("plot_pareto.py")
    p = _fake(tmp_path)
    out = tmp_path / "fig.png"
    monkeypatch.setattr("sys.argv", ["plot_pareto.py", "--predictions", str(p["preds"]), "--corpus", str(p["corpus"]),
                                     "--scores", str(p["scores"]), "--report-json", str(tmp_path / "none.json"),
                                     "--B", "10", "--out", str(out)])
    plot.main()
    assert out.exists() and out.stat().st_size > 10_000
