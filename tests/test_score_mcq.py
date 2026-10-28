"""scripts/14_score_mcq.py: resumable multiple-choice scoring, tested with a stand-in scorer."""
import importlib.util
from pathlib import Path

import pandas as pd

from router import schema

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_mcq", ROOT / "scripts" / "14_score_mcq.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeScorer:
    """Always prefers option B; counts calls."""

    def __init__(self):
        self.calls = 0

    def option_logprobs(self, prompt, labels):
        self.calls += 1
        return {lab: (-0.1 if lab == "B" else -3.0) for lab in labels}


def _corpus(n=6):
    return pd.DataFrame({"qid": [f"q{i}" for i in range(n)],
                         "prompt": ["Pick one.\nA) x\nB) y\nC) z\nD) w"] * n,
                         "reference": ["B", "A"] * (n // 2)})


def test_scores_and_writes_contract_files(tmp_path):
    done = mod.score_corpus(_corpus(), FakeScorer(), tmp_path / "p.jsonl", log=lambda *_: None)
    s_path, o_path = mod.write_outputs(done, "qwen3.5-4b", tmp_path)
    s = schema.read(s_path, schema.SCORES)
    assert len(s) == 6 and s.loc[s.qid == "q0", "score"].item() > 0.8 and s.loc[s.qid == "q1", "score"].item() < 0.1
    o = pd.read_parquet(o_path)
    assert (o["chosen"] == "B").all() and o["correct"].mean() == 0.5


def test_resumes_without_rescoring(tmp_path):
    p = tmp_path / "p.jsonl"
    first = FakeScorer()
    mod.score_corpus(_corpus(4), first, p, log=lambda *_: None)
    with p.open("a") as fh:
        fh.write('{"qid": "q9", "logpro')               # a crash mid-write leaves a broken last line
    second = FakeScorer()
    done = mod.score_corpus(_corpus(6), second, p, log=lambda *_: None)
    assert first.calls == 4 and second.calls == 2 and len([r for r in done.values() if "score" in r]) == 6
