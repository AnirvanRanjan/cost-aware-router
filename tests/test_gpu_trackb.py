"""scripts/gpu_trackb_vllm.py: output parsing, tested with stand-ins for vLLM's result objects."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS

from router import schema

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_trackb", ROOT / "scripts" / "gpu_trackb_vllm.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_gsm8k_rows_follow_the_responses_contract():
    ro = NS(prompt_token_ids=[1] * 40, outputs=[
        NS(text="The answer is 5.", token_ids=[1] * 10, cumulative_logprob=-2.0),
        NS(text="The answer is 6.", token_ids=[1] * 8, cumulative_logprob=None)])
    import pandas as pd
    df = pd.DataFrame(mod.gsm8k_rows([ro], ["q1"], "qwen3.5-4b"), columns=schema.RESPONSES.names)
    df["mean_logprob"] = df["mean_logprob"].astype(float)
    df["latency_ms"] = df["latency_ms"].astype(float)
    schema.validate(df, schema.RESPONSES)
    assert list(df["sample_idx"]) == [0, 1] and df.loc[0, "mean_logprob"] == -0.2
    assert math.isnan(df.loc[1, "mean_logprob"]) and df.loc[0, "n_in"] == 40


def _top(pairs):
    return {i: NS(logprob=lp, decoded_token=tok) for i, (tok, lp) in enumerate(pairs)}


def test_option_logprobs_take_the_best_spelling_and_floor_missing_labels():
    top = _top([("B", -0.1), (" B", -0.5), (" A", -2.0), ("The", -3.0)])
    lps = mod.option_logprobs_from_top(top, ["A", "B", "C", "D"])
    assert lps["B"] == -0.1 and lps["A"] == -2.0
    assert lps["C"] == lps["D"] == -4.0          # smallest seen (-3) minus 1


def test_mcq_record_and_files(tmp_path):
    prompt = "Q?\nA) x\nB) y\nC) z\nD) w"
    recs = [mod.mcq_record("q1", prompt, "B", _top([("B", -0.05), ("A", -3.0)])),
            mod.mcq_record("q2", prompt, "A", _top([("B", -0.05), ("A", -3.0)])),
            mod.mcq_record("q3", "no options here", "A", _top([("A", -0.1)]))]
    assert recs[0]["correct"] == 1.0 and recs[1]["correct"] == 0.0 and "error" in recs[2]
    s_path, o_path = mod.write_mcq(recs, "qwen3.5-4b", tmp_path)
    s = schema.read(s_path, schema.SCORES)
    assert len(s) == 2 and s.set_index("qid").loc["q1", "score"] > 0.9
