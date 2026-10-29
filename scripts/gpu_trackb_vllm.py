#!/usr/bin/env python3
"""Track B on a CUDA GPU with vLLM: all of Qwen's answers in one job (minutes on an H100).

    python scripts/gpu_trackb_vllm.py                     # everything: GSM8K k=3 + MCQ
    python scripts/gpu_trackb_vllm.py --only mcq          # multiple choice only
    python scripts/gpu_trackb_vllm.py --limit 50          # quick check first

Does the same job as scripts/09_generate.py (GSM8K) and scripts/14_score_mcq.py (MMLU, ARC,
Winogrande) on the Mac, with the same prompts, the same chat template (Qwen thinking OFF), the
same grading and the same output files, so scripts/10_build_trackb.py reads them unchanged:

  artifacts/trackb/qwen_responses.parquet      RESPONSES contract: k sampled GSM8K answers per question,
                                               with mean token log-probability (Qwen's confidence)
  artifacts/trackb/qwen_mcq_scores.parquet     SCORES contract: P(gold option)
  artifacts/trackb/qwen_mcq_options.parquet    every option's log-probability, chosen letter, its probability
  artifacts/trackb/gpu_manifest.json           model, vLLM version, GPU, sampling settings, timings

Model: Qwen/Qwen3.5-4B in bf16 (needs vLLM >= 0.17). NOTE for the write-up: the Mac demo runs the
4-bit MLX build of the same model, so its answers can differ slightly from these labels. Measure it:
run the Mac on ~200 of the same test questions and compare (scripts/check_gsm8k_responses.py).

GSM8K is processed in chunks; each finished chunk is saved, so a killed job resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import graders, schema  # noqa: E402
from router.generate import build_prompt  # noqa: E402

MCQ_SOURCES = ("mmlu", "arc-challenge", "winogrande")


# ---------------------------------------------------------------------------------------------------
# Pure helpers (tested without a GPU in tests/test_gpu_trackb.py)
# ---------------------------------------------------------------------------------------------------

def gsm8k_rows(request_outputs, qids, model_name: str) -> list[dict]:
    """RESPONSES rows from vLLM RequestOutputs (one per question, n samples each)."""
    rows = []
    for qid, ro in zip(qids, request_outputs):
        n_in = len(ro.prompt_token_ids or [])
        for j, co in enumerate(ro.outputs):
            n_out = len(co.token_ids)
            lp = co.cumulative_logprob
            mean_lp = (lp / n_out) if (lp is not None and n_out) else None
            if mean_lp is not None and not math.isfinite(mean_lp):
                mean_lp = None
            rows.append({"qid": qid, "model": model_name, "sample_idx": j, "text": co.text,
                         "n_in": n_in, "n_out": n_out,
                         "mean_logprob": None if mean_lp is None else min(0.0, float(mean_lp)),
                         "latency_ms": None})
    return rows


def option_logprobs_from_top(top: dict, labels: list[str]) -> dict[str, float]:
    """Map vLLM's top-k first-token log-probs to option labels ("A" or " A", whichever is higher).

    `top` is {token_id: Logprob(logprob=..., decoded_token=...)}. A label missing from the top-k gets
    the smallest log-prob seen minus 1 (an upper bound on its true value), so softmax over options
    still works.
    """
    best: dict[str, float] = {}
    floor = math.inf
    for lp in top.values():
        val = float(lp.logprob)
        floor = min(floor, val)
        tok = (lp.decoded_token or "").strip().strip("()").upper()
        if tok in labels:
            best[tok] = max(best.get(tok, -math.inf), val)
    floor = (floor - 1.0) if math.isfinite(floor) else -30.0
    return {lab: best.get(lab, floor) for lab in labels}


def mcq_record(qid: str, prompt: str, gold: str, top: dict | None) -> dict:
    labels = graders.parse_options(prompt)
    if not labels or top is None:
        return {"qid": qid, "error": "no options found" if not labels else "no logprobs"}
    lps = option_logprobs_from_top(top, labels)
    chosen = max(lps, key=lps.get)
    probs = np.exp(np.array(list(lps.values())) - max(lps.values()))
    probs = probs / probs.sum()
    return {"qid": qid, "logprobs": lps, "chosen": chosen, "p_chosen": float(probs.max()),
            "score": graders.mcq_score(lps, gold), "correct": float(chosen == str(gold).strip().upper()),
            "n_in": len(str(prompt)) // 4}


def write_mcq(records: list[dict], model_name: str, out_dir: Path) -> tuple[Path, Path]:
    ok = [r for r in records if "score" in r]
    scores = pd.DataFrame({"qid": [r["qid"] for r in ok], "model": model_name,
                           "score": [r["score"] for r in ok], "grader": "mcq_logprob",
                           "grader_version": graders.VERSION,
                           "mean_n_in": [float(r["n_in"]) for r in ok], "mean_n_out": 1.0})
    options = pd.DataFrame({"qid": [r["qid"] for r in ok], "model": model_name,
                            "chosen": [r["chosen"] for r in ok], "p_chosen": [r["p_chosen"] for r in ok],
                            "correct": [r["correct"] for r in ok],
                            "logprobs": [json.dumps(r["logprobs"]) for r in ok]})
    s_path = schema.write(scores, schema.SCORES, out_dir / "qwen_mcq_scores.parquet")
    o_path = out_dir / "qwen_mcq_options.parquet"
    options.to_parquet(o_path, index=False)
    return s_path, o_path


def messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------------------------------------------
# GPU job
# ---------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Hugging Face id or a local folder")
    ap.add_argument("--model-name", default="qwen3.5-4b", help="name written to the output files")
    ap.add_argument("--only", choices=("all", "gsm8k", "mcq"), default="all")
    ap.add_argument("--k", type=int, default=3, help="GSM8K samples per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2000, help="GSM8K questions per saved chunk")
    ap.add_argument("--limit", type=int, default=None, help="first N questions per task (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/trackb"))
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    corpus = schema.read(args.corpus, schema.CORPUS).sort_values("qid")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, seed=args.seed,
              gpu_memory_utilization=0.90)
    chat_kwargs = {"enable_thinking": False}
    manifest = {"model": args.model, "model_name": args.model_name, "dtype": "bfloat16",
                "vllm": vllm.__version__, "host": platform.node(), "thinking": False,
                "gsm8k": {"k": args.k, "temperature": args.temperature, "top_p": args.top_p,
                          "top_k": args.top_k, "max_tokens": args.max_tokens,
                          "prompt_template": build_prompt("gsm8k", "{question}")},
                "seconds": {}}
    try:
        import torch
        manifest["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        manifest["gpu"] = "unknown"

    # ---- GSM8K: k sampled answers per question, with log-probs ----
    if args.only in ("all", "gsm8k"):
        t0 = time.time()
        g = corpus[corpus["source"] == "gsm8k"]
        if args.limit:
            g = g.head(args.limit)
        params = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                max_tokens=args.max_tokens, logprobs=1, seed=args.seed)
        parts = []
        for start in range(0, len(g), args.chunk):
            tag = f"_limit{args.limit}" if args.limit else ""   # smoke tests never pose as finished chunks
            part_path = args.out_dir / f"gsm8k_part_{start // args.chunk:03d}{tag}.parquet"
            if part_path.exists():
                parts.append(pd.read_parquet(part_path))
                print(f"GSM8K chunk {start // args.chunk}: already done", flush=True)
                continue
            chunk = g.iloc[start:start + args.chunk]
            convs = [messages(build_prompt("gsm8k", p)) for p in chunk["prompt"]]
            outs = llm.chat(convs, params, chat_template_kwargs=chat_kwargs, use_tqdm=True)
            df = pd.DataFrame(gsm8k_rows(outs, chunk["qid"].tolist(), args.model_name),
                              columns=schema.RESPONSES.names)
            df.to_parquet(part_path, index=False)
            parts.append(df)
            print(f"GSM8K chunk {start // args.chunk}: {len(chunk):,} questions done", flush=True)
        resp = pd.concat(parts, ignore_index=True)
        resp["mean_logprob"] = resp["mean_logprob"].astype(float)
        resp["latency_ms"] = resp["latency_ms"].astype(float)
        schema.write(resp, schema.RESPONSES, args.out_dir / "qwen_responses.parquet")
        cut = float((resp["n_out"] >= args.max_tokens).mean())
        manifest["gsm8k"].update({"questions": int(resp["qid"].nunique()), "answers": len(resp),
                                  "cut_off_share": cut, "median_out_tokens": float(resp["n_out"].median())})
        manifest["seconds"]["gsm8k"] = round(time.time() - t0, 1)
        print(f"GSM8K: {len(resp):,} answers, median {resp['n_out'].median():.0f} tokens, "
              f"{cut:.1%} cut off at max-tokens", flush=True)

    # ---- MCQ: one token, top-20 log-probs, read the option letters ----
    if args.only in ("all", "mcq"):
        t0 = time.time()
        m = corpus[corpus["source"].isin(MCQ_SOURCES) & corpus["reference"].notna()]
        if args.limit:
            m = m.groupby("source").head(args.limit)
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        convs = [messages(p) for p in m["prompt"]]
        outs = llm.chat(convs, params, chat_template_kwargs=chat_kwargs, use_tqdm=True)
        records = []
        for qid, prompt, gold, ro in zip(m["qid"], m["prompt"], m["reference"], outs):
            lp = ro.outputs[0].logprobs
            records.append(mcq_record(qid, prompt, gold, lp[0] if lp else None))
        s_path, o_path = write_mcq(records, args.model_name, args.out_dir)
        ok = [r for r in records if "score" in r]
        acc = float(np.mean([r["correct"] for r in ok])) if ok else float("nan")
        top = pd.Series([r["chosen"] for r in ok]).value_counts(normalize=True)
        manifest["mcq"] = {"questions": len(m), "scored": len(ok), "accuracy": acc,
                           "most_chosen_letter": top.index[0] if len(top) else None,
                           "most_chosen_share": float(top.iloc[0]) if len(top) else None}
        manifest["seconds"]["mcq"] = round(time.time() - t0, 1)
        print(f"MCQ: {len(ok):,}/{len(m):,} scored, accuracy {acc:.3f}, most-chosen letter "
              f"{manifest['mcq']['most_chosen_letter']} at {manifest['mcq']['most_chosen_share']:.0%}", flush=True)

    manifest["seconds"]["total"] = round(time.time() - t_start, 1)
    (args.out_dir / "gpu_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done in {manifest['seconds']['total']:.0f} s -> {args.out_dir}/")


if __name__ == "__main__":
    main()
