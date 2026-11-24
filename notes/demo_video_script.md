# Demo video and dry-run script (T48, T49)

*For the M2. The video and the dry runs use `--offline`: every demo answer is cached, so the free
Gemini tier cannot fail mid-demo. Target length: 6–7 minutes.*

## Before you press record (5 minutes)

```bash
git checkout main && git pull origin main
pip install -r requirements.txt                       # scikit-learn 1.8.0
python scripts/11_build_bundle.py --mode cascade_t61 --samples 3     # only if the bundle is missing
python scripts/20_demo_preflight.py                   # must end with PASS
python scripts/serve_demo.py --offline                # leave this terminal running
```

- Open http://127.0.0.1:8000 in Safari or Chrome, full screen, zoom 125% so text is readable.
- Close other apps (the model needs memory); plug in the charger; turn on Do Not Disturb.
- Restart the server before each take, so the savings counter starts from zero.
- Keep this file open on a second screen or printed.

## The run (one question at a time; paste from the list, wait for the answer)

| # | Paste this question (from `configs/demo_set.yaml`) | Expect | Say |
|--:|---|---|---|
| 0 | — | header shows `τ* = 8.219 (rule v2)` | "This is the router from the thesis. Its threshold was fixed on held-out data before we looked at the test set, with a guarantee: at least 95% of GPT-4's quality advantage is kept." |
| 1 | easy-02 (Joel and his dad) | LOCAL, 3/3 agree | "Qwen runs on this Mac first. It answers three times; all three agree, so the router trusts it. This answer cost nothing." |
| 2 | easy-03 (Emily's shirt and coat) | LOCAL, 3/3 agree | "Same again: an everyday maths question the small model handles." |
| 3 | easy-08 (Cold War, multiple choice) | LOCAL, option probability ≈ 1.00 | "For multiple choice it reads how confident Qwen is in each option. It is almost certain, so it stays local." |
| 4 | hard-03 (Tom's sandbag) | FRONTIER, answers disagree | "Here Qwen's three answers disagree, and in fact it gets this one wrong. The router sends it to the frontier model." |
| 5 | hard-07 (Marge Simpson's hair) | FRONTIER, low option probability | "Qwen is unsure between options, so the question is escalated. Qwen would have picked the wrong colour." |
| 6 | hard-05 (Bob and Jim running) | FRONTIER, 3/3 agree | "Here all three answers agree, and they are right, but the router still escalates: its confidence is just under the certified bar. It plays safe, which costs money, not quality." |
| 7 | free-01 (capital of Australia) | label **not a certified question type** | "Free-form questions are outside what the router was certified on, and the page says so. The guarantee only covers question types it was calibrated on." |
| 8 | chat-03 (what a model router does) | FRONTIER | "Open-ended chat usually goes to the frontier: Qwen's three answers rarely agree." |
| 9 | — | point at the sidebar | "The counter shows what this traffic would have cost if everything went to the frontier. This is demo traffic, not our certified number: on the 4,707-question test set the certified saving is 41.3%, at 102.8% of GPT-4's quality advantage kept." |

Close (10 seconds): "Every decision takes about 5 milliseconds; the wait you see is Qwen answering on
the Mac. The same code, replayed on 200 test questions, matched the offline evaluation on all 200."

## Rules while recording

- Do not type new questions live: only the demo set has cached frontier answers (`--offline`
  returns "not cached" for anything else).
- If a question routes differently from the table, stop and re-run `scripts/20_demo_preflight.py`;
  do not explain it away on camera.
- Do not quote the 63.3% from the soak test as a result: it is observed demo traffic on the free
  tier, not certified.

## Dry runs (T49)

Run the same script twice, timed, with the whole team, from the combined deck
(`slides/cost_aware_router_deck.pptx`, 27 slides: opening by A, then lanes A, B, C, D, the live demo,
close):

| Run | Date | Where | Check |
|---|---|---|---|
| 1 | **Fri 27 Nov 2026** | Kartiken's M2, laptop screen | timing per lane (aim ≤ 5 min each + 7 min demo), handovers, preflight PASS before starting |
| 2 | **Tue 1 Dec 2026** | M2 on a projector | page readable from the back (light theme), green/red route colours visible, backup video plays from USB |

Before each run: `python scripts/20_demo_preflight.py` must print PASS, then
`python scripts/serve_demo.py --offline`. Note anything that runs over and cut it. Move the dates if
the guide's schedule needs it; keep at least two days between the runs to fix what run 1 finds.
