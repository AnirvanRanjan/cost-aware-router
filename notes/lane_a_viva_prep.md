# Lane A viva notes (Anirvan): data, evaluation pipeline, demo router

*Short honest answers. Every number is in the repo; the file is named after each answer.*

**1. Why not just use RouterBench's scores as they are?**
Because some of them don't measure correctness. Its GSM8K scores take only five values and GPT-4
never scores 1.0 (mean 0.66 against ~0.95 real accuracy), so I re-graded GSM8K from the gold answers
(hand check 48/50 = 96%). Mistral-7B answers "A" on 94% of MMLU, a formatting failure, so it was
dropped. A router trained on those labels would learn the artefacts. (`notes/chapter_A_facts.md`, ch. 3)

**2. How did you get gold letters for multiple choice?**
Voted from every RouterBench model that answered correctly, kept only if all agree and the letter is
an offered option: 0 conflicts. Coverage 95.5% (MMLU) to 100% (Winogrande). The questions no model got
right have no letter, so Track B excludes 649 of the hardest; that makes the headline pool slightly
easier and is stated as a limitation. (ch. 3, `notes/trackb_results.md`)

**3. Why template-centred de-duplication?**
Near-copies across train and test would leak. Plain embedding similarity failed: all 895 MMLU
moral-scenarios questions share a preamble and chained into one group. Subtracting the dataset/subject
mean embedding first and using residual cosine ≥ 0.90 found 1,298 real pairs (largest group 15).
Together with a MinHash lexical pass: 1,883 pairs, then frozen grouped splits 60/20/20
(train 20,867 · cal 6,957 · test 6,954, D-11). (`notes/dedup_report.md`)

**4. Why three splits, and why is the test split used only once?**
Train fits the router, cal picks the threshold, test reports the result. If the threshold were picked
by looking at test results it would be optimistic: the test curve suggests 78.9%, the certified
threshold gives 41.3%. Every test number goes through one frozen τ*. (ch. 5, `scripts/08_threshold_report.py`)

**5. What does the guarantee actually say?**
With probability at least 95% over the calibration sample, the chosen threshold keeps at least 95% of
GPT-4's quality advantage (PGR ≥ 0.95) on new questions drawn like the calibration ones. It says nothing
about different traffic (free-form chat), which is why the demo labels those "not certified".

**6. Why PGR and not plain accuracy?**
PGR measures how much of the gap between the small and the large model the router recovers, so it is
comparable across pairs with very different gaps (Mixtral 0.237, Qwen 0.075). RouteLLM uses it too.

**7. Why did the bootstrap get replaced?**
On low-headroom data it failed its guarantee about 15% of the time where 5% was promised (Sarthak's
simulation). The betting (WSR) bound is valid in finite samples; Sarthak's seed-free version averages
it over 50 orderings so the answer doesn't depend on a random seed. (ch. 5)

**8. What is cost model M2, and why does it matter?**
M1 treats local answers as free. M2 bills the laptop's time ($1,199 over 3 years at 8 h/day, 20 W
assumed): 28.4% saved instead of 41.3%, because the 3-answer router generates three local answers per
maths question. Against the Batch API (half price) the router is about 17% more expensive, so routing
pays off for interactive use, not overnight jobs. (`configs/models.yaml`, ch. 5, ch. 7)

**9. How do you know the demo is the router you certified?**
D-15: the demo rebuilds the exact T61 router from the release files, computes Qwen's signals with
the same code as training (`router/signals.py`, tested against `16_trackb_preview.py`), and with
scikit-learn 1.8.0 reproduces τ* = 8.2187. Replaying 200 test questions through the server matched
the offline evaluation on 200/200. (`notes/d15_demo_router.md`, `notes/replay_report_d15.md`)

**10. Why is scikit-learn pinned?**
Refitting the same model with 1.9.1 gave a different router (τ* 9.657, 48.3% certified). Both are
valid for their own fit, but the thesis quotes the release model, so the version is pinned.

**11. Why did the first demo set fail, and what did you learn?**
The first router read the answer's wording and one answer's log-probability; easy free-form answers
looked "unsure" and were escalated (8/10, then 10/10). Moving the demo to the certified router and to
questions like the ones it was calibrated on fixed it (24/24). Lesson: demo questions must be checked
on the machine that runs the demo; the H100's answers don't predict the Mac's samples.

**12. Limitations to admit**
Benchmarks are not real traffic; GPT-4's answers are from Nov 2023 while the demo calls Gemini;
649 hardest MCQs excluded; one fitted router per configuration; the guarantee assumes calibration and
deployment questions come from the same distribution.
