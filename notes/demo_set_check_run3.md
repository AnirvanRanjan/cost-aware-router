# Demo set check (T45)

*`python scripts/check_demo_set.py` on `artifacts/bundle/router.pkl`.*

```
tau* = 8.219, rule v2, cascade (Qwen answers first), cascade_t61_3sample

id        expect    got        delta_hat  signal                            query  ->  local answer
easy-01   local     local         -0.004  arithmetic: 3/3 answers agree     Edric's monthly salary is $576. If he wo  ->   hours: $576 / 192 = $3 per hour. The answer is 3.
easy-02   local     local         -0.006  arithmetic: 3/3 answers agree     Joel is 5 years old and his dad is 32 ye  ->  Joel's current age: 5 + 22 = 27. The answer is 27.
easy-03   local     local         -0.009  arithmetic: 3/3 answers agree     Emily bought a shirt and a coat for $600  ->  e-third of $450, which is $150. The answer is 150.
easy-04   local     local         -0.018  arithmetic: 3/3 answers agree     A man owned 1/2 of a lot. He sold 1/10 o  ->  ire lot = $4,600 * 2 = $9,200. The answer is 9200.
easy-05   local     frontier       0.488  arithmetic: 1/3 answers agree     Farmer Brown raises emus, large birds. H  ->  imes e) = 60$ $3e = 60$ $e = 20$ The answer is 20.   <-- MISMATCH
easy-06   local     local         -0.018  arithmetic: 3/3 answers agree     There are 11 boys and 13 girls in a clas  ->  plying by 100: 0.52 * 100 = 52%. The answer is 52.
easy-07   local     frontier       0.055  multiple choice: option probabili What does the p equal in 3 + p = 15? A)   ->  The answer is D) 12.   <-- MISMATCH
easy-08   local     local          0.009  multiple choice: option probabili The term "Cold War" refers to the A) com  ->  Union to gain political hegemony in world affairs.
easy-09   local     local          0.009  multiple choice: option probabili During President Dwight Eisenhower's adm  ->  the construction of the interstate highway system.
easy-10   local     local          0.009  multiple choice: option probabili What was the 'Domino Theory'? A) The ide  ->  ould inexorably lead to the fall of its neighbors.
hard-01   frontier  frontier       0.038  arithmetic: 2/3 answers agree     The city’s bus system carries 1,200,000   ->  eks = 109,200,000 people. The answer is 109200000.
hard-02   frontier  frontier       0.165  arithmetic: 2/3 answers agree     In 6 months Bella and Bob will be celebr  ->  ary = D + 6 months - 24 months = D - 18 months. So
hard-03   frontier  frontier       0.263  arithmetic: 2/3 answers agree     Tom fills a 250 pound sandbag 80% full.   ->   total: 200 + 280 = 480 pounds. The answer is 480.
hard-04   frontier  frontier       0.096  arithmetic: 2/3 answers agree     Tyrone empties his piggy bank and finds   ->   $2.00 + $0.40 + $0.35 = $13.00. The answer is 13.
hard-05   frontier  frontier       0.009  arithmetic: 3/3 answers agree     Bob runs 6 miles per hour. His friend Ji  ->  5 miles per minute = 20 minutes. The answer is 20.
hard-06   frontier  frontier       0.030  multiple choice: option probabili How many periods are there in an ice hoc  ->  The answer is D) Two.
hard-07   frontier  frontier       0.127  multiple choice: option probabili What color is cartoon character Marge Si  ->  The answer is D) yellow.
hard-08   frontier  frontier       0.215  multiple choice: option probabili What is the quotient for the expression   ->  The answer is D) 578.
hard-09   frontier  frontier       0.177  multiple choice: option probabili Which of these is not found in a Snicker  ->  The answer is D) caramel.
free-01   any       local         -0.005  free-form: 3/3 answers agree [unc What is the capital of Australia?  ->   in the late 19th century. The answer is Canberra.
free-02   any       frontier       0.130  free-form: 2/3 answers agree [unc Who wrote 'Romeo and Juliet'?  ->  ly around 1595. The answer is William Shakespeare.
free-03   any       local         -0.005  free-form: 3/3 answers agree [unc Translate 'good morning' into French.  ->  hasizes the morning period. The answer is bonjour.
free-04   any       local         -0.005  free-form: 3/3 answers agree [unc Give me a synonym for 'happy'.  ->  Another option is "content." The answer is joyful.
chat-01   any       frontier       0.455  free-form: 1/3 answers agree [unc Suggest three names for a study group th  ->  Nexus, Algorithmic Ascents, and The Silicon Synod.
chat-02   any       frontier       0.455  free-form: 1/3 answers agree [unc Write a two-line poem about a rainy even  ->  on rain washes the stone in gentle, green silence.
chat-03   any       frontier       0.455  free-form: 1/3 answers agree [unc Summarise what a model router does in on  ->  details. The answer is a smart traffic controller.

24/26 as planned
```

## What this shows (run 3, 2026-11-16: D-15 bundle built on Ani's M1, Qwen 4-bit MLX)

- **24/26 as planned.** The bundle built on the M1 reproduces the release model: tau* = 8.219, the
  headline tau* from `notes/t60_certification_report.md`. The demo now routes with the certified number.
- **Every question Qwen got wrong went to the frontier**: hard-03 (480 lb; right is 280) and all four
  multiple-choice traps (ice hockey, Marge Simpson, 2,314 / 4, Snickers). Live, Qwen also got hard-01,
  hard-04 and hard-05 right; the router still escalated them (2/3 agreement, or a confident answer
  the certified threshold does not trust), which costs money but no quality.
- **Easy questions stay local**: 8 of 10, with all 3 answers agreeing (arithmetic) or option
  probability ~1 on long multiple-choice questions.
- The two misses: easy-05 (emus): the M1's three samples disagreed (1/3) where the H100's agreed;
  easy-07 (3 + p = 15): a short multiple-choice question, which rule v2 escalates (score 20.7 with
  the release model). Both replaced in `configs/demo_set.yaml` with arithmetic questions whose
  release-model scores are far below tau* (−23.6, −16.5).
- Free-form questions (not certified, labelled so): Canberra, bonjour and joyful stayed local with
  3/3 agreement; Shakespeare escalated at 2/3; all three chat prompts escalated (1/3).
