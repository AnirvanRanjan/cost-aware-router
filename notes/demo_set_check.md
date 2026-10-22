# Demo set check (T45), interim bundle

*`python scripts/check_demo_set.py` on the interim bundle (F0 + TF-IDF features, two-head trained on Mixtral vs GPT-4, tau* by WSR on cal). Run 2026-10-22.*

```
tau* = 0.084, rule v1

id        expect    got        delta_hat  query
easy-01   local     frontier       0.115  What is the capital of Australia?   <-- MISMATCH
easy-02   local     frontier       0.436  What is 15% of 80?   <-- MISMATCH
easy-03   local     local          0.072  Which planet is known as the Red Planet? A) Venus B) Mars C)
easy-04   local     frontier       0.115  Translate 'good morning' into French.   <-- MISMATCH
easy-05   local     frontier       0.208  A pencil costs $2 and a notebook costs $5. How much do 3 pen   <-- MISMATCH
easy-06   local     local          0.061  What gas do plants absorb from the air for photosynthesis?
easy-07   local     frontier       0.326  Give me a synonym for 'happy'.   <-- MISMATCH
easy-08   local     frontier       0.157  How many minutes are there in 3 hours?   <-- MISMATCH
easy-09   local     frontier       0.225  Who wrote 'Romeo and Juliet'?   <-- MISMATCH
easy-10   local     frontier       0.518  Is 17 a prime number?   <-- MISMATCH
hard-01   frontier  frontier       0.320  A store raises a price by 20%, then lowers the new price by 
hard-02   frontier  frontier       0.285  Prove that the square root of 2 is irrational.
hard-03   frontier  frontier       0.305  Three friends split a bill. Ana pays twice what Ben pays, an
hard-04   frontier  frontier       0.305  In how many ways can 8 people sit around a round table if tw
hard-05   frontier  frontier       0.393  A bat and a ball cost $1.10 in total. The bat costs $1.00 mo
hard-06   frontier  local         -0.030  Explain why the halting problem is undecidable, with the dia   <-- MISMATCH
hard-07   frontier  frontier       0.341  A train leaves at 9:40 at 72 km/h. A second train leaves the
hard-08   frontier  frontier       0.179  Which statement about the Treaty of Westphalia (1648) is mos
hard-09   frontier  frontier       0.515  Find all real x such that |x - 3| + |x + 1| = 6, and show yo
hard-10   frontier  frontier       0.215  A drug test is 99% sensitive and 95% specific. If 1% of peop
chat-01   any       frontier       0.234  Suggest three names for a study group that works on AI proje
chat-02   any       frontier       0.207  Write a two-line poem about a rainy evening in Dehradun.
chat-03   any       frontier       0.219  What are the pros and cons of running a language model on a 
chat-04   any       frontier       0.144  Summarise what a model router does in one sentence for someo
chat-05   any       frontier       0.234  Plan a 30-minute revision session for an operating systems e

16/25 as planned
```

## What this shows

**The interim router fails the demo: 8 of 10 easy questions go to the frontier.** It was trained on
benchmark prompts (long, templated MMLU / GSM8K items), so a short natural question like "Is 17 a
prime number?" looks unlike anything it saw, and it falls back on "send it out". This is DESIGN.md
risk 5 (distribution shift), showing up exactly where the audience will see it.

Fixes, in order of how much they help:

1. **Cascade (D-12).** Qwen answers first; the router looks at Qwen's self-agreement / confidence.
   That signal is about the *answer*, so it travels to new kinds of questions far better than
   query-text features. On the Mixtral pair it tripled the savings (`notes/cascade_explore.md`).
2. **Train the bundle on the Qwen pair** (Track B), not Mixtral, as D-07 already says.
3. **Add a few hundred short, natural questions** with Qwen and Gemini answers to the training data
   (cheap on the free tier), so "short and simple" stops being out of distribution.

Re-run this check after each of these; the demo set itself stays fixed.
