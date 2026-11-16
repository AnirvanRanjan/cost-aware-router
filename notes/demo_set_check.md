# Demo set check (T45)

*`python scripts/check_demo_set.py` on `artifacts/bundle/router.pkl`.*

```
tau* = 9.352, rule v2, cascade (Qwen answers first)

id        expect    got        delta_hat     lp  query  ->  local answer
easy-01   local     frontier       0.532  -0.43  What is the capital of Australia?  ->   was officially proclaimed as the capital in 1927.   <-- MISMATCH
easy-02   local     frontier       0.063  -0.09  What is 15% of 80?  ->  200}{100} = 12$$  **Answer:** 15% of 80 is **12**.   <-- MISMATCH
easy-03   local     frontier       0.294  -0.19  Which planet is known as the Red Planet? A) V  ->  em.  Therefore, the correct answer is **B) Mars**.   <-- MISMATCH
easy-04   local     frontier       0.146  -0.19  Translate 'good morning' into French.  ->  Bonjour   <-- MISMATCH
easy-05   local     local          0.004  -0.06  A pencil costs $2 and a notebook costs $5. Ho  ->  $5 = \$11$  **Answer:** The total cost is **$11**.
easy-06   local     frontier       0.089  -0.12  What gas do plants absorb from the air for ph  ->  H_{12}O_6 + 6O_2$$  **Answer:** **Carbon dioxide**   <-- MISMATCH
easy-07   local     frontier       0.324  -0.25  Give me a synonym for 'happy'.  ->  ul** or **cheerful** are usually the best choices.   <-- MISMATCH
easy-08   local     frontier       0.024  -0.09  How many minutes are there in 3 hours?  ->  0 \times 3 = 180$$  So, **3 hours = 180 minutes**.   <-- MISMATCH
easy-09   local     frontier       0.405  -0.24  Who wrote 'Romeo and Juliet'?  ->  ally lead to the reconciliation of their families.   <-- MISMATCH
easy-10   local     local         -0.014  -0.08  Is 17 a prime number?  ->  visors: 1 and 17. Therefore, it is a prime number.
hard-01   frontier  local          0.005  -0.07  A store raises a price by 20%, then lowers th  ->    *(Note: This happens because the 20% decrease is   <-- MISMATCH
hard-02   frontier  local          0.005  -0.07  Prove that the square root of 2 is irrational  ->   Step 3: - If $q^2$ is even, then $q$ must be even   <-- MISMATCH
hard-03   frontier  local         -0.063  -0.10  Three friends split a bill. Ana pays twice wh  ->  of payments: $$A + B + C = 114$$ $$(2B) + B + (B -   <-- MISMATCH
hard-04   frontier  local         -0.044  -0.09  In how many ways can 8 people sit around a ro  ->  xt{Together} $$ $$ \text{Valid arrangements} = 504   <-- MISMATCH
hard-05   frontier  local         -0.003  -0.06  A bat and a ball cost $1.10 in total. The bat  ->  on The bat and both balls cost **$1.16** together.   <-- MISMATCH
hard-06   frontier  frontier       0.449  -0.21  Explain why the halting problem is undecidabl  ->  uate $D(D)$.  There are only two possible outcomes
hard-07   frontier  local         -0.063  -0.10  A train leaves at 9:40 at 72 km/h. A second t  ->    So the equation becomes: $$72t = 90t - 37.5$$  R   <-- MISMATCH
hard-08   frontier  local          0.000  -0.22  Which statement about the Treaty of Westphali  ->  various Italian states.  **Correct Answer:** **B**   <-- MISMATCH
hard-09   frontier  local         -0.001  -0.04  Find all real x such that |x - 3| + |x + 1| =  ->   + 1) = 6$$ $$-x + x + 3 + 1 = 6$$ $$4 = 6$$  This   <-- MISMATCH
hard-10   frontier  local          0.004  -0.06  A drug test is 99% sensitive and 95% specific  ->  tives)** $$P(T^+ | D) \cdot P(D) = 0.99 \times 0.0   <-- MISMATCH
chat-01   any       frontier       0.290  -0.32  Suggest three names for a study group that wo  ->  e collective advancement of algorithms and models.
chat-02   any       frontier       0.714  -0.75  Write a two-line poem about a rainy evening i  ->  ills, Where the monsoon whispers through the pine.
chat-03   any       frontier       0.186  -0.33  What are the pros and cons of running a langu  ->  asily run massive models (like Llama 3 405B or GPT
chat-04   any       frontier       0.716  -0.57  Summarise what a model router does in one sen  ->   so everything stays connected and works smoothly.
chat-05   any       frontier       0.546  -0.33  Plan a 30-minute revision session for an oper  ->  wn one function each for the three subsystems.  **

8/25 as planned
```

## What this shows (run 1, 2026-11-16: cascade bundle, Qwen 4-bit MLX on Ani's M1, query sent as typed)

- **Qwen answered all 10 easy questions correctly, but 8 went to the frontier.** The cascade's
  confidence signal for free-form answers is the mean token log-probability of the whole answer.
  Long, predictable LaTeX working scores as "confident" (lp ≈ −0.06 to −0.10); a correct fact
  followed by a paragraph of background prose scores as unsure (Canberra: −0.43). The signal
  measures how predictable the text is, not whether the answer is right.
- **Most "hard" questions are not hard for Qwen3.5-4B**: it got hard-01, 03, 05, 08 and 09 right in
  the visible working. Keeping them local is the router doing its job; `expect: frontier` was wrong.
- **10 of 25 answers hit the 512-token limit** (30–47 s each on the M1), so the live page would show
  cut-off answers.
- chat-04 is a real catch: Qwen explained a *network* router; the cascade escalated it.

The router was trained on GSM8K answers written with the "briefly, plain text, 'The answer is N.'"
prompt, while the server sent the raw query. Fix (same commit): the demo now uses that style for
every query (`PROMPT_TEMPLATES["demo"]`; `--raw-prompt` restores the old behaviour). Run 2 below
checks whether that fixes the easy questions. If it does not, the hard set must be replaced with
questions Qwen actually gets wrong, and the confidence signal should be read from the final
answer line only.
