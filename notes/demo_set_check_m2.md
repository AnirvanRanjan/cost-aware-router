# Demo set check (T45)

*`python scripts/check_demo_set.py` on `artifacts/bundle/router.pkl`.*

```
tau* = 8.219, rule v2, cascade (Qwen answers first), cascade_t61_3sample

id        expect    got        delta_hat  signal                            query  ->  local answer
easy-01   local     local         -0.009  arithmetic: 3/3 answers agree     Three blue chips are in a jar which is 1  ->  4. 30 - 3 - 15 = 12 green chips. The answer is 12.
easy-02   local     local         -0.016  arithmetic: 3/3 answers agree     Joel is 5 years old and his dad is 32 ye  ->  Joel's current age: 5 + 22 = 27. The answer is 27.
easy-03   local     local         -0.006  arithmetic: 3/3 answers agree     Emily bought a shirt and a coat for $600  ->   the shirt cost: 450 / 3 = 150. The answer is 150.
easy-04   local     local         -0.003  arithmetic: 3/3 answers agree     A man owned 1/2 of a lot. He sold 1/10 o  ->  final result: 460 * 20 = 9200. The answer is 9200.
easy-06   local     local         -0.002  arithmetic: 3/3 answers agree     There are 11 boys and 13 girls in a clas  ->  Multiply 0.52 by 100 to get 52%. The answer is 52.
easy-08   local     local          0.009  multiple choice: option probabili The term "Cold War" refers to the A) com  ->  Union to gain political hegemony in world affairs.
easy-09   local     local          0.009  multiple choice: option probabili During President Dwight Eisenhower's adm  ->  the construction of the interstate highway system.
easy-10   local     local          0.009  multiple choice: option probabili What was the 'Domino Theory'? A) The ide  ->  ould inexorably lead to the fall of its neighbors.
easy-11   local     local         -0.010  arithmetic: 3/3 answers agree     Emma bought 3 dozens of macarons in addi  ->  w many were eaten: 46 - 15 = 31. The answer is 31.
hard-02   frontier  frontier       0.171  arithmetic: 2/3 answers agree     In 6 months Bella and Bob will be celebr  ->   4. 24 minus 6 equals 18 months. The answer is 18.
hard-03   frontier  frontier       0.110  arithmetic: 2/3 answers agree     Tom fills a 250 pound sandbag 80% full.   ->  terial: 200 + 280 = 480 pounds. The answer is 480.
hard-04   frontier  frontier       0.135  arithmetic: 2/3 answers agree     Tyrone empties his piggy bank and finds   ->  $12.65 - $12.65 + $0.35 = $13.00 The answer is 13.
hard-05   frontier  frontier       0.009  arithmetic: 3/3 answers agree     Bob runs 6 miles per hour. His friend Ji  ->  5 miles per minute = 20 minutes. The answer is 20.
hard-06   frontier  frontier       0.030  multiple choice: option probabili How many periods are there in an ice hoc  ->  The answer is D) Two.
hard-07   frontier  frontier       0.189  multiple choice: option probabili What color is cartoon character Marge Si  ->  The answer is D) yellow.
hard-08   frontier  frontier       0.263  multiple choice: option probabili What is the quotient for the expression   ->  The answer is D) 578.
hard-09   frontier  frontier       0.177  multiple choice: option probabili Which of these is not found in a Snicker  ->  The answer is D) caramel.
free-01   any       local         -0.005  free-form: 3/3 answers agree [unc What is the capital of Australia?  ->  ecame the capital in 1927. The answer is Canberra.
free-02   any       local         -0.005  free-form: 3/3 answers agree [unc Who wrote 'Romeo and Juliet'?  ->  ish literature. The answer is William Shakespeare.
free-03   any       local         -0.005  free-form: 3/3 answers agree [unc Translate 'good morning' into French.  ->  "bonjour et bonne matinée." The answer is bonjour.
free-04   any       frontier       0.130  free-form: 2/3 answers agree [unc Give me a synonym for 'happy'.  ->  " "delighted," or "pleased." The answer is joyful.
chat-01   any       frontier       0.455  free-form: 1/3 answers agree [unc Suggest three names for a study group th  ->   Nexus, Algorithm Allies, and The Prompt Pioneers.
chat-02   any       frontier       0.455  free-form: 1/3 answers agree [unc Write a two-line poem about a rainy even  ->  he valley grows quiet under a silver, weeping sky.
chat-03   any       frontier       0.455  free-form: 1/3 answers agree [unc Summarise what a model router does in on  ->  nages and organizes your home's Wi-Fi connections.

24/24 as planned
```
