# Dedup report

_`scripts/02_dedup.py`, dedup-1.0. Embedding pass: not yet (run again after T12)._

- Near-duplicate pairs found: **585**
- Groups merged away: **421**; rows now sharing a group: **716**; largest group: **6**
- Rows whose split changed: **11,266**
- Splits: {'train': 20866, 'cal': 6958, 'test': 6954}

## Pairs by dataset

| dataset / method | pairs |
|---|---|
| arc-challenge/minhash | 2 |
| gsm8k/minhash | 1 |
| mbpp/minhash | 7 |
| mmlu/minhash | 466 |
| winogrande/minhash | 109 |

## How to read this

Pairs are grouped so both items land in the same split. On RouterBench, almost all MMLU pairs are
different questions on the same reading passage, and almost all Winogrande pairs are the dataset's
deliberate twin sentences (one word changed, opposite answer). Neither is a copy, but a router that
saw one in training would have an unfair head start on the other, so grouping them is correct.

## Examples (lowest similarity first: the borderline calls worth checking)

### arc-challenge

- **0.80** (minhash) `arc-challenge.test.484` / `arc-challenge.test.819`
  - When you are getting up to go to school in Louisiana, a student on the other side of Earth is getting ready for bed. What is the reason for this? A) Earth rotat
  - You are getting up to go to school in Louisiana; a student on the other side of Earth is getting ready for bed. What is the reason for this? A) The Moon revolve
- **0.83** (minhash) `arc-challenge.test.340` / `arc-challenge.test.980`
  - El Cajon Pass in California is becoming higher than the land around it at a rate of nearly 1 centimeter each year. Which statement is true about the area? A) Er
  - El Cajon Pass in California is becoming higher than the land around it at a rate of nearly one centimeter each year. Which statement is true about the area? A) 

### gsm8k

- **0.96** (minhash) `grade-school-math.dev.2483` / `grade-school-math.dev.6691`
  - Josie received $50 as a gift. She plans to buy two cassette tapes that cost $9 each and a headphone set that costs $25. How much money will she have left?
  - Amanda received $50 as a gift. She plans to buy two cassette tapes that cost $9 each and a headphone set that costs $25. How much money will she have left?

### mbpp

- **0.80** (minhash) `mbpp.dev.271` / `mbpp.dev.44`
  - Write a python function to find the length of the longest sublists.
  - Write a python function to find the length of the longest word.
- **0.80** (minhash) `mbpp.dev.281` / `mbpp.dev.39`
  - Write a function to find the surface area of a cylinder.
  - Write a function to find the surface area of a sphere.
- **0.82** (minhash) `mbpp.dev.222` / `mbpp.dev.406`
  - Write a function that matches a string that has an a followed by one or more b's.
  - Write a function that matches a string that has an a followed by three 'b'.
- **0.82** (minhash) `mbpp.dev.406` / `mbpp.dev.412`
  - Write a function that matches a string that has an a followed by three 'b'.
  - Write a function that matches a string that has an 'a' followed by anything, ending in 'b'.

### mmlu

- **0.80** (minhash) `mmlu-abstract-algebra.val.71` / `mmlu-abstract-algebra.val.8`
  - Please answer with the letter of the correct answer.  Find the degree for the given field extension Q(sqrt(2)*sqrt(3)) over Q. A) 2 B) 0 C) 6 D) 4 Print only a 
  - Please answer with the letter of the correct answer.  Find the degree for the given field extension Q(sqrt(2) + sqrt(3)) over Q. A) 0 B) 2 C) 6 D) 4 Print only 
- **0.80** (minhash) `mmlu-high-school-european-history.val.6` / `mmlu-high-school-european-history.val.92`
  - Please answer with the letter of the correct answer.  This question refers to the following information. Florence is more beautiful and five hundred forty years
  - Please answer with the letter of the correct answer.  This question refers to the following information. Florence is more beautiful and five hundred forty years
- **0.80** (minhash) `mmlu-high-school-physics.val.36` / `mmlu-high-school-physics.val.66`
  - Please answer with the letter of the correct answer.  Traveling at an initial speed of 1.5 × 10^6 m/s, a proton enters a region of constant magnetic field of ma
  - Please answer with the letter of the correct answer.  Traveling at an initial speed of 1.5 × 10^6 m/s, a proton enters a region of constant magnetic field, B, o
- **0.80** (minhash) `mmlu-high-school-us-history.val.177` / `mmlu-high-school-us-history.val.42`
  - Please answer with the letter of the correct answer.  This question refers to the following information. "I observe the great and wonderful mistake, both our ow
  - Please answer with the letter of the correct answer.  This question refers to the following information. "I observe the great and wonderful mistake, both our ow

### winogrande

- **0.80** (minhash) `winogrande.dev.651` / `winogrande.dev.652`
  - Natalie committed a bunch of good deeds in Samantha's name, because _ wanted all the credit. A) Natalie B) Samantha Print only a single choice  from "A" or "B" 
  - Natalie committed a bunch of good deeds in Samantha's name, because _ didn't want the credit. A) Natalie B) Samantha Print only a single choice  from "A" or "B"
- **0.80** (minhash) `winogrande.dev.1003` / `winogrande.dev.1004`
  - Jack wanted to start a game, but couldn't decide between baseball or basketball. The _ game would needed fewer of friends. A) basketball B) baseball Print only 
  - Jack wanted to start a game, but couldn't decide between baseball or basketball. The _ game would call for more friends. A) basketball B) baseball Print only a 
- **0.81** (minhash) `winogrande.dev.1167` / `winogrande.dev.1168`
  - The pharmacy offered a product that could cure any disease, made of a new chemical and container, but the _ was not FDA approved. A) chemical B) container Print
  - The pharmacy offered a product that could cure any disease, made of a new chemical and container, but the _ could hold few drugs. A) chemical B) container Print
- **0.81** (minhash) `winogrande.dev.730` / `winogrande.dev.731`
  - The brutal boxing match was suitable for Christopher but not Samuel because _ did damage to his opponent. A) Christopher B) Samuel Print only a single choice  f
  - The brutal boxing match was suitable for Christopher but not Samuel because _ did not damage his opponent. A) Christopher B) Samuel Print only a single choice  
