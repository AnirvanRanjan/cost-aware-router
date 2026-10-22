# Dedup report

_`scripts/02_dedup.py`, dedup-1.1. Embedding pass: yes._

- Near-duplicate pairs found: **1,883**
- Groups merged away: **1,061**; rows now sharing a group: **1,899**; largest group: **15**
- Rows whose split changed: **19,122**
- Splits: {'train': 20867, 'cal': 6957, 'test': 6954}

## Pairs by dataset

| dataset / method | pairs |
|---|---|
| arc-challenge/embedding | 6 |
| arc-challenge/minhash | 2 |
| gsm8k/embedding | 32 |
| gsm8k/minhash | 1 |
| hellaswag/embedding | 194 |
| mbpp/embedding | 7 |
| mbpp/minhash | 7 |
| mmlu/embedding | 799 |
| mmlu/minhash | 466 |
| winogrande/embedding | 260 |
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
- **0.93** (embedding) `arc-challenge.test.405` / `arc-challenge.test.76`
  - Under which circumstance would an adaptation be useful that allows tadpoles to go through a shortened period of metamorphosis? A) when tadpoles need to blend in
  - Under which circumstance would an adaptation that shortens the time of a tadpole's metamorphosis be most useful? A) when there is a limited amount of water in t
- **0.98** (embedding) `arc-challenge.test.484` / `arc-challenge.test.819`
  - When you are getting up to go to school in Louisiana, a student on the other side of Earth is getting ready for bed. What is the reason for this? A) Earth rotat
  - You are getting up to go to school in Louisiana; a student on the other side of Earth is getting ready for bed. What is the reason for this? A) The Moon revolve

### gsm8k

- **0.90** (embedding) `grade-school-math.dev.2483` / `grade-school-math.dev.6691`
  - Josie received $50 as a gift. She plans to buy two cassette tapes that cost $9 each and a headphone set that costs $25. How much money will she have left?
  - Amanda received $50 as a gift. She plans to buy two cassette tapes that cost $9 each and a headphone set that costs $25. How much money will she have left?
- **0.91** (embedding) `grade-school-math.dev.2441` / `grade-school-math.dev.4338`
  - There are 30 fish in the tank. One third of them are blue, and half of the blue fish have spots. How many fish in the tank are blue, spotted fish?
  - There are many fish in the tank. One third of them are blue, and half of the blue fish have spots. If there are 10 blue, spotted fish, how many fish are there i
- **0.91** (embedding) `grade-school-math.dev.4138` / `grade-school-math.dev.7405`
  - Agatha has some money to spend on a new bike. She spends $15 on the frame, and $25 on the front wheel. If she has $20 left to spend on a seat and handlebar tape
  - Agatha has $60 to spend on a new bike. She spends $15 on the frame, and $25 on the front wheel. What does she have left, in dollars, to spend on a seat and hand
- **0.91** (embedding) `grade-school-math.dev.1946` / `grade-school-math.dev.856`
  - Randy has $200 in his piggy bank. He spends 2 dollars every time he goes to the store. He makes 4 trips to the store every month. How much money, in dollars, wi
  - Randy has some money in his piggy bank. He spends 2 dollars every time he goes to the store. He makes 4 trips to the store every month. If he had $104 dollars l

### hellaswag

- **0.90** (embedding) `hellaswag.val.1672` / `hellaswag.val.3160`
  - A man is standing with a fishing pole. He is fishing through a hole in the ice. he A) pulls his fishing line out of the hole and into the water. B) chews on an 
  - He is attempting to catch a fish through a hole in the ice. He reels in a fish, and removes the hook. he A) then presents the fish and reel in a hole. B) contin
- **0.90** (embedding) `hellaswag.val.1581` / `hellaswag.val.1582`
  - A woman is washing her face in a sink. she A) removes her contacts and puts them in her eyes to show us her concealer. B) starts dancing happily. C) looks in a 
  - A woman is washing her face in a sink. She looks in a mirror in front of her. she A) gets all soapy and looks all confused. B) continues wiping her face. C) hol
- **0.90** (embedding) `hellaswag.val.2275` / `hellaswag.val.522`
  - A man is trying to lift weights in a gym. He lifts the barbell up to his chest. he A) lifts it over his head, then back down. B) then lifts it over his head bef
  - A man lifts a barbell with weights from the ground up in a gym with other people nearby. The man lifts the barbell up above his head. the man A) lifts the barbe
- **0.90** (embedding) `hellaswag.val.2374` / `hellaswag.val.2375`
  - A green title page appears, with words showing that the video will be of a tenor saxophone player as well as the website to receive to lessons. a black male A) 
  - A green title page appears, with words showing that the video will be of a tenor saxophone player as well as the website to receive to lessons. A black male the

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

- **0.80** (minhash) `mmlu-high-school-us-history.val.177` / `mmlu-high-school-us-history.val.42`
  - Please answer with the letter of the correct answer.  This question refers to the following information. "I observe the great and wonderful mistake, both our ow
  - Please answer with the letter of the correct answer.  This question refers to the following information. "I observe the great and wonderful mistake, both our ow
- **0.80** (minhash) `mmlu-abstract-algebra.val.71` / `mmlu-abstract-algebra.val.8`
  - Please answer with the letter of the correct answer.  Find the degree for the given field extension Q(sqrt(2)*sqrt(3)) over Q. A) 2 B) 0 C) 6 D) 4 Print only a 
  - Please answer with the letter of the correct answer.  Find the degree for the given field extension Q(sqrt(2) + sqrt(3)) over Q. A) 0 B) 2 C) 6 D) 4 Print only 
- **0.80** (minhash) `mmlu-high-school-world-history.val.21` / `mmlu-high-school-world-history.val.4`
  - Please answer with the letter of the correct answer.  This question refers to the following information. Bonesteel's prime consideration was to establish a surr
  - Please answer with the letter of the correct answer.  This question refers to the following information. Bonesteel's prime consideration was to establish a surr
- **0.80** (minhash) `mmlu-high-school-physics.val.36` / `mmlu-high-school-physics.val.66`
  - Please answer with the letter of the correct answer.  Traveling at an initial speed of 1.5 × 10^6 m/s, a proton enters a region of constant magnetic field of ma
  - Please answer with the letter of the correct answer.  Traveling at an initial speed of 1.5 × 10^6 m/s, a proton enters a region of constant magnetic field, B, o

### winogrande

- **0.80** (minhash) `winogrande.dev.1003` / `winogrande.dev.1004`
  - Jack wanted to start a game, but couldn't decide between baseball or basketball. The _ game would needed fewer of friends. A) basketball B) baseball Print only 
  - Jack wanted to start a game, but couldn't decide between baseball or basketball. The _ game would call for more friends. A) basketball B) baseball Print only a 
- **0.80** (minhash) `winogrande.dev.651` / `winogrande.dev.652`
  - Natalie committed a bunch of good deeds in Samantha's name, because _ wanted all the credit. A) Natalie B) Samantha Print only a single choice  from "A" or "B" 
  - Natalie committed a bunch of good deeds in Samantha's name, because _ didn't want the credit. A) Natalie B) Samantha Print only a single choice  from "A" or "B"
- **0.81** (minhash) `winogrande.dev.1167` / `winogrande.dev.1168`
  - The pharmacy offered a product that could cure any disease, made of a new chemical and container, but the _ was not FDA approved. A) chemical B) container Print
  - The pharmacy offered a product that could cure any disease, made of a new chemical and container, but the _ could hold few drugs. A) chemical B) container Print
- **0.81** (minhash) `winogrande.dev.1233` / `winogrande.dev.1234`
  - The girl broke Leslie's heart but not Kenneth's because _ was never in love with her. A) Leslie B) Kenneth Print only a single choice  from "A" or "B" without e
  - The girl broke Leslie's heart but not Kenneth's because _ used to be in love with her. A) Leslie B) Kenneth Print only a single choice  from "A" or "B" without 
