---
title: "A Cost-Aware Model Router: Certified Savings from Routing Between a Laptop Model and a Frontier Model"
subtitle: "Major Project Report submitted in partial fulfilment of the requirements for the degree of Bachelor of Technology (Hons.) in Computer Science and Engineering"
author:
  - "Anirvan Ranjan (SAP ID 500119330)"
  - "Huwaid Siddiqui (SAP ID 500119722)"
  - "Sarthak Jakhmola (SAP ID 500122482)"
  - "Kartiken Sharma (SAP ID 500123941)"
date: "School of Computer Science, University of Petroleum and Energy Studies, Dehradun — 2026"
lang: en-GB
---

# Candidate's Declaration {.unnumbered}

We hereby declare that the work presented in this report, entitled "A Cost-Aware Model Router:
Certified Savings from Routing Between a Laptop Model and a Frontier Model", in partial fulfilment of
the requirements for the award of the degree of Bachelor of Technology (Hons.) in Computer Science and
Engineering, submitted to the School of Computer Science, University of Petroleum and Energy Studies,
Dehradun, is an authentic record of our work carried out under the guidance of
[guide's name and designation].

The use of generative AI tools is declared at the start of each chapter, in line with the University's
guidelines. The matter presented in this report has not been submitted for the award of any other
degree.

| Name | SAP ID | Signature |
|---|---|---|
| Anirvan Ranjan | 500119330 | |
| Huwaid Siddiqui | 500119722 | |
| Sarthak Jakhmola | 500122482 | |
| Kartiken Sharma | 500123941 | |

# Certificate {.unnumbered}

This is to certify that the above statement made by the candidates is correct to the best of my
knowledge.

[Guide's name], [designation], School of Computer Science, UPES Dehradun

Signature and date:

# Acknowledgement {.unnumbered}

We thank [guide's name] for guidance throughout the project, and the School of Computer Science,
UPES, for its support. We thank the authors of RouterBench, whose released model responses made the
first stage of this work possible, and the open-weight model community whose quantised models let the
system run on a laptop.

# Abstract {.unnumbered}

Frontier language models are accurate but charge for every token, while a small open-weight model can
answer on a laptop at no marginal cost. A router that sends each question to the cheaper model when it
is good enough can cut spending, but only if the loss of quality can be bounded, and published routing
results are usually reported at operating points chosen on the same data they are evaluated on.

This project builds and evaluates a cost-aware router between Qwen3.5-4B, run in 4-bit on an Apple
Silicon laptop, and GPT-4. We audit and re-grade the RouterBench corpus, remove near-duplicates with a
template-centred embedding pass before freezing grouped splits, and generate our own labels for the
small model on an H100 with the prompts and sampling used at serving time. We compare routers that see
only the question text with cascade routers that let the small model answer first and read its
confidence: the agreement of three sampled answers for arithmetic, and the probability of the chosen
option for multiple-choice questions. The decision threshold is chosen on a calibration split with a
Learn-then-Test procedure; because the betting-martingale bound we use depended on the order of the
calibration data, we average its terminal e-values over 50 orderings to obtain a seed-free certificate.

On 4,707 held-out test questions the three-answer cascade keeps 102.8% of GPT-4's quality advantage
over the small model while cutting API spend by 41.3% (95% CI 39.0–43.5%), against 2.1% for the best
question-only router under the same guarantee. The certified router is served live on a MacBook with a
streaming web interface, offline mode and a pre-flight check, and makes the same decisions as the
offline evaluation on 200 of 200 replayed questions.

**Keywords:** model routing, cascades, LLM cost, conformal risk control, Learn-then-Test, e-values,
on-device inference
