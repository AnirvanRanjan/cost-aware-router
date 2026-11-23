# Chapter 8 — Conclusion

## 8.1 Summary of Findings and Architectural Insights

This thesis designed, evaluated, and deployed an end-to-end cost-aware model routing framework that dynamically arbitrates between an on-device lightweight language model (Qwen3.5-4B executing locally on consumer Apple Silicon hardware) and a cloud-based frontier model (GPT-4 accessed via API). Our central engineering objective was to determine whether significant API expenditure can be eliminated while mathematically guaranteeing that at least 95% of the frontier model’s performance advantage ($\text{PGR} \ge 0.95$) is preserved under rigorous finite-sample statistical certificates.

Three foundational insights emerge from this work:

1. **Empirical Data Hygiene Precedes Predictive Capability:** Public routing benchmarks contain severe methodological distortions. In raw RouterBench data, mathematical evaluation scripts systematically penalized valid numeric answers, certain open-weight architectures collapsed to chance-level guessing due to prompt formatting incompatibilities, and identical problem templates leaked across evaluation partitions. Implementing robust numeric extraction, recovering consensus gold targets for multiple-choice benchmarks, and executing template-centered embedding deduplication proved indispensable to building a truthful, leak-free experimental foundation.
2. **Text Features Alone Cannot Forecast Edge Model Competence:** Pre-execution query routers that inspect only prompt syntax and semantics achieve negligible certified utility on modern edge architectures, yielding only 0.5% to 2.1% certified savings on the Qwen3.5-4B / GPT-4 pair. As open-weight models improve and headroom contracts, intrinsic problem difficulty remains largely unobservable from input text alone. In contrast, evaluating the model’s internal generation dynamics—specifically multi-sample consensus and top-choice output probability—provides direct access to epistemic uncertainty, unlocking **41.3% [39.0%, 43.5%] certified API spend reduction** at a verified test PGR of **102.8% [99.1%, 107.0%]**.
3. **Statistical Guarantees Impose a Measurable, Necessary Premium:** Retrospectively scanning empirical test curves suggests that the cascade router can achieve a 78.9% cost reduction (a 4.7× improvement). However, locking operational thresholds prior to deployment via distribution-free Learn-then-Test and seed-free betting martingales bounds the certified cost reduction at **41.3% (a 1.7× improvement)**. The divergence between these figures represents the mathematical cost of eliminating deployment risk. We demonstrated that standard asymptotic bootstrap approximations are dangerous—failing three times more frequently than promised on low-headroom tasks—whereas terminal betting martingales provide exact, distribution-free safety.

---

## 8.2 Practical System Deployment

The complete routing pipeline was instantiated as a fully operational, interactive edge runtime on a MacBook Air (Apple M2, 16 GB unified RAM):
- **Local Edge Inference:** Qwen3.5-4B executes in 4-bit MLX quantization, maintaining parity with the 16-bit bfloat reference model across evaluation benchmarks while generating at 28.5 tokens/second.
- **Routing Engine Latency:** The feature extraction and decision pipeline executes in about **6 ms (99th percentile)** (`notes/replay_report_d15.md`, `notes/demo_perf.md`), introducing imperceptible computational latency.
- **Full Hardware Amortization:** Counting the laptop's purchase price and power for every local answer under Cost Model M2, the system keeps a net **28.4% cost reduction**, comfortably surpassing the economic break-even threshold against hyper-optimized commercial APIs.

---

## 8.3 Concluding Perspective

A machine learning router is only as trustworthy as the mathematical protocol used to calibrate its decision boundary. Optimistic Pareto frontiers published without statistical risk control offer false comfort to system designers. By uniting robust data curation, generation-level cascade diagnostics, and distribution-free risk control, this thesis demonstrates that cost-effective, multi-tier LLM inference can be deployed with rigorous mathematical safety. Moving forward, transitioning these concentration bounds from static benchmark calibration sets to non-stationary, streaming production traffic represents the critical next step toward truly adaptive, self-calibrating artificial intelligence infrastructure.
