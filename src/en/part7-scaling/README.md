<div class="part-banner">
<h1 style="color: white; margin: 0 0 8px 0;" font-size: 1.7rem;>📗 Part 7: Scaling — Generative Ranking Models</h1>
<p style="color: rgba(255,255,255,0.85); margin: 0;">Starting from HSTU, the first validation of the Scaling Law in recommender systems, trace how industry players polished the generative paradigm into deployable, scalable, hardware-efficient ranking engines.</p>
<p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 8px 0 0 0;">📚 5 sections · ⏱️ Estimated 3 weeks · 🎯 Target: understand the engineering that makes "bigger model = better recommendations" work in the ranking stage</p>
</div>

Traditional deep learning recommendation models (DLRMs) have long been the "exception" to deep learning **Scaling Laws**: throw in more parameters and more data, and metrics plateau almost immediately. This part follows the thread from Meta's HSTU — the first validation of the Scaling Law in recommender systems — and then unpacks the follow-up work from Xiaohongshu, Meituan, Alibaba, and ByteDance, so you can see how industry turned "generative ranking" from paper numbers into a reality serving billions of users.

---

## What This Part Covers

| Section | Topic | The Big Idea |
|------|-------|--------------|
| **7.1** | HSTU: The First Exploration of the Scaling Law | Treat user behavior history as a "language"; a unified sequence + autoregressive training + an efficient architecture prove for the first time that recommendation can scale |
| **7.2** | The Overall Generative Ranking Paradigm (GenRank) | The autoregressive mechanism is what is essential; Action-Oriented sequence organization halves sequence length and speeds up training by ~79% |
| **7.3** | MTGR: Hybrid Paradigm Modeling | Use a "generative architecture + discriminative objective" to retain cross features, solving the missing-feature problem of pure generative approaches |
| **7.4** | RankMixer: Hardware Efficiency Optimization | Derive the architecture from GPU hardware characteristics; Token Mixing / Per-Token FFN / Sparse MoE push MFU from 4% to 45% |
| **7.5** | OneTrans: A Unified Transformer | A single Transformer backbone does both sequence modeling and feature interaction, and reuses LLM system optimizations such as KV Caching |

---

## What You'll Be Able to Do After This Part

- 🟢 **Explain** why traditional DLRMs struggle to scale, and how HSTU broke through the bottleneck with user-level sequence modeling
- 🟢 **Distinguish** the respective contributions of the "autoregressive mechanism" versus "training paradigm details" within the generative paradigm (see Section 7.2)
- 🟡 **Explain** how MTGR's hybrid paradigm stays compatible with traditional cross features while retaining efficiency (see Section 7.3)
- 🟡 **Analyze** how RankMixer's hardware-aware design raised MFU from 4% to 45% (see Section 7.4)
- 🔴 **Recount** how OneTrans achieves end-to-end scalability with a unified Transformer + Pyramid Stack + Cross-Request KV Caching (see Section 7.5)
- 🔴 **Compare** the five works' different trade-offs on the "unification vs efficiency vs compatibility" triangle

---

## Core Concepts

| Concept | Section | Relevance |
|----------|---------|-----------|
| Behavior sequence modeling (user-level) | 7.1 | Treating recommendation as "language" is the prerequisite for the Scaling Law |
| Pointwise Aggregation / relative time bias | 7.1 | HSTU's three architectural innovations for recommendation |
| The autoregressive essence at the core of generative models | 7.2 | The dividing line between "means" and "ends" |
| Action-Oriented organization | 7.2 | The key trick that halves sequence length |
| Hybrid paradigm (generative architecture + discriminative objective) | 7.3 | A new way to stay compatible with cross features |
| Group LayerNorm / Dynamic Masking | 7.3 | Let heterogeneous tokens coexist in one Transformer |
| Token Mixing / Per-Token FFN / Sparse MoE | 7.4 | Hardware-aware restructuring of the recommendation computation graph |
| Unified Tokenization / Mixed Parameterization / Pyramid Stack | 7.5 | Deep fusion of sequences and features within a single backbone |

---

## Prerequisites

- Having read [Part 1 Introduction](./../part1-introduction/) and the discriminative paradigm foundations in [Part 3 Ranking](./../part3-ranking/feature-crossing.md)
- Familiarity with Transformer basics: self-attention, LayerNorm, residual connections
- Knowing what the Scaling Law means in NLP/CV (performance improves as a power law with compute/data/parameters)

> This part is the second stop in the second half of the "generative recommendation storyline". If you have not yet read [Part 6 Generative Paradigm Fundamentals](./../part6-gr-basic/), we recommend building up the background on generative retrieval and semantic IDs (RQ-VAE) first.

---

## Tips for This Part

1. **Separate "means" from "ends".** The generative architecture (Transformer + sequences) is a powerful representational tool, but it does not have to serve a generative objective — that is exactly the insight of MTGR in 7.3.
2. **Every work answers the same question**: how can a recommendation model truly enjoy the dividends of the Scaling Law? Keep cross-checking from the four angles of architecture, training, features, and hardware.
3. **Lean on the figures rather than memorizing formulas.** This part is on the frontier; the priority is understanding "why it was designed this way" rather than deriving every formula precisely.

---

Let's dive in! 🚀
