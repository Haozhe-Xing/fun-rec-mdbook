<div class="part-banner">
<h2 style="color: white; margin: 0 0 8px 0;">📦 Part 8: End-to-End Generative Applications</h2>
<p style="color: rgba(255,255,255,0.85); margin: 0;">Unify recommendation, search, and advertising into end-to-end generative systems that go directly from "user input → generated results."</p>
<p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 8px 0 0 0;">📚 3 sections · ⏱️ Estimated 2 weeks · 🎯 Target: master the architectures and alignment methods of industrial-grade end-to-end generative recommendation/search/advertising</p>
</div>

For over a decade, recommendation, search, and advertising have almost all been built on the **Multi-stage Cascading Architecture (MCA)**: retrieval, pre-ranking, ranking, re-ranking... data is filtered layer by layer like a funnel. This design historically balanced efficiency and complexity, but as data scales exploded and user-experience expectations rose, its structural flaws became increasingly apparent — conflicting objectives, information loss, and fragmented computation.

This part stops treating each sub-task in isolation. Instead, along the **generative paradigm** thread, we look at how industry uses a unified neural network to **generate final results directly from user input**, completely overturning the "cascading funnel." We focus on real deployments in three core business scenarios: Kuaishou's **OneRec** (end-to-end generative recommendation), e-commerce search's **OneSug + OneSearch** (end-to-end generative search), and online advertising's **EGA + GPR** (end-to-end generative advertising).

![Three structural dilemmas of the traditional cascading architecture](../images/part8-mca-pain.svg)

> 💡 **Key Insight:** The end-to-end practices across the three scenarios share one common thread — **semantic IDs are the bridge connecting generative models to business data; the Encoder-Decoder is the workhorse architecture for fusing context; and reinforcement learning is the key tool for aligning the generation process with online business objectives**.

---

## What This Part Covers

| Section | Topic | The Big Idea |
|---------|-------|--------------|
| **8.1** | End-to-End Generative Recommendation | OneRec-V1/V2 use semantic IDs + Encoder-Decoder + reinforcement learning to redefine recommendation as a generation task: "user context → semantic ID sequence" |
| **8.2** | End-to-End Generative Search | OneSug handles query completion and OneSearch handles product retrieval, covering the full e-commerce search pipeline with a unified generative architecture |
| **8.3** | End-to-End Generative Advertising | EGA embeds the auction mechanism into generation and GPR uses pre-training to unify ultra-long heterogeneous sequences across scenarios, deeply integrating mechanism constraints with the generative model |

---

## What You'll Be Able to Do After This Part

- 🟢 **Explain** the three structural flaws of the traditional MCA and how they gave rise to the end-to-end generative paradigm
- 🟢 **Describe** how semantic IDs compress hundreds of millions of items into a finite vocabulary, making generative recommendation mathematically feasible
- 🟡 **Contrast** the fundamental difference in compute allocation between OneRec-V1's Encoder-Decoder and V2's Lazy Decoder-Only
- 🟡 **Distinguish** the essential difference between the search scenario's "relevance first, personalization second" and the recommendation scenario's optimization objective
- 🔴 **Explain** how EGA embeds incentive compatibility (IC) and individual rationality (IR) constraints into the generation process
- 🔴 **Outline** how GPR's heterogeneous hierarchical decoder and value-guided Beam Search solve the cross-scenario and ultra-long-sequence challenges
- 🏆 Complete the tiered practice problems in each section, working through semantic ID encoding, reward modeling, and constrained decoding by hand

---

## Core Concepts

| Concept | Section | Relevance |
|----------|---------|-----------|
| Multi-stage Cascading Architecture (MCA) | 8.1 | The traditional industrial skeleton overturned by the end-to-end paradigm |
| Semantic ID | 8.1 | The core bridge connecting generative models to discrete items/products/ads |
| Encoder-Decoder generative architecture | 8.1 / 8.2 / 8.3 | The workhorse structure for fusing context and autoregressively generating sequences |
| Lazy Decoder-Only | 8.1 | Concentrates compute on target tokens, cutting decoding cost by 94% |
| RL alignment (ECPO/GBPO/DPO) | 8.1 / 8.2 / 8.3 | Aligns the generation process with online multi-objective business signals |
| Incentive compatibility (IC) and individual rationality (IR) | 8.3 | Economic constraints from mechanism design in the advertising scenario |
| Value-guided Trie Beam Search | 8.3 | Embeds constraints into decoding and improves inference efficiency |

---

## Prerequisites

- You have finished **1.1** (the two paradigms and the motivation for end-to-end generation) and **2.x** (two-tower models and a first look at semantic IDs)
- Familiarity with the basic Transformer structure (self-attention, cross-attention, encoder/decoder)
- A preliminary understanding of reinforcement learning basics (policy, reward, advantage) — this part develops them gradually through case studies

> This part is the industrial-deployment chapter of the generative thread and leans Advanced — there are many formulas, but remember: **better to lean on the figures than to wrestle with the formulas**. The point is to understand the "why" behind each architectural choice.

---

## Tips for This Part

1. **Follow the hidden thread of "semantic IDs."** All three sections — recommendation, search, advertising — repeatedly solve the same problem: how to turn discrete business objects into token sequences a generative model can "speak out." Understand semantic IDs first and the rest follows naturally.
2. **Read the three sections comparatively.** Each follows "MCA pain points → generative solution → alignment with online objectives," but the business constraints differ: recommendation pursues interests, search protects relevance first, and advertising must additionally satisfy economic mechanisms.
3. **Pay attention to the engineering trade-offs between compute and constraints.** OneRec-V2's Lazy architecture and GPR's Trie-constrained decoding are both classic examples of trading architecture for efficiency and compliance.

---

Let's dive in! 🚀
