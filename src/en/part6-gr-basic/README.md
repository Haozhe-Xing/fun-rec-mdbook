<div class="part-banner">
<h2 style="color: white; margin: 0 0 8px 0;">📜 Part 6: Foundations of the Generative Recommendation Paradigm</h2>
<p style="color: rgba(255,255,255,0.85); margin: 0;">The paradigm shift from "discriminative scoring" to "generative sequences" — the underlying foundation of architectures, LLM training pipelines, and semantic IDs.</p>
<p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 8px 0 0 0;">📚 4 sections · ⏱️ Estimated 2–3 weeks · 🎯 Target: build a complete foundational understanding of generative recommendation</p>
</div>

Generative recommendation is shifting recommender systems from the discriminative paradigm of "scoring and ranking a candidate set" to the generative paradigm of "directly generating recommendation results." This is more than an upgrade in model architecture — it is a fundamental rethinking of the modeling philosophy. As the starting point of the "generative recommendation mainline," this part systematically builds four pillars: from **paradigm motivation** (why generative is needed), to **architectural foundations** (which models implement generation), then to the **LLM modeling pipeline** (how to train such models), and finally to **Tokenizer technology** (how recommendation data adapts to the interfaces of generative models). The four pillars build on one another, together forming the complete foundation you need for the later chapters — from scaling architectures to end-to-end generation, from thinking recommenders to diffusion models.

---

## What This Part Covers

| Section | Topic | The Big Idea |
|---------|-------|--------------|
| **6.1** | Introduction to the Generative Recommendation Paradigm | Discriminative vs. generative: local scoring decisions vs. global probability modeling; three inherent limitations driving the paradigm shift |
| **6.2** | Foundations of Generative Architectures | Transformer self-attention / positional encoding / two architectural paradigms / causal masking, complemented by Diffusion |
| **6.3** | LLM Foundations | The pretraining–instruction tuning–preference alignment three-stage paradigm, and its mapping and challenges for generative recommendation |
| **6.4** | Codebook Quantization and Semantic IDs | Sparse ID / text / semantic ID paradigms; VQ-VAE, RQ-VAE, RQ-Kmeans, and RQ-OPQ industrial solutions |

---

## What You'll Be Able to Do After This Part

- 🟢 **Distinguish** the core formulas and "questions asked" of discriminative vs. generative models, and list the three inherent limitations of the discriminative paradigm
- 🟢 **Explain** the Q/K/V mechanism of self-attention, positional encoding (including time-aware variants), and the role of causal masking
- 🟡 **Compare** the strengths, weaknesses, and applicable scenarios of Encoder-Decoder vs. Decoder-Only architectures
- 🟡 **Restate** the three-stage LLM paradigm (pretraining / SFT / RLHF / DPO) and map it to recommendation scenarios
- 🔴 **Derive** the three VQ-VAE losses and RQ-VAE residual quantization, and explain the three key values of semantic IDs
- 🔴 Complete 18+ tiered practice problems across 4 chapters, consolidating the full chain from paradigm to semantic ID

---

## Core Concepts

| Concept | Section | Relevance |
|----------|---------|-----------|
| Discriminative / generative paradigms | 6.1 | The master switch of the book's generative mainline |
| Self-attention / positional encoding / causal masking | 6.2 | Core mechanisms of Transformer generative architectures |
| Encoder-Decoder vs. Decoder-Only | 6.2 | The fundamental trade-off in generative architecture selection |
| Diffusion models | 6.2 | A generation mechanism complementary to Transformer |
| Pretraining / SFT / RLHF / DPO | 6.3 | The methodological framework of LLM training paradigms |
| Item tokenization / semantic IDs | 6.4 | The bridge connecting recommendation data to generative models |
| VQ-VAE / RQ-VAE / RQ-Kmeans / RQ-OPQ | 6.4 | The technology spectrum for discretizing semantic IDs |

---

## Prerequisites

- You have finished **Part 1** (the two paradigms in [1.1](./../part1-introduction/recommender-system-basics.md), the technology map in [1.2](./../part1-introduction/book-overview.md))
- You have finished **Part 2** (the inner product and vector-space intuition of the [two-tower](./../part2-retrieval/two-tower.md) model in 2.3)
- Basic linear algebra (matrices, inner products), probability (softmax, KL divergence), and general neural network knowledge

> This part covers frontier material with recent terminology (generative retrieval, semantic IDs, RQ-VAE, Scaling Laws, etc.). Every term is given in both Chinese and English on first appearance, with mental models and diagrams.

---

## Tips for This Part

1. **Look at the motivation before the technology.** The "why generative" discussion in 6.1 is the key to everything that follows — each technology responds to a specific discriminative limitation.
2. **Grasp the "unified architecture" mainline.** From the Transformer in 6.2, to the LLM in 6.3, to semantic IDs in 6.4, "unified, scalable, end-to-end" is the design philosophy throughout.
3. **Get hands-on with the visualizations.** The interactive HTML and SVG demos in each chapter are worth walking through yourself, grounding abstract formulas in intuition about "how sequences are generated" and "how vectors are quantized."
4. **Think against the discriminative paradigm.** Whenever you meet a generative component, first ask: "which discriminative limitation does it solve?"

---

Let's dive in! 🚀
