<div class="part-banner">
<h2 style="color: white; margin: 0 0 8px 0;">📘 Part 3: Precise Preference Prediction (Ranking)</h2>
<p style="color: rgba(255,255,255,0.85); margin: 0;">Dive into the main battlefield where deep generalization plays out in industrial ranking models — from memorization and generalization to fine-grained modeling of multiple objectives and scenarios.</p>
<p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 8px 0 0 0;">📚 5 sections · ⏱️ Estimated 2 weeks · 🎯 Target: master the design motivations and structural differences of deep ranking models</p>
</div>

After the retrieval stage narrows hundreds of millions of items down to a few thousand candidates, **ranking** takes over the most critical job: precise scoring. Its goal is to compute, for every candidate, a predicted score that comes as close as possible to true user preference (typically click-through rate, conversion rate, and so on), then sort candidates into the best order. Ranking is the main battlefield for the generalization power of deep networks — the five chapters in this Part advance layer by layer along the thread of "how to make the model stronger, more flexible, and better aligned with the business."

This Part doesn't rush to pile up model names. Instead, it keeps asking one question: **what shortcoming of the previous method does each new model solve?** Only by reading with "motivation" in mind will you know which model to pick when facing a real business problem.

---

## What This Part Covers

| Section | Topic | The Big Idea |
|---------|-------|--------------|
| **3.1** | Wide & Deep | Joint training of "linear memorization + deep generalization" sets the foundational ranking framework |
| **3.2** | Feature Crossing | From FM's second-order crossing, through DeepFM and xDeepFM, toward automatic high-order crossing |
| **3.3** | Sequence Modeling | DIN dynamically activates history per candidate; DIEN explicitly models the temporal evolution of interests |
| **3.4** | Multi-Objective Optimization | MMoE and PLE balance multiple objectives; ESMM's entire-space modeling resolves dependency bias |
| **3.5** | Multi-Scenario Modeling | Multi-tower and dynamic weights adapt to distribution shifts across scenarios, capturing both commonality and specificity |

---

## What You'll Be Able to Do After This Part

- 🟢 **Explain** the division of labor between "memorization" and "generalization" in Wide & Deep, and why joint training matters
- 🟢 **Distinguish** the different motivations of FM's second-order crossing, xDeepFM's vector-wise high-order crossing, and AutoInt's adaptive crossing
- 🟡 **Explain** how DIN's local activation breaks through the "fixed-length user vector" bottleneck, and how DIEN further models interest evolution
- 🟡 **Differentiate** the essential difference and modeling strategies between multi-task (multiple objectives in one scenario) and multi-scenario (one objective across different scenarios)
- 🔴 **Design** appropriate multi-objective / multi-scenario architectures for businesses with dependency relationships (e.g., CTR × CVR) or seesaw conflicts
- 🟡 **Locate** each model in this Part on the chain of "solving the previous method's shortcomings"

---

## Core Concepts

| Concept | Section | Relevance |
|----------|---------|-----------|
| Memorization vs generalization / joint training | 3.1 | The foundational design philosophy of deep ranking models |
| Factorized crossing (FM) parameter sharing | 3.2 | The core technique for easing sparsity and parameter explosion |
| Vector-wise / adaptive high-order crossing | 3.2 | Letting the model automatically capture feature interactions of arbitrary order |
| Local activation (attention) | 3.3 | The key to modeling user interests that shift with the candidate |
| Interest evolution / session modeling | 3.3 | Upgrading a static "bag of items" into a dynamic sequence |
| Negative transfer / seesaw / gating | 3.4 | Multi-objective conflicts and mitigation mechanisms |
| Entire-space modeling (sample selection bias) | 3.4 | Resolving training bias caused by CVR dependencies |
| Scenario-private/shared parameters / dynamic modulation | 3.5 | Balancing commonality and differences when transferring across scenarios |

---

## Prerequisites

- You have read **Part 1** (recommender system paradigms and the three-stage pipeline) and **Part 2** (the retrieval algorithm family)
- You have read [1.3 Feature and Embedding Basics](../part1-introduction/feature-embedding-basics.md), understand sparse / dense features, bucketing, and embedding lookup; and know about MLPs, activation functions, and backpropagation
- You roughly know what business metrics like CTR (click-through rate) and CVR (conversion rate) mean

> This Part is the natural downstream of retrieval in Part 2 — the candidates are ready, and now we learn how to score them "precisely."

---

## Tips for This Part

1. **Read every model with its "motivation" in mind.** When you meet a new model, first ask: what shortcoming of the previous method does it solve? Where do the structural differences lie?
2. **Formulas serve intuition.** Understand "why it was designed this way" before looking at the math; formulas are just the precise notation for a design idea.
3. **Read the Advanced chapters comparatively.** 3.2 and 3.3 are dense with structurally similar models — comparing them side by side in a table gets you twice the result with half the effort.
4. **Run the interactive demos.** 3.2 and 3.3 each include an interactive HTML — use the "next step" button to get a hands-on feel for how high-order crossings and attention activation unfold.

---

Let's dive in! 🚀
