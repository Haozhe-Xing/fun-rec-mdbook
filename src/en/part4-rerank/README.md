<div class="part-banner">
<h2 style="color: white; margin: 0 0 8px 0;">🔶 Part 4: Diversity Modeling in Re-ranking</h2>
<p style="color: rgba(255,255,255,0.9); margin: 0;">Optimize the experience of the entire recommendation list while preserving relevance — from greedy heuristics to end-to-end personalization.</p>
<p style="color: rgba(255,255,255,0.75); font-size: 0.9em; margin: 8px 0 0 0;">📚 2 sections · ⏱️ Estimated 1 week · 🎯 Target: master the relevance–diversity trade-off at the re-ranking stage</p>
</div>

When a ranking model emits a list sorted by descending CTR, you often see an awkward phenomenon: the head of the list is dominated by items of the same category and the same style. Ranking pursues **point-wise accuracy**, but what users want is **a great experience across the whole screen**. This part stands at the very end of the three-stage funnel — **re-ranking** — and studies how to bridge the gap where "the highest-scoring list ≠ the best-experience list."

We take two routes. One is the **greedy-based** family of lightweight rule methods (MMR, DPP): intuitive, interpretable, and easy to deploy. The other is the **personalization-based** family of data-driven methods (PRM, PRS), which use models to automatically learn the high-order mutual influence among items.

---

## What This Part Covers

| Section | Topic | The Big Idea |
|---------|-------|--------------|
| **4.1** | Greedy-based re-ranking | MMR trades off relevance and diversity with a linear combination; DPP uses a determinant framework for more precise control of diversity |
| **4.2** | Personalized re-ranking | PRM models item mutual influence with a Transformer; PRS directly optimizes the experiential gain of permutations |

---

## What You'll Be Able to Do After This Part

- 🟢 **Explain** why homogenized ranking output is the fundamental motivation for re-ranking, and the two kinds of cost it incurs
- 🟢 **Write out** MMR's marginal gain formula, and hand-compute a top-k list on a given similarity matrix with the greedy procedure
- 🟡 **Derive** the DPP kernel matrix $L = \text{Diag}(r)\cdot S \cdot \text{Diag}(r)$, and articulate how the determinant measures diversity
- 🟡 **Distinguish** the essential difference between MMR (heuristic linear combination) and DPP (precise determinant control) in diversity modeling
- 🔴 **Describe** how PRM achieves end-to-end list re-ranking with a Transformer plus personalized vectors (PV)
- 🔴 **Understand** why PRS introduces permutation-variant influence, and how its PMatch / PRank two-stage design resolves the $n!$ combinatorial explosion
- Complete 8 leveled practice problems to consolidate the core methods of the two chapters

---

## Core Concepts

| Concept | Section | Relevance |
|----------|---------|-----------|
| List homogenization / diversity | 4.1 | The reason re-ranking exists: breaking up head repetition and protecting the long tail |
| MMR (Maximal Marginal Relevance) | 4.1 | The most classic and most deployable greedy diversity re-ranking |
| DPP (Determinantal Point Process) | 4.1 | Precisely models set-level diversity through the geometric meaning of the determinant |
| PRM (Personalized Re-ranking Model) | 4.2 | Learns item mutual influence end-to-end with a Transformer |
| PRS (permutation-based re-ranking) | 4.2 | Directly optimizes the experiential gain brought by ordering |

---

## Prerequisites

- You have read the scoring function $f$ in [Part 3 Ranking](../) and understand how the ranking stage outputs a candidate list with relevance scores
- Some matrix basics (determinants, positive semi-definiteness, Cholesky decomposition) will help you fully digest the DPP derivation
- Familiarity with the Transformer self-attention mechanism will help you understand the PRM encoding layer
- Basic Python and vector representation knowledge

> This part is the last link of the three-stage pipeline — we recommend building the full "retrieval → ranking → re-ranking" picture from Parts 1–3 first.

---

## Tips for This Part

1. **Motivation first, formulas second.** Every method in this part exists to solve a "list-level experience" problem; learning formulas detached from their motivation is an easy way to get lost.
2. **Work the examples by hand.** The MMR hand-computation table and the DPP kernel matrix construction in 4.1 — push them through yourself once; it sticks better than ten readings.
3. **Compare the two routes.** After finishing 4.2, look back and compare: greedy methods rely on "hand-crafted objective functions," while personalized methods rely on "models learning from data."
4. **Remember the visualizations.** The accompanying SVGs and interactive HTML in this part turn abstract formulas into observable processes — watch them often and drag the sliders.

---

Let's dive in! 🚀
