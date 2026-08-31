<div class="part-banner">
<h1 style="color: white; margin: 0 0 8px 0;" font-size: 1.7rem;>📗 Part 2: Fast Candidate Retrieval</h1>
<p style="color: rgba(255,255,255,0.85); margin: 0;">The first gate of the recommender pipeline — filtering a thousand candidates out of a billion-item corpus within milliseconds.</p>
<p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 8px 0 0 0;">📚 5 sections · ⏱️ Estimated 2 weeks · 🎯 Target: master the family of retrieval algorithms, from statistical co-occurrence to vector search</p>
</div>

Retrieval is the starting point of the "retrieval — ranking — re-ranking" three-stage funnel. It must quickly narrow a universe of hundreds of millions of items down to a few thousand candidates within millisecond-level latency — following the principle of "rather over-include than miss", its goal is **coverage**, not precision. Even a mediocre retriever can be tolerated, but if it misses the truly relevant items, the downstream ranking and re-ranking stages can do nothing to recover them.

This part unfolds along the technical evolution in five chapters: starting from the classic **collaborative filtering**, moving to **vector retrieval (I2I)**, which ports sequence modeling ideas into recommendation, then the **two-tower model (U2I)**, which uses deep networks for efficient retrieval, followed by the **sequential/temporal information** ignored by the previous methods (sequential retrieval), and finally stepping outside the "compress inside the model" paradigm to preserve full historical interests with a **streaming index**. Together they form the methodological map of the industrial retrieval layer.

---

## What This Part Covers

| Section | Topic | The Big Idea |
|---------|-------|--------------|
| **2.1** | Collaborative Filtering | Co-occurrence statistics over user–item interactions: from ItemCF item similarity, through Swing's industrial optimization and UserCF's user perspective, to matrix factorization opening the door to vectorization |
| **2.2** | Vector Retrieval (I2I) | Porting Word2Vec sequence modeling to recommendation: from Item2Vec's direct transfer, to EGES fusing attributes, to Airbnb baking business objectives into the objective |
| **2.3** | Two-Tower Model (U2I) | Users and items encoded separately as vectors, represented by FM, DSSM, and YouTubeDNN, enabling efficient vector search |
| **2.4** | Sequential Retrieval | Focusing on temporal information: MIND represents diverse interests with multiple vectors, SDM separates long- and short-term preferences and fuses them dynamically with gating |
| **2.5** | Streaming Index Retrieval | Stepping outside compression inside the model: Trinity preserves full interests with cluster statistics, Streaming VQ keeps the index adapting in real time |

---

## What You'll Be Able to Do After This Part

- 🟢 **Distinguish** neighborhood-based collaborative filtering (ItemCF / UserCF) from model-based matrix factorization, and articulate their respective strengths against sparsity
- 🟢 **Explain** how Swing exploits bipartite-graph structure to filter noise, and how EGES uses item-specific attention to solve cold start
- 🟡 **Derive** the $O(kn)$ simplification of FM's second-order interaction term, and show how it can be reorganized into a two-tower inner product
- 🟡 **Contrast** the fundamental difference between two-tower models and sequential retrieval (MIND / SDM) in terms of "user representation": single vector vs. multiple vectors / long-short fusion
- 🔴 **Analyze** how Trinity and Streaming VQ use cluster statistics and streaming indexes to solve "interest amnesia" and "index staleness"
- 🔴 Complete 25+ graded practice problems across the 5 chapters, consolidating the full path from co-occurrence to vector search

---

## Core Concepts

| Concept | Section | Relevance |
|----------|---------|-----------|
| Item/user similarity, co-occurrence matrix | 2.1 | The cornerstone of collaborative filtering and a key industrial retrieval channel |
| Swing score, Surprise | 2.1 | Similarity optimizations aimed at industrial robustness and complementary items |
| Latent vectors, low-rank assumption | 2.1 | The turning point from statistics to representation learning |
| Skip-Gram, sequence modeling | 2.2 | Applying the "sentence = behavior sequence" idea to I2I retrieval |
| Item-specific attention (EGES) | 2.2 | The key mechanism for solving cold start with attributes |
| Two towers, inner-product retrieval, ANN | 2.3 | The engineering backbone of efficient U2I retrieval |
| Multi-interest capsules (MIND), gated fusion (SDM) | 2.4 | Sequential retrieval that captures diverse interests and recency |
| Cluster histograms, VQ index, EMA | 2.5 | The statistical and real-time-update foundations of streaming index retrieval |

---

## Prerequisites

- You have finished **Part 1** (especially the three-stage funnel in [1.1](./../part1-introduction/recommender-system-basics.md) and the technology map in [1.2](./../part1-introduction/book-overview.md))
- Basic linear algebra (vector inner products, matrices), probability (softmax, cosine similarity), and general neural network knowledge
- Familiarity with Python and the basic concept of embeddings

> Retrieval-layer methods are relatively lightweight and coverage-oriented, with mostly controllable complexity; but the vectorization approaches (matrix factorization, two-tower, sequential) require you to be comfortable with embeddings and gradient descent.

---

## Tips for This Part

1. **Understand the motivation before the formulas.** Each method exists to fix a limitation of its predecessor — ItemCF is dominated by popular items, hence Swing; co-occurrence is sparse, hence matrix factorization.
2. **Follow the main thread of "how users/items are represented".** From CF's ID co-occurrence, to MF's latent vectors, to the two-tower's independent encoders, to sequential retrieval's multiple vectors, the representations grow ever more refined.
3. **Mind retrieval efficiency.** Methods that can precompute item vectors offline (two-tower, I2I) are usually the easiest to scale.
4. **Practice with the visualizations.** Every chapter's interactive HTML and SVG pieces are worth clicking through yourself, grounding abstract formulas in the intuition of "how the candidate pool shrinks".

---

Let's dive in! 🚀
