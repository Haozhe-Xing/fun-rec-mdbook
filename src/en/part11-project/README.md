<div class="part-banner">
<h2 style="color: white; margin: 0 0 8px 0;">🚀 Part 11: Generative Recommender Systems in Practice</h2>
<p style="color: rgba(255,255,255,0.85); margin: 0;">Build a runnable, industrial-grade movie recommender from scratch, covering offline training, online inference, frontend interaction, and containerized deployment end to end.</p>
<p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 8px 0 0 0;">📚 6 sections · ⏱️ Estimated 3–4 weeks · 🎯 Target: turn discrete algorithms into an end-to-end servable recommender</p>
</div>

The preceding chapters covered the core algorithm modules — retrieval, ranking, and re-ranking. But a model that runs in a paper is not the same as a model you can deploy in a real setting — a gap nearly every recommender-system learner runs into. This part uses an **end-to-end movie recommender** project to string the scattered algorithms into a complete system that runs, serves, and deploys, answering the engineering question: how do you build a production-grade recommender from scratch?

---

## Chapters

| Chapter | Topic | The Big Idea |
|---------|-------|--------------|
| **11.1** | Project Introduction and Goals | Clarify the gap between offline evaluation and online deployment; settle the technology choices and the learning path |
| **11.2** | System Architecture Design | Decouple offline from online; the classic funnel of retrieval → ranking → re-ranking |
| **11.3** | Offline Pipeline | Feature engineering, YoutubeDNN/DeepFM training, embedding generation, feature ingestion, and model deployment |
| **11.4** | Online Pipeline | Cold start (UCB), multi-route retrieval (Snake Merge), DeepFM ranking, diversity re-ranking |
| **11.5** | Frontend and Interaction | Five Vue 3 pages, Pinia state, search debouncing, a rating-driven data feedback loop |
| **11.6** | Deployment and Operations | Orchestrate five services with one Docker Compose command; health checks and troubleshooting |

---

## What You Will Be Able to Do After This Part

- 🟢 **Describe** the responsibility boundary between the offline and online systems, and how the storage layer decouples them
- 🟢 **Explain** why the funnel architecture is necessary: light models filter candidates in retrieval, heavy models score precisely in ranking
- 🟡 **Implement** the training loop for YoutubeDNN retrieval and DeepFM ranking, and understand why item embeddings are precomputed
- 🟡 **Design** a cold-start strategy (UCB exploration + preferred genres + popular fallback) and multi-route retrieval fusion (Snake Merge)
- 🔴 **Deploy** a multi-container system with PostgreSQL/Redis/Elasticsearch/backend/frontend, and troubleshoot common problems
- 🟢 **Complete** the tiered practice problems in each section to consolidate the engineering essentials

---

## Key Concepts

| Concept | Section | Relevance |
|----------|---------|-----------|
| Offline vs. online | 11.2 | The fundamental boundary of recommender engineering; the quality-vs-latency trade-off |
| Funnel architecture (retrieval → ranking → re-ranking) | 11.2 | The backbone of industrial recommendation |
| Item embedding precomputation | 11.3 | The prerequisite for millisecond-level online vector search |
| Cold start / UCB | 11.4 | The exploration-exploitation balance for new users with no behavior |
| Multi-route retrieval + Snake Merge | 11.4 | Fusion compensates for the coverage gaps of any single strategy |
| Data feedback loop | 11.5 | Frontend behavior feedback drives feature updates and recommendation improvements |

---

## Prerequisites

- The three-stage pipeline mental model from [1.1](./project-intro.md), and an understanding of how retrieval/ranking/re-ranking divide the work
- The basics of the two-tower model in [2.3](./project-architecture.md) (YoutubeDNN) and the ranking models in [3.x](./project-architecture.md) (DeepFM)
- Working knowledge of Python, basic neural networks, and SQL; familiarity with Docker basics is a plus

> The code for this project lives in the `web_project/` directory of the `datawhalechina/fun-rec` repository — you can run it as you read.

---

## Tips for This Part

1. **Get it running before nitpicking the details.** Launch the full project with one Docker Compose command and build overall intuition first.
2. **Grasp the offline/online boundary.** This is the core mental framework for engineered systems.
3. **Pay attention to the steps papers never mention:** how features move across systems, how models update without downtime, and how cold start degrades gracefully.

---

Let's build it! 🛠️
