<div class="part-banner">
<h2 style="color: white; margin: 0 0 8px 0;">💰 Part 12: Computational Advertising</h2>
<p style="color: rgba(255,255,255,0.85); margin: 0;">The twin sibling of recommender systems — from the ad ecosystem and billing models to auction mechanisms, adding the economics of traffic monetization to your toolkit.</p>
<p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 8px 0 0 0;">📚 7 sections · ⏱️ Estimated 5.5 hours · 🎯 Target: Build a panoramic understanding of computational advertising and master the full chain of auctions, smart bidding, prediction calibration, and data loops</p>
</div>

Recommendation and advertising share the same technical foundation — retrieval, ranking, feature engineering, CTR estimation — but advertising layers something on top that recommendation lacks: an **economic mechanism**. Ad slots are scarce resources; every impression must be allocated among multiple advertisers, and the winner pays a fee. The design of the allocation and pricing rules directly determines "how bidding pays off" for each advertiser, and in turn the stability of the whole market and the platform's long-term revenue.

This part follows the framework of Liu Peng's *Computational Advertising* course and extends it with recent industrial papers from KDD/SIGIR/RecSys, filling in the advertising picture for readers with a recommendation background: first a bird's-eye view of **the evolution of ad delivery models and the programmatic ecosystem** (12.1), then the thread that runs through every advertising system — **billing models and core metrics** (12.2), and then the centerpiece of this part — **auction mechanisms** (12.3): why first-price auctions destabilize the market, why the generalized second price became the industrial mainstream for two decades, and what it costs to make VCG's "truth-telling" work. The final two chapters turn to the engineering side: **smart bidding and budget control** (12.4) covers how platforms bid on behalf of advertisers, smooth out budget spending, and shade bids in first-price markets; **bias and calibration** (12.5) covers why ad prediction must be *absolutely accurate* rather than merely *correctly ranked*, and how position bias, sample selection bias, and industrial calibration pipelines are handled. Finally, **open-loop and closed-loop advertising** (12.6) closes the part from the angle of data observability: whether conversions happen inside the platform's observable domain decides how deep the platform can optimize — closed loops train deep pCVR models and support deep conversion bidding, while open loops are stuck with attribution and privacy. Finally, **online allocation and traffic management** (12.7) supplies the algorithmic foundation of contract advertising — the supply/demand bipartite graph, the compact allocation plan that keeps only $O(|A|)$-level dual variables, and the HWM heuristic; this "constrained optimization + dual pricing" framework is also the theoretical origin of 12.4's budget bidding.

![The programmatic advertising ecosystem at a glance](../images/part12-ecosystem.svg)

> 💡 **Key Insight:** A recommender system optimizes "the match between one user and one item"; an advertising system additionally optimizes "the game rules among multiple advertisers." **Mechanism design is the biggest watershed between ads and recommendation** — once you understand the GFP → GSP → VCG storyline, you can read 8.3's EGA with new eyes: why it embeds incentive-compatibility (IC) constraints directly into the generative process.

---

## What This Part Covers

| Section | Topic | The Big Idea |
|---------|-------|--------------|
| **12.1** | The Advertising Panorama and Ecosystem | From "advertiser ↔ publisher direct deals" to "DSP-ADX-SSP programmatic trading" — three leaps in delivery models, each integrating supply and demand and refining the granularity of bidding |
| **12.2** | Billing Models and Core Metrics | The CPT→CPM→CPC→CPA spectrum is fundamentally a transfer of **risk allocation**; eCPM is the common yardstick — the platform sorts by it, advertisers game it |
| **12.3** | Auction Mechanisms: From First-Price to Second-Price | First-price (GFP) has no stable equilibrium, second-price (GSP) became the industrial mainstream, VCG makes everyone honest but is hard to deploy — mechanism design is a trade-off between stability and incentive compatibility |
| **12.4** | Smart Bidding and Budget Control | oCPC/oCPM platform-managed bidding, PID feedback control for budget pacing, and bid shading in first-price markets maximizing expected surplus — the four-layer bidding stack |
| **12.5** | Bias and Calibration in Ad Systems | Position bias (PAL), sample selection bias (ESMM), winner's curse and delayed feedback; the Platt/isotonic calibration pipeline — absolute prediction accuracy is the foundation of ad systems |
| **12.6** | Open-Loop and Closed-Loop Advertising | Whether conversion happens inside the platform's observable domain decides how deep it can optimize — closed loops train deep pCVR models, open loops are stuck with attribution and privacy |
| **12.7** | Online Allocation and Traffic Management | Guaranteed-volume contracts written as constrained optimization on a bipartite graph: the compact allocation plan recovers $O(|E|)$-level allocation rates from $O(|A|)$-level dual variables, and HWM is the heuristic genuinely running in engineering |

---

## What You'll Be Able to Do After This Part

- 🟢 **Explain** the essential differences between ads and recommendation: non-homogeneous matching, conversion as the endpoint, ROI orientation, and mechanism-design constraints
- 🟢 **Describe** the responsibilities of DSP / ADX / SSP / DMP and the complete timeline of one RTB auction
- 🟡 **Distinguish** risk allocation under each billing model: who makes the decision, who bears the effect uncertainty
- 🟡 **Derive** the eCPM ranking logic and the GSP payment formula $p_i = b_{i+1} x_{i+1} / x_i$
- 🔴 **Prove** that truthful bidding is a dominant strategy in the single-slot second-price auction, and explain why multi-slot GSP loses strict incentive compatibility
- 🔴 **Compare** GSP and VCG on revenue, equilibrium properties, and industrial feasibility, and understand why programmatic markets moved "back to first-price"
- 🔴 **Distinguish** the open-loop/closed-loop criterion, and explain why closed loops support deep conversion bidding while open loops depend on postbacks and attribution models, and how ATT/SKAN collapse deterministic attribution
- 🔴 **Model** the online allocation problem: write out the supply/demand constraints and the compact plan's recovery formula $x_{ia} = \max(0, \theta_a(1+\alpha_a-\beta_i))$, and explain HWM's priority-and-scale-down logic
- 🏆 Verify "what happens if you misreport" with the interactive auction simulator, find the optimal bid with the bid-shading simulator, and compare five attribution models with the attribution simulator — then finish the tiered exercises in each section

---

## Core Concepts

| Concept | Section | Relevance |
|----------|---------|-----------|
| Advertising effectiveness model | 12.1 | Exposure→attention→comprehension→acceptance→retention→decision — a six-stage map of ad effect |
| Programmatic ecosystem (DSP/ADX/SSP/DMP) | 12.1 | The infrastructure of modern ad delivery |
| RTB (Real-Time Bidding) | 12.1 | Open bidding at per-impression granularity; Cookie Mapping is the prerequisite |
| eCPM | 12.2 | The common yardstick that makes cross-billing-model ranking possible |
| Guaranteed Delivery & online allocation | 12.7 | The bipartite-graph optimization framework of contract advertising: compact allocation plan + HWM |
| Compact Allocation Plan / SHALE | 12.7 | Stores only contract-level dual variables α and recovers allocation rates; solved by primal-dual iteration, supporting incremental contracts |
| Position Auction | 12.3 | The unified model for multi-slot allocation and pricing |
| Generalized Second Price (GSP) | 12.3 | The dominant pricing mechanism of search advertising for two decades |
| VCG mechanism | 12.3 | The theoretically optimal truth-telling pricing, and the benchmark for industrial trade-offs |
| Incentive Compatibility (IC) / Individual Rationality (IR) | 12.3 | The two properties of mechanism design, and the core constraints of EGA in 8.3 |
| Closed-loop / open-loop advertising | 12.6 | The binary of whether conversion happens inside the platform's observable domain, which caps optimization depth |
| Attribution | 12.6 | A rule for crediting a conversion to a channel — a convention, not an objective measurement |
| SKAdNetwork (SKAN) | 12.6 | Apple's privacy-preserving attribution: aggregated, delayed, crowd-anonymized — the collapse of deterministic attribution |

---

## Prerequisites

- You have read **1.1** (What is a Recommender System) — this part repeatedly uses recommendation as the reference frame
- Basic probability and expected-value computation ($E[\cdot]$); 12.3 contains light game-theoretic derivation but requires no prior game theory
- The IC/IR material in **12.3** and **8.3** (end-to-end generative advertising) are mirror images of each other — reading either first deepens the other

> This part is a **special topic** that does not depend on any chapter of the generative track; but the mechanism-design perspective of 12.3 will in turn illuminate why EGA in 8.3 decouples "allocation" from "payment."

---

## Tips for This Part

1. **Read 12.2 asking "who bears the risk."** Every billing model can be summarized in one sentence: who holds the decision right, and who absorbs the effect uncertainty. This thread beats memorizing formulas.
2. **Compute by hand in 12.3 — don't just read.** The GSP formula $p_i = b_{i+1} x_{i+1} / x_i$ looks simple, but the position-CTR conversion trips people up constantly. Work through the three-slot numerical example in 12.3.3 by hand first, then play with the interactive simulator.
3. **Treat mechanisms as institutions, not formulas.** The difference between first- and second-price lies not in algebra but in the advertiser behavior they incentivize: oscillating bidding wars vs. stable equilibria. After 12.3 you should be able to explain why Overture's 1998 market was chaotic — and what Google changed in 2002.

---

Let's dive in! 🚀
