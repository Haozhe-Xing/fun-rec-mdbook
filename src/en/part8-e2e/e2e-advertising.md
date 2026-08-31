<div class="badge-row">
  <span style="background: #f0f4ff; color: #4A6CF7; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(74,108,247,0.2);">📖</span>
  <span style="background: #fef2f2; color: #dc2626; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(100,100,100,0.15);">⏱️ ~50 min read</span>
  <span style="background: #fef2f2; color: #dc2626; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(100,100,100,0.15);">🎯 Advanced</span>
</div>

# End-to-End Generative Advertising

> 📝 **Before You Continue:** Read [8.1](./e2e-recommendation.md) first for semantic IDs / Enc-Dec / RL alignment, and [8.2](./e2e-search.md) for hard-constraint retrieval — the advertising scenario stacks both sets of technical challenges on top of each other, and additionally carries **economic constraints**.

[8.1](./e2e-recommendation.md) and [8.2](./e2e-search.md) solved the performance bottlenecks of cascaded systems with end-to-end generative architectures. But **online advertising** faces more complex constraints: the system must optimize user experience while balancing platform revenue and advertiser interests, satisfying the economic constraints of the auction mechanism. The traditional advertising system's multi-stage architecture of "retrieval → ranking → creative selection → auction → slot allocation" fragments objectives and struggles to adapt to fast-changing markets.

End-to-end generative advertising must break through three core challenges: **how to deeply integrate the auction mechanism into the generation process**, **how to guarantee advertisers' Incentive Compatibility (IC)**, and **how to efficiently model user intent in ultra-long heterogeneous sequences**. This section covers two industrial solutions: **EGA** unifies the auction mechanism with the generative model, embedding IC/IR constraints through a two-tier design of token-level bidding and POI-level payment; **GPR** achieves unified multi-scenario modeling over ultra-long heterogeneous sequences in the WeChat ecosystem through a heterogeneous hierarchical decoder and pre-training.

After reading this chapter, you will be able to:

- Explain the **triple constraints** of the advertising scenario relative to recommendation/search (IC/IR, POI + creative joint generation, decoupling of allocation and payment)
- Describe **EGA**'s dual-modality semantic IDs, probability-decomposed generation, and token-level auction mechanism
- Explain how **ex-post regret** and **Lagrangian optimization** approximately guarantee incentive compatibility
- Outline **GPR**'s four token types, RQ-Kmeans+, heterogeneous hierarchical decoder, and value-guided Trie Beam Search
- Complete 5 tiered practice problems consolidating bid aggregation, the payment network, and hierarchical policy optimization

---

## 8.3.0 The Triple Constraints of Ad Generation

Consider a scenario to understand how advertising fundamentally differs from recommendation and search: a user scrolls a local-life platform feed, and the system must insert one ad at slot 3. The candidates are nearby restaurants, gyms, and beauty salons; each merchant submits a different bid $b_i$ and has several creative images. A single forward pass must make four decisions — which merchant to show (POI), which creative to use, how to compute payment, and how to guarantee fairness. This reveals a triple set of constraints:

**Constraint one: Incentive Compatibility (IC) and Individual Rationality (IR).** Advertisers are independent players who adjust bids according to the rules. IC requires truthful bidding to be the optimal strategy: for true valuation $v_i$ and reported bid $b_i$, utility is maximized when $b_i=v_i$:

$$u_i(v_i; v_i, \boldsymbol{b}_{-i}) \geq u_i(v_i; b_i, \boldsymbol{b}_{-i}), \quad \forall b_i \in \mathbb{R}^+$$

Utility is $u_i = (v_i - p_i) \cdot \text{pCTR}_i$ (click value minus payment). IR requires payment not to exceed the bid, $p_i \leq b_i$. The traditional GSP auction preserves IC by charging "the next position's price," but it assumes ads are independent and cannot handle position externalities.

**Constraint two: joint generation of POI and creative.** One POI (restaurant) can have multiple creative images, and different users prefer different creatives. The system must jointly decide "which POI to show" and "which creative to use" — the POI determines the content subject, and the creative optimizes the presentation.

**Constraint three: decoupling allocation and payment.** Directly using bids as weights on generation probability causes a "winner's curse": the highest-bidding ad pays according to its own bid, so advertisers tend to under-bid. EGA resolves this conflict by separating **allocation** (bids guide generation probability) and **payment** (an independent network learns the IC payment function) into two modules.

> 💡 **Key Insight:** The end-to-end difficulty of advertising is that the generative model must "incidentally" satisfy an economic mechanism — this affects not just the objective function but also requires architecturally decoupling "allocation" from "payment" before IC/IR can be guaranteed mathematically.

---

## 8.3.1 EGA: Unifying Auction and Generation

### Dual-Modality Semantic IDs and Probability Decomposition

EGA uses **RQ-VAE** to discretize continuous representations of POIs and creatives into multi-layer semantic IDs (two independent semantic spaces). The raw POI representation includes category, geolocation, statistical features, and text description; the creative representation includes visual features, OCR copy, and creative type. With $C=3$ residual quantization layers and codebook size $W=1024$, each POI is encoded into 3 tokens:

$$\boldsymbol{a}_i^{\text{poi}} = (a_i^{1}, a_i^{2}, a_i^{3}), \quad a_i^{j} \in \{1, 2, \ldots, 1024\}$$

Creatives likewise yield $\boldsymbol{a}_i^{\text{img}}$. The user's interaction history is represented as a sequence of (POI, creative) pairs.

**Probability decomposition strategy.** The intuitive idea is to concatenate the 6 tokens of the POI and creative and generate autoregressively, but EGA found this causes POI-creative mismatches ("Restaurant A's POI + Gym B's creative"). So it decomposes:

$$P(\boldsymbol{a}_{t+1}^{\text{poi}}, \boldsymbol{a}_{t+1}^{\text{img}} \mid \mathcal{S}^u_{1:t}) = P(\boldsymbol{a}_{t+1}^{\text{poi}} \mid \mathcal{S}^u_{1:t}) \cdot P(\boldsymbol{a}_{t+1}^{\text{img}} \mid \boldsymbol{a}_{t+1}^{\text{poi}}, \mathcal{S}^u_{1:t})$$

Intuition: the POI decides "what to show," the creative decides "how to present it." First generate the POI from interests, then choose the creative based on the POI's characteristics and user preferences.

### Encoder-Decoder with Dual Decoders

EGA uses the classic Enc-Dec but with **two decoders** generating the POI and creative respectively. The encoder processes the historical sequence $\mathcal{S}^u$ mixing ads and organic content (each item labeled type∈{ad, organic}), outputting $\mathcal{S}^e = \text{Encoder}(\mathcal{S}^u)$. The **POI decoder** autoregressively generates the 3-layer semantic ID; the **creative decoder** generates the creative ID conditioned on the generated POI tokens — its input contains the POI token sequence, letting the model choose a matching creative based on POI semantics.

**MTP module.** A standard decoder predicts only the next token at each step; EGA uses **MTP (Multi-Token Prediction)** to jointly supervise both decoders at each step, letting them share underlying representations, accelerating convergence and improving consistency:

$$\mathcal{L}_{\text{pre-train}} = \mathcal{L}_{\text{NTP}}^{\text{POI}} + \mathcal{L}_{\text{MTP}}^{\text{Creative}}$$

![EGA architecture: dual-decoder joint generation + token-level bidding + POI-level payment network](../images/part8-ega-framework.svg)

### Permutation-Aware Reward Model: Handling Position Externalities

The pre-trained model doesn't know "which ad is better." Auction-based fine-tuning needs a reward model, and the advertising scenario must handle **position externalities** — ads are not independent: position effects (CTR at slot 1 is far higher than slot 5), adjacency effects (two adjacent restaurant ads suppress each other), and contrast effects (a low-quality ad following a high-quality one sees CTR drop). Mathematically:

$$\text{pCTR}_i = f(\text{user}, \text{item}_i, \mathcal{Y}_{-i}, \text{pos}_i)$$

Traditional point-wise models (DeepFM, Wide&Deep) cannot model sequence-level dependencies. EGA uses a **permutation-aware** design, using Self-Attention to let every ad "see" the other ads in the sequence:

$$\boldsymbol{h}_i = [\text{Embed}(\boldsymbol{a}_i^{\text{poi}}); \text{Embed}(\boldsymbol{a}_i^{\text{img}}); \boldsymbol{e}_i^{\text{poi}}], \quad \boldsymbol{h}_f = \text{SelfAttention}(\boldsymbol{h} W^Q, \boldsymbol{h} W^K, \boldsymbol{h} W^V)$$

Three independent towers predict **POI-CTR / Creative-CTR / CVR** respectively, with the composite reward:

$$\hat{r}_i = \lambda_1 \hat{r}_i^{\text{pctr-poi}} + \lambda_2 \hat{r}_i^{\text{pctr-img}} + \lambda_3 \hat{r}_i^{\text{pcvr}}$$

> **Analysis:** The permutation-aware reward model is EGA's key difference from OneRec's P-Score — it models "sequence-level position externalities" into the reward rather than making point-wise predictions. The costs are Self-Attention's $O(K^2)$ in sequence length and training an additional three-tower reward model.

### Token-Level Bidding: Max Aggregation

A generative framework outputs token sequences, and the token-ad relationship is **many-to-many** (one ad is encoded into multiple tokens; one token may correspond to multiple ads), so traditional item-level bidding doesn't apply. EGA uses a two-tier design:

**Token-level bid aggregation (max).** For the ad set $\{x_1,\ldots,x_{N_i}\}$ corresponding to layer-$j$ token $a_i^j$, bids are aggregated with the **maximum**:

$$b(a_i^j) = \max(b_1, b_2, \ldots, b_{N_i})$$

Why max rather than avg? If a token corresponds to a high-bidding ad, generating it carries high commercial value and its probability should be boosted; avg would be diluted by low bids. Based on this, the allocation probability is defined as:

$$z(a_i^j) = \frac{w(a_i^j) \cdot e^{a_i^j}}{\sum_{k=1}^W [w(a^{j,k}) \cdot e^{a^{j,k}}]}, \quad w(a_i^j) = [b(a_i^j)]^\alpha + \beta$$

- $\alpha$: the bid influence weight. $\alpha=0$ degenerates to pure interest-based recommendation; $\alpha\to\infty$ becomes pure bid-based ranking.
- $\beta$: the ratio of ads to organic content. Larger $\beta$ gives higher generation probability to organic content (bid 0).

### POI-Level Payment Network: Learning IC-Compliant Payments

Paying directly by generation probability $z$ is problematic: the probability is non-differentiable and hard to keep IC. EGA **decouples allocation from payment**: allocation is bid-guided, while payment uses an independent neural network to learn an IC payment function. The payment network's inputs include the POI sequence representation, a self-excluding bid matrix (depending only on others' bids and one's own allocation — the key to IC), and the expected value (allocation probability × pCTR). A Sigmoid outputs the payment rate:

$$\hat{p} = \sigma(\text{MLP}(\mathcal{S}^*; \mathcal{B}^-; \mathcal{Z} \cdot \Theta)), \quad p_i = \hat{p}_i \cdot b_i$$

The Sigmoid guarantees $\hat{p}_i \in [0,1]$, thereby satisfying IR $p_i \leq b_i$.

**Ex-post regret constraint.** Borrowing from mechanism design, IC violations are quantified: for advertiser $i$, truthful-bidding utility is $u_i(v_i; v_i, \boldsymbol{b}_{-i}) = (v_i - p_i)\cdot\text{pCTR}_i$, and the maximum gain from misreporting is the regret:

$$\text{rgt}_i = \max_{b'_i} \{u_i(v_i; b'_i, \boldsymbol{b}_{-i}) - u_i(v_i; v_i, \boldsymbol{b}_{-i})\}$$

When $\text{rgt}_i=0$, truthful bidding is optimal. In practice, candidate bids are sampled to approximate this. EGA solves the constrained optimization (maximize revenue, regret constrained near 0) with a **Lagrangian dual**:

$$\mathcal{L}_{\text{Pay}} = -\frac{1}{|\mathcal{D}|}\sum_{d} \left( \sum_i p_i \hat{r}_i^{\text{pctr}} - \sum_i \lambda_i \widehat{\text{rgt}}_i - \frac{\rho}{2} \sum_i (\widehat{\text{rgt}}_i)^2 \right)$$

Alternating updates: fix $\lambda$ and optimize the payment network; fix the network and update $\lambda_i^{\text{new}} = \lambda_i^{\text{old}} + \rho \cdot \widehat{\text{rgt}}_i$. For advertisers with high regret, $\lambda$ increases, forcing the loss to focus more on reducing their regret.

### Two-Stage Joint Training

**Stage one, interest-based pre-training**: ignore bids, train the NTP+MTP joint loss on exposure sequences, obtaining the base generative model $\mathcal{F}$.

**Stage two, auction-based post-training**: introduce bids, the reward model, and the payment network, alternating among three sub-tasks: (1) the reward model trains multi-task BCE on real feedback and is frozen as the evaluator; (2) **Policy Gradient** — non-autoregressive policy gradient with marginal-contribution reward $r_{y_i} = \sum b_j \hat{r}_j^{\text{pctr}} - \sum_{y_j \in \mathcal{S}^*_{-i}} b_j \hat{r}_j^{\text{pctr}}$ and loss $-\sum r_{y_i} \log z_{y_i}$; (3) the payment network minimizes ex-post regret via the Lagrangian.

> **Analysis:** EGA's core value is turning the "auction mechanism" from an external rule into a differentiable internal part of the generative model — token-level bidding guides allocation, and the POI-level payment network guarantees IC. Compared with OneRec, the differences are the introduction of bid signals, IC constraints, and permutation awareness. Limitations: RQ-VAE and Enc-Dec target a single scenario and struggle to unify across scenarios; a standard Transformer's input is limited and $O(L^2)$ struggles with sequences of tens of thousands; Beam Search generates many invalid candidates, adding latency. These gave rise to GPR.

---

## 8.3.2 GPR: Pre-training-Driven Ad Generation

EGA emphasizes "auction-driven"; **GPR (Generative Pre-trained Recommender)** adopts a "pre-train + fine-tune" paradigm — first learning general interest representations on massive unsupervised data, then aligning with business objectives through value-aware fine-tuning and RL. It tackles cross-scenario, ultra-long-sequence, and 100ms real-time challenges in the WeChat ecosystem (Channels/Moments/Official Accounts/Mini Programs).

### Unified Input Representation: Four Token Types

GPR encodes the user's complete behavioral journey as a mixed sequence of four token types:

1. **U-Token (User)** — static attributes and long-term preferences (demographics, spending power, interest tags)
2. **O-Token (Organic)** — browsed organic content (short-video RQ-VAE semantic IDs, article text representations, multimodal representations of friends' updates)
3. **E-Token (Environment)** — immediate environment (time, geolocation, device, scene identifier)
4. **I-Token (Item)** — interacted ad items (RQ-VAE semantic IDs, including POI + creative)

This representation provides: scene unification (content from different scenes shares one token system), temporal coherence (a cross-scene timeline), and rich context (each I-Token is surrounded by O/E-Tokens providing context).

### RQ-Kmeans+: Solving Codebook Collapse

When quantizing O/I-Tokens, traditional RQ-VAE faces **codebook collapse**: with randomly initialized codebooks, some codes are never activated, and utilization is only 60–70%. **RQ-Kmeans+** combines RQ-Kmeans's high-quality initialization with RQ-VAE's end-to-end optimization:

**Step 1** RQ-Kmeans builds initial codebooks by running K-means on residuals (guaranteeing every code is assigned at least some samples, avoiding dead codes).
**Step 2** Use these as RQ-VAE initial weights, add a residual connection on the encoder side $\boldsymbol{z} = \text{Encoder}(\boldsymbol{e}) + \alpha \cdot \boldsymbol{e}$ (with learnable $\alpha\in[0,1]$), then train end-to-end with the standard RQ-VAE loss. Result: codebook utilization rises from 65% to 92%, and reconstruction error drops 15%.

### Heterogeneous Hierarchical Decoder (HHD)

EGA's Enc-Dec tightly couples the encoder and decoder, and sequences of tens of thousands hit memory/compute bottlenecks. GPR proposes the **HHD (Heterogeneous Hierarchical Decoder)**, decoupling into three layers to achieve "understand first, then reason, then generate":

![GPR's heterogeneous hierarchical decoder (HSD intent understanding / PTD reasoning-generation / HTE value evaluation)](../images/part8-gpr-hd.svg)

**Layer one, HSD (Sequence-wise Decoder) — intent understanding.** Uses an improved HSTU architecture with three designs:
- **Hybrid Attention Mask** — bidirectional attention within the U/O/E-Token (Prompt) region for full interaction; causal attention within the I-Token (Target) region to guarantee autoregression; Targets can attend to the full Prompt.
- **Token-Aware Normalization** — the four token types U/O/E/I have vastly different distributions, so each gets an independent LayerNorm and FFN, projecting into its own semantic subspace.
- **MoR (Mixture-of-Recursions)** — the same layer recursively calls itself $R$ times (with learnable weights $w_r$), increasing reasoning depth without adding parameters, akin to "multiple rounds of thinking."

HSD outputs **intent embeddings** $\mathcal{S}^e$.

**Layer two, PTD (Token-wise Decoder) — reasoning and generation.** Designed as a "Thinking-Refining-Generation" three-stage process:
- **Thinking**: generates $K=4$ Thinking Tokens (learnable query vectors extract key signals from the intent embeddings via Cross-Attention, filtering out irrelevancies).
- **Refining**: drawing on Self-Reflection, Gaussian noise is added to the Thinking Tokens and a conditional denoising Transformer iteratively refines them (similar to Stable Diffusion), improving complex-user generation quality by 2–3%.
- **Generation**: autoregressively generates the target ad's semantic IDs (3 RQ layers) from the refined representation.

**Layer three, HTE (Token-wise Evaluator) — value evaluation.** Outputs a value estimate **at every token-generation layer**, $v^l = \text{MLP}_{\text{value}}([\boldsymbol{h}^{(l)}; \text{Embed}(a^l)])$, with the final ad value $final\_value = w_1\cdot\text{pCTR}+w_2\cdot\text{pCVR}+w_3\cdot\text{eCPM}$. HTE is used both for Beam Search pruning and as the Critic in Policy Optimization.

### Value-Guided Trie Beam Search

EGA's standard Beam Search generates many invalid candidates (exhausted budgets, targeting mismatches, geo restrictions). GPR proposes **Value-Guided Trie-based Beam Search**, integrating value estimation and constraint filtering into decoding:

**Trie tree constraints.** Filter a valid ad subset $\mathcal{X}_{\text{valid}}$ by user profile and ad-targeting constraints (age/targeting/budget/geo), and build a Trie prefix tree from each ad's 3-layer semantic IDs. When decoding layer $l$, sampling comes only from the Trie's current node's children rather than the full codebook ($W=1024$), shrinking the search space from $W^3$ to $|\mathcal{X}_{\text{valid}}|$.

**Value-based dynamic beam width.** Standard Beam Search uses a fixed beam width $B$; GPR adjusts it dynamically based on HTE values:

$$B_{\text{next}}(a^l) = \max\left(B_{\text{min}}, B_{\text{base}} \times \exp\left(\frac{v^l - \bar{v}}{\tau}\right)\right)$$

Branches with value far above the mean get wider beams to explore more; low-value branches shrink early. Actual results: inference latency dropped from 150ms to 80ms (down 47%), the valid-candidate share rose from 40% to 95%, and Top-1 accuracy improved 3.2%.

![Value-guided Trie Beam Search: constraints and values embedded in decoding](../images/part8-gpr-beamtrie.svg)

Left: the Trie prefix tree filters a valid ad subset by user profile and targeting constraints; decoding expands only on legal child nodes, shrinking the search space from $W^3$ to $|\mathcal{X}_{\text{valid}}|$. Right: each layer dynamically adjusts beam width by HTE value estimates — high-value branches are retained, low-value ones pruned.

The interactive demo below lets you feel the Beam Search decoding of generative retrieval: starting from the root, each layer branches among (Trie-constrained) candidate tokens; branches with high HTE values are retained and low-value ones pruned, ultimately outputting a valid ad semantic ID sequence. Click "Next" to watch the layer-by-layer expansion.

<iframe src="../viz/part8-beamsearch.html?embed&vizId=part8-beamsearch" style="width:100%; height:520px; border:none; display:block;" loading="lazy"></iframe>

Note the "pruning" at each step: candidates that fail the Trie constraints (e.g., geo mismatch) or have too-low HTE values are dropped early in generation. This is exactly how GPR turns "legality" and "value" into hard decoding constraints and cuts latency nearly in half.

### Multi-Stage Training Strategy

**Stage one, MTP pre-training**: massive WeChat all-scene behavior logs (Channels/Moments/Official Accounts/ads), with objective $\mathcal{L}_{\text{pre-train}} = \mathcal{L}_{\text{NTP}}^{\text{POI}} + \mathcal{L}_{\text{MTP}}^{\text{Creative}}$ — hundreds of millions of users, hundreds of billions of interactions, up to 8B parameters.

**Stage two, value-aware fine-tuning**: freeze HSD/PTD, train only the HTE multi-task towers on real feedback (BCE loss), introducing click/conversion business supervision.

**Stage three, HEPO (Hierarchy Enhanced Policy Optimization)**: policy gradients at both token level and item level simultaneously. Token-level advantage $A_{\text{token}}^l = v^l - \bar{v}^l$ (variance far smaller than item level); item-level reward $R_{\text{item}} = b_i \cdot \hat{r}_i^{\text{pctr}} + \lambda \cdot \hat{r}_i^{\text{pcvr}}$; hierarchical aggregation $A_{\text{item}} = \sum_{l=1}^{C} \gamma^l A_{\text{token}}^l$. The loss:

$$\mathcal{L}_{\text{HEPO}} = -\mathbb{E} \left[ \sum_{l=1}^C A_{\text{token}}^l \log \pi_\theta(a^l) + \beta \cdot A_{\text{item}} \log \pi_\theta(\text{item}) \right]$$

Benefits: low variance (small token space), fine-grained control (locating which token layer causes low value), and fast convergence (dense token-level gradient signals).

### Design Trade-offs

GPR fully launched on WeChat Channels ads. Compared with the cascaded system: GMV and CTCVR improved, inference latency dropped from 200ms+ to 80ms, and the model count went from 5 independent models down to 1. The trade-offs:
- **Architectural complexity vs. scene generality**: HHD's three layers + Thinking-Refining-Generation take more than 2× EGA's code volume, but buy cross-scene unification (Channels/Moments/Official Accounts share one model).
- **Pre-training cost vs. zero-shot transfer**: pre-training consumes thousands of GPU cards for weeks, but launching a new scene requires only light fine-tuning.
- **End-to-end optimization vs. interpretability**: the black box makes anomalies hard to localize, partially mitigated by visualizing Thinking Tokens and HTE's layered value outputs.

---

## ⚠️ Common Mistakes in 8.3

| # | Mistake | Example | Why It's Wrong | Fix |
|---|---------|---------|---------------|-----|
| 1 | Ignoring advertising's economic constraints | "Ads just optimize CTR too" | Advertisers game their bids; IC/IR needed | Use a payment network + ex-post regret to preserve IC |
| 2 | Concatenated generation of POI and creative | "Autoregress over the 6 tokens together" | Easily generates POI-creative mismatches | Probability decomposition: POI first, then creative |
| 3 | Avg aggregation for token bids | "Take the ad set's average bid" | High-bid signals get diluted by low bids | Use max aggregation to highlight high-value tokens |
| 4 | Paying directly by generation probability | "p_i ∝ z(a_i^j)" | Non-differentiable and hard to keep IC | Decouple allocation/payment; independent payment network |
| 5 | All RQ-VAE causing codebook collapse | "Randomly initialized codebook, end-to-end" | Dead codes leave utilization at only 65% | RQ-Kmeans+ first for high-quality initialization |
| 6 | Unconstrained Beam Search | "Decode over the full codebook W^3" | Generates many invalid candidates, adding latency | Trie constraints + HTE value-guided pruning |

---

## Chapter Summary

### 📌 Key Takeaways

| Concept | Key Points | Why It Matters |
|---------|-----------|----------------|
| Triple constraints | IC/IR, POI + creative joint generation, allocation-payment decoupling | Economic challenges unique to advertising vs. recommendation/search |
| EGA | Dual decoders + token-level max bidding + POI-level payment network | Deep unification of auction mechanism and generative model |
| ex-post regret + Lagrangian | Sampling approximates regret; dual updates of λ | Approximately guarantees IC while balancing revenue |
| Permutation-aware reward | Self-Attention models position externalities | Ads are not independent; point-wise estimation fails |
| GPR | Four token types + RQ-Kmeans+ + HHD + value-guided Trie Beam Search | Unified ad generation across scenes and ultra-long sequences |
| HEPO | Token-level + item-level hierarchical policy gradients | Low variance, fine-grained control, fast convergence |

### ❓ FAQ

**Q1: Why does EGA's token bidding use max rather than avg?**
> A: One semantic token may correspond to multiple ads. If one of them bids high, generating that token carries high commercial value and its probability should be boosted. Avg dilutes the high-bid signal with the low-bid ads in the same group; max highlights the value peak.

**Q2: Why must allocation and payment be decoupled?**
> A: If you pay directly by generation probability, the probability is non-differentiable and the "winner's curse" pushes advertisers to under-bid. Decoupled, allocation uses bid-guided generation (differentiable Softmax) and payment uses an independent network learning the IC function (Sigmoid preserves IR) — only then can IC be approximately guaranteed with mathematical constraints.

**Q3: What makes GPR's Trie Beam Search better than standard Beam Search?**
> A: Standard Beam Search expands over the full codebook $W^3$, generating many invalid candidates (exhausted budgets/targeting mismatches/geo restrictions) requiring post-processing. The Trie pre-filters valid ads by constraints so decoding walks only legal branches early; then the beam width is dynamically adjusted by HTE values, cutting latency 47% and raising the valid-candidate share to 95%.

### 🔗 Connections to Later Chapters

- **8.1** (end-to-end generative recommendation) provides the semantic ID / Enc-Dec / RL alignment foundations for EGA and GPR.
- **8.2** (end-to-end generative search) covers hard-constraint retrieval (KHQE, constrained Beam Search), carried forward in GPR's Trie-constrained decoding.
- **6.x** (generative foundations) covers RQ-VAE quantization, appearing here in two forms: EGA's RQ-VAE and GPR's RQ-Kmeans+.
- **9.1–9.3** (generative thinking/reasoning) further discuss how "reasoning steps" like Thinking Tokens improve generation quality, complementing GPR's PTD Thinking-Refining stage.

---

## Practice Problems

Work through all problems in order — they get progressively harder. Each has a complete solution you can reveal after trying it yourself.

---

**Problem 8.3.1 — Token-Level Bid Aggregation** 🟢 Easy

A semantic token $a^j$ corresponds to 3 ads with bids $b_1=2.0, b_2=0.5, b_3=3.0$. Find (a) the token bid $b(a^j)$ under max aggregation; (b) the result under avg aggregation; (c) why is max more reasonable?

<details>
<summary>💡 Solution (click to reveal)</summary>

**Approach:** Apply the aggregation formula directly.

- (a) $b(a^j) = \max(2.0, 0.5, 3.0) = 3.0$.
- (b) avg = $(2.0+0.5+3.0)/3 = 5.5/3 \approx 1.83$.
- (c) This token contains a high-bidding ad ($b=3.0$), so generating it has high commercial value; max concentrates probability mass on this value peak, while avg is diluted by $b=0.5$, weakening the high-bid signal — exactly max's design motivation.

**Key points:**
- Max highlights value peaks; avg smooths away extremes.
- Bid aggregation in generative advertising is fundamentally a strategy for handling the "many-to-many" mapping.

</details>

---

**Problem 8.3.2 — Payment Rate and the IR Constraint** 🟢 Easy

An advertiser reports bid $b_i=5.0$, and the payment network outputs payment rate $\hat{p}_i=0.6$. Compute the actual payment $p_i$, and determine whether the individual rationality (IR) constraint $p_i \leq b_i$ holds.

<details>
<summary>💡 Solution (click to reveal)</summary>

**Approach:** $p_i = \hat{p}_i \cdot b_i$.

$p_i = 0.6 \times 5.0 = 3.0$. Since the Sigmoid guarantees $\hat{p}_i \in [0,1]$, we have $p_i = 0.6 \times 5.0 \leq 5.0 = b_i$ — the IR constraint holds.

**Key points:**
- The payment rate naturally falls in [0,1] via Sigmoid, so $p_i \leq b_i$ holds automatically.
- IR is the basic precondition for advertisers to participate in the auction (they never pay more than their bid).

</details>

---

**Problem 8.3.3 — ex-post regret intuition** 🟡 Medium

Advertiser $i$ has true valuation $v_i=10$. With truthful bidding $b_i=10$, the payment is $p_i=4$ and pCTR=0.5, so utility $u=(10-4)\times0.5=3$. If they misreport $b'_i=6$, the new payment is $p'_i=2$ with pCTR unchanged, giving utility $u'=(10-2)\times0.5=4$. Compute the ex-post regret $\text{rgt}_i$, and state whether this mechanism approximately satisfies IC.

<details>
<summary>💡 Solution (click to reveal)</summary>

**Approach:** regret = maximum gain from misreporting − truthful utility.

$\text{rgt}_i = \max_{b'_i}\{u_i(v_i; b'_i, \boldsymbol{b}_{-i}) - u_i(v_i; v_i, \boldsymbol{b}_{-i})\} = 4 - 3 = 1 > 0$.

The mechanism **does not satisfy** IC: the advertiser obtained higher utility by misreporting (shading down the bid) (4 > 3), yielding positive regret. EGA's goal is precisely to press $\widehat{\text{rgt}}_i$ toward 0 via Lagrangian optimization — in this example, the payment network must be adjusted so that truthful bidding becomes the optimal strategy.

**Key points:**
- $\text{rgt}_i=0$ is the criterion for IC to hold.
- Positive regret means the mechanism can be gamed; the payment network must learn to correct it.

</details>

---

**Problem 8.3.4 — Value-Guided Beam Width** 🔴 Hard

At Beam Search layer $l$, a token has value $v^l=0.8$; the mean value across all current branches is $\bar{v}=0.5$; the temperature is $\tau=0.3$; the base beam width is $B_{\text{base}}=8$ and the minimum beam width is $B_{\text{min}}=2$. Compute this branch's next-layer beam width $B_{\text{next}}$. If another branch has $v^l=0.45$ (below the mean), what is its beam width?

<details>
<summary>💡 Solution (click to reveal)</summary>

**Approach:** Apply the value-based dynamic adjustment formula.

Branch 1 ($v^l=0.8$):
$\exp((0.8-0.5)/0.3) = \exp(1.0) \approx 2.718$
$B_{\text{next}} = \max(2, 8 \times 2.718) = \max(2, 21.7) = 21.7$

Branch 2 ($v^l=0.45$):
$\exp((0.45-0.5)/0.3) = \exp(-0.167) \approx 0.846$
$B_{\text{next}} = \max(2, 8 \times 0.846) = \max(2, 6.77) = 6.77$

**Answer:** The high-value branch's beam width expands to about 21.7 (exploring more), and the low-value branch shrinks to about 6.77 (but still keeps $B_{\text{min}}=2$, so it isn't abandoned entirely).

**Key points:**
- Higher value means wider beams, achieving "explore deep on high value, retract early on low value."
- $B_{\text{min}}$ guarantees even low-value branches retain a little exploration, avoiding premature misses.

</details>

---

**🏆 Challenge: Arguing the Case for End-to-End Advertising**

A local-life platform's ad system is currently a five-stage cascade of "retrieval → ranking → creative → auction → allocation," training a separate model for each of three scenes: video, feed, and search. Write roughly 170 words arguing, when introducing a GPR-style end-to-end generative architecture: (1) how the four token types unify the three scenes; (2) versus EGA, which two designs give GPR its breakthroughs on ultra-long sequences and inference efficiency; (3) what new risks to watch for?

<details>
<summary>💡 Hint</summary>

(1) The four token types (U/O/E/I) represent the content and ads of video, feed, and search in one semantic system, forming a coherent cross-scene behavioral timeline that breaks data silos and model fragmentation. (2) Ultra-long sequences rely on HSD's Hybrid Mask + MoR recursive reasoning and Q-Former-style compression; inference efficiency relies on value-guided Trie Beam Search filtering invalid candidates early in decoding, cutting latency nearly in half. (3) New risks: HHD's architecture and the Thinking-Refining paradigm take 2×+ EGA's code volume with high training cost; the end-to-end black box offers poor interpretability, making bad cases hard to localize (mitigated by visualizing Thinking Tokens and HTE's layered value outputs).

</details>
