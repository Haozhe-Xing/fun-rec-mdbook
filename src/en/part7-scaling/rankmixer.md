<div class="badge-row">
  <span style="background: #f0f4ff; color: #4A6CF7; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(74,108,247,0.2);">📖</span>
  <span style="background: #fef2f2; color: #dc2626; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(100,100,100,0.15);">⏱️ ~50 min read</span>
  <span style="background: #fef2f2; color: #dc2626; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(100,100,100,0.15);">🎯 Advanced</span>
</div>

# RankMixer: Hardware Efficiency Optimization

> 📝 **Before You Continue:** You have read [7.3 MTGR](./mtgr.md) (feature compatibility) and [3.2 Feature Crossing](./../part3-ranking/feature-crossing.md) (cross-feature expressiveness). This chapter switches perspective — looking at scaling not from "can it be modeled" but from "is it worth what the GPU pays", with hardware-aware architecture design at its core.

A traditional DLRM's **MFU (Model FLOPs Utilization)** on GPU is typically only 4–5%, while large language models reach 40–60%. This ten-fold efficiency gap directly prevents recommendation models from enjoying the Scaling Law's dividends — even as parameter counts grow, most of the added compute is wasted on inefficient memory access.

The root cause is that the traditional DLRM inherited its architecture from the CPU era, exposing three fundamental problems on GPU: (1) the core operations are predominantly **memory-bound** — embedding lookups, feature crossing, and sequence modeling move far more memory than they compute; (2) the computation graph is **highly fragmented** — many independent hand-crafted modules chained together, with kernel launch overhead and global memory transfers accumulating into a bottleneck; (3) **Tensor Cores cannot be fully utilized** — most operations are small vector ops or irregular memory accesses that cannot leverage the matrix-multiply acceleration units.

**RankMixer** solves this at the root through hardware-aware architecture design. The core principle: **derive the architecture from hardware characteristics**, restructuring the recommendation model as a unified, GPU-friendly computation graph — **Token Mixing** replaces Self-Attention to cut complexity, **Per-Token FFN** captures feature heterogeneity, and **Sparse MoE** enables parameter-efficient scaling.

---

## 7.4.0 The RankMixer Architecture

The model's core is $L$ stacked RankMixer Blocks, each containing Multi-head Token Mixing (replacing Self-Attention) and a Per-Token FFN (capturing feature heterogeneity). Input features are tokenized into $T$ tokens of uniform dimension, pass through $L$ blocks, and produce an output via mean pooling.

![RankMixer overall architecture: Tokenization → L Blocks → Mean Pooling](../images/part7-rankmixer-arch.svg)

After input features are tokenized, they pass through multiple RankMixer Blocks (each = Token Mixing + Per-Token FFN), and mean pooling finally produces the logit. All core operations are matrix multiplications.

Each RankMixer Block's forward pass:

$$\begin{aligned}
\boldsymbol{S}_{n-1} &= \text{LN}(\text{TokenMixing}(\boldsymbol{X}_{n-1}) + \boldsymbol{X}_{n-1}) \\
\boldsymbol{X}_n &= \text{LN}(\text{PFFN}(\boldsymbol{S}_{n-1}) + \boldsymbol{S}_{n-1})
\end{aligned}$$

Overall complexity is $O(LTD^2)$. In the Sparse MoE version, the Per-Token FFN can be replaced with expert networks, expanding parameters while keeping inference cost. The design follows three principles: (1) all core operations are matrix multiplications, fully exploiting Tensor Cores; (2) the computation graph stays as simple as possible to reduce kernel launch overhead; (3) the expressiveness required by recommendation tasks is preserved.

---

## 7.4.1 The Token Mixing Mechanism

Self-Attention complexity is $O(T^2D)$ (from computing the full token-pair similarity matrix $QK^T$). In recommendation, feature counts can reach hundreds or thousands, making the $T^2$ term a significant bottleneck. RankMixer's core insight: **what recommendation tasks need is information mixing between tokens, not similarity-based dynamic weighting (attention)**. For example, learning high-order interactions like "young users in tier-1 cities prefer tech items" is essentially fusing information from multiple tokens into new representations — it does not necessarily require explicitly computing token-pair similarities.

Token Mixing's core idea: **mix along the feature dimension rather than the token dimension**. Given input $\boldsymbol{X} \in \mathbb{R}^{T\times D}$, there are two steps:

**Step 1, Multi-head decomposition** — each token is decomposed into $H$ heads: $[\boldsymbol{x}_t^{(1)} \| \cdots \| \boldsymbol{x}_t^{(H)}] = \text{SplitHead}(\boldsymbol{x}_t)$, where $\boldsymbol{x}_t^{(h)} \in \mathbb{R}^{D/H}$ is head $h$ of token $t$.

**Step 2, Token-wise mixing** — within each head, concatenate that head's portion across all tokens: $\boldsymbol{s}^{(h)} = \text{Concat}(\boldsymbol{x}_1^{(h)}, \ldots, \boldsymbol{x}_T^{(h)}) \in \mathbb{R}^{TD/H}$.

The key: **change the data layout from "per token" to "per head"**. The original $\boldsymbol{X}$ is $T$ vectors of length $D$; after SplitHead+Concat it becomes $H$ vectors of length $TD/H$, with different tokens' features densely packed within each head, creating the conditions for mixing. In practice $H=T$, so each "head" holds a slice of every token's features. After mixing the token count is unchanged, enabling residual connections.

![Token Mixing: reorganize by head, then mix along the feature dimension, avoiding the $T^2$ term](../images/part7-rankmixer-token-mixing.svg)

Left: Self-Attention computes the full $T\times T$ similarity matrix ($O(T^2D)$); right: Token Mixing rearranges by head and applies a linear transformation along the feature dimension ($O(TD)$), with no softmax.

Complexity-wise, Token Mixing costs $O(TD)$ (mainly memory rearrangement). Compared with self-attention's $O(T^2D+TD^2)$, this avoids the $T^2$ term when $T$ is large (hundreds to thousands in recommendation) — a significant reduction. And Token Mixing has no softmax normalization (which requires an extra reduction kernel), further cutting kernel overhead.

The key question: **without explicitly computing token-pair similarities, how are interactions still captured?** The answer is **stacking multiple layers**. A single Token Mixing layer is "feature-level mixing" — the same feature dimension across tokens influences each other (because they are concatenated into one vector). With multiple stacked layers, layer 1's per-token output fuses first-order information from all tokens; layer 2's input is already a mixed result where each token contains other tokens' information, so layer 2's mixing achieves second-order interactions:

$$\boldsymbol{X}_1 = f(\boldsymbol{X}_0),\quad \boldsymbol{X}_2 = f(f(\boldsymbol{X}_0)),\quad \ldots,\quad \boldsymbol{X}_L = \underbrace{f\circ\cdots\circ f}_{L\text{ times}}(\boldsymbol{X}_0)$$

Each Token Mixing layer applies a linear transformation along the feature dimension (followed by FFNs), so stacking $L$ layers can model $L$-th order polynomial interactions between tokens. In practice $L$ is usually 6–12 layers, enough to capture the needed high-order crossings.

From the hardware's perspective, Token Mixing's core is data rearrangement (SplitHead, Concat) that can be implemented with efficient kernels, designed as contiguous memory reads/writes (coalesced memory access) to fully use bandwidth. Compared with attention's softmax (a global normalization), Token Mixing is local and parallel. More importantly, SplitHead, Concat, and the FFN can be fused into a single kernel, reducing kernel launch overhead — the key to the MFU improvement.

> **Analysis:** Token Mixing replaces "token-pair similarity + softmax" with "feature-dimension mixing + multi-layer stacking", dropping $O(T^2D)$ to $O(TD^2)$ overall, with all operations fusable into matrix-multiply kernels. What is sacrificed is attention's "dynamic similarity routing"; what is gained is GPU utilization — for recommendation, where features are numerous and interaction patterns fairly fixed, this trade is well worth it.

---

## 7.4.2 Per-Token FFN

The standard Transformer's FFN uses the same weights for all tokens: $\text{FFN}(\boldsymbol{x}) = \boldsymbol{W}_2\cdot\text{GELU}(\boldsymbol{W}_1\boldsymbol{x}+\boldsymbol{b}_1)+\boldsymbol{b}_2$. This is reasonable for LLMs (all tokens share one semantic space). But recommendation features live in entirely different semantic spaces: user IDs imply implicit preferences, item categories are coarse-grained classes, click rates follow long-tail distributions, timestamps have periodicity. Forcing the same FFN on them loses parameter efficiency.

RankMixer's core design: **each token gets its own FFN parameters**. For the $t$-th token:

$$\boldsymbol{v}_t = \boldsymbol{W}_{\text{pffn}}^{t,2}\cdot\text{GELU}(\boldsymbol{W}_{\text{pffn}}^{t,1}\boldsymbol{s}_t + \boldsymbol{b}_{\text{pffn}}^{t,1}) + \boldsymbol{b}_{\text{pffn}}^{t,2}$$

Each token's $\boldsymbol{W}_{\text{pffn}}^{t,i}$ is independent, so: (1) each token learns a transformation specific to its semantic space; (2) high-information tokens are automatically allocated more parameter capacity; (3) different semantic spaces no longer interfere.

Per-Token FFN is fundamentally different from MMoE. In MMoE, multiple experts share the same input, and gating dynamically weights a combination of expert outputs: $\boldsymbol{y}=\sum_i g_i(\boldsymbol{x})\cdot\text{Expert}_i(\boldsymbol{x})$ (all experts see the same $\boldsymbol{x}$). Per-Token FFN gives each token its own input and its own FFN: $\boldsymbol{v}_t=\text{FFN}_t(\boldsymbol{s}_t)$ (each FFN sees a different $\boldsymbol{s}_t$). Parameter isolation ensures learning in different feature spaces stays independent, preventing high-frequency features from dominating low-frequency ones.

For parameter efficiency, with $T$ tokens, Per-Token FFN has total parameters $\text{Param}=2TkD^2$ — $T$ times the shared FFN's ($2kD^2$). But **computational complexity is unchanged**: $\text{FLOPs}=2TkD^2$, identical to the shared FFN (which must also compute once per token for $T$ tokens). The added parameters are "specialized" — each block serves only one token, giving higher learning efficiency.

Cross-feature-space interaction happens through the Token Mixing layers. Per-Token FFN focuses on deep modeling within each space; Token Mixing ensures information flows between tokens. This "mixing + per-token processing" combination preserves parameter isolation while achieving thorough cross-space interaction through multi-layer stacking.

---

## 7.4.3 Sparse MoE Scaling

With Token Mixing and Per-Token FFN in place, how do we scale to billions or even tens of billions of parameters? Directly adding depth/width scales compute linearly, and inference latency grows proportionally — industrially unacceptable. **Sparse MoE (Sparse Mixture of Experts)** provides the solution: not all parameters participate in every sample's computation; a subset of experts is dynamically selected per sample. The model can have a huge parameter count while per-sample compute stays fixed (activating only a few experts).

The ideal MoE pattern has each expert specialize in some sample pattern. But achieving effective expert specialization in recommendation faces three challenges: (1) the input is high-dimensional sparse features whose combinatorial space is exponential; representations are scattered across the high-dimensional space without clear cluster structure, making stable routing hard for the gating to learn; (2) data is extremely imbalanced — head users account for 50% of samples; if gating routes many head samples to one expert early on, that expert receives more gradients, gating keeps sending it more, and a few experts end up processing most samples (expert overload) while the rest go nearly unused (expert underutilization); (3) even if training is load-balanced, request distributions at inference may differ, making some experts bottlenecks and increasing latency variance.

RankMixer counters with two complementary training strategies. First, **ReLU Routing** — standard MoE uses Top-$k$ + Softmax routing, activating a fixed $k$ experts per token. RankMixer's ReLU Routing lets each token activate a variable number of experts:

$$G_{i,j} = \text{ReLU}(h(\boldsymbol{s}_i)),\quad \boldsymbol{v}_i = \sum_{j=1}^{N_e} G_{i,j}\cdot e_{i,j}(\boldsymbol{s}_i)$$

ReLU outputs can be 0 (not activated) or positive (activated), so high-information tokens may activate more experts. To control sparsity, a regularization is added: $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda\mathcal{L}_{\text{reg}}$, where $\mathcal{L}_{\text{reg}} = \sum_{i,j} G_{i,j}$ and $\lambda$ controls the average number of activated experts.

Second, **Dense-Training / Sparse-Inference (DTSI-MoE)** — Per-Token FFN already multiplies parameters by $T$; adding MoE on top expands expert count further, easily causing expert under-training. DTSI-MoE uses two routers: during training, a dense router $h_{\text{train}}$ activates all or most experts to ensure sufficient training; at inference, a sparse router $h_{\text{infer}}$ activates only a few experts to cut compute. Both routers train simultaneously; only $h_{\text{infer}}$ is constrained by $\mathcal{L}_{\text{reg}}$:

$$G_{i,j}^{\text{train}} = \text{ReLU}(h_{\text{train}}(\boldsymbol{s}_i)),\quad G_{i,j}^{\text{infer}} = \text{ReLU}(h_{\text{infer}}(\boldsymbol{s}_i))$$

During training, the forward pass uses $G^{\text{train}}$ while $G^{\text{infer}}$ is computed alongside and given the sparsity regularization; at inference, only $h_{\text{infer}}$ is used. This achieves sufficient training, efficient inference, and consistent strategy.

Load balancing is achieved through the soft constraint of $\mathcal{L}_{\text{reg}}$. Expert $j$'s total activation in a batch is $A_j = \sum_i G_{i,j}$, and the regularization can be rewritten as $\mathcal{L}_{\text{reg}} = \sum_j A_j$. When some expert's $A_j$ grows too large, the gradient $\partial\mathcal{L}_{\text{reg}}/\partial h_{\text{infer}}$ suppresses its activation probability, achieving load balance. Compared with hard constraints (capacity limits), a soft constraint never force-assigns suboptimal experts when one is saturated, preserving routing flexibility.

![RankMixer's hardware efficiency: MFU from 4% to 45%, with unifying everything as matrix multiplications at the core](../images/part7-rankmixer-moe.svg)

Left: the traditional DLRM's fragmented computation graph (memory-bound embedding lookups, small kernels, launch overhead), with effective GEMM at only 5%; right: RankMixer's core operations are all GEMMs — Token Mixing + PFFN take ~85% of compute, and MFU reaches 45%.

The key to RankMixer's high MFU: **all core operations are compute-bound large matrix multiplications**. Token Mixing and PFFN take about 85% of compute time, all GEMMs that efficiently use Tensor Cores (a single GEMM kernel reaches 60–80% MFU). By contrast, in the traditional DLRM, embedding lookups (40% of time, memory-bound), small kernels (35% of time, MFU<10%), and launch overhead (20% of time) dominate, with effective GEMM at only 5% — which is exactly why a DLRM's MFU is 4–5% while RankMixer reaches 45%.

> 💡 **Key Insight:** RankMixer moves the recommendation model from fragmented design to a unified architectural paradigm. Algorithmically, Token Mixing drops complexity from $O(T^2D)$ to $O(TD^2)$, Per-Token FFN captures feature heterogeneity, and Sparse MoE achieves parameter-efficient scaling through ReLU Routing and DTSI-MoE. Systematically, unifying all core operations as matrix multiplications raises MFU from 4–5% to 45%, making the recommendation model a "first-class citizen" on GPU that can directly use Tensor Cores and the mature LLM toolchain, opening a path to sustained scaling. But RankMixer focuses on compute efficiency inside the model; the pipeline still has other fragmentation — sequence modeling separate from feature interaction, retrieval separate from ranking, multi-task fragmentation. The next section, OneTrans, breaks through these remaining walls.

---

## ⚠️ Common Mistakes in 7.4

| # | Mistake | Example | Why It's Wrong | Fix |
|---|---------|---------|---------------|-----|
| 1 | Assuming low MFU just means few parameters | "Add GPUs and utilization is solved" | It is architectural fragmentation + memory-bound access, not a shortage of compute | Use hardware-aware unification into GEMMs |
| 2 | Assuming Token Mixing loses interactions | "Without token-pair similarity there are no crossings" | Multi-layer stacking achieves $L$-th order polynomial interactions | Look at the $f\circ f\circ\cdots$ stacking |
| 3 | Treating Per-Token FFN as MMoE | "One expert per token is just MoE" | MMoE shares the input and weights combinations; PFFN gives each token independent input/parameters | Distinguish parameter isolation from routed weighting |
| 4 | Using Top-k Softmax routing | "MoE should always activate a fixed k" | Sparse recommendation features make stable routing hard, and imbalance causes overload | Use ReLU Routing for dynamic activation |
| 5 | Ignoring the necessity of DTSI-MoE | "Just train sparse directly" | PFFN already multiplies parameters by T; pure sparse training under-trains experts | Two routers: dense training, sparse inference |

---

## Chapter Summary

### 📌 Key Takeaways

| Concept | Key Points | Why It Matters |
|---------|-----------|----------------|
| The MFU bottleneck | DLRM only 4–5%, LLM 40–60% | The hardware root cause of recommendation's scaling trouble |
| Token Mixing | Feature-dimension mixing replaces $QK^T$, complexity $O(TD^2)$ | Removes the $T^2$ term + fusable kernels |
| Per-Token FFN | Independent FFN parameters per token, unchanged complexity | Captures feature heterogeneity, parameter isolation |
| Sparse MoE | ReLU Routing + DTSI-MoE | Parameter-efficient scaling to the billion level |
| Unify as GEMM | All core operations are matrix multiplications | MFU up to 45%, a first-class GPU citizen |

### ❓ FAQ

**Q1: Without computing similarities, can Token Mixing really replace Self-Attention?**
> A: Yes. High-order crossings in recommendation are essentially "fusing multiple tokens' information into new representations"; a single layer mixes along the feature dimension, and stacking layers achieves $L$-th order polynomial interactions. The cost is losing dynamic similarity routing, but with many features and relatively fixed patterns in recommendation, the gained GPU utilization is the better deal.

**Q2: Per-Token FFN multiplies parameters by T — why don't FLOPs change?**
> A: A shared FFN also computes once per token for T tokens (one FFN per token), so FLOPs were already $2TkD^2$; Per-Token merely swaps the shared weights for T independent sets, still computing once per token — same FLOPs. What grows is "specialized" parameters with higher learning efficiency.

**Q3: Why does recommendation MoE use ReLU Routing instead of Top-k?**
> A: Recommendation inputs are high-dimensional, sparse, and unevenly distributed; Top-k's fixed activation count easily overloads a few experts while the rest go underutilized. ReLU lets each token activate a variable number of experts based on information content, and with the regularization it achieves soft load balancing — better suited to recommendation data.

### 🔗 Connections to Later Chapters

- **7.3 (MTGR)** — also handles heterogeneous features, but with GLN + Dynamic Masking; compare with RankMixer's Per-Token FFN (separate parameters) approach.
- **7.5 (OneTrans)** — further breaks the fragmentation of "sequence modeling separate from feature interaction", extending RankMixer's hardware thinking to a unified architecture.
- **3.2 (Feature Crossing)** — the high-order crossings of DCN/xDeepFM are re-implemented in RankMixer via multi-layer Token Mixing, and in a more GPU-friendly way.

---

## Practice Problems

Work through all problems in order — they get progressively harder. Each has a complete solution you can reveal after trying it yourself.

---

**Problem 7.4.1 — The Root Cause of Low MFU** 🟢 Easy

A traditional DLRM's MFU is about 4–5%, an LLM's about 40–60%. Name the three architecture-level causes of the DLRM's low MFU.

<details>
<summary>💡 Solution (click to reveal)</summary>

**Approach:** Map to the three fundamental problems in the text.

1. Core operations are memory-bound (embedding lookup, feature crossing — memory traffic >> compute);
2. The computation graph is highly fragmented (many independent modules chained; kernel launch + global memory transfer overhead);
3. Tensor Cores are underutilized (small vectors/irregular access, not GEMM).

**Key points:**
- It is not a lack of compute; the compute is not "worth it".
- RankMixer's unification into GEMM pulls MFU to 45%.

</details>

---

**Problem 7.4.2 — Token Mixing Complexity** 🟢 Easy

Self-Attention complexity is $O(T^2D)$ and Token Mixing about $O(TD)$ (rearrangement). With $T=1000, D=256$, by about how many times do the orders of magnitude differ?

<details>
<summary>💡 Solution (click to reveal)</summary>

**Approach:** Substitute and estimate (ignore constants).

- Self-Attention: $T^2D = 10^6 \times 256 = 2.56\times10^8$.
- Token Mixing: $TD = 1000 \times 256 = 2.56\times10^5$.
- The ratio is $\approx 1000$x (i.e., $T$ times).

**Key points:**
- Token Mixing removes the $T^2$ term and scales linearly with token count.
- In recommendation, T is hundreds to thousands — the gain is significant.

</details>

---

**Problem 7.4.3 — Per-Token FFN vs MMoE** 🟡 Medium

Why are Per-Token FFN and MMoE "fundamentally different"? Summarize each one's parameter organization in a sentence.

<details>
<summary>💡 Solution (click to reveal)</summary>

**Approach:** Distinguish "parameter isolation" from "routed weighting".

- MMoE: multiple experts share the same input $\boldsymbol{x}$, and gating dynamically weights the output combination $\boldsymbol{y}=\sum_i g_i(\boldsymbol{x})\text{Expert}_i(\boldsymbol{x})$ — all experts see the same input.
- Per-Token FFN: each token has its own input $\boldsymbol{s}_t$ and its own FFN $\boldsymbol{v}_t=\text{FFN}_t(\boldsymbol{s}_t)$ — parameter isolation, preventing high-frequency features from dominating low-frequency ones.

**Key points:**
- One is "same input, weighted expert selection"; the other is "different inputs, each with its own FFN".
- Both aim to handle heterogeneous features; the mechanisms differ.

</details>

---

**Problem 7.4.4 — DTSI-MoE Design** 🔴 Hard

Explain why DTSI-MoE needs two routers ($h_{\text{train}}$ and $h_{\text{infer}}$), and what would happen if you trained with only $h_{\text{infer}}$.

<details>
<summary>💡 Solution (click to reveal)</summary>

**Approach:** Start from the tension between "sufficient training vs efficient inference".

Per-Token FFN already multiplies parameters by T, and MoE expands the expert count further; if training also activates sparsely, many experts never receive enough gradients → expert under-training. DTSI-MoE uses $h_{\text{train}}$ during training to activate most experts (sufficient training), while only $h_{\text{infer}}$ is constrained by the sparsity regularization and used at inference. Training with only $h_{\text{infer}}$ would under-train the experts, and the deployed model would perform poorly.

**Key points:**
- Dense training preserves quality; sparse inference preserves efficiency.
- The two routers train simultaneously with a consistent strategy.

</details>

---

**🏆 Challenge: Hardware-Aware Restructuring**

You must restructure a fragmented DLRM (embedding lookup + hand-crafted crossing + DIN + MLP) into a GPU-friendly architecture. Within 150 words, state which three RankMixer components you would substitute for the existing modules, and the expected MFU change plus its precondition.

<details>
<summary>💡 Hint</summary>

Replace Self-Attention/hand-crafted crossing with Token Mixing (removes $T^2$, fusable kernels), replace the shared FFN with Per-Token FFN (captures feature heterogeneity), and scale parameters with Sparse MoE. The precondition is first tokenizing all features and unifying the computation graph as matrix multiplications; expect MFU to rise from ~5% to ~45%. This corresponds exactly to RankMixer's hardware-aware restructuring approach.

</details>
