# Glossary

> This glossary collects the core terms used in this book, grouped by part. Format: **Term** — a one-sentence definition, with cross-references to related chapters where helpful.

---

## Basic Concepts (All Parts)

- **Candidate Set** — The set of items that survive the retrieval stage and await precise ranking by the ranking model, typically on the order of thousands.
- **Discriminative Recommendation** — The paradigm that formulates recommendation as "given a user-item-context triple, predict the interaction probability"; its core is a scoring function $Score = f(User, Item, Context)$.
- **Generative Recommendation** — The paradigm in which the model directly "creates" a recommendation sequence from the user's history and the context; its core is a generation function $[I_1, I_2, \ldots, I_k] = g(User, Context)$.
- **Retrieval** — The first stage of the recommendation pipeline, which filters a corpus of hundreds of millions of items down to a few thousand candidates within milliseconds.
- **Ranking** — The second stage of the recommendation pipeline, where a complex model computes a precise prediction score for every candidate.
- **Re-ranking** — The third stage of the recommendation pipeline, which optimizes list-level experience metrics such as diversity and novelty while preserving relevance.

---

## Part 2 · Fast Candidate Retrieval

- **ItemCF (Item-based Collaborative Filtering)** — A collaborative filtering method that spreads candidates from seed items via cosine similarity over item co-occurrence (Ch2.1).
- **UserCF (User-based Collaborative Filtering)** — A collaborative filtering method that predicts a target user's interests by aggregating the behavior of similar users (Ch2.1).
- **Swing** — An industrial-grade similarity algorithm that analyzes substructures of the user-item bipartite graph and filters popularity noise via "specific co-occurrence" (Ch2.1).
- **Bipartite Graph** — A graph whose two node types are users and items, with interactions as edges; Swing analyzes its substructures to filter noise (Ch2.1).
- **FunkSVD** — The foundational matrix factorization model, which decomposes the rating matrix into user/item latent vectors and predicts ratings by inner product (Ch2.1).
- **BiasSVD** — A matrix factorization model that improves FunkSVD by introducing a global mean $\mu$, a user bias $b_u$, and an item bias $b_i$ (Ch2.1).
- **MF (Matrix Factorization)** — The family of methods that decompose user-item interactions into low-rank latent vectors, with vector distances reflecting preferences (Ch2.1).
- **Surprise** — A Swing derivative that mines complementary items at three levels: category, product, and cluster (Ch2.1).
- **Word2Vec (Skip-Gram)** — A sequence modeling method that predicts context from a center word and learns dense word vectors efficiently with negative sampling; the theoretical foundation of I2I vector retrieval (Ch2.2).
- **Item2Vec** — An I2I method that transfers Word2Vec Skip-Gram directly to recommendation, treating each user's interaction history as a "sentence" to learn item vectors (Ch2.2).
- **EGES (Enhanced Graph Embedding with Side Information)** — An I2I vector method that fuses side information over random-walk item graphs and aggregates it with item-specific attention weights, addressing cold start and sparsity (Ch2.2).
- **Two-Tower Model** — A U2I retrieval architecture in which the user and the item are encoded into vectors by independent towers that interact only through a final inner product (FM/DSSM/YouTubeDNN) (Ch2.3).
- **YouTubeDNN** — An industrial two-tower model that formulates retrieval as "predict the user's next watch" and uses an asymmetric two-tower design with temporal splits (Ch2.3).
- **MIND (Multi-Interest Network with Dynamic Routing)** — A sequential retrieval model that represents a user's diverse interests with multiple interest capsules, each retrieving independently before the results are merged (Ch2.4).
- **Dynamic Routing** — The iterative algorithm in capsule networks that determines the connection strength between lower- and higher-level capsules; MIND uses it to softly cluster behaviors into interest capsules (Ch2.4).
- **Squash Function** — The function MIND uses to nonlinearly compress a vector's norm into $[0,1)$ while keeping its direction, with the norm representing the probability that the interest exists (Ch2.4).
- **Label-Aware Attention** — The attention mechanism MIND uses during training, where the target item vector serves as the query that picks the most relevant interest capsule (Ch2.4).
- **SDM (Sequential Deep Matching)** — A sequential retrieval model that separately models short-term (LSTM + multi-head) and long-term (feature-dimension attention) interests and fuses them with a dynamic gate (Ch2.4).
- **LSTM (Long Short-Term Memory)** — The recurrent network SDM uses to handle temporal dependencies within session sequences and suppress random mis-clicks (Ch2.4).
- **Multi-Head Self-Attention** — The mechanism SDM applies after the LSTM, running multiple attention paths in parallel to capture multiple interests within a sequence (Ch2.4).
- **Trinity** — A retrieval framework that explicitly preserves full historical interests via hierarchical VQ clustering plus statistical histograms, curing interest amnesia; it contains the M/LT/L retrievers (Ch2.5).
- **Hierarchical Clustering** — The structure Trinity maintains during training via VQ: two levels of learnable cluster centers (main 128 / sub 1024) (Ch2.5).
- **Streaming VQ Index** — An index structure that quantizes items into clusters in real time, with cluster centers continuously adapting via EMA and no need to interrupt and rebuild (Ch2.5).
- **Exponential Moving Average (EMA)** — The mechanism that smoothly updates cluster centers with weighted averages of member item embeddings, letting them adapt to distribution shifts (Ch2.5).
- **Interest Amnesia** — The phenomenon where an online learning framework fits recent samples and the memory of sparse long-tail interests decays; Trinity aims to fix this (Ch2.5).
- **Trinity-L (Long-term Interest Retrieval)** — The Trinity retriever that selects seed items with a light two-tower model and then performs I2I retrieval by embedding similarity (Ch2.5).
- **Trinity-LT (Long-tail Interest Retrieval)** — The Trinity retriever that tracks long-tail clusters with streaming frequency estimation and boosts retrieval of salient long-tail behaviors (Ch2.5).
- **Merge-Sort Serving** — The serving policy of Streaming VQ that decomposes the score into "cluster-level personalization + within-cluster popularity" and uses a max-heap K-way merge to guarantee candidate contributions from every cluster (Ch2.5).

---

## Part 3 · Accurate Preference Prediction

- **Memorization** — The model learning and remembering feature combinations that co-occur frequently in history (e.g., "people who buy A also buy B"); corresponds to the Wide part of Wide&Deep (Ch3.1).
- **Generalization** — The model learning deep relations between features and handling combinations rarely seen in training; corresponds to the Deep part of Wide&Deep (Ch3.1).
- **Cross-product Features** — New features manually composed from several independent features, used by the Wide part to capture specific co-occurrence patterns `AND(a, b)` (Ch3.1).
- **Joint Training** — The Wide and Deep parts are updated simultaneously by a single loss, as opposed to training them separately and then ensembling (Ch3.1).
- **Factorization Machine (FM)** — Models second-order crossings with inner products of per-feature low-dimensional latent vectors, cutting parameters from $O(n^2)$ to $O(nk)$ and mitigating sparsity (Ch3.2).
- **Parameter Sharing** — FM expresses crossing weights as inner products of latent vectors, so features that never co-occurred can still generalize through their own vectors (Ch3.2).
- **Shared Embedding** — In DeepFM, the FM and DNN components share the same feature embeddings, balancing low-/high-order interactions against training efficiency (Ch3.2).
- **Cross Network (DCN)** — Each layer crosses with the original input residually, $x_{l+1}=x_0 x_l^\top w_l + b_l + x_l$, explicitly producing element-wise high-order crossings (Ch3.2).
- **CIN (Compressed Interaction Network, xDeepFM)** — Performs Hadamard products at the vector level and compresses them with learned weights, explicitly producing vector-wise high-order crossings layer by layer (Ch3.2).
- **Local Activation (DIN)** — The user's interest representation changes dynamically with the candidate ad, obtained by attention-weighting historical behaviors (Ch3.3).
- **Auxiliary Loss (DIEN)** — Forces the GRU hidden state to predict the next behavior so that it learns meaningful interest representations (Ch3.3).
- **AUGRU (Attention Update Gate GRU, DIEN)** — Scales the GRU update gate by attention scores, letting relevant interests pass through smoothly while suppressing interest drift (Ch3.3).
- **Session (DSIN)** — A behavior unit with concentrated intent over a time span; DSIN uses sessions as the basic unit for hierarchical sequence modeling (Ch3.3).
- **Negative Transfer / Seesaw** — The phenomenon in multi-task hard sharing where conflicting task gradients improve one objective at the cost of another (Ch3.4).
- **MMoE (Multi-gate Mixture-of-Experts)** — Each task gets a dedicated gate that weights and fuses shared experts, softly isolating gradients to ease conflicts (Ch3.4).
- **PLE / CGC (Progressive Layered Extraction / Customized Gate Control)** — Explicitly separates shared experts from task-specific experts, physically cutting off cross-task gradient interference paths (Ch3.4).
- **Sample Selection Bias (ESMM)** — The CVR model trains on clicked samples but predicts over all exposures, so the training and serving distributions mismatch (Ch3.4).
- **Entire Space Modeling (ESMM)** — Jointly optimizes on the exposure space with $pCTCVR = pCTR \times pCVR$, resolving the bias and sparsity of CVR (Ch3.4).
- **Uncertainty Weight (UWL)** — Dynamically adjusts loss weights by task uncertainty, down-weighting tasks whose uncertainty is low but whose loss is high (Ch3.4).
- **GradNorm** — Dynamically balances multi-task losses by gradient magnitude and relative training progress (Ch3.4).
- **Pareto Optimization** — Treats loss weights as learnable variables under KKT conditions, steering optimization toward the Pareto frontier (Ch3.4).
- **Multi-scenario Modeling** — Predicting the same target across different scenarios/distributions, balancing shared and scenario-specific patterns (as opposed to multi-task) (Ch3.5).
- **STAR FC (Star Topology FCN)** — Each layer's parameters fuse shared and scenario-private parameters via element-wise product, $W_p^\star = W_p \otimes W$ (Ch3.5).
- **Partitioned Normalization (PN)** — Computes Batch Norm statistics separately per scenario, avoiding cross-scenario distribution confusion (Ch3.5).
- **Gate NU (PEPNet)** — A lightweight gating unit that generates dynamic scaling weights from prior features to modulate shared parameters (Ch3.5).
- **EPNet / PPNet (PEPNet)** — EPNet personalizes embeddings at the scenario level; PPNet personalizes task-tower parameters at the sample level (Ch3.5).
- **APG (Adaptive Parameter Generation)** — Dynamically generates parameter matrices from sample-aware inputs and controls cost with low-rank factorization (Ch3.5).
- **M2M (Meta-learning Multi-scenario Multi-task)** — Uses a meta-learner to dynamically generate task-model parameters from scenario/input features (Ch3.5).

---


- **STAR FCN (Star Topology FCN)** — Each layer's parameters fuse shared and scenario-private parameters via element-wise product, $W_p^\star = W_p \otimes W$.
## Part 4 · Re-ranking for Diversity

- **List Homogenization** — The phenomenon where point-wise ranking optimization makes the top of the list highly similar; the fundamental motivation for re-ranking.
- **MMR (Maximal Marginal Relevance)** — The marginal-gain formula $\lambda\cdot\text{Rel}-(1-\lambda)\max\text{Sim}$ that greedily trades off relevance against diversity (Ch4.1).
- **MMR with Window** — An MMR variant that computes the similarity penalty using only the last $w$ selected items, cutting cost for long lists (Ch4.1).
- **DPP (Determinantal Point Process)** — A probabilistic model that measures set diversity with the determinant of a kernel matrix, precisely characterizing the mutual repulsion among multiple items (Ch4.1).
- **Kernel Matrix ($L$)** — The positive semi-definite matrix in DPP that fuses relevance and diversity, constructed as $L=\text{Diag}(r)\cdot S\cdot\text{Diag}(r)$ (Ch4.1).
- **Cholesky Acceleration** — An efficient solver that exploits the factorization $L_{Y_g}=VV^\top$ to reduce DPP greedy selection to taking $\arg\max\log(d_i^2)$ each round (Ch4.1).
- **Personalized Re-ranking** — The re-ranking paradigm that deeply integrates user personalization signals into list-level optimization and lets a model learn the optimal list end-to-end (Ch4.2).
- **PRM (Personalized Re-Ranking Model)** — A model that encodes the list with a Transformer and fuses personalization vectors (PV) to achieve end-to-end personalized re-ranking (Ch4.2).
- **Personalization Vector (PV)** — The user-item preference vector extracted from hidden-layer activations of a pretrained CTR model; the core of PRM's personalization (Ch4.2).
- **Permutation-Variant Influence** — The phenomenon where the same items in a different order lead to different user behaviors; the motivation for PRS (Ch4.2).
- **PRS (Permutation Retrieve System)** — A re-ranking model that directly optimizes the experience gain of the ordering, using a PMatch + PRank two-stage design to defuse the $n!$ combinatorial explosion (Ch4.2).
- **FPSA (Fast Permutation Searching Algorithm)** — The algorithm for PRS's PMatch stage, which uses beam search with dual CTR/Next models to quickly generate candidate permutations (Ch4.2).
- **DPWN (Deep Permutation-Wise Network)** — The network for PRS's PRank stage, which scores candidate permutations with a Bi-LSTM and picks the best by List Reward (LR) (Ch4.2).
- **List Reward (LR)** — The sum of predicted click probabilities over all positions of a permutation in PRS, used to compare the whole-list gain of candidate permutations (Ch4.2).

---


- **Re-ranking** — The tail end of the three-stage funnel, which applies list-level optimization (diversity, novelty, business rules) to the ranked candidate list to maximize the whole-screen experience.
## Part 5 · Frontier Trends

- **Data Bias** — Systematic distortion introduced at data-collection time by system policies, user habits, and similar factors (Ch5.1).
- **Selection Bias** — In explicit feedback, users only rate content they are interested in, so the observed data does not represent true attitudes (MNAR) (Ch5.1).
- **Exposure Bias** — In implicit feedback, users only see items that were recommended, so a non-interaction may stem from never being exposed rather than disinterest (Ch5.1).
- **Conformity Bias** — Users are influenced by group opinion and echo it with ratings that are neither independent nor genuine (Ch5.1).
- **Position Bias** — In list recommendation, users pay more attention to items near the top, so clicks are driven by position rather than relevance (Ch5.1).
- **MNAR (Missing Not At Random)** — The observed ratings are not a random sample, causing statistical bias (Ch5.1).
- **Popularity Bias** — The model over-learns the interaction patterns of popular items, biasing recommendations toward hits and burying the long tail (Ch5.1).
- **Feedback Loop** — Recommendations influence future user behavior, and that behavior becomes new training data, snowballing the bias (Ch5.1).
- **Matthew Effect** — The rich-get-richer vicious cycle in which popular items get more exposure, then more interactions, and then even more exposure (Ch5.1).
- **IPS (Inverse Propensity Score)** — Uses the inverse of the observation probability as a sample weight to reverse selection/exposure bias and obtain unbiased risk estimates (Ch5.1).
- **Weight Clipping** — Caps extreme IPS weights to trade off between unbiasedness and low variance (Ch5.1).
- **PAL (Position-bias Aware Learning)** — Decomposes the click probability into "probability of being seen × probability of clicking given seen", architecturally decoupling position from preference (Ch5.1).
- **Cold Start** — The predicament of new items or new users lacking interaction history, which classic collaborative filtering struggles to serve (Ch5.2).
- **CB2CF (Content-Based to Collaborative Filtering)** — Learns a mapping from content features to collaborative-filtering representations so that new items directly obtain CF-quality representations (Ch5.2).
- **Mapping Network** — The core of CB2CF: multi-layer fully connected networks that learn a nonlinear map from content space to CF embedding space (Ch5.2).
- **MetaEmbedding** — Uses meta-learning to optimize an embedding generator that can adapt quickly, improving item cold start (Ch5.2).
- **MAML (Model-Agnostic Meta-Learning)** — The "learning how to learn" meta-learning framework that learns good initializations for fast adaptation to new tasks with few samples (Ch5.2).
- **MeLU (Meta-Learned User preference estimator)** — A MAML-based user cold-start method that treats each user as an independent task and adapts quickly (Ch5.2).
- **POSO (Personalized Cold Start Modules)** — An architectural approach that uses population-specific sub-modules plus personalized gating to solve user cold start (Ch5.2).
- **Generative Retrieval** — The generative paradigm that redefines recommendation as sequence generation, autoregressively predicting the next item (Ch5.3).
- **Event Stream** — The heterogeneous sequence representation that HSTU encodes from user attributes, behaviors, and timestamps (Ch5.3).
- **Semantic ID** — The structured token tuple that TIGER encodes item content into with RQ-VAE, carrying semantics and enabling knowledge sharing and cold start (Ch5.3).
- **RQ-VAE (Residual Quantization VAE)** — A residual-quantization variational autoencoder that quantizes residuals layer by layer to generate semantic IDs (Ch5.3).
- **End-to-end Generative** — The generative form in which a single model covers the whole flow from retrieval to ranking (Ch5.3).
- **MoE (Mixture-of-Experts)** — In OneRec's decoder, activates a small number of expert sub-networks to add capacity without adding compute (Ch5.3).
- **IPA (Iterative Preference Alignment)** — The mechanism by which OneRec constructs chosen/rejected pairs from multiple candidates with a reward model and aligns preferences via DPO (Ch5.3).
- **DPO (Direct Preference Optimization)** — Optimizes preferences directly from "chosen/rejected" contrastive pairs without a separate critic (Ch5.3).


---

# Volume II Terms · The Generative Recommendation Track


- **Result Bias** — Bias in biased data carried into the recommendation results after model training.
- **Unfairness** — The system systematically discriminating against certain user groups or item categories.
- **Propensity Score** — The probability that a user-item interaction is observed, $P(O_{u,i}=1)$; the denominator of IPS weighting.
- **Naive Estimator** — The estimator that directly averages over observed data; it is biased under selection bias.
- **Semi-synthetic Experiment** — Completes a real dataset into ground truth, then samples it according to a bias model, creating a "known answer" for quantifying debiasing effects.
- **ProbSeen Module** — The lightweight module in PAL that takes only position as input and outputs the probability of being seen.
- **pCTR Module** — The deep module in PAL that excludes position information and models true user preference; used alone at inference time for debiasing.
- **Submergence** — When new users are far fewer than existing users, their personalization signals are drowned out by a training process dominated by the majority's data.
- **Item Cold Start** — New items lack user interactions, so collaborative filtering cannot compute their similarities.
- **User Cold Start** — New users lack behavior history and can only receive generic popularity-based recommendations.
- **Constraint Optimization** — In CB2CF, uses cosine-similarity constraints to ensure that mapped embeddings stay semantically consistent with true CF embeddings.
- **Meta Loss** — The loss in MetaEmbedding that balances initial quality against post-adaptation performance, e.g., $l_{meta}=\alpha l_a+(1-\alpha)l_b$.
- **Parameter Separation** — MeLU separates the shared embedding parameters $\theta_1$ from the decision parameters $\theta_2$ used for fast adaptation.
- **Personalized Gating** — The network in POSO that outputs the weights of each sub-module from user features (e.g., is_new_user).
- **Pointwise Aggregation** — The attention aggregation HSTU uses, which drops softmax normalization and preserves the strength of user preferences.
- **Generative Ranking** — Bringing autoregressive generation ideas into the ranking stage (e.g., GenRank, MTGR).
- **Action-oriented** — GenRank predicts the probability of user actions on candidates rather than item IDs, reducing computational cost.
- **User Sample Aggregation** — MTGR aggregates all of a user's candidates into a single sample, sharing the user-feature computation.
- **GLN (Group Layer Normalization)** — MTGR normalizes tokens from different semantic spaces separately.
- **Session-level Generation** — OneRec directly generates an ordered set of recommendation lists (a "session") rather than a single next item.
## Part 6 · Foundations of the Generative Recommendation Paradigm

- **Generative Recommendation** — Redefines recommendation as a sequence generation task: the model directly learns the generation probability of user interaction sequences and autoregressively produces item sequences, rather than scoring candidates one by one.
- **Discriminative Recommendation** — The modeling paradigm that learns the conditional probability $p(y=1\mid u,i,c)$, predicting the probability of a positive interaction for a given candidate item.
- **Autoregressive Modeling** — The generation scheme in which the current prediction depends on all previously generated outputs, letting information circulate along the time dimension and naturally capture sequential dependencies.
- **Atomic ID** — The random unique number that traditional recommendation assigns to each item; such IDs are mutually orthogonal with no semantic relation and generalize poorly to new items.
- **Semantic ID (SID)** — Represents an item as a fixed-length discrete token sequence, with each token drawn from a semantic codebook of controllable size, encoding hierarchical semantics while retaining collaborative information.
- **Item Tokenization** — The key technique for converting items in a recommender system into token sequences that generative models can understand and generate; the bridge between traditional recommendation data and generative models.
- **Transformer** — The deep architecture based on self-attention, adept at capturing long-range dependencies in parallel; the mainstream backbone of generative recommendation and LLMs.
- **Self-Attention** — Attention computed via the Query/Key/Value "query-match-aggregate" mechanism, letting each position of the sequence dynamically attend to information at any other position.
- **Multi-Head Attention** — Attention that computes multiple independent Q/K/V groups in parallel, each learning a different attention pattern — like multiple "experts" understanding the sequence from different angles.
- **Positional Encoding** — Encoding that injects order information into each position, in absolute (sinusoidal/learnable) and relative (bias) flavors; often extended to time-aware encodings in recommendation.
- **Relative Temporal Positional Encoding** — The time encoding used by HSTU and others, which models inter-behavior intervals with $\log(\Delta t+1)$ so the model can balance long- and short-term interests.
- **Encoder-Decoder Architecture** — The two-tower generation architecture: the encoder understands the input bidirectionally, the decoder generates autoregressively under causal masking, and cross-attention dynamically queries the input — well suited to heterogeneous inputs and outputs.
- **Decoder-Only Architecture** — The unified single-tower generation architecture: input and output are concatenated into one continuous sequence and generated autoregressively via causal self-attention alone; parameter-efficient, high MFU, and compatible with the LLM ecosystem.
- **Causal Masking** — Applying a $-\infty$ mask to future positions in the attention matrix, ensuring that the $t$-th token is predicted only from the previous $t-1$ tokens; this enables autoregression while supporting parallel training.
- **Diffusion Model** — The generative paradigm that recovers data from noise via forward noising and iterative reverse denoising, in data-space and latent-space variants, complementary to the Transformer.
- **Scaling Law** — The empirical regularity that model performance keeps improving as parameters, data, and compute grow, underpinning the parameter scaling of generative models.
- **Emergent Abilities** — Zero-shot/few-shot and other abilities that suddenly appear once model scale and data pass a threshold.
- **Pre-training** — The first LLM stage: causal language modeling (next-token prediction) on large-scale unlabeled text to build general language generation ability.
- **Instruction Tuning / SFT** — The second LLM stage: conditional language modeling on "instruction-input-output" triples, computing loss only on the outputs so the model learns to follow instructions.
- **Preference Alignment** — The third LLM stage: making outputs better match human values and preferences, via methods including RLHF and DPO.
- **RLHF (Reinforcement Learning from Human Feedback)** — Trains a reward model from human preference pairs, then optimizes the generation policy with PPO while constraining it to a reference model via KL divergence.
- **DPO (Direct Preference Optimization)** — An alignment method that needs no explicit reward model or reinforcement learning, implicitly representing reward via the ratio between the policy and a reference model and optimizing preferences in a supervision-like way.
- **VQ-VAE (Vector Quantized Variational Autoencoder)** — An autoencoder that uses a learnable codebook to discretize continuous semantic vectors into a single codebook index; the foundational technique behind semantic ID discretization.
- **Codebook** — The learnable or clustered set of discrete vectors in the VQ/RQ family; each vector (codeword) corresponds to one semantic token.
- **Straight-Through Estimator (STE)** — The trick for training quantization models: the forward pass performs discrete quantization while the backward pass approximates it as an identity mapping to pass gradients.
- **RQ-VAE (Residual Quantized VAE)** — Encodes a continuous vector into a length-$L$ token sequence via multi-layer residual quantization, reaching capacity $K^L$ with naturally emergent hierarchical semantics.
- **RQ-Kmeans** — A residual quantization scheme that builds the codebook with K-means clustering instead of gradient learning, decoupling representation learning from codebook construction; new items can be assigned SIDs via vector retrieval.
- **RQ-OPQ** — A hybrid encoding scheme in which RQ handles hierarchical semantics and OPQ (Optimized Product Quantization) handles the distinctive attributes in the last residual layer, balancing retrieval against precise long-tail discrimination.
- **SID Collision** — The phenomenon where quantization information loss maps different items to the same SID sequence; mitigated by uniform assignment or hybrid encoding for disambiguation.

## Part 7 · Scaling Generative Ranking

- **Scaling Law** — The regularity that, under a suitable architecture, model performance improves as a predictable power law with compute, data, and parameters, often of the form $L = L_0 + \beta\ln C$.
- **DLRM (Deep Learning Recommendation Model)** — Traditional deep recommendation models that rely on handcrafted features, heterogeneous modules, and item-level per-candidate scoring; representative of the long failure of Scaling Laws.
- **Generative Recommender / GR** — The paradigm proposed by Meta that treats recommendation as a stochastic process interleaving content and actions, using unified sequences plus autoregressive training for user-level modeling.
- **HSTU (Hierarchical Sequential Transduction Unit)** — The sequence model Meta customized for recommendation, with three innovations: Pointwise Aggregation, relative temporal bias, and gated feed-forward; the first to verify a Scaling Law for recommendation.
- **Pointwise Aggregation** — HSTU replaces standard attention with element-wise SiLU aggregation (no Softmax normalization), preserving the "absolute strength" of interests.
- **Relative Attention Bias / RAB (rab)** — HSTU adds learnable biases to attention scores, jointly considering position difference, time difference, and token type to model non-uniform temporal patterns.
- **Stochastic Length** — HSTU's training trick: randomly truncate over-long sequences with some probability, reducing $O(n^2)$ complexity while acting as regularization; the parameter $\alpha$ controls the aggressiveness.
- **M-FALCON** — HSTU's inference algorithm: a three-layer optimization of Batched Inference → Microbatching → KV Caching that speeds up multi-candidate ranking inference by hundreds of times.
- **Action-Oriented Organization** — GenRank's sequence organization, which makes actions the subject and items the attributes ($[a_i^{(x_i)}]$), halving sequence length and speeding up training by about 79%.
- **ALiBi (Attention with Linear Biases)** — A parameter-free relative position bias that penalizes distant query-key pairs proportionally to distance and can be fused directly into the FlashAttention kernel.
- **MTGR (Meituan Generative Recommendation)** — Meituan's hybrid paradigm: a generative architecture (Transformer + user-level aggregation) performing discriminative ranking while keeping traditional crossed features.
- **GLN (Group Layer Normalization)** — MTGR normalizes independently by token-type group (User/Seq/RT/Cand), resolving the semantic-space conflicts of heterogeneous tokens.
- **Dynamic Masking** — MTGR dynamically generates attention masks from each sample's actual token timestamps: statically fully visible, dynamically causal, and mutually independent across candidates — preventing information leakage.
- **MFU (Model FLOPs Utilization)** — The fraction of a GPU's theoretical compute spent on effective matrix multiplications; traditional DLRMs reach about 4–5%, LLMs about 40–60%.
- **RankMixer** — Alibaba's hardware-aware architecture: Token Mixing instead of Self-Attention, Per-Token FFN for heterogeneity, and Sparse MoE for parameter scaling, pushing MFU to 45%.
- **Token Mixing** — RankMixer's core operation: mixing information along the feature dimension (reorganized by head) instead of token-pair similarity, reducing complexity from $O(T^2D)$ to $O(TD^2)$.
- **Per-Token FFN** — RankMixer equips each token with dedicated FFN parameters to capture heterogeneous feature spaces; the computational complexity matches a shared FFN, but the parameters are more specialized.
- **ReLU Routing / DTSI-MoE** — RankMixer's sparse expert strategy: ReLU routing dynamically activates a variable number of experts; Dense-Training/Sparse-Inference uses dual routers to combine thorough training with efficient inference.
- **OneTrans** — ByteDance's unified architecture: a single Transformer backbone performs both sequence modeling and feature interaction, ending the module fragmentation of encode-then-interaction.
- **Mixed Parameterization** — OneTrans's parameter organization: S-tokens (sequential) share parameters while NS-tokens (non-sequential) get dedicated ones, resolving token heterogeneity conflicts.
- **Pyramid Stack** — OneTrans's progressive distillation: keep only the trailing query tokens layer by layer, with KV over all tokens, distilling information into the tail while cutting compute.
- **Cross-Request KV Caching** — OneTrans reuses the user-side KV cache across requests (appending only new events), making per-request sequence computation nearly $O(1)$ in the number of candidates.
- **encode-then-interaction** — The traditional separated paradigm: a sequence module first encodes into a fixed-length vector that is then concatenated with static features for feature interaction; information flow is constrained and execution is fragmented.

## Part 8 · End-to-End Generative Applications

- **Multi-stage Cascading Architecture (MCA)** — The funnel-style multi-module architecture (retrieval → pre-ranking → ranking → re-ranking) used by traditional recommendation/search/advertising systems, with each stage optimized independently and objectives that may conflict.
- **Semantic ID** — Encodes discrete business objects (items/products/ads) into multi-level discrete token sequences from coarse to fine, letting generative models "speak" the object within a controllable vocabulary.
- **RQ-Kmeans (Residual Quantization K-means)** — A hierarchical quantization method that runs K-means layer by layer on residuals to build the codebook; unlike end-to-end-trained RQ-VAE, RQ-Kmeans builds the codebook directly and non-end-to-end.
- **RQ-VAE (Residual Quantized VAE)** — A residual-quantization variational autoencoder trained end-to-end to discretize continuous representations into multi-level semantic IDs, commonly seen in EGA.
- **Encoder-Decoder Generation Architecture** — The unified generation structure in which the encoder bidirectionally fuses user/query context and the decoder autoregressively generates the target semantic ID sequence.
- **Lazy Decoder-Only** — The OneRec-V2 architecture: preprocesses context into static key-value pairs (Context Processor), and the decoder computes loss only on target tokens, concentrating compute where gradients are produced.
- **Scaling Law** — The predictable power-law decay of model loss with parameter count; OneRec-V2 verified this law on a recommendation model.
- **Squeezing Effect** — After reinforcement learning, the model squeezes probability mass onto its current best outputs, pressing the probabilities of some legal tokens down to levels close to illegal ones, making them hard to distinguish.
- **Format Reward** — Assigns advantage to legal generated samples and drops illegal ones, mitigating the squeezing effect and ensuring that generated sequences map to real objects.
- **P-Score (Preference Score)** — The personalized multi-objective preference score that OneRec learns with a neural network, used as the reward signal for reinforcement-learning alignment.
- **ECPO (Early Clipped GRPO)** — A preference optimization algorithm that pre-clips the policy ratio of negative-advantage samples, avoiding GRPO's gradient explosion.
- **GBPO (Gradient-Bounded Policy Optimization)** — A policy optimization algorithm that bounds RL gradients with the stable gradient of a BCE loss, supporting full sample utilization with bounded-gradient stabilization.
- **PRE (Prefix2Query Representation Enhancement)** — OneSug's prefix representation enhancement module, which retrieves co-occurring queries to enrich short-prefix semantics.
- **RWR (Reward-Weighted Ranking)** — OneSug's reward-weighted ranking strategy, which constructs preference pairs from six levels of interaction feedback and injects business value into ranking.
- **KHQE (Keyword-enhanced Hierarchical Quantization Encoding)** — OneSearch's keyword-enhanced hierarchical quantization encoding: the first 3 RQ-Kmeans layers preserve the semantic hierarchy, and the last 2 OPQ layers preserve product distinctiveness.
- **OPQ (Optimized Product Quantization)** — Optimized product quantization, which splits residuals into sub-vectors quantized independently, encoding the distinctive attributes of products.
- **Mu-Seq (Multi-view behavior Sequence injection)** — OneSearch's strategy for injecting user behavior from three views: constructed from user ID, short-term sequences, and long-term sequences.
- **PARS (Preference-Aware Reward System)** — OneSearch's preference-aware reward system with multi-stage SFT and adaptive reward models, amplifying the relevance weight 10×.
- **Incentive Compatibility (IC)** — The mechanism-design property that truthful bidding is the advertiser's optimal strategy.
- **Individual Rationality (IR)** — The mechanism-design property that an advertiser pays no more than its declared bid ($p_i \leq b_i$).
- **Position Externality** — An ad's CTR is affected by the other ads and positions in the sequence, rather than being mutually independent.
- **EGA (End-to-end Generative Advertising)** — An end-to-end generative advertising system that unifies the auction mechanism with a generative model, achieving IC/IR through token-level bidding and POI-level payment.
- **POI (Point of Interest)** — A point of interest such as a restaurant or gym — the content subject in ad generation.
- **Token-level Bidding** — The allocation mechanism that projects ad bids onto semantic tokens via max aggregation, steering the distribution of generation probability.
- **POI-level Payment Network** — An independent neural network that learns payment functions satisfying the IC constraint, decoupled from allocation.
- **Ex-post Regret** — The maximum extra utility an advertiser could gain by misreporting its bid; when it is zero, the mechanism satisfies IC.
- **Lagrangian Optimization** — Uses dual multipliers to turn "maximize revenue + regret constraint" into a loss amenable to alternating optimization.
- **GPR (Generative Pre-trained Recommender)** — An end-to-end generative advertising system using the "pre-train + fine-tune" paradigm with unified multi-scenario ultra-long sequences.
- **Four Token Types (U/O/E/I-Token)** — GPR's unified input representation: User, Organic, Environment, Item (ads).
- **RQ-Kmeans+** — Combines the high-quality initialization of RQ-Kmeans with the end-to-end optimization of RQ-VAE, mitigating codebook collapse.
- **HHD (Heterogeneous Hierarchical Decoder)** — GPR's three-layer heterogeneous hierarchical decoder: HSD for intent understanding, PTD for reasoning generation, and HTE for value assessment.
- **MoR (Mixture-of-Recursions)** — A mechanism that recursively calls the same layer multiple times to deepen inference without adding parameters.
- **Value-Guided Trie-based Beam Search** — The decoding algorithm that builds a Trie prefix tree from constraints and dynamically adjusts beam width and pruning with HTE values.
- **HEPO (Hierarchy Enhanced Policy Optimization)** — A reinforcement-learning algorithm that applies hierarchical policy gradients at both the token level and the item level.

## Part 9 · Thinking and Reasoning in Recommendation

- **Collaborative Semantics** — The meaning of item representations that a recommender system learns from behavioral co-occurrence, encoded in discrete IDs and carrying no textual semantics (9.1).
- **Language Semantics** — The lexical/syntactic meanings that large language models (LLMs) acquire from pre-training on text (9.1).
- **Semantic Gap** — The divide that prevents direct alignment between collaborative semantics (discrete IDs) and language semantics (natural language) (9.1).
- **Semantic Index / Semantic ID** — Encodes items into discrete token sequences via hierarchical quantization (e.g., `<A37><B12><C5><D8>`) that are both understandable by LLMs and carry collaborative semantics (9.1).
- **Uniform Semantic Mapping** — LC-Rec's mechanism of introducing a uniformity constraint on the last quantization layer and using optimal transport (Sinkhorn-Knopp) to mitigate index collisions (9.1).
- **Multimodal Embedding Concatenation** — PLUM concatenates text/visual/audio/collaborative embeddings, fusing heterogeneous signals to build semantic IDs (9.1).
- **Multi-Resolution Codebook** — PLUM uses codebooks of different sizes at different quantization levels (128/256/512/1024), matching the coarse-to-fine principle from information theory (9.1).
- **Explicit Reasoning** — The model first generates a structured, auditable reasoning chain before outputting recommendations, unlike implicit black-box scoring (9.2).
- **Reasoning Scaffolding** — OneRec-Think's mechanism of progressive prompt templates and tasks that guide the model to "learn to think" (9.2).
- **Multi-Validity** — In recommendation, one user typically has multiple valid recommendations, with no single correct answer (9.2).
- **Recommendation-Specific Reward** — A multi-dimensional reward function combining collaborative similarity, content relevance, reasoning coherence, and user feedback (9.2).
- **GRPO (Group Relative Policy Optimization)** — A reinforcement-learning method that samples multiple rollouts per sample and updates the policy by relative reward rather than an absolute standard (9.2).
- **Think-Ahead Architecture** — The deployment strategy of offloading dense reasoning from the online critical path to asynchronous pre-computation whenever user behavior updates (9.2).
- **Autonomous Reasoning** — The model evolves its reasoning strategy autonomously from task feedback alone, without manual templates or teacher demonstrations (9.3).
- **Imitation Learning** — The reasoning-learning paradigm, as in OneRec-Think, that depends on manual templates or teacher knowledge (9.3).
- **Exploratory Learning** — The learning paradigm, as in RecZero, that relies on reinforcement-learning trial-and-error with feedback to discover strategies autonomously (9.3).
- **Think-before-Recommendation Template** — RecZero's prompt that defines only a four-step frame of "analyze the user / analyze the items / match / score", leaving the content for the model to explore (9.3).
- **Cold-start SFT** — RecOne initializes reasoning ability with a small number of high-quality (bias-corrected) reasoning samples (9.3).
- **Hybrid Paradigm** — The reasoning-learning idea that supervision provides the "language" and reinforcement learning provides the "wisdom" — framework first, refinement later (9.3).

## Part 10 · Diffusion Models for Recommendation

- **Diffusion Model** — A generative model that learns the data distribution via forward noising and learned reverse denoising (10.1).
- **Pixel-Space Diffusion** — Adds and removes noise directly in the raw data space (pixels/interaction vectors); DDPM is representative; computationally expensive (10.1).
- **Latent Diffusion (LDM)** — Encodes into a low-dimensional latent space before diffusing, then decodes at the end; Stable Diffusion is representative; more commonly used in recommendation (10.1).
- **Forward Diffusion** — Gradually adds Gaussian noise to the data along a Markov chain so that x_T approaches a standard Gaussian (10.1).
- **Reverse Denoising** — Trains a denoising network to recover x_0 from x_T step by step (10.1).
- **Reparameterization Trick** — Samples noised data at any t directly from x₀ via x_t = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε (10.1).
- **ε-prediction** — The denoising network predicts the added noise; the standard DDPM parameterization (10.1).
- **x₀-prediction** — The denoising network directly predicts the original data; better suited to recommendation scenarios (10.1).
- **Classifier-Guided** — Uses the gradients of a pretrained classifier to steer generation toward a target class (10.1).
- **Classifier-Free Guidance** — Randomly drops the condition during training and linearly combines conditional/unconditional predictions at inference (10.1).
- **v-prediction** — The parameterization predicting the "velocity" v = αₜε − σₜx₀; more stable training (10.3).
- **Sequential Augmentation** — Generates "prior" interactions for short-history users to expand their history; DiffuASR is representative (10.2).
- **SU-Net (Sequential U-Net)** — DiffuASR's U-Net variant that treats the embedding sequence as a multi-channel "image" (10.2).
- **Rounding** — Maps denoised continuous embeddings back to the nearest discrete item IDs (10.2).
- **Multi-Scenario Augmentation** — Borrows knowledge from data-rich scenarios to augment cold-start scenarios; Diff-MSR is representative (10.2).
- **Segmented Noise** — Diff-MSR keeps structure with small β early on, then grows it linearly to converge to a Gaussian (10.2).
- **Asymmetric Diffusion** — The forward pass uses discrete dropout in the raw feature space while the reverse pass denoises in latent space; AsymDiffRec is representative (10.3).
- **Feature Dropout** — AsymDiffRec's forward pass randomly drops features to mimic real missingness, fitting recommendation better than Gaussian noise (10.3).
- **Step Embedding** — A binary vector marking which features are missing, guiding completion in latent space (10.3).
- **Slate** — A set of items consumed as a whole (e.g., a playlist or a bundle), requiring coordination and diversity (10.3).
- **DMSG** — A model that generates diverse slates from text prompts with conditional diffusion, using v-prediction (10.3).
- **DDIM Acceleration** — Deterministic sampling acceleration that cuts inference from thousands of steps down to tens (10.3).

## Part 11 · Building a Generative Recommendation System

- **Offline System** — The "production" subsystem: processes full historical data, trains models, and computes item vectors and similarities, prioritizing quality over latency and producing model files and feature indexes.
- **Online System** — The "serving" subsystem: receives real-time requests, invokes models, and assembles recommendation results, targeting hundred-millisecond latency and depending on the models and features produced offline.
- **Funnel Architecture** — The classic structure of industrial recommendation: light models filter candidates quickly in retrieval, heavy models score precisely in ranking, and re-ranking optimizes experience, narrowing the candidate pool stage by stage.
- **Snake Merge** — A multi-source retrieval fusion strategy that takes candidates round-robin from each source (A→B→C→C→B→A…), ensuring every source is represented in ranking and improving diversity and coverage.
- **Cold Start** — New users/items lack behavioral data, breaking traditional collaborative filtering and vector retrieval; this project handles it with a dedicated cold-start flow (UCB/preference/popularity).
- **UCB (Upper Confidence Bound)** — An algorithm balancing exploration and exploitation: score = historical average reward + exploration bonus, giving under-explored categories more exploration opportunities and avoiding filter bubbles.
- **Exploration vs Exploitation** — The fundamental trade-off in recommendation: exploitation recommends known high-quality content while exploration tries new categories to discover potential interests; UCB unifies both in a single formula.
- **Hard Negatives** — Items the user was exposed to but did not interact with positively; hard to distinguish, they sharpen the model's discrimination.
- **Random Negatives** — Sampled randomly from items the user has not interacted with to expand the negative pool; this project mixes them with hard negatives at 1:2 to reach a 1:3 positive-negative ratio.
- **Sliding Window Samples** — How YoutubeDNN training samples are built: given the user's first $k$ views, predict the $k+1$-th, simulating "predict the next watch".
- **Left Padding** — Zero-pads variable-length behavior sequences on the left up to a fixed length so that the most recent behavior sits at the right end, matching temporal order and fitting RNN/Transformer.
- **Item Vector Pre-computation** — Offline batch computation and normalization of all item vectors by the item tower for millisecond-level online vector retrieval; the key to two-tower scalability.
- **Version Pointer (active.json)** — The pointer file in the deployment directory recording which model version to load; updates deploy the new version first and then flip the pointer, achieving transparent hot updates and rollback.
- **Consecutive Dispersion** — A diversity re-ranking strategy that forbids more than $N$ consecutive items sharing an attribute (genre/era), improving list diversity while preserving order.
- **Order Preservation** — The re-ranking algorithm keeps the original order as much as possible under the constraints — high scores still come first with only minor position adjustments — balancing relevance and diversity.
- **Pinia** — Vue's officially recommended state-management library, centralizing cross-component shared state (e.g., user authentication) in Stores, with state changes driving dependent components to re-render.
- **Debounce** — A frontend request-rate control technique: the request fires only after the user stops typing for a while (300 ms in this project), keeping live search from hammering the API.
- **Singleton** — The design for online resource loading: one process-level instance with lazy loading — models and vocabularies load once and are shared by all requests, avoiding repeated loading and memory bloat.
- **Graceful Degradation** — Falls back to a suboptimal strategy when a model is unavailable (e.g., ranking by retrieval score when ranking fails), keeping the service highly available instead of erroring out.
- **Data Loop** — User behavior is collected by the frontend and written back to the backend and storage; the updated features then influence the next recommendation so the system keeps improving — the frontend is the collection end of the loop.
- **Docker Compose** — A declarative multi-container orchestration tool that describes all services plus their dependencies, networks, and volumes in a single YAML and starts the whole system with one command.
- **Multi-stage Build** — A Dockerfile technique: build the artifacts in a builder image (e.g., Node), then copy them into a lightweight runtime image (e.g., Nginx); the final image contains no dev dependencies.
- **Named Volume** — Docker persistence that stores container data in a named volume on the host, so the data survives container deletion; stateful services (PG/Redis/ES) must mount one.
- **Healthcheck** — The container periodically runs a probe command (e.g., `redis-cli ping`) and is judged unhealthy only after consecutive failures, letting dependent services wait for readiness and guaranteeing startup order.
- **Service-name DNS** — Within a Docker network, service names (e.g., `postgres`) resolve to container IPs — the correct way for containers to communicate (not `localhost`).

## Part 12 · Computational Advertising

### 12.1

- **Computational Advertising (计算广告)** — The technical and business system that matches and optimizes over the user, context, and ad triple with the goal of maximizing ROI.
- **Sponsor (出资人)** — One of the three elements of the advertising definition: the advertiser who pays for ad delivery and has explicit commercial objectives.
- **Publisher (媒介)** — One of the three elements of the advertising definition: the medium or product that carries ads and holds the user's attention.
- **Audience (受众)** — One of the three elements of the advertising definition: the group of target users the advertising message reaches.
- **Brand Awareness (品牌广告)** — An ad type focused on long-term influence and building recognition; typical metrics are exposure and awareness.
- **Direct Response (效果广告)** — An ad type pursuing short-term conversion actions (clicks, sign-ups, orders).
- **Ad Effectiveness Model (广告有效性模型)** — The six-stage funnel describing how ads take effect: Exposure→Attention→Comprehension→Acceptance→Retention→Decision, grouped into the selection, interpretation, and attitude phases.
- **ROI (Return on Investment)** — The ratio of return to spend in ad delivery; the core optimization objective of computational advertising.
- **eCPM (effective Cost Per Mille)** — Expected revenue per thousand impressions, obtained by multiplying the click-through rate and the click value; the unified yardstick for ad ranking and traffic valuation.
- **CPM Market (CPM 市场)** — A market form billing per impression, where the decisions (and risks) of click-through rate and click value are handed entirely to the advertiser.
- **CPC Market (CPC 市场)** — A market form billing per click, where click value is judged by the advertiser through bidding and the click-through rate is dynamically estimated by the platform.
- **CPA/CPS Market (CPA/CPS 市场)** — A market form billing per action/sale, where decisions and risks fall entirely on the platform; suits markets whose advertisers have highly uniform conversion processes.
- **Advertising System Value Formula (广告系统价值公式)** — Advertising system value = conversion efficiency × pricing mechanism × resource volume × delivery efficiency; the master framework for understanding advertising technology evolution.
- **Ad Network (广告网络)** — A closed intermediary system under the 2.0 delivery model that aggregates multi-media traffic and sells audiences rather than ad slots, mainly billing on CPC.
- **Programmatic Trade (程序化交易)** — Automated, single-impression-granularity ad trading completed via DSP-ADX-SSP under the 3.0 delivery model.
- **Ad Exchange (ADX, 广告交易平台)** — The trading hub that connects ads with (context, users) via real-time bidding and settles auctions at impression granularity.
- **Demand-Side Platform (DSP, 需求方平台)** — The demand-side technology platform serving advertisers, providing customized audience segmentation, cross-media traffic procurement, and RTB bidding.
- **Supply-Side Platform (SSP, 供应方平台)** — The supply-side technology platform serving media; its core function is yield optimization, uniformly optimizing multiple monetization methods.
- **Data Management Platform (DMP, 数据管理平台)** — A platform providing websites with data processing and external trading capabilities, characterized by customized audience segmentation and a unified data interface.
- **Trading Desk (广告购买平台)** — A demand-side tool allowing advertisers to buy across ad networks with ROI optimization, often incubated by agencies.
- **Real-Time Bidding (RTB, 实时竞价)** — The programmatic trading mechanism that queries multiple DSPs in real time for every ad impression, with the highest bidder winning.
- **Cookie Mapping (用户身份匹配)** — The up-front RTB phase, initiated by the DSP, that builds the lookup table between media Cookies and DSP user IDs; the mapping table is stored on the demand side.
- **Ad Call (广告请求)** — The RTB auction phase: the ADX broadcasts the bid request, DSPs return bids, and the highest bidder wins the impression.
- **Guaranteed Delivery (担保式投送)** — A premium-sale trading form based on contracts, with make-goods for unmet guaranteed impression volumes; CPM settlement, volume over quality.
- **Preferred Deal (优选)** — A one-on-one negotiated trading method where advertisers pick traffic first at an agreed price, with no open auction.
- **Network Optimization (网络优化)** — A trading method where the medium hands traffic to an ad network for wholesale monetization; a portfolio optimization problem.
- **Targeting (定向)** — The technology of finding an ad's target audience within the broad population; the professional term for audience-ad matching.
- **Contextual Targeting (上下文定向)** — A targeting technology matching ads based on page content and scenario information; implemented in engineering as a near-line context system.
- **Behavioral Targeting (行为定向)** — A targeting technology based on user behavior logs; behaviors are ordered by information strength, and behaviors closer to demand and more active are more effective.
- **Retargeting (重定向)** — A system-based targeting technology where the advertiser provides audience information and the system recovers these already-reached users from supply-side traffic.
- **Personalized Retargeting (个性化重定向)** — The vertical extension of retargeting: pushing item-granularity personalized ads to old users; equivalent to an offsite recommendation engine.
- **Search Retargeting (搜索重定向)** — The horizontal extension of retargeting: directing users who searched specific keywords to the advertiser's site.
- **Look-alike (新客推荐)** — A targeting technology where the advertiser provides a seed audience and the DSP finds potential new users by behavioral similarity among the supply-side audience.
- **Seed Audience (种子用户)** — The high-value target audience sample provided by the advertiser for look-alike targeting.
- **Feed Ads (信息流广告)** — An ad form mixed into the user's reading feed with a form similar to content; a positive example of balancing ad effectiveness and user experience.
- **Native Ads (植入式原生广告)** — An ad form blended into product content and services, deeply integrated with content.
- **Click Value (点击价值)** — The expected revenue brought by one click; together with the click-through rate it constitutes eCPM.
- **Bid Landscape Prediction (竞价行情预估)** — The core DSP problem of forecasting the traffic bidding distribution to decide procurement strategy; the traffic it receives is a function of its bids.
- **Yield Optimizer (收益管理)** — The SSP's core function, uniformly optimizing premium sales, network, and RTB traffic to maximize the medium's revenue.

### 12.2

- **Advertiser** — The party that pays for ads and derives the value of a single ad backward from final outcomes; the decision-making subject on the demand side.
- **Media / Supply Side** — The content or application owner holding ad slots and traffic, concerned with how much revenue each unit of ad inventory generates.
- **Direct Response** — An advertising form oriented toward short-term conversion actions; the supply side computes ad volume from ad performance, billed by outcomes and traded through auctions.
- **Brand Awareness** — An advertising form focused on long-term brand impact, billed by impressions and traded through contracts, commonly seen in premium placements such as core banners.
- **CPT (Cost Per Time)** — A model charging by the duration an ad slot is occupied (monthly or weekly); hassle-free but crude in measurement, unable to guarantee client interests.
- **CPD (Cost Per Day)** — A billing model that buys out an ad slot by the day, mostly seen in contracted brand advertising; modest prerequisites for cooperation, but less real-time and effective than CPS in the long run.
- **CPM (Cost Per Mille)** — A billing model charging per thousand impressions, computed as spend/impressions×1000; common in RTB, with risk borne mainly by the advertiser.
- **CPC (Cost Per Click)** — A billing model charging per click, computed as spend/clicks; the compromise point of risk between advertiser and platform, common in keyword advertising and RTB.
- **CPA (Cost Per Action)** — A billing model charging on user actions such as registration or ordering; both CTR and value are dynamic, with decisions and risk falling on the platform.
- **CPS (Cost Per Sales)** — A billing model converting ad fees into commissions on actual sales; advertisers hedge fee risk, commonly seen in affiliate networks.
- **dCPM (dynamic CPM)** — The settlement system widely adopted by DSPs; the bid for each impression is computed in real time from campaign performance, optimizing for advertisers by performance while settling with media by impressions.
- **flat CPM** — The traditional CPM with a fixed per-thousand-impression price, in contrast to dCPM.
- **Spend** — The advertiser's cost of running ads; the numerator in formulas such as CPM, CPC, and ROI.
- **CTR (Click-Through Rate)** — Clicks divided by impressions; measures the average number of user clicks an ad receives across multiple impressions.
- **CVR (Conversion Rate)** — Orders divided by clicks; measures the relationship between user clicks and final orders.
- **ROI (Return On Investment)** — Order value divided by spend; measures the return relationship between ad cost and generated order value.
- **eCPM (effective/expected CPM)** — Expected revenue per thousand impressions; equals pCTR×bid×1000 under CPC billing and the bid under CPM billing — the unified ranking measure across billing models.
- **Guaranteed Delivery (GD)** — A contract-based ad delivery mechanism: agreed impression volume unmet requires compensation, volume before quality, CPM settlement, and server-side decisions.
- **Online Allocation** — Modeling the matching of ads to (Context, User) traffic as a bipartite-graph optimization of Ad→(Context,User), allocating impressions under each contract's volume constraint; the classic solution constructs the dual problem.
- **Traffic Forecasting** — Estimation of future volumes of targeted traffic; can be viewed as an inverted retrieval problem with the ad as the query over the (u,c) space, requiring u and c to be handled separately.
- **Exclusivity** — Brand advertisers' exclusionary demands on exposure in contract sales (e.g., competitor exclusion), further tightening the feasible space of online allocation.
- **Ad hierarchy (creative/solution/campaign/advertiser)** — The hierarchical organization from creative through delivery unit, campaign, to advertiser, used for back-off prior estimation of new ads' CTR.
- **Back-off** — An estimation strategy that climbs to coarser levels to borrow statistics when data is missing; used for CTR estimation in new-ad cold start.
- **Dynamic Features** — Click-feedback statistical features aggregated along label-combination dimensions; fast-responding with strong back-off for new combinations, but costly in online storage and updates.
- **Online Learning** — A scheme where the model updates in a streaming fashion on new data to capture dynamic behavior; forms the "adjust the model vs. adjust the features" trade-off with dynamic features.
- **E&E (Exploration & Exploitation)** — A framework that creates impression opportunities for long-tail (a,u,c) combinations to accumulate statistics and thus estimate CTR more accurately; the volume and effectiveness of exploration must be strictly controlled.
- **ε-greedy** — A multi-armed bandit strategy that explores randomly on an ε fraction of traffic and exploits the current best on the rest.
- **UCB (Upper Confidence Bound)** — A strategy that computes an upper confidence bound on each candidate's expected reward and picks the highest; the more selections, the closer the bound approaches the true expectation.
- **Contextual Bandit** — An E&E method that makes decisions on arms' feature vectors instead of the arms themselves to reduce dimensionality; well suited to ad scenarios with huge candidate spaces.
- **GSP (Generalized Second Pricing)** — An auction mechanism where the winner pays a price converted from the next-ranked ad; simple to implement and widely adopted by online ad systems, but the market as a whole is not truth-telling (see 12.3).
- **Individual Rationality (IR)** — The basic participation constraint that an advertiser pays no more than its bid, e.g., GSP payment ≤ winner's bid.

### 12.3

- **Auction Mechanism** — the institutional design governing how ad slots are allocated and priced, consisting of an allocation rule and a pricing rule.
- **Position Auction** — the auction model in which multiple advertisers compete for multiple slots differing only in click-through rate; expected value $u_{is} = v_i \cdot x_s$.
- **Valuation** — the advertiser's true value judgment of one click; private information invisible to the platform.
- **Bid** — the per-click price the advertiser declares to the platform as willing to pay (on a CPC basis).
- **Position CTR** — the click-through rate $x_s$ of slot $s$; larger for more forward positions, and the only difference between slots.
- **Allocation Rule** — the rule within a mechanism deciding "who wins which slot"; in auction advertising, usually assignment in descending order of bid (times quality score).
- **Pricing Rule** — the rule within a mechanism deciding "how much the winner pays"; it determines whether advertisers are willing to bid truthfully.
- **Generalized First Price (GFP)** — the mechanism that allocates slots by ranking bids with everyone paying their own bid; has no pure-strategy Nash equilibrium, causes market oscillation, and is now obsolete.
- **Nash Equilibrium** — a strategy profile in which no player can gain by unilaterally changing its own strategy.
- **Pure-Strategy Nash Equilibrium** — a Nash equilibrium in which each player commits to one deterministic strategy; none exists under GFP.
- **Second-Price Auction / Vickrey Auction** — a single-slot auction where the highest bidder wins and pays the second-highest bid; truthful bidding is a dominant strategy.
- **Dominant Strategy** — a strategy that is optimal regardless of how opponents act; in the second-price auction, truth-telling is a dominant strategy.
- **Generalized Second Price (GSP)** — the mechanism where the rank-$i$ advertiser pays the next bidder's eCPM converted as $p_i = b_{i+1}x_{i+1}/x_i$ and the last rank pays the reserve price; widely adopted by online advertising systems.
- **Reserve Price** — the minimum transaction price set by the platform; paid by the last-ranked advertiser or when there is no competitor.
- **Incentive Compatibility (IC)** — the property that truthfully reporting one's valuation is a dominant strategy: misreporting cannot raise utility.
- **Individual Rationality (IR)** — participation in the auction never leaves the participant with negative utility, i.e., payment does not exceed the declared value $p_i \le b_i$.
- **Truth-telling** — the behavior of bidding one's true valuation $b_i = v_i$; satisfied by the VCG market as a whole, not by GSP.
- **Symmetric Nash Equilibrium (SNE)** — the stable equilibrium that exists under GSP, satisfying the envy-free property.
- **Envy-free** — the allocation property that in equilibrium no one wants to swap positions with another: taking another's position requires paying their price, yielding no utility gain.
- **VCG Mechanism (Vickrey-Clarke-Groves)** — the mechanism charging each participant the externality damage it imposes on the others; the market as a whole is truth-telling, and it degenerates to second-price with a single slot.
- **Externality** — the welfare loss that one participant's presence imposes on all other participants, i.e., "how much more the others could have earned without you."
- **Winner's Curse** — the situation of winning a slot above one's own valuation through an inflated bid and suffering negative utility; common under first-price auctions.
- **First-Price Auction** — the auction where the winner pays their own bid; re-adopted around 2019 by leading ADXs in programmatic open auctions.
- **Header Bidding** — the technique where publishers send traffic to multiple demand sides for pre-bidding before the main auction; its spread fueled the multi-level resale chain and the return to first-price.
- **Bid Shading** — the bidding strategy under first-price auctions by which a DSP presses its bid toward "the lowest price that still wins" based on the win-probability distribution; the core competency of the first-price era.
- **Smart Bidding** — the bidding product form where the platform bids on the advertiser's behalf (e.g., OCPC by target conversion cost) and converts the bid into the ranking model.

### 12.4

- **Smart Bidding** — the product form where the platform manages the per-impression bid on the advertiser's behalf: the advertiser reports only a goal (target CPA/ROI), and the bid is jointly determined by the platform's value estimation, budget control, and mechanism-adaptation modules.
- **Conversion Bidding (oCPC / oCPM)** — products that bid by conversion goal: the bid formula is $\text{eCPM} = 1000 \cdot \text{pCTR} \cdot \text{pCVR} \cdot \text{Bid}_{\text{CPA}}$; oCPC bills by click, oCPM bills by impression.
- **Target CPA** — the target cost the advertiser is willing to pay for one conversion; the only value anchor in the bidding stack input directly by the advertiser.
- **Value Bid** — the expected value of a single impression converted from the conversion goal via pCTR × pCVR × targetCPA; the input to downstream shading and pacing.
- **Two-Phase Rollout** — the cold-start convention for oCPC/oCPM: the first phase stays with CPC bidding to accumulate conversion data, switching to conversion bidding once the model is confident.
- **Budget Pacing** — the control problem of spending the daily budget evenly in step with time progress, avoiding front-loaded spending that misses premium evening-peak traffic.
- **Reference Trajectory** — the control target of pacing, $r(t) = G \cdot t / T$: the straight line of "spending progress in sync with time progress."
- **Probabilistic Throttling** — one pacing implementation: decide whether to participate in each auction with probability $\alpha$; a 0/1 hard gate (LinkedIn, KDD 2014).
- **Bid Scaling** — the other pacing implementation: scale the bid $b' = \alpha \cdot b$ with a multiplier $\alpha \in [0,1]$, preserving participation at the cost of per-auction competitiveness.
- **Pacing Multiplier** — the control action of the budget controller, squashed by a sigmoid into $[0,1]$ and multiplied directly onto the bid.
- **PID Control** — the proportional–integral–derivative feedback controller: P responds to error immediately, I removes the steady-state error, D damps anticipatorily; in ad pacing the D term is universally dropped because it amplifies discrete-request noise, leaving only PI.
- **Log-Ratio Error** — the error form $e(t) = -\log(N(t)/r(t))$, normalizing deviation to a relative value so that plans of different budget scales can share the same control gains.
- **Feedforward Compensation** — beyond feedback control, adjusting the control action in advance using predictable disturbances (such as intraday traffic patterns); Verizon DSP's integral control is equipped with feedforward.
- **Expected Surplus** — the expected profit of bid $b$ under a first-price auction, $\mathbb{E}[S] = (v-b) \cdot P(\text{win} \mid b)$; the optimization target of bid shading.
- **Minimum Winning Price** — the price that just barely wins an auction; its distribution (CDF) determines the win rate $P(\text{win} \mid b) = F(b)$.
- **Bid Landscape / Win-Price Distribution** — the probability distribution of the minimum winning price across traffic; the core estimation object of bid shading, with log-normal fitting its long tail best.
- **Censored Data** — samples where only partial information is observed: in sealed auctions, the true winning price of lost auctions is never visible, requiring survival analysis.
- **Survival Analysis** — the statistical method for censored observations; DDN uses it to estimate the win-price distribution from the incomplete data of "whether we won + the minimum price when we won."
- **Golden Section Search** — a gradient-free interval extremum search retaining 0.618 of the interval per iteration; DDN uses it to find the surplus-peak bid $b^*$ in milliseconds.
- **DDN (Deep Distribution Network)** — Verizon Media's distribution-estimation network (Zhou et al., KDD 2021): the network outputs win-price distribution parameters; online surplus improved 14.3%, serving hundreds of billions of requests daily.
- **Distributionally Robust Bidding** — a bidding-robustness method that uses KL-divergence uncertainty sets for max-min optimization when the estimation noise in valuations and win-price distributions is large.
- **Error Propagation Chain** — the property that the modules of the bidding stack are coupled in series, so biases in upstream predictions (pCTR/pCVR) propagate losslessly to the final bid; the motivation for the calibration problem of 12.5.

### 12.5

- **Calibration** — the consistency between predicted values and true probabilities: $\Pr(Y{=}1 \mid f(x)=p) = p$; i.e., about $100p\%$ of the samples scored $p$ are positive.
- **Discrimination** — a model's ability to rank positives above negatives, measured by AUC-type metrics, invariant to monotonic transformations of the scores.
- **size-accuracy** — the accuracy of the absolute magnitude of predictions; critical for precise bidding, auction stability, and mixed-delivery fairness.
- **Overconfidence** — the general tendency of deep models' predictions to systematically exceed the true probabilities (Guo et al., 2017).
- **Position bias** — the bias in which the click advantage of forward positions is misattributed to the ad's own quality.
- **Examination hypothesis** — the decomposition assumption click = seen × worth clicking: $P(\text{click}) = P(\text{seen}\mid\text{position}) \cdot P(\text{relevant}\mid\text{user, ad})$.
- **Inverse propensity weighting (IPW)** — a debiasing method that weights samples by the reciprocal of propensity scores to restore an unbiased distribution.
- **Propensity score** — the probability of a sample being assigned to a position / being selected; the source of IPW weights, usually requiring random traffic to estimate.
- **PAL (position-bias-aware learning)** — the structured debiasing framework proposed by Huawei: bCTR = ProbSeen(position) × pCTR(user, ad, context); joint training, online only the pCTR tower (Guo et al., RecSys 2019).
- **Cascade model** — a position modeling approach assuming users browse front to back in order, stop upon clicking, and click at most once per session; the examination probability depends on preceding content.
- **Sample selection bias (SSB)** — the distribution mismatch caused by training CVR on the click subspace while inferring on the entire impression space.
- **Data sparsity (DS)** — insufficient training signal caused by conversion samples being far fewer than click samples (clicks are only about 4% of impressions).
- **ESMM (Entire Space Multi-Task Model)** — Alibaba's entire-space multi-task model: joint training on all impression samples with pCTCVR = pCTR × pCVR, solving SSB and DS simultaneously (Ma et al., SIGIR 2018).
- **pCTCVR** — the probability from impression to conversion, equal to pCTR × pCVR; defined on the entire impression space and directly supervisable.
- **Implicit learning** — the learning regime in ESMM where the CVR tower has no direct loss term and is updated only by L_ctcvr's gradients through the product.
- **Winner's bias** — selection bias caused by auction logs recording only winners' outcomes while losers have no labels; requires exploration traffic to supply unbiased signals.
- **Exploration traffic** — a traffic allocation strategy that deliberately lets ads that would have lost occasionally win, to generate unbiased feedback.
- **Delayed feedback** — the phenomenon of conversion labels arriving hours or days after the click.
- **Label window** — the observation-period convention for calibration data extraction (e.g., 1-day clicks, 7-day conversions); extracting before maturity is necessarily biased.
- **Reliability diagram** — a diagnostic plot of bucketed predicted probabilities vs the actual positive rate per bucket; points on the diagonal mean perfect calibration.
- **Expected calibration error (ECE)** — the sample-size-weighted average of the gap between the actual positive rate and the mean prediction per bucket.
- **Platt scaling** — a calibration method fitting a logistic transform (two parameters) to the model scores; suitable for small samples.
- **Isotonic regression** — a calibration method fitting a free-form monotone step function; suitable for large samples, prone to overfitting in sparse regions.
- **PAVA (Pool Adjacent Violators Algorithm)** — the classic algorithm for solving isotonic regression: repeatedly merging adjacent blocks that violate monotonicity and averaging them.
- **Prior correction** — a calibration method that restores the true base rate after negative sampling with the closed-form formula $\hat{p}'=\hat{p}/(\hat{p}+(1-\hat{p})/w)$ (Facebook ADKDD'14).
- **Negative sampling** — a technique of downsampling negatives during training to accelerate or balance samples; it raises the training base rate and requires correction before use.
- **Distribution drift** — the phenomenon of historical calibration becoming inaccurate as traffic mix, ad inventory, and user behavior change.
- **PCOC (Predicted-over-Posterior Click rate)** — the ratio of predicted CTR to posterior CTR; the closer to 1 the better.
- **Cal-N** — an overall calibration bias measure aggregated from multi-cluster PCOC.
- **GC-N** — a calibration evaluation metric weighted across dimensions.
- **SIR (Smoothed Isotonic Regression)** — the starting point of Alimama's calibration system: bucketing + isotonic regression + linear scaling.
- **Bayes-SIR** — a calibration algorithm introducing Bayesian priors on top of SIR to address cold start and instability in sparse buckets.
- **RTW-BSIR** — a calibration algorithm adding real-time fluctuation correction on top of Bayes-SIR to fight distribution drift.
- **PCCEM** — a calibration algorithm that uses short-term post-click signals to predict long-term conversions and address delayed feedback; deployed online by Alimama since 2018.
- **AdaCalib** — a field-level fine-grained calibration framework: a family of isotonic functions + adaptive guidance from posterior statistics (Wei et al., SIGIR 2022).
- **observed-vs-predicted guardrail** — an operations mechanism that continuously monitors "actual positive rate ÷ predicted positive rate" online and alerts and refits when the deviation from 1 exceeds a threshold.


### 12.6

- **Data Observability** — the degree to which an advertising platform can directly observe user conversion behavior; the second axis that determines how deep the platform's optimization can go.
- **Closed-loop Advertising** — advertising in which the entire chain of impression, click, order, and payment happens inside the platform's domain, with no data leaving the platform's ecosystem; also called the inner loop.
- **Open-loop Advertising** — advertising in which the conversion happens outside the platform's domain (App Store, brand website, offline store), where the platform must rely on postbacks to learn about conversions; also called the outer loop.
- **Inner-loop / Outer-loop** — the industry-wide alternative names (Douyin, Kuaishou, Facebook) for closed-loop/open-loop advertising.
- **Semi-closed-loop** — the compromise form in which the advertiser posts back only some events (e.g., only activation, not payment), giving the platform an incomplete label set for partial optimization.
- **Deep conversion bidding** — bidding that optimizes toward back-funnel behaviors such as payment, ROI, next-day retention, and 7-day ROI; feasible only when the platform can observe those behaviors.
- **Shallow-funnel goal / Deep-funnel goal** — the two sets of optimization targets: the front funnel (impression, click, activation, form, registration) and the back funnel (payment, ROI, next-day retention, 7-day ROI), corresponding to the capability boundary of open-loop and closed-loop respectively.
- **pDeepCVR** — the deep conversion probability of "click → payment/next-day retention", sitting deeper in the conversion funnel, with sparser samples and higher latency.
- **Attribution** — the process of identifying which ad/channel brought about the key behavior in the advertising behavior chain.
- **Attribution Model** — the allocation rule that decides how conversion credit is distributed across touchpoints; a convention, not objective fact.
- **Last-click** — the attribution model that gives 100% credit to the last touchpoint before conversion; the mobile default.
- **First-click** — the attribution model that gives 100% credit to the first touchpoint; used to measure top-of-funnel discovery.
- **Linear Attribution** — the attribution model that divides credit equally among all touchpoints.
- **Time-decay** — the attribution model that gives more credit to touchpoints closer to conversion; suits short-cycle intent-driven journeys.
- **Position-based** — the attribution model that gives more credit to the first and last touchpoints and less to the middle (U-shaped); balances discovery and closing.
- **Data-driven Attribution** — the attribution model in which an algorithm assigns credit automatically based on observed contributions; requires large amounts of conversion data.
- **clickid** — the unique identifier issued by the media at the ad touchpoint (impression/click), used to bind the conversion to a specific ad during postback.
- **Conversion postback** — the process by which the advertiser posts the device ID, clickid, and timestamp back to the media via SDK/API to report "this user has converted".
- **Fallback attribution** — the attribution approach of falling back to ip + ua fuzzy matching when a device ID is unavailable; lower precision.
- **Self-attribution** — the approach in which the platform/media completes attribution itself and claims the conversion; prone to double counting when multiple networks run in parallel.
- **Non-self-attribution** — the approach in which the advertiser matches users to media information itself and completes attribution independently.
- **Mobile Measurement Partner (MMP)** — a neutral third-party attribution/analytics platform (AppsFlyer, Adjust, Branch, Singular, Kochava) that arbitrates between the advertiser and the various ad networks.
- **ATT (App Tracking Transparency)** — Apple's authorization framework since iOS 14.5; apps must show a prompt to access IDFA, with an opt-in rate of only about 25%.
- **IDFA (Identifier for Advertisers)** — Apple's device-level advertising identifier; largely collapsed after ATT, and the unique user ID that deterministic attribution depended on.
- **SKAdNetwork (SKAN)** — Apple's privacy-preserving install attribution framework that posts conversion data back in an aggregated, randomly delayed, crowd-anonymized way.
- **Crowd Anonymity** — SKAN's privacy mechanism; it returns less information when install volume is low, preventing any single user from being reverse-identified.
- **conversion value** — the user-interaction conversion value reported by the app through `updateConversionValue` in SKAN; SKAN 4.0 introduces coarse and fine variants.
- **Postback window** — SKAN 4.0's postback cadence: roughly 0–2 days, 3–7 days, and 8–35 days; conversion data flows back in batches with random delay.
- **Hierarchical source identifier** — SKAN 4.0's 4-digit hierarchical source identifier (first 2 digits campaign, 3rd digit position, 4th digit placement); more digits returned as the crowd anonymity level rises.
- **Android Privacy Sandbox** — Google's cookieless attribution solution, containing the Attribution Reporting API and the Topics API.
- **Attribution Reporting API** — the Privacy Sandbox component that provides event-level and aggregated attribution reports, with differential privacy noise on aggregated reports.
- **Differential Privacy** — the technique of injecting calibrated noise into aggregate statistics to protect individual privacy; used in Privacy Sandbox attribution reports.
- **Deterministic attribution / Probabilistic attribution** — precise attribution relying on a device-level unique identifier vs statistical attribution relying on aggregated/fuzzy signals; the privacy wave pushes the former to collapse into the latter.
- **First-party Data** — data that an enterprise collects directly from and lawfully with its users; the strategic direction after cross-app tracking is restricted.
- **Modeling-based Estimation** — the estimation method that trains a model on the observable portion (SKAN + authorized deterministic data) and extrapolates to fill the "unattributable" gap.


### 12.7

- **Online Allocation** — an algorithmic framework that decides in real time, for every ad impression, how to allocate it so as to optimize overall product revenue subject to volume constraints; offline planning + online execution is its standard shape.
- **Guaranteed Delivery (GD)** — the delivery system for impression contracts: contracts commit to a targeting audience and an impression volume, the system must guarantee full delivery by the deadline, and its core computational problem is constrained online allocation.
- **Scheduling System** — the non-personalized system managing CPT ad-slot contracts: creatives are delivered directly through the CDN front end by schedule, with no real-time server-side decisions.
- **House Ad (fallback ad)** — the default creative rendered by the CDN when dynamic ad serving times out or errs, guaranteeing the ad slot is never blank.
- **Bipartite Graph** — the problem modeling of online allocation: a matching structure $G=(I \cup A, E)$ between supply nodes (traffic pools with identical labels) and demand nodes (contracts).
- **Supply Node** — a node on one side of the bipartite graph, representing a block of traffic inventory whose labels are all identical, with total volume $s_i$; node count grows geometrically with targeting-condition combinations.
- **Demand Node** — a node on the other side of the bipartite graph, representing one ad contract, with committed volume $d_a$.
- **Demand Constraint** — the constraint that the revenue (or volume) allocated to a contract is no less than its committed value: $\sum_{i \in \Gamma(a)} s_i x_{ia} q_{ia} \ge d_a$.
- **Supply Constraint** — the constraint that the ratios allocated out of each supply node sum to at most 1: $\sum_{a \in \Gamma(i)} x_{ia} \le 1$; violating it means overselling.
- **Allocation Ratio** — the decision variable $x_{ia}$: what fraction of supply node $i$'s traffic is allocated to contract $a$.
- **AdWords Problem (bidding with budget constraints)** — the online allocation instance of maximizing market revenue in a CPC auction given each advertiser's budget; its dual variables are "the marginal value of traffic to a budget," the theoretical prototype of the pacing multiplier.
- **Traffic Forecasting** — the technique of estimating the winnable impression volume of a future period given audience label combinations and an eCPM threshold; in engineering it uses the "inverted index" scheme (documents = traffic aggregated by labels, queries = targeting conditions).
- **Frequency Capping** — controlling the number of impressions for the combination $(a,u)$ within a period; implemented via client-side cookie/SDK or server-side in-memory cache, it is the main factor breaking the per-impression separability assumption.
- **Dual Variable** — the variable corresponding to a constraint in the LP dual problem: $\alpha_a$ (contract scarcity) and $\beta_i$ (supply-side opportunity cost), on the order of the contract count and the supply node count respectively.
- **Compact Allocation Plan** — an allocation plan that keeps only contract-level dual variables $\alpha$ ($O(|A|)$-level) and recovers $\beta$ and $x_{ia} = \max(0, \theta_a(1+\alpha_a-\beta_i))$ via the KKT conditions; stateless, zero synchronization across machines.
- **Demand-Supply Ratio (θ)** — $\theta_a = d_a / \sum_{i \in \Gamma(a)} s_i$, measuring how tight a contract is relative to all its candidate traffic; appears in both the compact plan and HWM.
- **SHALE** — the primal-dual iterative algorithm for online allocation: alternately updates $\alpha$ and $\beta$ to solve the dual problem, and supports incrementally inserting new contracts.
- **High Water Mark (HWM)** — the engineering heuristic allocation scheme: determines contract priority in descending order of $\theta$ and scales down candidate supply remains layer by layer to get allocation ratios; online decisions are made randomly by cumulative ratio.
- **Competitive Ratio** — if an online policy achieves a factor of $\epsilon$ of the offline globally optimal objective in the worst case, it is called $\epsilon$-competitive; the optimal upper bound for online allocation is $1-1/e$.
- **Free Disposal** — the assumption that over-delivering brings neither gain nor loss; it matches the reality of most ad contracts and is the source of online allocation algorithms' tolerance.


---

_The glossary covers the full book: Volume I (Ch0–Ch4, Parts 1–5), Volume II (Ch5–Ch10, Parts 6–11), and the Special Topic (Part 12, Computational Advertising)._
