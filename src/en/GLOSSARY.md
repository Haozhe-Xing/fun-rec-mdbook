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
- **OpenRTB** — the IAB real-time bidding communication specification: standardizes the Bid Request (inquiry carrying slot/context/user identifiers/floor price) and Bid Response (bid/creative reference/tracking URL); decoupling bids from creatives is the key engineering constraint.
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
- **Deep Conversion Bidding** — the bid form pushing the optimization target from activation to key post-install behaviors (next-day retention / 7-day payment / repurchase / card binding); the difficulty is the delayed-feedback problem from slowly maturing labels.
- **LTV Bidding** — the bid form anchoring on user lifetime value instead of single-conversion value: $\widehat{\text{LTV}}(u)$ is typically decomposed into "retention probability × per-period value," modeled separately and recombined, to handle the heavy-tailed sparse distribution.
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
- **Attribution Window** — the protocol parameter for how long after a touchpoint a conversion still earns credit; the click window is commonly 7 days and the view window 1 day; longer windows give the channel more opportunity to claim credit.
- **Idempotent Deduplication** — deduplication with set semantics on "device × event type × dedup key" at the postback receiving end, preventing duplicate conversions from network retries.
- **Event Time vs Arrival Time** — when the conversion actually happened (carried in the postback body) vs when the platform received it; the difference between the two is the delayed-feedback problem itself, and training samples must be organized by the former.
- **Click Flooding** — an attribution-fraud scheme of mass fabricated/low-quality clicks that inflates the "probability of being credited"; hallmarks are anomalous click density and suspiciously short click-to-install intervals.
- **Conversion Value Encoding** — the information-compression problem of squeezing the funnel progress you want to observe into SKAN's 64 fine values and three coarse tiers, e.g., fine encodes retention flags and coarse encodes payment-amount tiers.
- **SKAN-side Modeling Pipeline** — the technical stack that trains an "aggregate distribution → true funnel" mapping model on concurrent opt-in deterministic data, restoring the noisy, delayed SKAN postbacks into an optimizable signal.
- **Delayed-feedback Three Solution Families** — importance sampling (Zhang, CIKM 2016), fake-negative correction (Chen, 2020), and streaming FTRL data correction; the choice depends on the delay-distribution shape and training real-time-ness.
- **Shallow Proxy + Deep Correction** — the true form of an open-loop platform's deep bidding: the bid formula uses a shallow goal (pCTR × pCVR), and postback deep data periodically calibrates the mapping from the shallow goal to the true deep goal.
- **Propensity Score Weighting** — modeling and weighting the selection behavior of "whether to post back," mitigating the selection bias that open-loop training samples come only from advertisers who post back.
- **Incentivized Postback** — the platform's "trading product capability for data" mechanism design: deep bidding abilities such as payment bidding unlock only once payment events are posted back.
- **Incrementality Measurement** — the causal measurement answering "would conversions have happened anyway without the ads"; complementary to attribution (accounting), it governs budget decisions.
- **Geo Experiment** — the experimental method that splits geographies into test/control with the control receiving no ads at all, directly measuring incremental conversions; causally the cleanest.
- **Synthetic Control** — the quasi-experimental method that synthesizes a counterfactual baseline from similar unexposed geographies/periods to estimate the increment during delivery.
- **Marketing Mix Modeling (MMM)** — the measurement method that decomposes sales into channel inputs via macro time-series regression, requiring no user-level data; enjoying a revival in the privacy era.
- **"Attribution for accounting, incrementality for decisions"** — the division-of-labor principle: attribution settles cross-channel accounts, while incrementality measurement governs budget-reallocation decisions.


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


### 12.8

- **Audience Targeting** — The process of extracting meaningful features (labels) along the three dimensions of ad $a$, user $u$, and context $c$; one of the core driving forces of display advertising.
- **Contextual Targeting** — The $t(c)$ class of targeting: assigning labels instantly based on the page the user is currently visiting or request parameters (geo, channel, URL, keywords, topics).
- **Behavioral Targeting (BT)** — The $t(u)$ class of targeting: mapping a user onto some targeting label based on the history of the user's online behaviors over a period of time.
- **Customized Labels ($t(a,u)$)** — User labels produced for a specific advertiser (e.g., retargeting, look-alike); their count grows proportionally with the number of advertisers, making them suitable for direct supply by the demand side in programmatic trading.
- **Taxonomy** — A predefined, interpretable set of labels sold to advertisers; the dual metrics of effectiveness and scale require it to cover both the "broad and large" end and the "precise and small" end.
- **Semi-online Crawler** — The page-crawling scheme for contextual targeting: no offline crawling; crawling and labeling are triggered only after an ad request, managed with cache + TTL, and temporarily empty labels are allowed.
- **Weak Consistency** — A business property of ad systems: as long as most decisions are optimal, a few suboptimal or even random decisions are acceptable; the basis for low-cost system design.
- **Demand-driven Keywords** — A keyword selection method that obtains a commercially valuable keyword list and IDF from advertiser descriptions, then computes TF-IDF together with page TF.
- **Latent Semantic Analysis (LSA)** — A topic model that takes the SVD of the document-term matrix and keeps the dominant singular values; its two transformation matrices are not guaranteed non-negative, which is intuitively unsatisfying.
- **Probabilistic Latent Semantic Indexing (PLSI)** — The probabilistic version of LSA: modeled as a generative process of "document picks a topic, topic generates words"; solvable with distributed EM.
- **Latent Dirichlet Allocation (LDA)** — The Bayesian version of PLSI, adding a Dirichlet prior to the topic distribution; more robust under noisy data or short documents, commonly solved with Gibbs sampling.
- **word2vec / Word Embedding** — A representation-learning method mapping words into dense real-valued vectors; CBOW + Huffman tree (hierarchical softmax) reduces output complexity from $O(\|V\|)$ to $O(\log \|V\|)$ — the origin of the embedding idea.
- **Embedding-based Labeling** — The mainstream label-production route of 2026: using representation models (two-tower / graph embeddings) or LLMs to map content into vectors or structured labels, having replaced topic-model labeling.
- **Time Decay** — The behavior accumulation method $\tilde{x}(d) = \alpha\tilde{x}(d-1) + x(d)$; an exponential window filters raw behaviors, only the previous slice's state needs storing, and it is superior to the sliding window method in engineering.
- **Sliding Window** — An accumulation method that sets a window length $D$ and sums behavior intensities within the window; rectangular in shape, and it must store all behaviors inside the window.
- **Feature Selection Function ($x_{tn}(b)$)** — The function that maps raw behavior $b$ onto label $t$ with an intensity; the most critical link in behavioral targeting's feature generation.
- **User Label Score ($\lambda_t$)** — The linear score of the behavioral targeting GLM, $\log\lambda_t = w_t \cdot \tilde{x}$, controlling how frequently clicks arrive; updated online via the recursion $\lambda(d) = \alpha\lambda(d-1) + \sum_n w_{tn} x_{tn}(d)$.
- **Demographic Prediction** — A classification task predicting fixed user characteristics such as gender and age from behavior; a rejection threshold is mandatory, and training-set quality matters more than the model.
- **reach/CTR Curve** — The semi-quantitative evaluation tool for behavioral targeting: the curve of label population size (reach) versus that population's CTR; it should decrease monotonically, the head slope reflects discriminative power, and the far-right CTR is fixed at the full-population level.
- **AUC (see 12.5)** — The metric of a model's discriminative power (relative ordering); the steepness of the reach/CTR curve's head is its projection in targeting evaluation, and scores entering the arithmetic must still pass calibration.

### 12.9

- **Ad Retrieval** — the computational stage that, under millisecond-level constraints, finds from hundreds of millions of ad candidates the few eligible to participate in this auction; the qualification round before eCPM ranking (12.2).
- **Disjunctive Normal Form (DNF)** — the standard representation of ad targeting conditions: a union of several Conjunctions, where hitting any one Conjunction means hitting the ad.
- **Conjunction** — a group of assignments joined by AND within a DNF; the retrieval algorithm builds its inverted index over Conjunctions (not over whole ads).
- **Assignment** — a minimal constraint on a single label (belonging or not belonging to some value set), e.g. age ∈ {3}; size counts only assignments containing "∈".
- **Boolean Retrieval** — retrieval that evaluates targeting conditions over an inverted index: a two-layer index (Conjunction inverted index + Conj→AD auxiliary index) plus size-tier pruning, first taking a candidate superset then doing exact evaluation; pure-"∉" Conjunctions hang on the special key Z as a fallback.
- **Inverted Index** — a data structure mapping "keys (labels/keywords)" to "the list of documents containing each key"; ad retrieval extends it into a size-tiered Conjunction index.
- **Size-Tier Pruning** — building the index tiered by the number of "∈" assignments in a Conjunction; when a request's label count is below a tier's size, that entire tier is skipped — the most powerful pruning in boolean retrieval.
- **Exact Match** — the strictest tier of keyword matching in search ads: triggered only when the query is identical to the bidding keyword; contrasted with phrase match and broad match (which triggers query expansion).
- **Query Expansion** — the technique in search ads of expanding a short query into a set of biddable keywords; three routes are collaborative filtering, topic models, and historical eCPM performance mining, with over-generalization harming relevance.
- **Ad Placement** — the decision in search ads of how many ads the North/East zones carry: revenue optimization under an average-ad-count constraint, with personalized adjustment via the ratio of user click-through rates.
- **Relevance Retrieval** — retrieval for extremely long queries targeting "query-document similarity" rather than boolean matching; requires the evaluation function to be linear with non-negative weights to support fast upper-bound pruning.
- **WAND (Weak AND / weight AND)** — a Top-K pruning retrieval algorithm: precompute keyword contribution upper bounds, exactly score only when the accumulated bound exceeds the heap threshold, with a min-heap maintaining the current best K results; its "rough upper bound + threshold positive feedback" idea carries into the pre-ranking layer.
- **Semantic Recall** — recall that maps queries/users and ads into the same semantic vector space with a DNN and replaces keyword matching with nearest neighbor search; resolves the matching blind spot caused by different wording.
- **DSSM (Deep Semantic Similarity Model)** — a deep model using clicks as weak supervision to learn query and document semantic vectors end to end: word embedding → multi-layer network projecting into semantic space → cosine similarity + softmax/pairwise ranking loss.
- **Two-Tower Model** — a recall architecture where the user tower and item tower independently encode vectors and online serving computes only vector inner products; the modern form of the DSSM/YouTube model, standardly trained with in-batch negatives.
- **Approximate Nearest Neighbor (ANN)** — the umbrella term for nearest neighbor search techniques that accept some recall loss in exchange for millisecond-level vector retrieval, in three families: hashing (LSH), vector quantization (PQ/HKM), and graphs (NSW/HNSW).
- **Locality-Sensitive Hashing (LSH)** — divide-and-conquer ANN based on "the closer in the original space, the easier the hash collision"; with random projection, the same-bucket probability is 1−θ/π; recall is boosted by LSH forest (trading memory) or multi-probe (trading query time); superseded in engineering by graph indexes, but its intuition remains the conceptual origin of ANN.
- **HNSW (Hierarchical Navigable Small World)** — the hierarchical version of NSW: sparse upper layers for fast navigation, a dense base layer for precision; the most widely used ANN graph index in industry today, implemented in faiss/hnswlib.
- **IVF-PQ** — an ANN index that first coarse-clusters into buckets with K-means (IVF), then compresses within buckets with product quantization (PQ); memory-efficient, suited to candidate pools above hundreds of millions.
- **Retrieval Funnel** — the full system view of candidates narrowing layer by layer: recall (10⁴~10⁵) → pre-ranking (10²~10³) → fine-ranking (10¹~10²) → auction (1~3 ads shown), each layer trading between "shrinking candidates" and "raising scoring precision"; the modern recall form is **multi-channel recall** — boolean targeting, semantic vectors, collaborative/behavioral, and popularity fallback run in parallel each taking Top-K, merged and deduplicated before entering ranking.

### 12.10

- **First-party Data** — Data generated on an advertiser's own channels (CRM, orders, website visitor behavior); small in volume but with the clearest semantics, it is the "soul" of all data.
- **Second-party Data** — Behavioral and delivery data generated by users on the media/ad platform and held by the platform itself; the mainstay guiding delivery under the ad network model.
- **Third-party Data** — Data owned and circulated by providers that do not directly participate in ad trading (small and mid-sized media, data companies, etc.); large in volume but of uneven quality.
- **User Identifier** — The basis for linking "which behaviors come from the same user," such as cookie, IDFA, Android ID/IMEI; identity is the 1 in front of a string of 0s.
- **Decision Behavior** — Conversions and pre-conversions (searching, browsing, price comparison, cart addition, and other pre-order actions), occurring on the advertiser's own site; the clearest intent orientation and the highest value.
- **Semi-active Behavior** — Weak-purpose content consumption behaviors such as sharing and page views; they capture the domain of interest with limited precision, and their volume is the largest of all behavior classes.
- **Cookie Mapping** — The technique of aligning the same user's cookie identities across different domain systems with one party's consent; it has become historical infrastructure since third-party cookies' exit.
- **Data Management Platform (DMP)** — A product that organizes and processes raw data into directly usable user labels and supports monetization; it comes in two models — first-party (hosting and processing for a service fee) and third-party (processing and selling for monetization).
- **Customer Data Platform (CDP)** — First-party data infrastructure that unifies a brand's own touchpoints (website, app, CRM) into persistent customer profiles; in its modern form it has replaced most scenarios of the old first-party DMP.
- **Audience Segment** — A set of users selected by labels; the standard "trading unit" of data trading and audience targeting.
- **Data Trading Platform (Third-party DMP)** — A product that aggregates raw behavioral data from multiple sources, processes it into labels under its own logic, sells it to monetize, and shares revenue with data providers; the representative case is BlueKai.
- **Unified ID 2.0 (UID2)** — An open identity framework led by The Trade Desk, rooted in hashed email addresses/phone numbers; the cross-domain identity solution replacing third-party cookies.
- **Data Trading** — The market mechanism in which labels are relayed through the ADX, attached to bid requests and priced on a CPM basis, and delivered on the DSP's actually won impressions.
- **Data Clean Room** — A compliant collaboration environment where multi-party data is matched and analyzed under the premise that neither side can see the other's raw records, outputting only aggregated results (often with differential privacy added); the mainstream form of data collaboration in the 2020s.
- **Quasi-identifier** — A set of attributes individually unidentifying but capable of locating a specific person in combination (e.g., age + city + job title); a high leakage risk even without PII.
- **K-Anonymity** — Generalizing quasi-identifiers so that every group of quasi-identifier instances in the dataset has K records identical to it; not applicable to extremely sparse behavioral data.
- **Differential Privacy** — A technique that modifies the dataset to a certain degree so as to minimize privacy leakage risk with as little loss of query accuracy as possible.
- **Demand-side Data Security** — The risk that an advertiser's first-party data (such as visitor sets) is obtained and exploited by the platform or competitors in RTB; the typical tactic is merging visitor sets under vague labels and reselling them.
- **GDPR** — The EU General Data Protection Regulation (effective 2018): a sensitive-data list, explicit consent, and four rights (access / erasure / restriction of processing / portability), with the penalty cap being the higher of 20 million euros or 4% of global annual turnover.
- **PIPL** — China's Personal Information Protection Law (effective November 2021): establishes principles such as informed consent, minimal necessity, and withdrawable consent, and likewise distinguishes sensitive personal information.

### 12.11

- **Programmatic Creative** — an optimization approach in which a program assembles the key reasons for pushing the ad (geo, search term, featured product, etc.) into the creative online at delivery time, under the premise that the ad's basic appeal stays stable.
- **Click Heatmap** — a tool that visualizes the click density of each position of a creative; used both to guide creative iteration semi-quantitatively and to detect machine click flooding through distribution shape (too uniform / too concentrated).
- **A/B Testing** — an experimental method that splits real traffic into a control group and a treatment group running the original and the new scheme respectively, with online metrics adjudicating which is better.
- **Experimentation Framework** — the online system supporting A/B testing, responsible for traffic splitting, parameter distribution, and metric collection; the infrastructure underlying the evolution speed of an ad system.
- **Experiment Layer and Domain** — a layer is a container of experiment parameters divided by system module (retrieval / ranking / display); a domain is a traffic subset split within a layer by user-ID hashing; all of a user's requests deterministically land in the same domain (mutual exclusion within a layer).
- **Layered Experimentation** — a framework that expands experiment capacity by exploiting the relative independence of modules: mutual exclusion within layers, orthogonality across layers, a reserved non-overlapping domain for cross-layer joint tuning, and a companion publishing layer for gray-scale release.
- **Orthogonality** — the traffic splits of different experiment layers are independent of one another, so the same traffic can be reused by multiple layers and experiment capacity grows linearly with the number of layers.
- **AA Test** — a controlled experiment in which both groups run under exactly identical configurations, used to verify even splitting and consistent log definitions; a significant AA difference means the experimentation framework itself is biased.
- **Ad Monitoring** — a service in which the demand side commissions an independent third party to perform verification measurement of impressions, clicks, or conversions (about 1% of brand campaign budgets); its core vehicle is the **monitoring URL** that packs together ad/media/user information.
- **Brand Safety** — the requirement that ads not appear on content that damages the brand image, implemented by **advertising verification** (stop serving and switch creatives upon detecting unsafe content; the engineering core is iframe penetration to obtain the top-level URL).
- **Viewability** — verification of whether an ad impression was actually seen by the user (rendered); defined today by MRC-style dual thresholds on area and duration, and one of the settlement metrics for brand buying.
- **Non-Human Traffic (NHT)** — fraud in which the impressions, clicks, or conversions themselves are fabricated; the mainstream of CPM/CPC ad fraud, subdivided by method into machine fraud and human-operated fraud.
- **Attribution Fraud** — fraud that credits traffic from other channels or organic traffic to oneself; common in CPA/CPS advertising, where fabricating conversions is expensive.
- **Click Spam / Click Flooding** — a tactic that fabricates clicks for a large number of users and waits for their organic downloads to be attributed to the channel; smoking guns are CVR 1–2 orders of magnitude low and a near-uniform click-to-conversion time distribution.
- **Click Injection** — a tactic that exploits the Android install broadcast to fire a make-up click at the instant an app is installed, snatching attribution for the subsequent activation; signature: abnormally high CVR and an extremely short click-to-activation gap.
- **Cookie Stuffing** — attribution fraud in CPS affiliate advertising: silently planting a source cookie via hidden requests without the user clicking, hijacking organic conversions.
- **Traffic Hijacking** — quasi-fraud committed by operators of underlying network services that forcibly place ads where they have no right to serve or tamper with creatives/landing pages (channel pop-ups, creative replacement, search redirection, landing-page source hijacking); the first three harm media, source hijacking harms advertisers.
- **Device Farm** — a human-operated fraud form in which real people with real devices mass-produce browse-click-convert sequences; every dimension of the data looks genuine, requiring device clustering and association networks for detection.
- **Device Fingerprint** — a unique device identifier assembled from hardware and environment characteristics, used to track fraud sources across IPs/cookies and build device reputation scores.
- **Graph-based Fraud Detection** — connecting devices, IPs, accounts, and payment paths into an association network and exploiting the highly clustered structure of fraud rings to expose group characteristics that no single record can disguise.

### 12.12

- **Agreement Advertising (Contract Advertising)** — an ad transaction form with contractually guaranteed impression delivery: audience, volume, and unit price written into the contract, fulfillment responsibility on the supply side; opposed to auction advertising cleared by market competition.
- **Position Contract** — the earliest form of online ad selling: certain slots deliver a specified advertiser's ads exclusively over a period; no audience targeting, but retains brand-impact and competitor-exclusion premium on high-exposure slots.
- **CPT (Cost per Time)** — billing by time period for slot contracts, typically bought wholesale per slot; low technology on both sides, executed via agency media buying.
- **CPD (Cost per Day)** — per-day billing for slot selling; today's splash-screen and brand-resource scheduling contracts still use this convention.
- **Rotation Selling** — labeling successive visits to one slot with cyclic rotation numbers (e.g., {1, 2, 3, 4}) and selling same-number impressions as virtual slots; used when exclusive inventory is short but advertisers need deterministic display rules.
- **Random Rotation Start** — the key detail of rotation selling: a user's first impression draws its number uniformly at random from all numbers before cycling, so each rotation receives equal traffic.
- **Scheduling System** — a non-personalized tool that executes delivery automatically per contract schedule (e.g., DFP, Allyes, Baidu Ad Manager); with dynamic allocation and RTB added, it approaches the SSP.
- **Blank-Slot-Prevention Ad (fallback creative)** — the default creative rendered via CDN when a dynamic ad times out or errors, ensuring the slot is never blank; engineering details in 12.7.0.
- **Inventory Contract (GD contract)** — a contract selling a total impression volume under agreed audience conditions at an agreed unit price, i.e., guaranteed delivery; shortfalls may trigger media compensation.
- **Guaranteed Delivery (GD)** — the umbrella term for the delivery system and selling market of impression contracts; the "guarantee" is the volume; its algorithmic core is online allocation (see 12.7).
- **Audience-based Selling** — selling slot traffic sliced by audience labels as the object of sale: data participates directly in selling for the first time, spawning structured hierarchical taxonomies as sales catalogs.
- **Audience Package** — a sellable traffic unit defined by a label combination; audience packages overlap pervasively, the source of guaranteed-allocation complexity.
- **Selling Geo** — the geo-targeting clause of a contract; geo is the most basic selling dimension that every ad system must support.
- **Minimum Commit** — the minimum daily audience volume for a label to enter the contract catalog; labels below it cannot be sold with guarantees and should be bundled or pushed to auctions.
- **Traffic Forecasting** — estimation of the function $t(u, b)$ (label combination × bid), supporting pre-sales guidance, online allocation, and bid guidance; the engineering scheme is in 12.7.2.
- **Traffic Shaping** — proactively influencing audience traffic distribution by tuning user-product funnels (e.g., homepage links) to help contracts close.
- **Showcase Effect** — the continuous shaping of brand value and conversion by long-term exclusive occupation of premium slots; a core selling point of exclusive selling.
- **Category Exclusivity** — a contract's added service promising no competing ads on the same page; the source of slot-selling premium.

### 12.13

- **Native Ads** — the product direction of uniformly producing or jointly ranking commercial and non-commercial content, also "content as ad"; advertorials, search ads, and feed ads each reflect one facet.
- **Feed Ads** — an ad form satisfying two conditions: the ad interacts coupled with the content; the content segments separated by the ad have no direct relation. In product essence, a multi-slot, freely placed auction product.
- **Content as Ad** — the product philosophy of native advertising: ads are no longer independent of content but part of content production and ranking.
- **Splash Ad** — a full-screen ad shown during app load; the user has no active task while waiting, so annoyance is low and brand value high; mostly sold by contract.
- **Interstitial Ad** — a form appearing on app pause, similar to video pause ads; like mobile banners, inflated CTR with relatively poor conversion.
- **Offerwall** — a direct-push ad form for app-download promotion, analogous to off-platform recommendation.
- **Points Wall** — an incentive ad granting points redeemable for virtual goods after download and activation; clicks and activations look good but downstream retention is poor; once used for chart-climbing and game launches.
- **Rewarded Video** — a native ad granting a virtual reward after the user watches an unskippable 15–30 second video; rewards viewing only, not downloads — hence the native form with intact user quality and the highest eCPM.
- **Expressive Native** — the aspiration to make the ad's display style consistent with content, requiring the media to control ad display form (including font and color adaptation).
- **Scene Native** — the aspiration to keep ad targeting decisions consistent with content production, triggered by user scenario and intent; search ads are native in both senses.
- **Embedded Native Advertising** — the native platform mechanism where the media requests structured paid content via structured queries (e.g., "type=hotel; location=Lhasa") and assembles it in its own templates.
- **Structured Paid Content** — the form of native platform inventory: not finished creatives but per-industry structured field material for the media to assemble into seamlessly fused creatives.
- **oCPX** — the smart-delivery mode separating billing from bidding: billing stays CPM/CPC while the advertiser expresses a conversion bid and the platform takes over estimation and auction conversion, lowering the barrier for small clients.
- **Conversion Tracking** — the recording and reporting chain from impression, click, to conversion events; on mobile, conversions across industries converge to app-store downloads, enabling cross-industry CVR modeling.
- **Mixing** — the decision problem of organic content and ads competing for display positions as one candidate pool under comparable scores; feed ads evolving to unified-criterion ranking is the origin of the mixing problem.
- **Feed Density (S/K)** — the two parameters controlling ad placement: S is the first ad's position, K the gap between ads; tune S/K under an average-ad-count constraint to optimize overall CTR.

_The glossary covers the full book: Volume I (Ch0–Ch4, Parts 1–5), Volume II (Ch5–Ch10, Parts 6–11), and the Special Topic (Part 12, Computational Advertising)._
