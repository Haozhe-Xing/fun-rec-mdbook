# 术语表 (Glossary)

> 本表收录本书出现的核心术语，按中文首字母排序。格式：**术语** — 一句话定义，必要时交叉引用相关章节。

---

## 基础概念（全篇）

- **候选集 (Candidate Set)** — 经过召回阶段筛选后、待排序模型精排的物品集合，规模通常为千级。
- **判别式推荐 (Discriminative Recommendation)** — 将推荐定义为「给定用户-物品-场景三元组，预测交互概率」的范式，核心为打分函数 $Score = f(User, Item, Context)$。
- **生成式推荐 (Generative Recommendation)** — 让模型根据用户历史与场景直接「创作」推荐序列的范式，核心为生成函数 $[I_1, I_2, \ldots, I_k] = g(User, Context)$。
- **召回 (Retrieval / Recall)** — 推荐流水线的第一阶段，在毫秒级延迟内从亿级物品库快速筛选千级候选。
- **排序 (Ranking)** — 推荐流水线的第二阶段，用复杂模型为每个候选计算精确预测分数。
- **重排 (Re-ranking)** — 推荐流水线的第三阶段，在保持相关性的前提下优化整张列表的多样性、新颖性等体验指标。

---

## Part 2 · 快速候选召回

- **ItemCF (基于物品的协同过滤)** — 通过物品共现余弦相似度、由种子物品扩散候选的协同过滤方法（Ch2.1）。
- **UserCF (基于用户的协同过滤)** — 通过用户相似度聚合邻居行为来预测目标用户兴趣的协同过滤方法（Ch2.1）。
- **Swing** — 通过分析用户-物品二部图子结构、以「特异性共现」过滤热门噪声的工业级相似度算法（Ch2.1）。
- **二部图 (Bipartite Graph)** — 用户与物品为两类节点、交互为边的图结构，Swing 在其上分析子结构以过滤噪声（Ch2.1）。
- **FunkSVD** — 矩阵分解基础模型，把评分矩阵分解为用户/物品隐向量并以内积预测评分（Ch2.1）。
- **BiasSVD** — 在 FunkSVD 基础上引入全局均值 $\mu$、用户偏置 $b_u$、物品偏置 $b_i$ 的矩阵分解改进模型（Ch2.1）。
- **MF（矩阵分解）** — 将用户-物品交互分解为低秩隐向量、以向量距离反映偏好的方法家族（Ch2.1）。
- **Surprise 算法** — Swing 的衍生算法，从类别/商品/聚类三层面挖掘互补商品（Ch2.1）。
- **Word2Vec (Skip-Gram)** — 用中心词预测上下文、以负采样高效学稠密词向量的序列建模方法，是 I2I 向量召回的理论基础（Ch2.2）。
- **Item2Vec** — 把 Word2Vec Skip-Gram 直接迁移到推荐、将用户交互历史当「句子」学物品向量的 I2I 方法（Ch2.2）。
- **EGES (Enhanced Graph Embedding with Side Information)** — 在随机游走访物品图上融合属性、用商品特定注意力加权聚合以解决冷启动/稀疏的 I2I 向量方法（Ch2.2）。
- **双塔模型 (Two-Tower)** — 用户与物品分别由独立塔编码为向量、仅在最终内积交互的 U2I 召回架构（FM/DSSM/YouTubeDNN）（Ch2.3）。
- **YouTubeDNN** — 把召回定义为「预测用户下一观看」、采用非对称双塔与时序分割的工程化双塔模型（Ch2.3）。
- **MIND (Multi-Interest Network with Dynamic Routing)** — 用多兴趣胶囊分别代表用户多元兴趣、各胶囊独立检索再合并的序列召回模型（Ch2.4）。
- **动态路由 (Dynamic Routing)** — 胶囊网络中确定低层与高层胶囊连接强度的迭代算法，MIND 借其把行为软聚类为兴趣胶囊（Ch2.4）。
- **squash 函数** — MIND 把向量模长非线性压缩到 $[0,1)$、方向不变、模长表兴趣存在概率的函数（Ch2.4）。
- **标签感知注意力 (Label-Aware Attention)** — MIND 训练时用目标物品向量作查询、从多兴趣胶囊中挑最相关者的注意力机制（Ch2.4）。
- **SDM (Sequential Deep Matching)** — 分别建模短期（LSTM+多头）与长期（特征维度注意力）兴趣、以门控动态融合的序列召回模型（Ch2.4）。
- **LSTM（长短期记忆）** — SDM 用以处理会话序列时序依赖、抑制随机误点击的循环网络（Ch2.4）。
- **多头自注意力 (Multi-Head Self-Attention)** — SDM 在 LSTM 之后并行多路注意力以捕捉序列内多重兴趣的机制（Ch2.4）。
- **Trinity** — 用层次化 VQ 聚类 + 统计直方图显式保留全量历史兴趣、根治兴趣遗忘的召回框架，含 M/LT/L 三召回器（Ch2.5）。
- **层次化聚类 (Hierarchical Clustering)** — Trinity 训练阶段用 VQ 维护两级可学习聚类中心（主 128 / 次 1024）的结构（Ch2.5）。
- **流式向量量化索引 (Streaming VQ)** — 让物品实时量化到聚类、中心经 EMA 持续自适应、无需中断重建的索引结构（Ch2.5）。
- **Exponential Moving Average (EMA)** — 用所属物品 Embedding 加权平均平滑更新聚类中心、使其适应分布变化的机制（Ch2.5）。
- **兴趣遗忘 (Interest Amnesia)** — 在线学习框架拟合近期样本、致稀疏长尾兴趣记忆衰退的现象，Trinity 试图解决（Ch2.5）。
- **长期兴趣召回 (Trinity-L)** — Trinity 用轻量双塔选种子物品、再基于 Embedding 相似度做 I2I 检索的召回器（Ch2.5）。
- **长尾兴趣召回 (Trinity-LT)** — Trinity 用流式频率估计追踪长尾聚类、对长尾主题显著行为增强推送的召回器（Ch2.5）。
- **归并排序服务策略 (Merge-Sort Serving)** — Streaming VQ 把 score 拆为「聚类级个性化 + 聚类内流行度」、用最大堆 K 路归并保证每聚类贡献候选（Ch2.5）。

---

## Part 3 · 精准偏好预测

- **记忆（Memorization）** — 模型学习并记住历史中频繁共现的特征组合（如"买 A 的人也买 B"），对应 Wide&Deep 中 Wide 部分（Ch3.1）。
- **泛化（Generalization）** — 模型学到特征间的深层关系，能处理训练时罕见的组合，对应 Wide&Deep 中 Deep 部分（Ch3.1）。
- **交叉特征（Cross-product Features）** — 将多个独立特征人工组合成的新特征，用于 Wide 部分捕捉特定共现模式 `AND(a, b)`（Ch3.1）。
- **联合训练（Joint Training）** — Wide 与 Deep 两部分由同一损失同时更新全部参数，区别于分别训练再集成（Ch3.1）。
- **因子分解机 / FM（Factorization Machine）** — 用每个特征的低维隐向量内积建模二阶交叉，把参数量从 $O(n^2)$ 降到 $O(nk)$ 并缓解稀疏（Ch3.2）。
- **参数共享（Parameter Sharing）** — FM 把交叉权重表示为隐向量内积，使未共现特征也能通过各自隐向量泛化预测（Ch3.2）。
- **共享 Embedding（Shared Embedding）** — DeepFM 中 FM 与 DNN 组件共用同一份特征 Embedding，兼顾低阶/高阶交互与训练效率（Ch3.2）。
- **Cross Network（DCN）** — 每层与原始输入做残差交叉 $x_{l+1}=x_0 x_l^\top w_l + b_l + x_l$，明确产出元素级高阶交叉（Ch3.2）。
- **CIN（Compressed Interaction Network, xDeepFM）** — 在向量级做哈达玛积并加权压缩，逐层显式产出向量级高阶交叉（Ch3.2）。
- **局部激活（Local Activation, DIN）** — 用户兴趣表示随候选广告动态变化，由注意力机制加权历史行为得到（Ch3.3）。
- **辅助损失（Auxiliary Loss, DIEN）** — 逼 GRU 隐状态预测下一步行为，使其学到有意义的兴趣表示（Ch3.3）。
- **AUGRU（Attention Update Gate GRU, DIEN）** — 用注意力得分缩放 GRU 更新门，让相关兴趣顺畅传递、抑制兴趣漂移（Ch3.3）。
- **会话（Session, DSIN）** — 一段时间内意图集中的行为单元；DSIN 以会话为基本单元做分层序列建模（Ch3.3）。
- **负迁移 / 跷跷板（Negative Transfer / Seesaw）** — 多任务硬共享时，任务梯度冲突导致提升一目标损害另一目标的现象（Ch3.4）。
- **MMoE（Multi-gate Mixture-of-Experts）** — 每任务配专属门控，对共享专家加权融合，实现梯度软隔离以缓解冲突（Ch3.4）。
- **PLE / CGC（Progressive Layered Extraction / Customized Gate Control）** — 显式分离共享专家与任务专家，物理切断跨任务梯度干扰路径（Ch3.4）。
- **样本选择偏差（Sample Selection Bias, ESMM）** — CVR 模型在点击样本训练、全量曝光预估，导致训练/预估分布不一致（Ch3.4）。
- **全空间建模（Entire Space, ESMM）** — 用 $pCTCVR = pCTR \times pCVR$ 在曝光空间联合优化，化解 CVR 的偏差与稀疏（Ch3.4）。
- **Uncertainty Weight（UWL）** — 按任务不确定性动态调损失权重，不确定小且损失大时降权（Ch3.4）。
- **GradNorm** — 按梯度量级与相对训练速率动态平衡多任务损失（Ch3.4）。
- **帕累托优化（Pareto Optimization）** — 用 KKT 条件把损失权重作为可学习变量，把优化引向帕累托前沿（Ch3.4）。
- **多场景建模（Multi-scenario）** — 不同场景/分布下预估同一目标，需兼顾共性与特性（区别于多任务）（Ch3.5）。
- **STAR FC（Star Topology FCN）** — 每层参数由共享与场景私有参数元素积融合 $W_p^\star = W_p \otimes W$（Ch3.5）。
- **分区归一化（Partitioned Normalization, PN）** — 按场景分别统计 Batch Norm 的均值/方差，避免跨场景分布混淆（Ch3.5）。
- **Gate NU（PEPNet）** — 轻量门控单元，由先验特征生成动态缩放权重调制共享参数（Ch3.5）。
- **EPNet / PPNet（PEPNet）** — EPNet 做场景级 Embedding 个性化，PPNet 做样本级任务塔参数个性化（Ch3.5）。
- **APG（Adaptive Parameter Generation）** — 由样本感知输入动态生成参数矩阵，并用低秩分解控制开销（Ch3.5）。
- **M2M（元学习多场景多任务）** — 用元学习器根据场景/输入特征动态生成任务模型参数（Ch3.5）。

---


- **STAR FCN（Star Topology FCN）** — 每层参数由共享与场景私有参数元素积融合 $W_p^\star = W_p \otimes W$。
## Part 4 · 重排多样性建模

- **列表同质化（List Homogenization）** — 精排逐点最优化导致头部物品高度相似的现象，是重排存在的根本动机。
- **最大边际相关（MMR, Maximal Marginal Relevance）** — 用 $\lambda\cdot\text{Rel}-(1-\lambda)\max\text{Sim}$ 的边际收益公式，以贪心方式在相关性与多样性间权衡（Ch4.1）。
- **滑动窗口 MMR（MMR with Window）** — 只用最近 $w$ 个已选物品计算相似度惩罚的 MMR 变体，用于长列表降成本（Ch4.1）。
- **行列式点过程（DPP, Determinantal Point Process）** — 用核矩阵行列式度量集合多样性的概率模型，能精确刻画多物品间的相互排斥（Ch4.1）。
- **核矩阵（Kernel Matrix, $L$）** — DPP 中融合相关性与多样性的半正定矩阵，构造为 $L=\text{Diag}(r)\cdot S\cdot\text{Diag}(r)$（Ch4.1）。
- **Cholesky 加速** — 利用 $L_{Y_g}=VV^\top$ 分解，把 DPP 贪心选择简化为每轮取 $\arg\max\log(d_i^2)$ 的高效求解法（Ch4.1）。
- **个性化重排（Personalized Re-ranking）** — 把用户个性化信号深度融入列表级优化、由模型端到端学习最优列表的重排范式（Ch4.2）。
- **PRM（Personalized Re-Ranking Model）** — 用 Transformer 编码列表、融合个性化向量（PV）实现端到端个性化重排的模型（Ch4.2）。
- **个性化向量（PV, Personalization Vector）** — PRM 中由预训练 CTR 模型隐藏层激活提取的用户-物品偏好向量，是 PRM 个性化的核心（Ch4.2）。
- **排列变异影响（Permutation-Variant Influence）** — 相同物品、不同排列顺序导致用户行为不同的现象，是 PRS 的动机（Ch4.2）。
- **PRS（Permutation Retrieve System）** — 直接优化排列顺序体验收益的重排模型，采用 PMatch + PRank 两阶段化解 $n!$ 组合爆炸（Ch4.2）。
- **FPSA（Fast Permutation Searching Algorithm）** — PRS 的 PMatch 阶段算法，用 beam search 结合 CTR/Next 双模型快速生成候选排列（Ch4.2）。
- **DPWN（Deep Permutation-Wise Network）** — PRS 的 PRank 阶段网络，用 Bi-LSTM 评估候选排列质量，以 List Reward（LR）为指标选最优（Ch4.2）。
- **List Reward（LR）** — PRS 中排列所有位置预测点击概率之和，用于比较候选排列的整列收益（Ch4.2）。

---


- **重排（Re-ranking）** — 三阶段漏斗最末端，对精排输出的候选列表做列表级优化（多样性、新颖性、业务规则）以最大化整屏体验。
## Part 5 · 前沿趋势

- **数据偏差（Data Bias）** — 推荐数据在收集阶段就因系统策略、用户习惯等带来的系统性失真（Ch5.1）。
- **选择偏差（Selection Bias）** — 显式反馈中用户只评分感兴趣内容，导致观测数据不代表真实态度（MNAR）（Ch5.1）。
- **曝光偏差（Exposure Bias）** — 隐式反馈中用户只能看到被推荐的物品，未交互可能源于未曝光而非不感兴趣（Ch5.1）。
- **从众偏差（Conformity Bias）** — 用户受群体意见影响而附和给出非独立、非真实的评价（Ch5.1）。
- **位置偏差（Position Bias）** — 列表推荐中用户更关注排在前面的物品，使点击受位置而非相关性影响（Ch5.1）。
- **非随机缺失（MNAR, Missing Not At Random）** — 观测到的评分并非随机样本，导致统计偏差（Ch5.1）。
- **流行度偏差（Popularity Bias）** — 模型过度学习热门物品交互模式，致推荐偏向热门、长尾难被发现（Ch5.1）。
- **反馈闭环（Feedback Loop）** — 推荐结果影响用户未来行为，行为又成新训练数据，使偏差被滚雪球放大（Ch5.1）。
- **马太效应（Matthew Effect）** — 热门物品因曝光多而交互多、再获更多曝光的「富者愈富」恶性循环（Ch5.1）。
- **逆倾向得分（IPS, Inverse Propensity Score）** — 用观测概率的倒数作样本权重，逆转选择/曝光偏差，得到无偏风险估计（Ch5.1）。
- **权重截断（Weight Clipping）** — 对 IPS 极端权重做上限截断，在无偏与低方差间折中（Ch5.1）。
- **位置感知学习（PAL, Position-bias Aware Learning）** — 把点击概率分解为「看到概率 × 看到后点击概率」，在架构上解耦位置与偏好（Ch5.1）。
- **冷启动（Cold Start）** — 新物品或新用户缺乏历史交互，传统协同过滤难以服务的困境（Ch5.2）。
- **CB2CF（Content-Based to Collaborative Filtering）** — 学习内容特征到协同过滤表示的映射，使新物品直接获得 CF 质量表示（Ch5.2）。
- **映射网络（Mapping Network）** — CB2CF 核心，多层全连接，学内容空间到 CF 嵌入空间的非线性映射（Ch5.2）。
- **MetaEmbedding** — 用元学习优化「可快速适应」的初始 embedding 生成器，改善物品冷启动（Ch5.2）。
- **MAML（Model-Agnostic Meta-Learning）** — 「学会如何学习」的元学习框架，学良好初始化以少量样本快速适应新任务（Ch5.2）。
- **MeLU（Meta-Learned User preference estimator）** — 基于 MAML 的用户冷启动方法，把每用户当独立任务快速适应（Ch5.2）。
- **POSO（Personalized Cold Start Modules）** — 用人群专用子模块 + 个性化门控解决用户冷启动的架构方法（Ch5.2）。
- **生成式召回（Generative Retrieval）** — 把推荐重定义为序列生成，自回归预测下一个物品的生成式范式（Ch5.3）。
- **事件流（Event Stream）** — HSTU 把用户属性、行为、时间戳统一编码成的异构序列表示（Ch5.3）。
- **语义 ID（Semantic ID）** — TIGER 用 RQ-VAE 把物品内容编码成的结构化 Token 元组，携带语义、支持知识共享与冷启动（Ch5.3）。
- **RQ-VAE（Residual Quantization VAE）** — 残差量化变分自编码器，逐层量化残差生成语义 ID（Ch5.3）。
- **端到端统一生成（End-to-end Generative）** — 用单一模型完成从召回到排序全流程的生成式形态（Ch5.3）。
- **稀疏专家混合（MoE, Mixture-of-Experts）** — OneRec 解码器中激活少数专家子网络以增容量不增算力（Ch5.3）。
- **迭代偏好对齐（IPA, Iterative Preference Alignment）** — OneRec 用奖励模型从多候选中构造选择/拒绝对、以 DPO 对齐偏好的机制（Ch5.3）。
- **DPO（Direct Preference Optimization）** — 直接偏好优化，无需独立 critic 即可用「选择/拒绝」对比对优化策略（Ch5.3）。


---

# 下篇术语 · 生成式推荐主线


- **结果偏差（Result Bias）** — 有偏数据经模型训练后体现在推荐结果中的偏差。
- **不公平性（Unfairness）** — 系统对某些用户群体或物品类别产生系统性歧视。
- **倾向得分（Propensity Score）** — 用户-物品交互被观测到的概率 $P(O_{u,i}=1)$，IPS 加权的分母。
- **朴素估计量（Naive Estimator）** — 直接在观测数据上求平均的评估量，选择偏差下是有偏的。
- **半合成实验（Semi-synthetic Experiment）** — 用真实数据集补全为 ground truth 再按偏差模型采样，制造「已知答案」以量化纠偏效果。
- **ProbSeen 模块** — PAL 中仅输入位置、输出被看到概率的轻量模块。
- **pCTR 模块** — PAL 中不含位置信息、建模用户真实偏好的深度模块；推理时单独使用以去偏。
- **个性化淹没（Submergence）** — 新用户远少于老用户时，其个性化信号被多数老用户数据主导的训练过程淹没。
- **内容冷启动（Item Cold Start）** — 新物品缺用户交互，协同过滤无法计算其相似度。
- **用户冷启动（User Cold Start）** — 新用户缺行为历史，只能获得基于流行度的通用推荐。
- **约束优化模块（Constraint Optimization）** — CB2CF 中用余弦相似度约束保证映射后与真实 CF 嵌入语义一致。
- **元损失（Meta Loss）** — MetaEmbedding 中平衡初始质量与适应后性能的损失，如 $l_{meta}=\alpha l_a+(1-\alpha)l_b$。
- **参数分离（Parameter Separation）** — MeLU 把共享 embedding 参数 $\theta_1$ 与快速适应用的决策参数 $\theta_2$ 分开。
- **个性化门控（Personalized Gating）** — POSO 中按用户特征（如 is_new_user）输出各子模块权重的网络。
- **点向聚合（Pointwise Aggregation）** — HSTU 摒弃 softmax 归一化、保留用户偏好强度的注意力聚合机制。
- **生成式排序（Generative Ranking）** — 把自回归生成思想引入排序阶段（如 GenRank、MTGR）。
- **动作导向（Action-oriented）** — GenRank 预测用户对候选的动作概率而非物品 ID，降低计算开销。
- **用户样本聚合（User Sample Aggregation）** — MTGR 把用户全部候选聚为单样本，共享用户特征计算。
- **组层归一化（GLN, Group Layer Normalization）** — MTGR 对不同语义空间 token 分别归一化。
- **会话级生成（Session-level Generation）** — OneRec 直接生成一组有序推荐列表（一个「会话」）而非单一下一个物品。
## Part 6 · 生成式推荐范式基础

- **生成式推荐（Generative Recommendation）** — 把推荐重定义为序列生成任务，模型直接学习用户交互序列的生成概率并自回归产出物品序列，而非对候选逐一打分。
- **判别式推荐（Discriminative Recommendation）** — 学习条件概率 $p(y=1\mid u,i,c)$，对给定候选物品预测正向交互概率的建模范式。
- **自回归建模（Autoregressive Modeling）** — 当前预测依赖之前所有已生成输出的生成方式，使信息在时间维度形成循环流动、天然捕捉序列依赖。
- **原子 ID（Atomic ID）** — 传统推荐为每个物品分配的随机唯一数字编号，彼此正交无语义关联，难以泛化到新物品。
- **语义 ID（Semantic ID, SID）** — 把物品表示为固定长度的离散 token 序列，每个 token 来自可控大小的语义码本，既编码层次化语义又保留协同信息。
- **物品 Token 化（Item Tokenization）** — 将推荐系统中的物品转化为生成式模型可理解、可生成的 token 序列的关键技术，是连接传统推荐数据与生成式模型的桥梁。
- **Transformer** — 基于自注意力机制的深度架构，擅长并行捕捉长程依赖，是生成式推荐与 LLM 的主流骨干。
- **自注意力（Self-Attention）** — 通过 Query/Key/Value 的「查询—匹配—聚合」机制，让序列每个位置动态聚焦任意其他位置信息的注意力计算。
- **多头注意力（Multi-Head Attention）** — 并行计算多组独立 Q/K/V、各自学习不同关注模式的注意力机制，类似多个「专家」从不同角度理解序列。
- **位置编码（Positional Encoding）** — 为序列各位置注入顺序信息的编码，分绝对（正弦/可学习）与相对（偏置）两类，推荐中常扩展为时间感知编码。
- **相对时间位置编码（Relative Temporal Positional Encoding）** — HSTU 等采用的时间编码，以 $\log(\Delta t+1)$ 建模行为间时间间隔，使模型兼顾长短期兴趣。
- **Encoder-Decoder 架构** — 双塔生成架构：Encoder 双向理解输入、Decoder 因果自回归生成，并以交叉注意力动态查询输入，适合异构输入输出。
- **Decoder-Only 架构** — 统一单塔生成架构：输入与输出拼接为连续序列、仅靠因果自注意力自回归生成，参数高效、MFU 高、兼容 LLM 生态。
- **因果掩码（Causal Masking）** — 在注意力矩阵中对未来位置施加 $-\infty$ 的掩码，保证预测第 $t$ 个 token 时只能依赖前 $t-1$ 个，实现自回归并支持训练并行。
- **Diffusion 模型（扩散模型）** — 通过前向逐步加噪、反向迭代去噪从噪声恢复数据的生成范式，分数据空间与潜在空间两类，与 Transformer 互补。
- **Scaling Law（规模化效应）** — 模型性能随参数量、数据量、计算量增长而持续提升的经验规律，支撑生成式模型的参数扩展。
- **涌现能力（Emergent Abilities）** — 模型规模与数据达到阈值后突然出现的零样本/少样本等能力。
- **预训练（Pre-training）** — LLM 第一阶段，在大规模无标注文本上以因果语言建模（下一 token 预测）学习目标，建立通用语言生成能力。
- **指令微调（Instruction Tuning / SFT）** — LLM 第二阶段，用「指令—输入—输出」三元组做条件语言建模，仅对输出算损失，使模型学会遵循指令。
- **偏好对齐（Preference Alignment）** — LLM 第三阶段，让输出更符合人类价值观与偏好，方法含 RLHF 与 DPO。
- **RLHF（基于人类反馈的强化学习）** — 收集人类偏好对训练奖励模型，再以 PPO 强化学习优化生成策略并用 KL 散度约束参考模型。
- **DPO（直接偏好优化）** — 无需显式奖励模型与强化学习，用策略与参考模型比值隐式表示奖励、以似监督方式优化偏好的对齐方法。
- **VQ-VAE（向量量化变分自编码器）** — 用可学习码本将连续语义向量离散化为单一码本索引的自编码器，是语义 ID 离散化的奠基技术。
- **码本（Codebook）** — VQ/RQ 系列中可学习或聚类的离散向量集合，每个向量（码字）对应一个语义 token。
- **直通估计器（Straight-Through Estimator, STE）** — 训练量化模型时，前向执行离散量化、反向把量化近似为恒等映射以传递梯度的技巧。
- **RQ-VAE（残差量化 VAE）** — 通过多层残差量化把连续向量编码为长度 $L$ 的 token 序列、容量达 $K^L$ 并自然涌现层次化语义的模型。
- **RQ-Kmeans** — 用 K-means 聚类替代梯度学习来构建码本的残差量化方案，将表示学习与码本构建解耦，新物品可经向量检索分配 SID。
- **RQ-OPQ** — RQ 处理层次化语义、OPQ（优化乘积量化）处理最后一层残差中独特属性的混合编码方案，兼顾召回与长尾精确区分。
- **SID 冲突（SID Collision）** — 量化信息损失导致不同物品映射到相同 SID 序列的现象，可通过均匀分配或混合编码消歧缓解。

## Part 7 · Scaling 生成式排序

- **Scaling Law（缩放定律）** — 在合适架构下，模型性能随计算量、数据量、参数量增加呈可预测幂律提升的规律，形式常为 $L = L_0 + \beta\ln C$。
- **DLRM（Deep Learning Recommendation Model）** — 传统深度学习推荐模型，依赖手工特征、异构模块与 item-level 逐候选打分，是 Scaling Law 长期失效的代表。
- **Generative Recommender / GR（生成式推荐）** — Meta 提出的范式，把推荐视为内容与行为交织的随机过程，用统一序列 + 自回归训练实现 user-level 建模。
- **HSTU（Hierarchical Sequential Transduction Unit）** — Meta 为推荐场景定制的序列模型，三大创新：Pointwise Aggregation、相对时间偏置、门控前馈，首次验证推荐界 Scaling Law。
- **Pointwise Aggregation** — HSTU 用 SiLU 逐元素聚合（非 Softmax 归一化）替代标准 attention，保留兴趣的「绝对强度」信息。
- **相对注意力偏置 / RAB（rab）** — HSTU 在 attention score 中加入可学习偏置，同时考虑位置差、时间差与 token 类型，建模非均匀时间模式。
- **Stochastic Length（随机长度）** — HSTU 的训练技巧：以一定概率随机截断超长序列，降 $O(n^2)$ 复杂度并起正则化作用，参数 $\alpha$ 控制激进度。
- **M-FALCON** — HSTU 的推理算法：Batched Inference → Microbatching → KV Caching 三层优化，把多候选排序推理数百倍加速。
- **Action-Oriented Organization（行为导向组织）** — GenRank 的序列组织：把行为作主体、物品作属性（$[a_i^{(x_i)}]$），序列长度减半、训练提速约 79%。
- **ALiBi（Attention with Linear Biases）** — 无参数相对位置偏置：给距离远的 query-key 对施加与距离成正比的惩罚，可直接融入 FlashAttention kernel。
- **MTGR（Meituan Generative Recommendation）** — 美团的混合范式：用生成式架构（Transformer + user-level 聚合）做判别式排序，保留传统交叉特征。
- **Group Layer Normalization / GLN（分组层归一化）** — MTGR 按 token 类型（User/Seq/RT/Cand）分组独立归一化，解决异构 token 的语义空间冲突。
- **Dynamic Masking（动态掩码）** — MTGR 按每样本 token 实际时间戳动态生成 attention mask：静态全可见、动态按因果、候选间独立，防信息泄露。
- **MFU（Model FLOPs Utilization，模型浮点利用率）** — GPU 上有效矩阵乘法计算占总理论算力的比例；传统 DLRM 约 4–5%，LLM 约 40–60%。
- **RankMixer** — 阿里的 hardware-aware 架构：用 Token Mixing 替代 Self-Attention、Per-Token FFN 捕捉异质性、Sparse MoE 扩展参数，把 MFU 拉到 45%。
- **Token Mixing（令牌混合）** — RankMixer 的核心操作：在特征维度（按 head 重组）做信息混合而非 token 对相似度，复杂度从 $O(T^2D)$ 降到 $O(TD^2)$。
- **Per-Token FFN（逐令牌前馈）** — RankMixer 为每个 token 配备独立 FFN 参数以捕捉异构特征空间，计算复杂度与共享 FFN 相同但参数更专门化。
- **ReLU Routing / DTSI-MoE** — RankMixer 的稀疏专家策略：ReLU 路由动态激活不同数量 expert；Dense-Training/Sparse-Inference 用双 router 兼顾充分训练与高效推理。
- **OneTrans** — 字节的统一架构：单一 Transformer backbone 同时做序列建模与特征交互，打破 encode-then-interaction 的模块碎片化。
- **Mixed Parameterization（混合参数化）** — OneTrans 的参数组织：S-tokens（序列）共享参数、NS-tokens（非序列）独立参数，解决 token 异质性冲突。
- **Pyramid Stack（金字塔堆叠）** — OneTrans 的渐进式蒸馏：逐层只保留尾部 query token、KV 用全部 token，把信息蒸馏到尾部并降计算量。
- **Cross-Request KV Caching** — OneTrans 跨请求复用用户侧 KV 缓存（仅追加新增事件），使每请求序列计算近似 $O(1)$（相对候选数）。
- **encode-then-interaction（先编码后交互）** — 传统分离式范式：序列模块编码为定长向量后再与静态特征拼接做特征交互，信息流受限且执行碎片化。

## Part 8 · 端到端生成式应用

- **多阶段级联架构（Multi-stage Cascading Architecture, MCA）** — 传统推荐/搜索/广告系统采用的漏斗式多模块架构（召回→预排序→排序→重排），各阶段独立优化、目标可能冲突。
- **语义 ID（Semantic ID）** — 把离散业务对象（物品/商品/广告）编码为由粗到细的多层离散 Token 序列，使生成模型能在可控词表内「说出」该对象。
- **RQ-Kmeans（Residual Quantization K-means）** — 在残差表示上逐层做 K-means 聚类以构建码本的层次化量化方法；与端到端训练的 RQ-VAE 相对，RQ-Kmeans 直接建码本、非端到端。
- **RQ-VAE（Residual Quantized VAE）** — 残差量化变分自编码器，端到端训练把连续表示离散化为多层语义 ID，常见于 EGA。
- **Encoder-Decoder 生成架构** — 编码器双向融合用户/查询上下文、解码器自回归生成目标语义 ID 序列的统一生成结构。
- **Lazy Decoder-Only** — OneRec-V2 架构：把上下文预处理为静态键值对（Context Processor），解码器仅对目标 Token 计算损失，将算力集中到产生梯度的目标上。
- **Scaling Law（规模律）** — 模型损失随参数量幂律衰减的可预测规律；OneRec-V2 在推荐模型上验证了该规律。
- **挤压效应（Squeezing Effect）** — 强化学习后模型把概率质量压到当前最优输出，使部分合法 Token 概率被压到与非法 Token 相近，导致难分。
- **格式奖励（Format Reward）** — 对合法生成样本设优势、丢弃非法样本，以缓解挤压效应、保证生成序列可映射到真实对象。
- **P-Score（Preference Score）** — OneRec 用神经网络学习的个性化多目标偏好分数，作为强化学习对齐的奖励信号。
- **ECPO（Early Clipped GRPO）** — 对负优势样本的策略比率预先裁剪的偏好优化算法，避免 GRPO 的梯度爆炸。
- **GBPO（Gradient-Bounded Policy Optimization）** — 用 BCE 损失的稳定梯度界定 RL 梯度的策略优化算法，支持完整样本利用与有界梯度稳定化。
- **PRE（Prefix2Query Representation Enhancement）** — OneSug 的前缀表示增强模块，用共现查询检索增强短前缀语义。
- **RWR（Reward-Weighted Ranking）** — OneSug 的奖励加权排序策略，用六级交互反馈构造偏好对、把业务价值注入排序。
- **KHQE（Keyword-enhanced Hierarchical Quantization Encoding）** — OneSearch 的关键词增强分层量化编码：前 3 层 RQ-Kmeans 保语义层次、后 2 层 OPQ 保商品独特性。
- **OPQ（Optimized Product Quantization）** — 优化乘积量化，把残差切分子向量独立量化，编码商品的独特属性。
- **Mu-Seq（Multi-view behavior Sequence injection）** — OneSearch 从用户 ID 构造、短期序列、长期序列三视角注入用户行为的策略。
- **PARS（Preference-Aware Reward System）** — OneSearch 的偏好感知奖励系统，含多阶段 SFT 与自适应奖励模型，并把相关性权重放大 10 倍。
- **激励相容（Incentive Compatibility, IC）** — 机制设计性质：广告主如实出价是最优策略。
- **个体理性（Individual Rationality, IR）** — 机制设计性质：广告主支付不超过其申报竞价（$p_i \leq b_i$）。
- **位置外部性（Position Externality）** — 广告的 CTR 受其所在序列其他广告与位置影响，而非相互独立。
- **EGA（End-to-end Generative Advertising）** — 把竞价机制与生成模型统一、通过 Token 级竞价与 POI 级支付实现 IC/IR 的端到端生成式广告系统。
- **POI（Point of Interest）** — 兴趣点，如餐厅、健身房等商户实体，是广告生成中的内容主体。
- **Token 级竞价（Token-level Bidding）** — 用最大值聚合把广告竞价投影到语义 Token，引导生成概率的分配机制。
- **POI 级支付网络（POI-level Payment Network）** — 独立神经网络，学习满足 IC 约束的支付函数，与分配解耦。
- **Ex-post Regret** — 广告主通过谎报竞价所能获得的最大额外效用；为 0 时机制满足 IC。
- **Lagrangian 优化** — 用对偶乘子把「最大化收益 + regret 约束」转化为可交替优化的损失。
- **GPR（Generative Pre-trained Recommender）** — 用「预训练 + 微调」范式、统一多场景超长序列的端到端生成式广告系统。
- **四类 Token（U/O/E/I-Token）** — GPR 的统一输入表示：User（用户）、Organic（有机内容）、Environment（环境）、Item（广告）。
- **RQ-Kmeans+** — 结合 RQ-Kmeans 高质量初始化与 RQ-VAE 端到端优化，缓解码本坍塌（codebook collapse）。
- **HHD（Heterogeneous Hierarchical Decoder）** — GPR 的三层异构层次化解码器：HSD 意图理解、PTD 推理生成、HTE 价值评估。
- **MoR（Mixture-of-Recursions）** — 同层递归调用自身多次以增加推理深度、不增参数的机制。
- **价值引导 Trie Beam Search（Value-Guided Trie-based Beam Search）** — 依约束构建 Trie 前缀树、用 HTE 价值动态调整束宽与剪枝的解码算法。
- **HEPO（Hierarchy Enhanced Policy Optimization）** — 同时在 Token 级与 Item 级做层次化策略梯度的强化学习算法。

## Part 9 · 推荐中的思考与推理

- **协同语义 (Collaborative Semantics)** — 推荐系统通过用户行为共现学习到的物品表示含义，编码在离散 ID 中，本身不携带文本语义（9.1）。
- **语言语义 (Language Semantics)** — 大语言模型（LLM）通过预训练在文本上习得的词汇/句法关联含义（9.1）。
- **语义鸿沟 (Semantic Gap)** — 协同语义（离散 ID）与语言语义（自然语言）之间无法直接对齐的隔阂（9.1）。
- **语义索引 / 语义 ID (Semantic Index)** — 用层次化量化把物品编码为离散 token 序列（如 `<A37><B12><C5><D8>`），同时可被 LLM 理解并承载协同语义（9.1）。
- **均匀语义映射 (Uniform Semantic Mapping)** — LC-Rec 在最后一层量化引入均匀分布约束、以最优传输（Sinkhorn-Knopp）缓解索引冲突的机制（9.1）。
- **多模态嵌入拼接 (Multimodal Embedding Concatenation)** — PLUM 将文本/视觉/音频/协同四类嵌入拼接，融合异构信号构建语义 ID（9.1）。
- **多分辨率码本 (Multi-Resolution Codebook)** — PLUM 在不同量化层使用不同大小码本（128/256/512/1024），契合从粗到细的信息论原理（9.1）。
- **显式推理 (Explicit Reasoning)** — 模型在输出推荐前先生成结构化、可审查的推理链，区别于隐式黑箱打分（9.2）。
- **推理脚手架 (Reasoning Scaffolding)** — OneRec-Think 用渐进 prompt 模板与任务引导模型「学会思考」的机制（9.2）。
- **多有效性 (Multi-Validity)** — 推荐场景中同一用户常存在多个合理推荐，不存在唯一正确答案的特性（9.2）。
- **推荐特定奖励 (Recommendation-Specific Reward)** — 综合协同相似度、内容相关性、推理连贯性、用户反馈的多维奖励函数（9.2）。
- **GRPO (Group Relative Policy Optimization)** — 对同一样本采样多条 rollout，按相对奖励（而非绝对标准）更新策略的强化学习方法（9.2）。
- **Think-Ahead 架构** — 把密集推理从在线关键路径剥离、改为用户行为更新时异步预计算的部署策略（9.2）。
- **自主推理 (Autonomous Reasoning)** — 模型在无人工模板、无教师示范下，仅凭任务反馈自主演化推理策略（9.3）。
- **模仿学习 (Imitation Learning)** — OneRec-Think 等依赖人工模板/教师知识的推理学习范式（9.3）。
- **探索学习 (Exploratory Learning)** — RecZero 依赖强化学习试错与反馈、自主摸索策略的学习范式（9.3）。
- **Think-before-Recommendation 模板** — RecZero 只定义「分析用户/分析物品/匹配/评分」四步框架、内容留给模型探索的提示（9.3）。
- **冷启动监督微调 (Cold-start SFT)** — RecOne 用少量高质量（含纠偏）推理样本初始化推理能力（9.3）。
- **混合范式 (Hybrid Paradigm)** — 监督提供「语言」、强化提供「智慧」，先框架后精进的推理学习思路（9.3）。

## Part 10 · 扩散模型推荐

- **扩散模型 (Diffusion Model)** — 通过前向逐步加噪、反向学习去噪来建模数据分布的生成模型（10.1）。
- **数据空间扩散 (Pixel-Space Diffusion)** — 直接在原始数据空间（像素/交互向量）加噪去噪，代表为 DDPM；计算昂贵（10.1）。
- **潜在空间扩散 (Latent Diffusion, LDM)** — 先编码到低维潜在空间再扩散、最后解码，代表为 Stable Diffusion；推荐中更常用（10.1）。
- **前向扩散 (Forward Diffusion)** — 按马尔可夫链逐步向数据加高斯噪声，使 x_T 趋近标准高斯（10.1）。
- **反向去噪 (Reverse Denoising)** — 训练去噪网络从 x_T 逐步恢复 x_0（10.1）。
- **重参数化技巧 (Reparameterization Trick)** — 用 x_t = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε 直接从 x₀ 采样任意 t 的加噪数据（10.1）。
- **ε-prediction** — 去噪网络预测所加噪声；DDPM 标准参数化（10.1）。
- **x₀-prediction** — 去噪网络直接预测原始数据；推荐场景更合适（10.1）。
- **分类器引导 (Classifier-Guided)** — 用预训练分类器梯度把生成推向目标类别（10.1）。
- **无分类器引导 (Classifier-Free Guidance)** — 训练时随机丢弃条件，推理时线性组合条件/无条件预测（10.1）。
- **v-prediction** — 预测「速度」v = αₜε − σₜx₀ 的参数化，训练更稳定（10.3）。
- **序列增强 (Sequential Augmentation)** — 为短序列用户生成「前序」交互以扩充历史；DiffuASR 为代表（10.2）。
- **Sequential U-Net (SU-Net)** — DiffuASR 把嵌入序列当多通道「图像」处理的 U-Net 变体（10.2）。
- **舍入 (Rounding)** — 把去噪后的连续嵌入映射回最近离散物品 ID（10.2）。
- **跨场景增强 (Multi-Scenario Augmentation)** — 从数据丰富场景借知识增强冷启动场景；Diff-MSR 为代表（10.2）。
- **分段噪声策略 (Segmented Noise)** — Diff-MSR 前期小 β 保留结构、后期线性增长以收敛到高斯（10.2）。
- **不对称扩散 (Asymmetric Diffusion)** — 前向在原始特征空间用离散 dropout、反向在潜在空间去噪；AsymDiffRec 为代表（10.3）。
- **特征 dropout** — AsymDiffRec 前向用随机丢弃特征模拟真实缺失，比高斯噪声更贴合推荐（10.3）。
- **步长嵌入 (Step Embedding)** — 标记哪些特征缺失的二值向量，引导潜在空间补全（10.3）。
- **Slate** — 一组需整体消费的物品集合（如歌单、套装），需协调性与多样性（10.3）。
- **DMSG** — 用条件扩散从文本 prompt 生成多样化 slate 的模型，采用 v-prediction（10.3）。
- **DDIM 加速** — 把推理步数从上千减到数十的确定性采样加速（10.3）。

## Part 11 · 生成式推荐系统实战

- **离线系统（Offline System）** — 负责「生产」的子系统：处理全量历史数据、训练模型、计算物品向量与相似度，追求质量而非延迟，产出模型文件与特征索引。
- **在线系统（Online System）** — 负责「服务」的子系统：接收实时请求、调用模型、组装推荐结果，目标百毫秒级延迟，依赖离线产出的模型与特征。
- **漏斗式架构（Funnel Architecture）** — 工业推荐的经典结构：召回用轻模型快速筛候选、排序用重模型精打分、重排优化体验，逐级缩小候选规模。
- **Snake Merge（蛇形合并）** — 多路召回融合策略：从各路召回中轮流取候选（A→B→C→C→B→A…），确保每路都有代表进入排序，提升多样性与覆盖。
- **冷启动（Cold Start）** — 新用户/新物品缺乏行为数据，传统协同过滤与向量召回失效的现象；本项目用独立冷启动流程（UCB/偏好/热门）应对。
- **UCB（Upper Confidence Bound，置信上界）** — 平衡探索与利用的算法：分数 = 历史平均奖励 + 探索奖励，未充分推荐的类型获得更高探索机会，避免信息茧房。
- **探索与利用（Exploration vs Exploitation）** — 推荐中的基本权衡：利用是推荐已知高质量内容，探索是尝试新类型以发现潜在兴趣；UCB 用单一公式统一两者。
- **困难负样本（Hard Negatives）** — 用户曝光过但未正向交互的物品，区分难度大，用于提升模型辨别力。
- **随机负样本（Random Negatives）** — 从用户未交互物品中随机采样，扩充负样本数量；本项目与困难负样本按 1:2 混合成 1:3 正负比。
- **滑动窗口样本（Sliding Window Samples）** — YoutubeDNN 训练样本的构建方式：给定用户前 $k$ 次观影，预测第 $k+1$ 次，模拟「预测下一个观看」。
- **左填充（Left Padding）** — 把变长行为序列在左侧补零到固定长度，使最近行为落在序列右侧，符合时间顺序，适配 RNN/Transformer。
- **物品向量预计算（Item Vector Pre-computation）** — 离线用物品塔批量算出全量物品向量并归一化，供在线毫秒级向量检索；双塔可规模化的关键。
- **版本指针（Version Pointer，active.json）** — 部署目录中的指针文件，记录当前应加载的模型版本路径；更新时先部署新版本再翻指针，实现无感知热更新与回滚。
- **连续打散（Consecutive Dispersion）** — 多样性重排策略：不允许超过 $N$ 个相同属性（类型/年代）连续出现，在保序前提下提升列表多样性。
- **保序性（Order Preservation）** — 重排算法在满足约束下尽量保持原排序，高分仍靠前、仅微调位置，兼顾相关性与多样性。
- **Pinia** — Vue 官方推荐的状态管理库，以 Store 集中管理跨组件共享状态（如用户认证），状态变化驱动依赖组件自动重渲染。
- **防抖（Debounce）** — 前端控制请求频率的技术：用户停止输入一段时间（本项目 300ms）后才真正发起请求，避免实时搜索刷爆 API。
- **单例模式（Singleton）** — 在线资源加载的设计：进程级唯一实例，配合惰性加载，模型与词表只加载一次、被所有请求共享，避免重复加载与内存膨胀。
- **优雅降级（Graceful Degradation）** — 模型不可用时退化为次优策略（如排序失败用召回分排序），保证服务高可用而非直接报错。
- **数据闭环（Data Loop）** — 用户行为经前端采集回写后端与存储，更新特征后影响下一次推荐，使系统持续改进；前端是闭环的采集端。
- **Docker Compose** — 声明式多容器编排工具，用单个 YAML 描述所有服务及其依赖、网络与卷，一条命令启动完整系统。
- **多阶段构建（Multi-stage Build）** — Dockerfile 技巧：先用构建镜像（如 Node）生成产物，再把产物拷入轻量运行镜像（如 Nginx），最终镜像不含开发依赖。
- **命名卷（Named Volume）** — Docker 持久化机制：把容器数据存于宿主机具名卷，容器删除后数据不丢；有状态服务（PG/Redis/ES）必须挂载。
- **健康检查（Healthcheck）** — 容器定期执行探测命令（如 `redis-cli ping`），连续失败才判不健康，供依赖服务等待其就绪，保障启动顺序。
- **服务名 DNS（Service-name DNS）** — Docker 网络内用服务名（如 `postgres`）解析到容器 IP，是容器间通信的正确方式（不用 `localhost`）。

## Part 12 · 计算广告

### 12.1

- **计算广告（Computational Advertising）** — 以最大化 ROI 为目标，对用户、上下文、广告三元进行匹配优化的技术与商业体系。
- **出资人（Sponsor）** — 广告定义三要素之一，为广告投放付费、有明确商业诉求的广告主。
- **媒介（Publisher）** — 广告定义三要素之一，承载广告、掌握用户注意力的媒体或产品。
- **受众（Audience）** — 广告定义三要素之一，广告信息触达的目标用户群体。
- **品牌广告（Brand Awareness）** — 注重长期影响与认知建立的广告类型，典型度量是曝光与认知度。
- **效果广告（Direct Response）** — 以短期转化行为（点击、注册、下单）为诉求的广告类型。
- **广告有效性模型** — 描述广告产生效果的六阶段漏斗：曝光→关注→理解→接受→保持→决策，归入选择、解释、态度三个阶段。
- **ROI（Return on Investment，投资回报率）** — 广告投放的回报与投入之比，计算广告优化的核心目标。
- **eCPM（effective Cost Per Mille）** — 千次展示期望收益，由点击率与点击价值相乘得到，广告排序与流量价值评估的统一口径。
- **CPM 市场** — 按展示付费的市场形态，点击率与点击价值的决策（与风险）全部交给广告主。
- **CPC 市场** — 按点击付费的市场形态，点击价值由广告主出价判断，点击率由平台动态预估。
- **CPA/CPS 市场** — 按行为/销售付费的市场形态，决策与风险全部归于平台，适合广告主转化流程高度一致的市场。
- **广告系统价值公式** — 广告系统价值 = 转化效率 × 计价机制 × 资源量 × 投放效率，理解广告技术演进的总纲。
- **广告网络（Ad Network）** — 2.0 投放模式下聚合多媒体流量、卖人群不卖广告位的封闭中介系统，以 CPC 为主要计价方式。
- **程序化交易（Programmatic Trade）** — 3.0 投放模式下经 DSP-ADX-SSP 完成的自动化、单次展示粒度的广告交易。
- **广告交易平台（Ad Exchange, ADX）** — 用实时竞价方式连接广告与（上下文、用户）、按展示粒度竞价结算的交易中枢。
- **需求方平台（Demand-Side Platform, DSP）** — 服务广告主的需求侧技术平台，提供定制化用户划分、跨媒体流量采购与 RTB 出价。
- **供应方平台（Supply-Side Platform, SSP）** — 服务媒体的供给侧技术平台，核心功能是收益管理，统一优化多种变现方式。
- **数据管理平台（Data Management Platform, DMP）** — 为网站提供数据加工与对外交易能力的平台，以定制化用户划分和统一数据接口为特征。
- **广告购买平台（Trading Desk）** — 面向需求方、允许广告商跨广告网络采买与 ROI 优化的工具，常由代理公司孵化。
- **实时竞价（Real-Time Bidding, RTB）** — 每次广告展示实时向多家 DSP 询价、价高者得的程序化交易机制。
- **Cookie Mapping（用户身份匹配）** — RTB 前置阶段，由 DSP 发起建立媒体 Cookie 与 DSP 用户 ID 对照表，Mapping 表存于需求方端。
- **Ad Call（广告请求）** — RTB 竞价阶段：ADX 广播询价请求，DSP 返回出价，最高者赢得展示。
- **担保式投送（Guaranteed Delivery）** — 基于合约、约定展示量未完成需补偿的优先销售交易形态，以 CPM 结算、量优先于质。
- **优选（Preferred Deal）** — 广告主按约定价格优先挑选流量、无需公开竞价的一对一议价交易方式。
- **网络优化（Network Optimization）** — 媒体将流量交给广告网络整体变现的交易方式，属组合优化。
- **定向（Targeting）** — 从广大受众中找到广告目标受众的技术，是受众与广告匹配度的专业叫法。
- **上下文定向** — 根据页面内容与场景信息匹配广告的定向技术，工程上以 Near-line 上下文系统实现。
- **行为定向** — 基于用户行为日志的定向技术，行为按信息强度排序且越靠近需求、越主动的行为越有效。
- **重定向（Retargeting）** — 广告主提供受众信息、系统从供应方流量中找回这些已触达用户的系统定向技术。
- **个性化重定向（Personalized Retargeting）** — 重定向的纵向延伸：对老用户推送商品粒度的个性化广告，相当于站外推荐引擎。
- **搜索重定向（Search Retargeting）** — 重定向的横向延伸：把搜索过特定关键词的用户定向到广告主网站。
- **新客推荐（Look-alike）** — 广告主提供种子用户，DSP 通过行为相似性在供应方受众中寻找潜在新用户的定向技术。
- **种子用户（Seed Audience）** — 新客推荐中广告主提供的高价值目标人群样本。
- **信息流广告（Feed Ads）** — 混排在用户阅读信息流中、形态与内容相似的广告形式，平衡广告效果与用户体验的正面典型。
- **植入式原生广告（Native Ads）** — 融入产品内容与服务形态、与内容深度结合的广告形式。
- **点击价值（Click Value）** — 一次点击带来的期望收益，与点击率共同构成 eCPM。
- **竞价行情预估（Bid Landscape Prediction）** — DSP 预测流量竞价分布以决定采买策略的核心问题，其拿到的流量是出价的函数。
- **收益管理（Yield Optimizer）** — SSP 的核心功能，统一优化优先销售、网络与 RTB 流量以最大化媒体收益。

### 12.2

- **广告主（Advertiser）** — 出资购买广告并按最终效果反推单次广告价值的一方，是需求侧的决策主体。
- **媒体/供应方（Supply Side）** — 拥有广告位与流量的内容或应用方，关心单位广告资源能产生多少收入。
- **效果广告（Direct Response）** — 以短期转化行为为诉求的广告形态，供应方根据广告效果计算广告量，按效果计费、竞价交易。
- **品牌广告（Brand Awareness）** — 注重长期品牌影响的广告形态，根据展示量计费，以合约方式交易，常见于核心 banner 等优质位置。
- **CPT（Cost Per Time，包时段付费）** — 按广告位占用时长（包月、包周）收费的模式，省心但计量粗糙，无法保障客户利益。
- **CPD（Cost Per Day，包天付费）** — 按天买断广告位的计费模式，多见于合约的品牌广告；对合作条件要求低，但长期不如 CPS 实时有效。
- **CPM（Cost Per Mille，千次展示付费）** — 按每千次展示收费的计费模式，公式为消耗/展现×1000，常见于 RTB，风险主要由广告主承担。
- **CPC（Cost Per Click，每点击付费）** — 按每次点击收费的计费模式，公式为消耗/点击，是广告主与平台风险的折中点，常见于关键词广告与 RTB。
- **CPA（Cost Per Action，按行为付费）** — 按注册、下单等用户行为收费的计费模式，点击率与价值皆动态，决策与风险归于平台。
- **CPS（Cost Per Sales，按销售付费）** — 按实际销售提成换算广告金额的计费模式，广告主规避费用风险，常见于网络联盟。
- **dCPM（dynamic CPM，动态千次展示付费）** — DSP 普遍采用的结算体系，每次 impression 的出价依据投放效果实时计算，对广告主按效果优化、与媒体仍按展示结算。
- **flat CPM（固定千次展示付费）** — 与 dCPM 相对，千次展示价格固定不变的传统 CPM。
- **消耗（Spend）** — 广告主投放广告的花费，是 CPM、CPC、ROI 等公式的分子口径。
- **CTR（Click-Through Rate，点击率）** — 点击量与展现量之比，衡量一个广告在多次展现中用户的平均点击次数。
- **CVR（Conversion Rate，转化率）** — 下单量与点击量之比，衡量用户点击与最终下单之间的关系。
- **ROI（Return On Investment，投资回报率）** — 下单金额与消耗金额之比，衡量广告花费与带来成交额的回报关系。
- **eCPM（effective/expected CPM，千次展示期望收益）** — 每千次展示的期望收益；CPC 计费下等于 pCTR×bid×1000，CPM 计费下等于出价，是跨计费模式的统一排序度量衡。
- **担保式投送（Guaranteed Delivery, GD）** — 基于合约的广告投送机制：约定展示量未完成需补偿、量优先于质、按 CPM 结算、由服务端决策。
- **在线分配（Online Allocation）** — 把广告与 (Context, User) 流量的匹配建模为 Ad→(Context,User) 二部图优化，在满足各合约量的约束下分配展示，经典解法是构造对偶问题。
- **流量预测（Traffic Forecasting）** — 对未来各定向流量规模的估计，可视为以广告为 Query、对 (u,c) 空间的反向检索问题，需对 u、c 分别处理。
- **独占要求（Exclusivity）** — 合约销售中品牌广告主对曝光的排他性要求（如竞品排斥），进一步收紧在线分配的可行空间。
- **广告层级结构（creative/solution/campaign/advertiser）** — 从创意、投放单元、广告计划到广告主的层级组织，用于新广告 CTR 的 back-off 先验估计。
- **back-off（回退估计）** — 统计缺失时逐级上退到更粗层级借用统计量的估计策略，用于新广告冷启动的 CTR 估计。
- **动态特征（Dynamic Features）** — 在标签组合维度上聚合的点击反馈统计特征，响应快、对新组合 back-off 强，但在线存储与更新代价高。
- **在线学习（Online Learning）** — 让模型随新数据流式更新以捕捉动态特性的方案，与动态特征构成「调模型 vs 调特征」的权衡。
- **E&E（Exploration & Exploitation，探索与利用）** — 为长尾 (a,u,c) 组合创造展示机会以累积统计量、从而更准确估计 CTR 的框架，需严格控制探索的量与有效性。
- **ε-greedy** — 以 ε 比例流量随机探索、其余利用当前最优的多臂老虎机策略。
- **UCB（Upper Confidence Bound，上置信界）** — 为每个候选计算期望收益的上置信界并选最大者的策略，选择次数越多上界越逼近真实期望。
- **Contextual Bandit（上下文老虎机）** — 用 arm 的特征矢量代替 arm 本身做决策从而降维的 E&E 方法，适合候选空间巨大的广告场景。
- **GSP（Generalized Second Pricing，广义第二高价）** — 胜者按下一名折算价格支付的竞价机制，简单易行、被在线广告系统广泛采用，但整体市场并非 truth-telling（详见 12.3）。
- **个体理性（Individual Rationality, IR）** — 广告主支付不超过其出价的基本参与约束，如 GSP 支付价 ≤ 胜者 bid。

### 12.3

- **竞价机制（Auction Mechanism）** — 规定广告坑位如何分配与计费的制度设计，由分配规则与定价规则两部分构成。
- **位置拍卖（Position Auction）** — 多个广告主竞争多个仅有点击率差异的坑位的拍卖模型，期望价值 $u_{is} = v_i \cdot x_s$。
- **估值（Valuation）** — 广告主对一次点击的真实价值判断，是其私人信息，平台不可见。
- **出价（Bid）** — 广告主向平台申报的每次点击愿意支付的价格（CPC 口径）。
- **位置点击率（Position CTR）** — 坑位 $s$ 的点击率 $x_s$，位置越靠前越大，是坑位间唯一差异。
- **分配规则（Allocation Rule）** — 机制中「谁赢哪个坑」的规则，竞价广告中通常为按出价（乘质量分）降序分配。
- **定价规则（Pricing Rule）** — 机制中「赢家付多少」的规则，决定广告主是否愿意如实出价。
- **广义第一高价（Generalized First Price, GFP）** — 按出价排序分配坑位、每人按自己出价支付的机制，无纯策略纳什均衡、市场震荡，已被淘汰。
- **纳什均衡（Nash Equilibrium）** — 任何一方单独改变策略都无法获益的策略组合。
- **纯策略纳什均衡（Pure-Strategy Nash Equilibrium）** — 每个参与者选定一个确定性策略构成的纳什均衡；GFP 不存在。
- **二价拍卖（Second-Price Auction / Vickrey Auction）** — 最高出价者赢、按第二高出价支付的单坑位拍卖，如实出价是占优策略。
- **占优策略（Dominant Strategy）** — 无论对手如何行动都最优的策略；二价拍卖中说真话即占优策略。
- **广义第二高价（Generalized Second Price, GSP）** — 第 $i$ 位广告主按下一名 eCPM 折算支付 $p_i = b_{i+1}x_{i+1}/x_i$、末位付保留价的机制，被在线广告系统广泛采用。
- **保留价（Reserve Price）** — 平台设置的最低成交价，末位广告主或无竞争者时按其支付。
- **激励兼容（Incentive Compatibility, IC）** — 如实报告估值是占优策略的性质：谎报不能提高效用。
- **个体理性（Individual Rationality, IR）** — 参与拍卖不会使参与者效用为负，即支付不超过申报价值 $p_i \le b_i$。
- **Truth-telling（如实出价）** — 广告主按真实估值出价 $b_i = v_i$ 的行为；VCG 整体市场满足，GSP 不满足。
- **对称纳什均衡（Symmetric Nash Equilibrium, SNE）** — GSP 存在的稳定均衡，满足无妒忌性质。
- **无妒忌（Envy-free）** — 均衡中没有人想与他人互换位置的分配性质：顶到他人位置需付他人价格，效用不增。
- **VCG 机制（Vickrey-Clarke-Groves Mechanism）** — 按参与者给其他人造成的外部性损害收费的机制，整体市场 truth-telling，单坑位退化为二价。
- **外部性（Externality）** — 一个参与者的在场给其他所有参与者造成的福利损失，即「你不在场，其他人本可多赚多少」。
- **赢家诅咒（Winner's Curse）** — 以虚高出价赢下超过自身估值的坑位而招致负效用的情形，一价拍卖下常见。
- **一价拍卖（First-Price Auction）** — 赢家按自己出价支付的拍卖；2019 年前后被头部 ADX 在程序化公开竞价中重新采用。
- **头部竞价（Header Bidding）** — 媒体在主拍卖前把流量同时发给多家需求方预竞价的技术，其普及是多级转售链路与一价回归的推手。
- **出价遮蔽（Bid Shading）** — 一价拍卖下 DSP 基于赢得概率分布把出价压向「能赢的最低价」的出价策略，是一价时代的核心竞争力。
- **智能出价（Smart Bidding）** — 平台代广告主出价（如 OCPC 按目标转化成本），把出价换算进排序模型的竞价产品形态。

### 12.4

- **智能出价（Smart Bidding）** — 平台代广告主管理每次展示出价的产品形态：广告主只报目标（target CPA/ROI），出价由平台的价值估计、预算控制与机制适配模块联合决定。
- **转化出价（oCPC / oCPM）** — 按转化目标出价的产品：出价公式为 $\text{eCPM} = 1000 \cdot \text{pCTR} \cdot \text{pCVR} \cdot \text{Bid}_{\text{CPA}}$，oCPC 按点击计费、oCPM 按展示计费。
- **目标转化成本（Target CPA）** — 广告主愿意为一次转化支付的目标成本，是出价栈中唯一由广告主直接输入的价值锚点。
- **价值出价（Value Bid）** — 用 pCTR × pCVR × targetCPA 把转化目标折算成的单次展示期望价值，是下游 shading 与 pacing 的输入。
- **两阶段投放（Two-Stage Delivery）** — oCPC/oCPM 的冷启动惯例：第一阶段沿用 CPC 出价积累转化数据，模型置信后切换到转化出价。
- **预算平滑消耗（Budget Pacing）** — 把日预算按时间进度均匀花完的控制问题，避免前置消耗错过晚高峰优质流量。
- **参照轨迹（Reference Trajectory）** — pacing 的控制目标 $r(t) = G \cdot t / T$，即「消耗进度与时间进度同步」的直线。
- **概率节流（Probabilistic Throttling）** — pacing 的一种实现：以概率 $\alpha$ 决定是否参与每次竞价，是 0/1 的硬门控（LinkedIn，KDD 2014）。
- **出价缩放（Bid Scaling）** — pacing 的另一种实现：用乘子 $\alpha \in [0,1]$ 缩放出价 $b' = \alpha \cdot b$，保留参与度、牺牲单次竞争力。
- **Pacing 乘子（Pacing Multiplier）** — 预算控制器的操作量，经 sigmoid 压缩到 $[0,1]$，直接乘在出价上。
- **PID 控制（PID Control）** — 比例–积分–微分反馈控制器：P 即时响应误差，I 消除稳态偏差，D 超前抑制；广告 pacing 中 D 项因放大离散请求噪声被普遍砍掉，只留 PI。
- **对数比误差（Log-Ratio Error）** — 误差形式 $e(t) = -\log(N(t)/r(t))$，把偏离归一化为相对值，使不同预算规模的计划可共用同一套控制增益。
- **前馈补偿（Feedforward Compensation）** — 在反馈控制之外，用可预测的扰动（如流量日内规律）提前调整操作量，Verizon DSP 的积分控制即配有前馈。
- **期望剩余（Expected Surplus）** — 一价拍卖下出价 $b$ 的期望利润 $\mathbb{E}[S] = (v-b) \cdot P(\text{win} \mid b)$，bid shading 的优化目标。
- **最低赢价（Minimum Winning Price）** — 刚好能赢下一次拍卖的价格，其分布（CDF）决定赢率 $P(\text{win} \mid b) = F(b)$。
- **赢价分布（Bid Landscape / Win-Price Distribution）** — 最低赢价随流量变化的概率分布，bid shading 的核心估计对象；log-normal 对其长尾拟合最好。
- **删失数据（Censored Data）** — 只观测到部分信息的样本：封闭拍卖中输掉的拍卖看不到真实赢价，需用生存分析处理。
- **生存分析（Survival Analysis）** — 处理删失观测的统计方法，DDN 用它从「是否赢 + 赢时最低价」的不完全数据中估计赢价分布。
- **黄金分割搜索（Golden Section Search）** — 无需梯度的区间极值搜索，每次迭代保留 0.618 区间；DDN 用它在毫秒级求出 surplus 峰值出价 $b^*$。
- **DDN（Deep Distribution Network）** — Verizon Media 的分布估计网络（Zhou et al., KDD 2021）：网络输出赢价分布参数，线上 surplus 提升 14.3%，日服务数千亿次请求。
- **分布鲁棒出价（Distributionally Robust Bidding）** — 当估值与赢价分布估计噪声大时，用 KL 散度不确定性集做 max-min 优化的出价鲁棒化方法。
- **误差传导链（Error Propagation Chain）** — 出价栈各模块串行耦合，上游预测（pCTR/pCVR）的偏差无损传导到最终出价的性质，是 12.5 校准问题的动机。

### 12.5

- **校准（Calibration）** — 预测值与真实概率的一致性：$\Pr(Y{=}1 \mid f(x)=p) = p$，即打 $p$ 分的样本约 $100p\%$ 为正。
- **判别力（Discrimination）** — 模型把正例排在负例前面的能力，由 AUC 等指标度量，对分数的单调变换不变。
- **绝对精度（size-accuracy）** — 预测值绝对大小的准确度，对精确出价、竞价稳定与混投公平性至关重要。
- **过度自信（Overconfidence）** — 深度模型预测值系统性高于真实概率的普遍倾向（Guo et al., 2017）。
- **位置偏差（Position Bias）** — 位置靠前带来的点击优势被误记为广告自身质量的偏差。
- **检验假设（Examination Hypothesis）** — 点击 = 被看到 × 值得点击的分解假设：$P(\text{click}) = P(\text{seen}\mid\text{position}) \cdot P(\text{relevant}\mid\text{user, ad})$。
- **逆倾向加权（Inverse Propensity Weighting, IPW）** — 按倾向分倒数对样本加权以还原无偏分布的去偏方法。
- **倾向分（Propensity Score）** — 样本被分配到某位置/被选中的概率，IPW 的权重来源，常需随机流量估计。
- **PAL（Position-bias-aware Learning）** — 华为提出的结构化去偏框架：bCTR = ProbSeen(position) × pCTR(user, ad, context)，训练联合、线上只用 pCTR 塔（Guo et al., RecSys 2019）。
- **级联模型（Cascade Model）** — 假设用户从前往后顺序浏览、点击即停止、会话至多一次点击的位置建模方式，查看概率与前位内容相关。
- **样本选择偏差（Sample Selection Bias, SSB）** — CVR 在点击子空间训练却在全曝光空间推理导致的分布不一致。
- **数据稀疏（Data Sparsity, DS）** — 转化样本远少于点击样本（点击仅占曝光约 4%）造成的训练信号不足。
- **ESMM（Entire Space Multi-task Model）** — 阿里提出的全空间多任务模型：以 pCTCVR = pCTR × pCVR 在全部曝光样本上联合训练，同时解决 SSB 与 DS（Ma et al., SIGIR 2018）。
- **pCTCVR** — 曝光到转化的概率，等于 pCTR × pCVR，定义在全曝光空间、可直接监督。
- **隐式学习（Implicit Learning）** — ESMM 中 CVR 塔无直接损失项、仅由 L_ctcvr 经乘积梯度更新的学习方式。
- **胜出偏差（Winner's Bias）** — 竞价日志只记录赢家表现、输家无标签造成的选择偏差，需探索流量供给无偏信号。
- **探索流量（Exploration Traffic）** — 刻意让本会输的广告偶尔赢以产生无偏反馈的流量分配策略。
- **延迟反馈（Delayed Feedback）** — 转化标签在点击后数小时/数天才回流的标签晚到现象。
- **标签窗口（Label Window）** — 校准取数的观察期口径（如 1 天点击、7 天转化），未成熟即取数必有偏。
- **可靠性图（Reliability Diagram）** — 预测概率分桶 vs 桶内真实正例率的诊断图，落在对角线即校准完美。
- **期望校准误差（Expected Calibration Error, ECE）** — 分桶后实际正例率与平均预测值之差的样本量加权平均。
- **Platt scaling** — 对模型分数拟合 logistic 变换（两个参数）的校准方法，适合小样本。
- **保序回归（Isotonic Regression）** — 拟合自由形单调阶梯函数的校准方法，适合大样本、稀疏区易过拟合。
- **PAVA（Pool Adjacent Violators Algorithm）** — 保序回归的经典求解算法：反复合并违反单调性的相邻块并取均值。
- **先验修正（Prior Correction）** — 负采样后用闭式公式 $\hat{p}'=\hat{p}/(\hat{p}+(1-\hat{p})/w)$ 恢复真实基率的校准方法（Facebook ADKDD'14）。
- **负采样（Negative Sampling）** — 训练时下采样负例以加速/平衡样本的技术，会抬高训练基率、需修正后使用。
- **分布漂移（Distribution Drift）** — 流量结构、广告库与用户行为变化导致历史校准失准的现象。
- **PCOC（Predicted-over-Posterior Click rate）** — 预估 CTR 与后验 CTR 之比，越接近 1 越好。
- **Cal-N** — 多簇 PCOC 聚合得到的整体校准偏差度量。
- **GC-N** — 分维度加权的校准评估指标。
- **SIR（Smoothed Isotonic Regression）** — 阿里妈妈校准体系的起点：分桶 + 保序回归 + 线性缩放。
- **Bayes-SIR** — 在 SIR 上引入贝叶斯先验以解决冷启动与稀疏分桶不稳定的校准算法。
- **RTW-BSIR** — 在 Bayes-SIR 上叠加实时波动修正、对抗分布漂移的校准算法。
- **PCCEM** — 利用点击后短期信号预测长期转化、应对延迟反馈的校准算法，阿里妈妈 2018 年起线上部署。
- **AdaCalib** — 字段级（field-level）细粒度校准框架：保序函数族 + 后验统计自适应引导（Wei et al., SIGIR 2022）。
- **observed-vs-predicted 护栏** — 线上持续监控「实际正例率 ÷ 预估正例率」、偏离 1 超阈值即告警重拟合的运维机制。


### 12.6

- **数据可观测性（Data Observability）** — 广告平台能否直接观测到用户转化行为的程度，是决定平台优化深度的第二根轴。
- **闭环广告（Closed-loop Advertising）** — 曝光、点击、下单、支付全链路都发生在平台域内、数据不离开平台生态的广告，也称内循环。
- **开环广告（Open-loop Advertising）** — 转化发生在平台域外（App Store、品牌官网、线下门店）、平台须依赖回传才能获知转化的广告，也称外循环。
- **内循环 / 外循环** — 闭环/开环广告在业界（抖音、快手、Facebook）的通行别称。
- **半闭环** — 广告主只回传部分事件（如只回传激活、不回传付费）的折中形态，平台拿到不完整标签做部分优化。
- **深度转化出价** — 以付费、ROI、次留、7 日 ROI 等后链路行为为优化目标的出价，只有平台能观测这些行为时才可行。
- **浅层目标 / 深层目标** — 前链路（曝光、点击、激活、表单、注册）与后链路（付费、ROI、次留、7 日 ROI）两组优化目标，分别对应开环与闭环的能力边界。
- **pDeepCVR** — 「点击 → 付费/次留」的深度转化概率，位于转化漏斗更深处，样本更稀疏、延迟更高。
- **归因（Attribution）** — 在广告行为链路中识别关键行为由哪个广告/渠道带来的过程。
- **归因模型（Attribution Model）** — 决定转化功劳如何分配给各触点的分配规则，是约定而非客观事实。
- **最后点击（Last-click）** — 把 100% 功劳给转化前最后一次触点的归因模型，移动端默认。
- **首次点击（First-click）** — 把 100% 功劳给第一次触点的归因模型，用于度量漏斗顶部发现。
- **线性归因（Linear Attribution）** — 所有触点均分功劳的归因模型。
- **时间衰减（Time-decay）** — 越接近转化的触点分得越多功劳的归因模型，适合短周期意图驱动。
- **位置加权（Position-based）** — 首尾触点多分、中间少分（U 型）的归因模型，兼顾发现与收口。
- **数据驱动归因（Data-driven Attribution）** — 算法基于观测到的贡献自动分配功劳的归因模型，需大量转化数据。
- **clickid** — 媒体在广告触点（曝光/点击）时下发的唯一标识，用于后续转化回传时把转化绑定到具体广告。
- **转化回传** — 广告主通过 SDK/API 把设备 ID、clickid、时间戳回传给媒体、告知「该用户已转化」的过程。
- **兜底归因** — 拿不到设备 ID 时退而用 ip + ua 做模糊匹配的归因方式，精度较低。
- **自归因（Self-attribution）** — 平台/媒体自行完成归因、宣称转化由自己带来的方式，多网络并跑时易重复计数。
- **非自归因** — 广告主自行匹配用户与媒体信息、独立完成归因的方式。
- **移动归因伙伴（Mobile Measurement Partner, MMP）** — 中立第三方归因/分析平台（AppsFlyer、Adjust、Branch、Singular、Kochava），作为广告主与各广告网络之间的仲裁者。
- **ATT（App Tracking Transparency）** — 苹果 iOS 14.5 起的授权框架，App 访问 IDFA 须弹窗授权，opt-in 比例仅约 25%。
- **IDFA（Identifier for Advertisers）** — 苹果设备级的广告标识符，ATT 后大面积坍缩，是确定性归因赖以为生的唯一用户 ID。
- **SKAdNetwork（SKAN）** — 苹果的隐私保护安装归因框架，以聚合、随机延迟、群组匿名的方式回传转化数据。
- **群组匿名（Crowd Anonymity）** — SKAN 的隐私机制，安装量低时回传更少信息，防止单个用户被反向定位。
- **conversion value** — SKAN 中 App 通过 updateConversionValue 上报的用户互动转化值，SKAN 4.0 引入粗/细两种。
- **三次回传窗口** — SKAN 4.0 的回传节奏：约 0–2 天、3–7 天、8–35 天，转化数据分批、带随机延迟回流。
- **层级 source identifier** — SKAN 4.0 的 4 位分层来源标识（前 2 位活动、第 3 位位置、第 4 位屏位），随群组匿名级别升高返回更多位。
- **Android Privacy Sandbox** — Google 的无 cookie 归因方案，含 Attribution Reporting API 与 Topics API。
- **Attribution Reporting API** — Privacy Sandbox 中提供事件级与聚合归因报告、聚合报告带差分隐私噪声的组件。
- **差分隐私（Differential Privacy）** — 向聚合统计注入校准噪声以保护个体隐私的技术，Privacy Sandbox 归因报告使用。
- **确定性归因 / 概率性归因** — 依赖设备级唯一标识的精确归因 vs 依赖聚合/模糊信号的统计归因，隐私浪潮推动前者向后者坍缩。
- **第一方数据（First-party Data）** — 企业与用户直接发生、合法合规收集的数据，是跨 App 追踪受限后的战略方向。
- **建模估计（Modeling-based Estimation）** — 用可观测部分（SKAN + 授权确定性）训练模型、外推填补「无法归因」缺口的估计方法。


### 12.7

- **在线分配（Online Allocation）** — 在满足量的约束的前提下，对每次广告展示实时决策，以优化产品整体收益的算法框架；离线规划 + 在线执行是其标准形态。
- **担保式投送（Guaranteed Delivery, GD）** — 展示量合约对应的投放系统：合约约定定向人群与展示量，系统须保证到期足量交付，核心计算问题是带约束的在线分配。
- **排期系统（Scheduling System）** — 管理 CPT 广告位合约的非个性化系统：素材按排期经 CDN 前端直投，无需服务端实时决策。
- **防天窗广告（House Ad / 兜底广告）** — 动态广告服务超时或出错时，由 CDN 渲染的默认素材，保证广告位不空白。
- **二部图（Bipartite Graph）** — 在线分配的问题建模：供给节点（标签相同的流量库）与需求节点（合约）之间的匹配结构 $G=(I \cup A, E)$。
- **供给节点（Supply Node）** — 二部图一侧的节点，代表所有标签都相同的一块流量库存，总量记为 $s_i$；节点数随定向条件组合呈几何级数上升。
- **需求节点（Demand Node）** — 二部图另一侧的节点，代表一份广告合约，约定量记为 $d_a$。
- **需求约束（Demand Constraint）** — 分配合约的收益（或量）不低于约定值的约束 $\sum_{i \in \Gamma(a)} s_i x_{ia} q_{ia} \ge d_a$。
- **供给约束（Supply Constraint）** — 每个供给节点分配出去的比例之和不超过 1 的约束 $\sum_{a \in \Gamma(i)} x_{ia} \le 1$，违反即超卖。
- **分配比例（Allocation Ratio）** — 决策变量 $x_{ia}$：把供给节点 $i$ 的多大比例流量分配合约 $a$。
- **AdWords 问题（带预算约束的出价）** — CPC 竞价下给定各广告主预算、最大化市场收入的在线分配实例；其对偶变量是「流量对预算的边际价值」，是 pacing 乘子的理论原型。
- **流量预测（Traffic Forecasting）** — 给定受众标签组合与 eCPM 阈值，估算未来时段可获展示量的技术；工程上用「反向索引」方案（文档=标签聚合的流量，查询=定向条件）。
- **频次控制（Frequency Capping）** — 控制组合 $(a,u)$ 在周期内的展示次数；客户端 cookie/SDK 与服务端内存缓存两种实现，是破坏展示可分性假设的主要因素。
- **对偶变量（Dual Variable）** — LP 对偶问题中对应约束的变量：$\alpha_a$（合约稀缺度）与 $\beta_i$（供给机会成本），其量级分别正比于合约数与供给节点数。
- **紧凑分配方案（Compact Allocation Plan）** — 只保留合约级对偶变量 $\alpha$（$O(|A|)$ 级），经 KKT 条件恢复 $\beta$ 与 $x_{ia} = \max(0, \theta_a(1+\alpha_a-\beta_i))$ 的分配方案；无状态、多机零同步。
- **需求-供给比（θ, Demand-Supply Ratio）** — $\theta_a = d_a / \sum_{i \in \Gamma(a)} s_i$，衡量合约相对其全部候选流量的紧缺程度，同时出现在紧凑方案与 HWM 中。
- **SHALE** — 在线分配的原始对偶迭代算法：交替更新 $\alpha$ 与 $\beta$ 求解对偶问题，支持增量插入新合约。
- **高水位算法（High Water Mark, HWM）** — 工程启发性分配方案：按 $\theta$ 降序确定合约优先级，逐层折减候选供给余量得到分配比例；线上按累积比例随机决策。
- **竞争比（Competitive Ratio）** — 在线策略在最坏情形下能达到离线全局最优目标函数的 $\epsilon$ 倍，则称其为 $\epsilon$-competitive；在线分配的最优上限为 $1-1/e$。
- **Free Disposal** — 超投无收益也无损失的假设，符合多数广告合约现实，是在线分配算法宽容性的来源。

---


### 12.8

- **受众定向（Audience Targeting）** — 对广告 $a$、用户 $u$、上下文 $c$ 三个维度提取有意义的特征（标签）的过程，是展示广告最核心的驱动力之一。
- **上下文定向（Contextual Targeting）** — $t(c)$ 类定向：根据用户当前访问的页面或请求参数（地域、频道、URL、关键词、主题）即时打标签。
- **行为定向（Behavioral Targeting, BT）** — $t(u)$ 类定向：根据用户一段时期内的网络行为历史，将用户映射到某个定向标签上。
- **定制化标签（$t(a,u)$）** — 针对特定广告主加工的用户标签（如重定向、新客推荐），标签数与广告主数成正比，适合由需求方在程序化交易中直接提供。
- **标签体系（Taxonomy）** — 向广告主售卖的、预先定义且可解释的标签集合；效果与规模双指标要求它同时覆盖「泛而大」与「准而小」的两端。
- **半在线抓取系统（Semi-online Crawler）** — 上下文定向的页面抓取方案：不做离线抓取，广告请求触发后才抓取打标，用缓存 + TTL 管理，允许暂时返回空标签。
- **弱一致（Weak Consistency）** — 广告系统的业务特性：只要大多数决策最优，少量次优甚至随机决策可接受，是低成本系统设计的依据。
- **需求方驱动关键词（Demand-driven Keywords）** — 从广告主描述中得到商业价值高的词表与 IDF，再与页面 TF 一起计算 TF-IDF 选取关键词的方法。
- **潜在语义分析（Latent Semantic Analysis, LSA）** — 对文档-词矩阵做 SVD、保留主要奇异值的主题模型；两个变换矩阵不保证非负，直觉欠合理。
- **概率潜在语义索引（PLSI）** — LSA 的概率化：以「文档选主题、主题生成词」的生成过程建模，可用 EM 分布式求解。
- **潜在狄利克雷分配（Latent Dirichlet Allocation, LDA）** — 给 PLSI 的主题分布加 Dirichlet 先验的贝叶斯版本，噪声大或短文档下更稳健，常用吉布斯采样求解。
- **词嵌入（word2vec / Word Embedding）** — 把词映射为稠密实数向量的表示学习方法；CBOW + 哈夫曼树（层次 softmax）把输出复杂度从 $O(\|V\|)$ 降到 $O(\log \|V\|)$，是 embedding 思想的源头。
- **embedding 打标** — 2026 年主流的标签加工路线：用表示模型（双塔/图 embedding）或 LLM 把内容映射为向量或结构化标签，已取代主题模型打标。
- **时间衰减法（Time Decay）** — 行为累积方法 $\tilde{x}(d) = \alpha\tilde{x}(d-1) + x(d)$，指数窗过滤原始行为，只需保存上一时间片状态，工程上优于滑动窗法。
- **滑动窗法（Sliding Window）** — 设定窗长 $D$、累加窗口内行为强度的累积方法，窗型为矩形，需保存窗内全部行为。
- **特征选择函数（$x_{tn}(b)$）** — 把原始行为 $b$ 映射到标签 $t$ 上并给出强度的函数，是行为定向特征生成最关键的环节。
- **用户标签得分（$\lambda_t$）** — 行为定向 GLM 的线性打分 $\log\lambda_t = w_t \cdot \tilde{x}$，控制点击到达频繁性；线上按 $\lambda(d) = \alpha\lambda(d-1) + \sum_n w_{tn} x_{tn}(d)$ 递归更新。
- **人口属性预测（Demographic Prediction）** — 以行为为输入预测性别、年龄等用户确定特点的分类任务；必须有拒识门槛，训练集质量比模型更重要。
- **reach/CTR 曲线** — 行为定向的半定量评测工具：标签人群规模（reach）与该人群 CTR 构成的曲线，应单调下降，头部斜率体现鉴别力，最右端 CTR 固定为全量水平。
- **AUC（见 12.5）** — 衡量模型判别力（相对序）的指标；reach/CTR 曲线头部的陡峭程度是它在定向评测中的投影，进入算术的得分还需过校准。

### 12.9

- **广告检索（Ad Retrieval）** — 从亿级广告候选中，在毫秒级约束下找出可参与本次竞价的少数广告的计算环节；是 eCPM 排序（12.2）之前的入场资格赛。
- **析取范式（Disjunctive Normal Form, DNF）** — 广告定向条件的标准表示：若干个交集（Conjunction）的并，命中任一交集即命中该广告。
- **Conjunction（交集）** — DNF 中由「与」连接的一组赋值集；检索算法以它（而非整条广告）为单位建倒排索引。
- **Assignment（赋值集）** — 对单个标签的最小约束（属于或不属于某取值集合），如 age ∈ {3}；size 只统计含「∈」的赋值集数目。
- **布尔表达式检索（Boolean Retrieval）** — 在倒排索引上求值定向条件的检索方式：两层索引（Conjunction 倒排 + Conj→AD 辅助索引）加 size 分层剪枝，先取候选超集再做精确判定；纯「∉」型 Conjunction 挂在特殊键 Z 上兜底。
- **倒排索引（Inverted Index）** — 从「键（标签/关键词）」指向「包含该键的文档列表」的数据结构；广告检索把它扩展为按 size 分层的 Conjunction 索引。
- **size 分层剪枝** — 按 Conjunction 中「∈」赋值集数目分层建索引；请求标签数小于某层 size 时该层整体跳过，是布尔检索最有力的剪枝。
- **精确匹配（Exact Match）** — 搜索广告关键词匹配的最严格档位：查询与竞价关键词完全一致才触发；与之相对的是短语匹配与广泛匹配（触发查询扩展）。
- **查询扩展（Query Expansion）** — 搜索广告中把简短查询拓展为一组可竞价关键词的技术；三条思路为协同过滤、主题模型与历史 eCPM 效果挖掘，泛化过度会损害相关性。
- **广告放置（Ad Placement）** — 搜索广告中确定北区/东区广告条数的决策：平均条数约束下的营收优化，可用用户点击率比值做个性化调整。
- **相关性检索（Relevance Retrieval）** — 超长查询场景下以「查询-文档相似度」而非布尔匹配为目标的检索；要求评价函数线性且权重非负以支持快速上界剪枝。
- **WAND 算法（weight AND）** — Top-K 剪枝检索算法：预计算关键词贡献上界，累加超过堆阈值才精确评分，配合小顶堆维护当前最优 K 个结果；其「上界粗估 + 门槛正反馈」思想延续到粗排层。
- **语义召回（Semantic Recall）** — 用 DNN 把查询/用户与广告映射到同一语义向量空间，以最近邻查找代替关键词匹配的召回方式；解决用词不同导致的匹配盲区。
- **DSSM（Deep Semantic Similarity Model）** — 以点击为弱监督、端到端学习查询与文档语义向量的深度模型：词嵌入 → 多层网络投影语义空间 → 余弦相似度 + softmax/按对排序损失。
- **双塔模型（Two-Tower Model）** — 用户塔与物品塔各自独立编码为向量、在线只做向量内积的召回架构；DSSM/YouTube 模型的现代形态，标配 in-batch negatives 训练。
- **近似最近邻（Approximate Nearest Neighbor, ANN）** — 接受一定召回率损失、换取毫秒级向量检索速度的最近邻查找技术总称，分哈希（LSH）、向量量化（PQ/HKM）、图（NSW/HNSW）三类。
- **局部敏感哈希（Locality-Sensitive Hashing, LSH）** — 「原始空间越近、哈希越易同桶」的分治式 ANN：随机投影下同桶概率为 1−θ/π；扩召回用 LSH forest（换内存）或 multi-probe（换查询时间），工程上已被图索引取代但直觉仍是 ANN 的思想原点。
- **HNSW（分层可导航小世界）** — NSW 的分层版本：上层稀疏图快速导航、底层稠密图保精度；当前工业界最常用的 ANN 图索引，faiss/hnswlib 均有实现。
- **IVF-PQ** — 先用 K 均值粗聚类分桶（IVF）、桶内再做乘积量化压缩（PQ）的 ANN 索引；内存效率高，适合亿级以上候选池。
- **检索漏斗（Retrieval Funnel）** — 候选量级逐级收窄的系统全貌：召回（10⁴~10⁵）→ 粗排（10²~10³）→ 精排（10¹~10²）→ 竞价（1~3 条展示），每一层在「缩小候选」与「提高打分精度」间交换；现代召回形态是**多路召回**——布尔定向、语义向量、协同/行为、热门兜底并行各取 Top-K，合并去重后统一进入排序。

### 12.10

- **第一方数据（First-party Data）** — 广告主自有渠道产生的数据（CRM、订单、官网访客行为），量小而语义最明确，是所有数据的「灵魂」。
- **第二方数据（Second-party Data）** — 用户在媒体/广告平台上产生、由平台自己掌握的行为与投放数据，广告网络模式下指导投放的主力。
- **第三方数据（Third-party Data）** — 不直接参与广告交易的数据提供方（中小媒体、数据公司等）拥有并提供流通的数据，量大数据质量参差。
- **用户标识（User Identifier）** — 关联「哪些行为来自同一个用户」的基础，如 cookie、IDFA、Android ID/IMEI；身份是一串 0 前面的那个 1。
- **决策行为（Decision Behavior）** — 转化与预转化（搜索、浏览、比价、加购等下单前动作），发生在广告主站内，兴趣指向最明确、价值最高。
- **半主动行为（Semi-active Behavior）** — 分享、页面浏览等弱目的内容消费行为，把握兴趣领域但精度有限，数据量在各类行为中最大。
- **Cookie 映射（Cookie Mapping）** — 在一方同意的前提下，把不同域名体系下同一用户的 cookie 身份对应起来的技术；三方 cookie 退场后已成历史基建。
- **数据管理平台（DMP）** — 把原始数据整理加工成可直接利用的用户标签并支持变现的产品；分第一方（托管加工收服务费）与第三方（加工售卖变现）两种模式。
- **客户数据平台（CDP）** — 统一品牌自有触点（官网、App、CRM）数据为持久客户档案的一方数据基建，现代形态下取代了旧第一方 DMP 的大部分场景。
- **人群包（Audience Segment）** — 按标签圈出的一组用户集合，是数据交易与受众定向的标准「商品单元」。
- **数据交易平台（第三方 DMP）** — 聚合多方原始行为数据、按自有逻辑加工成标签后售卖变现并与数据提供方分成的产品，代表案例 BlueKai。
- **Unified ID 2.0（UID2）** — The Trade Desk 主导的开放身份框架，以哈希后的邮箱/手机号为根，替代三方 cookie 的跨域身份方案。
- **数据交易（Data Trading）** — 标签经 ADX 中转、附加在竞价请求上按 CPM 计价、按 DSP 实际成交展示量交割的市场机制。
- **数据清洁室（Data Clean Room）** — 互不可见明细的前提下做多方数据匹配与分析、只输出聚合结果（常叠加差分隐私）的合规协作环境，2020 年代数据协作的主流形态。
- **准标识符（Quasi-identifier）** — 单独看无辨识力、组合起来却能定位到具体人的属性集合（如年龄+城市+职位），即使不含 PII 也有高泄露风险。
- **K 匿名（K-Anonymity）** — 通过准标识符泛化，使数据集中每组准标识符实例都能找到 K 条与其相同的；对极稀疏的行为数据不适用。
- **差分隐私（Differential Privacy）** — 对数据集做适度修改，在尽量少损失查询准确率的前提下使隐私泄露风险最低的技术。
- **需求方数据安全（Demand-side Data Security）** — 广告主第一方数据（如访客集合）在 RTB 中被平台或对手获取利用的风险，典型手法是合并访客集合打模糊标签倒卖。
- **GDPR** — 欧盟《通用数据保护条例》（2018 生效）：敏感数据清单、明确同意、访问/被遗忘/限制处理/携带四项权利，罚则上限为 2000 万欧元与全球年营业额 4% 中的较高者。
- **PIPL** — 中国《个人信息保护法》（2021 年 11 月施行）：确立知情同意、最小必要、可撤回同意等原则，并区分敏感个人信息。

### 12.11

- **程序化创意（Programmatic Creative）** — 在投放时由程序把推送广告的关键原因（地域、搜索词、单品等）在线拼装进创意的优化方式，前提是广告基本诉求保持稳定。
- **点击热力图（Click Heatmap）** — 把创意各位置被点击的密度可视化的工具，既用于半定量地指导创意迭代，也可通过分布形态（过于均匀/集中）甄别机器刷量。
- **A/B 测试（A/B Testing）** — 从真实流量中切出对照与实验两组分别运行原方案与新方案、以线上指标裁决优劣的实验方法。
- **实验框架（Experimentation Framework）** — 支撑 A/B 测试的线上系统，负责流量切分、参数下发与指标收集，是广告系统进化速度的基础设施。
- **实验层（Layer）与域（Domain）** — 按系统模块（检索/排序/展现）划分的实验参数容器为层，层内按 user ID 哈希切出的流量子集为域；一个用户的所有请求固定落在同一域中（层内互斥）。
- **分层实验（Layered Experimentation）** — 利用模块间相对独立性扩展实验容量的框架：层内互斥、层间正交、预留非重叠域做跨层联合调参、配套发布层灰度放量。
- **正交（Orthogonality）** — 不同实验层的流量切分彼此独立，同一份流量可被多层重复使用，实验容量随层数线性增长。
- **AA 实验（AA Test）** — 两组按完全相同配置运行的对照实验，用于验证切分均匀与日志口径一致；AA 跑出显著差异说明实验框架本身有偏。
- **广告监测（Ad Monitoring）** — 需求方委托独立第三方对展示、点击或转化做核实性度量的服务（约占品牌投放预算 1%）；核心载体是拼装广告/媒体/用户三方信息的**监测 URL**。
- **品牌安全（Brand Safety）** — 保证广告不出现在损害品牌形象的内容上的诉求，由**广告投放验证**（发现不安全内容即停投换创意，工程核心是 iframe 穿透取顶层 URL）实现。
- **可见性（Viewability）** — 广告展示实际被用户看到（发生渲染）的程度验证；现代由 MRC 等标准的面积/时长双阈值定义，是品牌结算口径之一。
- **虚假流量作弊（Non-Human Traffic, NHT）** — 展示、点击或转化本身被伪造的作弊类型，是 CPM/CPC 广告作弊的主流，按手段又分机器作弊与人工作弊。
- **归因作弊（Attribution Fraud）** — 把其他渠道的流量或自然流量记在自己名下的作弊类型，常见于伪造转化成本高的 CPA/CPS 广告。
- **点击滥用（Click Spam / Click Flooding）** — 对大量用户伪造点击、坐等其自然下载被归因到自己渠道的手法；铁证是 CVR 低 1–2 个数量级、点击-转化时间分布近似均匀。
- **点击注入（Click Injection）** — 利用 Android 安装广播在应用安装瞬间补发点击、抢归因后续激活的手法；特征是 CVR 异常高、点击到激活间隔极短。
- **Cookie 填充（Cookie Stuffing）** — CPS 联盟广告中通过隐藏请求在用户不点击的情况下静默打上来源 cookie、劫持自然转化的归因作弊。
- **流量劫持（Traffic Hijacking）** — 网络底层服务提供者在无权投放处强行投放或篡改创意/落地页的准作弊（信道弹窗、创意替换、搜索重定向、落地页来源劫持）；前三种损媒体，来源劫持损广告主。
- **设备农场（Device Farm）** — 真人持真机批量伪造浏览-点击-转化的人工作弊形态，各维度数据接近真实，需靠设备聚集度与关联网络识别。
- **设备指纹（Device Fingerprint）** — 由硬件与环境特征拼出的设备唯一标识，用于跨 IP/cookie 追踪作弊源与建立设备信誉分。
- **图分析反作弊（Graph-based Fraud Detection）** — 把设备、IP、账号、支付路径连成关联网络，利用作弊团伙的高聚集结构识别单条记录无法伪装的群体特征。
_术语表覆盖上篇（Ch0–Ch4，Part 1–5）、下篇（Ch5–Ch10，Part 6–11）与专题（Part 12 计算广告）全书内容。_
