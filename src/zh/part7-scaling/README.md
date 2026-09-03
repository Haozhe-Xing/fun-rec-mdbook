<div class="part-banner">
<h1 style="color: white; margin: 0 0 8px 0;" font-size: 1.7rem;>📗 Part 7: Scaling — 生成式排序模型</h1>
<p style="color: rgba(255,255,255,0.85); margin: 0;">从 HSTU 首次验证推荐界的 Scaling Law 出发，看工业界如何把生成式范式打磨成可落地、可扩展、硬件高效的排序引擎。</p>
<p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin: 8px 0 0 0;">📚 5 节 · ⏱️ Estimated 3 weeks · 🎯 Target: 理解「更大模型=更好推荐」在排序阶段的工程实现</p>
</div>

传统深度学习推荐模型（DLRM）长期是深度学习 **Scaling Law（缩放定律）** 的「例外」：砸下更多参数、更多数据，指标却很快触顶。本部分沿着 Meta 的 HSTU 首次验证推荐界 Scaling Law 的脉络，逐一拆解小红书、美团、阿里、字节的后续工作，看清工业界如何把「生成式排序」从论文数字变成服务数十亿用户的现实。

---

## 本章涵盖

| 章节 | Topic | The Big Idea |
|------|-------|--------------|
| **7.1** | HSTU：Scaling Law 的首次探索 | 把用户行为历史当作「语言」，用统一序列 + 自回归训练 + 高效架构，首次证明推荐也能 Scale |
| **7.2** | 生成式排序总体范式（GenRank） | 自回归机制才是本质，Action-Oriented 序列组织让序列长度减半、训练提速 ~79% |
| **7.3** | MTGR：混合范式建模 | 用「生成式架构 + 判别式目标」保留交叉特征，破解纯生成式的特征缺失难题 |
| **7.4** | RankMixer：硬件效率优化 | 从 GPU 硬件特性反推架构，用 Token Mixing / Per-Token FFN / Sparse MoE 把 MFU 从 4% 拉到 45% |
| **7.5** | OneTrans：统一 Transformer | 单一 Transformer backbone 同时做序列建模与特征交互，并复用 KV Caching 等 LLM 系统优化 |

---

## What You'll Be Able to Do After This Part

- 🟢 **解释** 为什么传统 DLRM 难以 Scale，以及 HSTU 如何用 user-level 序列建模打破瓶颈
- 🟢 **区分** 生成式范式里「自回归机制」与「训练范式细节」各自的贡献（见 7.2）
- 🟡 **说明** MTGR 的混合范式如何在保留效率的同时兼容传统交叉特征（见 7.3）
- 🟡 **分析** RankMixer 的 hardware-aware 设计如何把 MFU 从 4% 提升到 45%（见 7.4）
- 🔴 **复述** OneTrans 如何用统一 Transformer + Pyramid Stack + Cross-Request KV Caching 实现整体可扩展（见 7.5）
- 🔴 **对比** 五个工作在「统一性 vs 效率 vs 兼容性」三角上的不同取舍

---

## 核心概念

| Concept | 章节 | Relevance |
|----------|---------|-----------|
| 行为序列建模（user-level） | 7.1 | 把推荐当作「语言」是 Scaling Law 的前提 |
| Pointwise Aggregation / 相对时间偏置 | 7.1 | HSTU 针对推荐场景的三大架构创新 |
| 自回归本质是生成式的核心 | 7.2 | 区分「手段」与「目的」的分水岭 |
| Action-Oriented 组织 | 7.2 | 序列长度减半的关键技巧 |
| 混合范式（生成架构 + 判别目标） | 7.3 | 兼容交叉特征的新思路 |
| Group LayerNorm / Dynamic Masking | 7.3 | 让异构 token 在同一 Transformer 中共存 |
| Token Mixing / Per-Token FFN / Sparse MoE | 7.4 | hardware-aware 重构推荐计算图 |
| 统一 Tokenization / Mixed Parameterization / Pyramid Stack | 7.5 | 序列与特征在单一 backbone 内深度融合 |

---

## 前置知识

- 已读完 [Part 1 引言](./../part1-introduction/) 与 [第 3 部分 排序](./../part3-ranking/feature-crossing.md) 的判别范式基础
- 了解 Transformer 的自注意力、LayerNorm、残差连接基本概念
- 知道 Scaling Law 在 NLP/CV 中的含义（性能随算力/数据/参数量呈幂律提升）

> 本部分是「生成式推荐主线」下篇的第二站。若尚未读 [Part 6 生成式范式基础](./../part6-gr-basic/)，建议先建立生成式检索与语义 ID（RQ-VAE）的背景。

---

## Tips for This Part

1. **区分「手段」与「目的」。** 生成式架构（Transformer + 序列）是强大的表征手段，但不必服务于生成式目标——这正是 7.3 MTGR 的洞见。
2. **每个工作都在回答同一个问题** ：推荐模型如何真正享受 Scaling Law 红利？从架构、训练、特征、硬件四个角度反复对照。
3. **多配图、少死记公式。** 本章偏前沿，重点理解「为什么这样设计」，而非精确推导每一条公式。

---

Let's dive in! 🚀
