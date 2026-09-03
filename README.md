<div align="center">
  <h1>RecSys Auto Research KB · 推荐系统自动研究知识库</h1>
  <p><strong>A Knowledge Base for Recommender-Systems &amp; Computational-Advertising Auto Research</strong></p>

  <p>
    <img alt="mdBook" src="https://img.shields.io/badge/Built%20with-mdBook-2f74c0?style=flat-square">
    <img alt="Language" src="https://img.shields.io/badge/Language-English%20%7C%20中文-16a34a?style=flat-square">
    <img alt="Topic" src="https://img.shields.io/badge/Topic-Recommender%20Systems%20%7C%20Computational%20Advertising-f97316?style=flat-square">
    <img alt="Deploy" src="https://img.shields.io/github/actions/workflow/status/Haozhe-Xing/fun-rec-mdbook/deploy.yml?style=flat-square&label=deploy">
    <a href="https://haozhe-xing.github.io/fun-rec-mdbook/"><img alt="Read Online" src="https://img.shields.io/badge/Read-Online-4A6CF7?style=flat-square"></a>
  </p>

  <p>
    <a href="https://haozhe-xing.github.io/fun-rec-mdbook/en/">🇺🇸 English</a>
    ·
    <a href="#-中文版说明">🇨🇳 中文说明</a>
    ·
    <a href="https://haozhe-xing.github.io/fun-rec-mdbook/zh/">📖 中文版在线阅读</a>
    ·
    <a href="https://github.com/Haozhe-Xing/fun-rec-mdbook">GitHub</a>
  </p>
</div>

---

## English

### What is this?

A bilingual (English/Chinese) **knowledge base for recommender systems and computational advertising**, built to provide rich, well-structured technical knowledge for **auto research** — AI agents (or humans) researching, designing, or auditing recommender systems can retrieve precise, citable material here instead of crawling scattered blog posts and papers.

The content covers the full stack of modern recommender systems and the advertising stack built on top of it, organized along three main threads:

- **Discriminative recommendation** — the industrial workhorse: candidate retrieval (collaborative filtering, two-tower, sequential recall), ranking (Wide & Deep, feature crossing, multi-objective, multi-scenario), re-ranking for diversity (MMR, DPP), debiasing, and cold start.
- **Generative recommendation** — the emerging paradigm: semantic IDs and codebook quantization, HSTU and scaling laws, end-to-end generative recommenders/search/ads (OneRec, EGA), reasoning-based recommendation (OneRec-Think, RecZero), and diffusion models.
- **Computational advertising** — the economics layer: auction mechanisms (GFP/GSP/VCG), smart bidding and budget pacing, prediction calibration, online allocation, audience targeting, ad retrieval (boolean indexing, WAND, ANN semantic recall), data trading, and anti-fraud.

### Built for auto research

Every chapter is a self-contained, citable unit with learning objectives, formula derivations, worked numerical examples, layered practice problems (with solutions), Common Mistakes tables, and FAQs. Three machine-facing entry points let an agent go from question to exact section with minimal token cost:

| Entry point | Use it for |
| --- | --- |
| [`llms.txt`](llms.txt) | Top-level map ([llmstxt.org](https://llmstxt.org) format): every part with a one-line summary |
| [`index/en.json`](index/en.json) / [`index/zh.json`](index/zh.json) | Machine-readable chapter index: numbering, difficulty, learning objectives, key terms, source paths |
| [`src/en/GLOSSARY.md`](src/en/GLOSSARY.md) | 600+ terms, one definition each, organized by part |

**Retrieval path:** look up a term → GLOSSARY; locate a chapter → `index/*.json`; read the full text → `src/{en,zh}/partX-*/`. See [CLAUDE.md](CLAUDE.md) for the complete agent guide.

### Content map

| Part | Topic |
| --- | --- |
| 1 | Recommender Systems Overview |
| 2 | Fast Candidate Retrieval (CF, two-tower, sequential) |
| 3 | Preference Prediction & Ranking (Wide&Deep, DIN, MMoE) |
| 4 | Re-ranking for Diversity (MMR, DPP) |
| 5 | Frontier Trends (debiasing, cold start) |
| 6 | Foundations of Generative Recommendation (semantic IDs) |
| 7 | Scaling Generative Ranking (HSTU, RankMixer, OneTrans) |
| 8 | End-to-End Generative Applications (OneRec, EGA) |
| 9 | Thinking & Reasoning in Recommendation |
| 10 | Diffusion Models for Recommendation |
| 11 | Building a Recommender System in Practice |
| 12 | Computational Advertising (auctions, bidding, targeting, retrieval) |

### Build locally

```bash
./serve.sh        # build both editions + local preview on :3000
```

Requires `cargo install mdbook mdbook-katex`. Deployment to GitHub Pages is automatic on every push to `main`.

---

## 🇨🇳 中文版说明

### 这是什么？

一个面向**推荐系统 Auto Research** 的中英双语知识库——为 AI agent(以及人类研究者)在做推荐系统研究、方案设计与技术审阅时,提供结构化、可精确引用的技术知识,而非散落的博客与论文。

内容覆盖推荐系统全栈及其上的广告变现体系,沿三条主线组织:

- **判别式推荐**:召回(协同过滤/双塔/序列召回)、排序(Wide&Deep/特征交叉/多目标/多场景)、重排(MMR/DPP)、去偏、冷启动
- **生成式推荐**:语义 ID 与码本量化、HSTU 与 Scaling Law、端到端生成式推荐/搜索/广告(OneRec/EGA)、推理式推荐(OneRec-Think/RecZero)、扩散模型
- **计算广告**:竞价机制(GFP/GSP/VCG)、智能出价与预算控制、预估校准、在线分配、受众定向、广告检索(布尔索引/WAND/ANN 语义召回)、数据交易、反作弊

### Agent 检索入口

| 入口 | 用途 |
| --- | --- |
| [`llms.txt`](llms.txt) | 知识库总入口(llmstxt.org 规范):章节地图 + 每部分一句话摘要 |
| [`index/zh.json`](index/zh.json) | 机器可读章节索引:编号、难度、学习目标、核心术语、源文件路径 |
| [`src/zh/GLOSSARY.md`](src/zh/GLOSSARY.md) | 600+ 术语的一句话定义,按 Part 组织 |

检索路径:查术语 → GLOSSARY;定位章节 → index/*.json;读全文 → `src/{zh,en}/partX-*/`。每章含学习目标、公式推导、带解答的分层练习题与 FAQ。完整指南见 [CLAUDE.md](CLAUDE.md)。

### 本地构建

```bash
./serve.sh   # 构建双语 + 本地预览 :3000
```

需要 `cargo install mdbook mdbook-katex`。push 到 main 自动部署 GitHub Pages。

---

## Acknowledgments · 致谢

Much of the content of this knowledge base is rewritten and expanded from the Datawhale open-source project [fun-rec](https://github.com/datawhalechina/fun-rec) (《推荐系统实战》 by Datawhale). We are grateful to the original authors and the community of contributors. The rewrite restructures the material around the auto-research use case: bilingual full-text, machine-readable indexes, a glossary, worked examples with verified solutions, and new sections covering generative recommendation and computational advertising.

本知识库的大量内容重写并扩展自 Datawhale 开源项目 [fun-rec](https://github.com/datawhalechina/fun-rec)。感谢原项目作者与社区贡献者。重写版围绕 auto research 场景重新组织:双语全文、机器可读索引、术语表、经过验算的例题与解答,并新增了生成式推荐与计算广告等章节。

内容以 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可发布。
