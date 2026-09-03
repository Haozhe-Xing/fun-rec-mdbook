<div align="center">
  <h1>深度推荐算法实践 · Deep Recommender Systems in Practice</h1>
  <p><strong>小麦书 · 从级联架构到生成式范式</strong></p>

  <p>
    <img alt="mdBook" src="https://img.shields.io/badge/Built%20with-mdBook-2f74c0?style=flat-square">
    <img alt="Language" src="https://img.shields.io/badge/Language-中文%20%7C%20English-16a34a?style=flat-square">
    <img alt="Topic" src="https://img.shields.io/badge/Topic-Recommender%20Systems%20%7C%20Computational%20Advertising-f97316?style=flat-square">
    <img alt="Deploy" src="https://img.shields.io/github/actions/workflow/status/Haozhe-Xing/fun-rec-mdbook/deploy.yml?style=flat-square&label=deploy">
    <a href="https://haozhe-xing.github.io/fun-rec-mdbook/"><img alt="Read Online" src="https://img.shields.io/badge/Read-Online-4A6CF7?style=flat-square"></a>
  </p>

  <p>
    一本系统讲解推荐系统技术演进的开源书：从传统级联架构，到生成式推荐、推理式推荐与端到端推荐系统实战。
  </p>
  <p>
    An open-source book on the evolution of recommender systems — from classic cascading architectures to generative, reasoning-based, and end-to-end recommender systems.
  </p>
  <p>
    <a href="https://haozhe-xing.github.io/fun-rec-mdbook/"><strong>📖 在线阅读 / Read Online</strong></a>
    ·
    <a href="https://haozhe-xing.github.io/fun-rec-mdbook/zh/">🇨🇳 中文版</a>
    ·
    <a href="https://haozhe-xing.github.io/fun-rec-mdbook/en/">🇺🇸 English</a>
    ·
    <a href="https://github.com/Haozhe-Xing/fun-rec-mdbook">GitHub</a>
  </p>
</div>

---

## 这是什么？ / What is this?

**《深度推荐算法实践》** 是一个推荐系统与计算广告的**双语知识库**——既是一本供人系统阅读的开源书(内容重写自 Datawhale 开源项目 [fun-rec](https://github.com/datawhalechina/fun-rec)),也是一个为 AI agent 优化的可检索知识源(机器可读索引、术语表、llms.txt 入口)。

A bilingual **knowledge base** for recommender systems and computational advertising — a book for humans, and a retrieval-friendly source for AI agents.

本书围绕三条核心主线展开：

- **判别式推荐**：召回、排序、重排、多目标、多场景、去偏、冷启动等工业级推荐系统基础能力。
- **生成式推荐**：语义 ID、生成式排序、端到端推荐、推荐推理、扩散模型与生成式推荐系统实战。
- **计算广告专题**：竞价机制（GFP/GSP/VCG）、智能出价、预估校准、在线分配、受众定向、广告检索、数据交易与反作弊。

## Agent 访问 / For AI Agents

本知识库为 agent 提供三层检索入口(详见 [CLAUDE.md](CLAUDE.md)):

| 入口 | 用途 |
| --- | --- |
| [`llms.txt`](llms.txt) | 知识库总入口（[llmstxt.org](https://llmstxt.org) 规范）：章节地图 + 每章一句话摘要 |
| [`index/zh.json`](index/zh.json) / [`index/en.json`](index/en.json) | 机器可读章节索引：编号、难度、学习目标、核心术语、源文件路径 |
| [`src/zh/GLOSSARY.md`](src/zh/GLOSSARY.md) | 600+ 术语的一句话定义，按 Part 组织 |

检索路径：查术语 → GLOSSARY；定位章节 → index/*.json；读全文 → `src/{zh,en}/partX-*/`。每章含学习目标、公式推导、带解答的分层练习题与 FAQ，适合精确引用。

---

## 双语结构 / Bilingual Structure

本项目提供**中英双语**版本，结构与 [agent_learning](https://github.com/Haozhe-Xing/agent_learning) 一致：

```text
.
├── src/
│   ├── zh/              # 中文源（authoritative source of truth）
│   │   ├── SUMMARY.md   # mdBook 目录
│   │   ├── part1-introduction/ … part11-project/
│   │   ├── appendix/
│   │   ├── images/      # SVG 图表
│   │   └── viz/         # 交互式可视化
│   └── en/              # 英文源（结构与 zh 1:1 对齐）
├── book.toml            # 中文版配置 → book/zh/
├── book-en.toml         # 英文版配置 → book/en/
├── root-index.html      # 语言选择首页（按浏览器语言自动跳转）
├── serve.sh             # 一键构建并本地预览
├── styles/              # 自定义样式
└── scripts/             # 自定义交互脚本
```

站点根路径会依据浏览器语言自动跳转到 `/zh/` 或 `/en/`。

---

## 本地构建 / Build Locally

```bash
# 一键构建中英双语并启动本地预览（默认 http://localhost:3000）
./serve.sh

# 或分别构建
mdbook build                    # 中文版 → book/zh/
cp book-en.toml book.toml && mdbook build  # 英文版 → book/en/（再恢复 book.toml）
```

依赖：`cargo install mdbook mdbook-katex`。

---

## 内容地图 / Content Map

| Part | 主题 / Topic |
| --- | --- |
| 1 | 推荐系统全景 / Recommender Systems Overview |
| 2 | 快速候选召回 / Fast Candidate Retrieval |
| 3 | 精准偏好预测 / Preference Prediction & Ranking |
| 4 | 重排多样性建模 / Re-ranking for Diversity |
| 5 | 前沿趋势 / Frontier Trends (debiasing, cold start) |
| 6 | 生成式推荐基础 / Foundations of Generative Recommendation |
| 7 | Scaling 生成式排序 / Scaling Generative Ranking |
| 8 | 端到端生成式应用 / End-to-End Generative Applications |
| 9 | 推荐中的思考与推理 / Thinking & Reasoning in Recommendation |
| 10 | 扩散模型推荐 / Diffusion Models for Recommendation |
| 11 | 生成式推荐系统实战 / Building a Generative Recommender in Practice |
| 12 | 计算广告专题 / Computational Advertising (auctions, bidding, targeting, retrieval) |

---

## 致谢 / Acknowledgments

本书重写自 Datawhale 开源项目 [fun-rec](https://github.com/datawhalechina/fun-rec)，感谢原项目作者与社区贡献者。
