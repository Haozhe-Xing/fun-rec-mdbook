<div align="center">
  <h1>深度推荐算法实践 · Deep Recommender Systems in Practice</h1>
  <p><strong>小麦书 · 从级联架构到生成式范式</strong></p>

  <p>
    <img alt="mdBook" src="https://img.shields.io/badge/Built%20with-mdBook-2f74c0?style=flat-square">
    <img alt="Language" src="https://img.shields.io/badge/Language-中文%20%7C%20English-16a34a?style=flat-square">
    <img alt="Topic" src="https://img.shields.io/badge/Topic-Recommender%20Systems-f97316?style=flat-square">
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

**《深度推荐算法实践》** 是一本面向推荐系统学习者与工程实践者的开源电子书，内容重写自 Datawhale 开源项目 [fun-rec](https://github.com/datawhalechina/fun-rec)。

本书围绕推荐系统的两条核心主线展开：

- **判别式推荐**：召回、排序、重排、多目标、多场景、去偏、冷启动等工业级推荐系统基础能力。
- **生成式推荐**：语义 ID、生成式排序、端到端推荐、推荐推理、扩散模型与生成式推荐系统实战。

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

---

## 致谢 / Acknowledgments

本书重写自 Datawhale 开源项目 [fun-rec](https://github.com/datawhalechina/fun-rec)，感谢原项目作者与社区贡献者。
