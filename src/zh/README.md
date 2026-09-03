<div align="center">
  <h1>推荐系统自动研究知识库</h1>
  <p><strong>RecSys Auto Research KB · 从级联架构到生成式范式的推荐与广告知识库</strong></p>

  <p><img alt="mdBook" src="https://img.shields.io/badge/Built%20with-mdBook-2f74c0?style=flat-square"> <img alt="Language" src="https://img.shields.io/badge/Language-中文-16a34a?style=flat-square"> <img alt="Topic" src="https://img.shields.io/badge/Topic-Recommender%20Systems-f97316?style=flat-square"> <img alt="Status" src="https://img.shields.io/badge/Status-In%20Progress-8b5cf6?style=flat-square"> <a href="https://haozhe-xing.github.io/fun-rec-mdbook/"><img alt="Read Online" src="https://img.shields.io/badge/Read-Online-4A6CF7?style=flat-square"></a></p>

  <p>
    一个面向推荐系统与计算广告的知识库:判别式推荐、生成式推荐、计算广告专题,为人类读者与 AI agent 的 auto research 提供结构化、可精确引用的技术知识。
  </p>

  <p>
    <a href="https://haozhe-xing.github.io/fun-rec-mdbook/"><strong>📖 在线阅读</strong></a>
    ·
    <a href="https://github.com/Haozhe-Xing/fun-rec-mdbook">GitHub 仓库</a>
  </p>
</div>

---

## 这是什么?

**《推荐系统自动研究知识库》** 是一个面向推荐系统 auto research 的知识库,内容重写并扩展自 Datawhale 开源项目 [fun-rec](https://github.com/datawhalechina/fun-rec)。

> 📖 **在线阅读**：访问 [https://haozhe-xing.github.io/fun-rec-mdbook/](https://haozhe-xing.github.io/fun-rec-mdbook/) 阅读最新版本。

本书围绕推荐系统的两条核心主线展开：

- **判别式推荐** ：召回、排序、重排、多目标、多场景、去偏、冷启动等工业级推荐系统基础能力。
- **生成式推荐** ：语义 ID、生成式排序、端到端推荐、推荐推理、扩散模型与生成式推荐系统实战。

如果你希望从「算法原理」走到「系统实践」，并理解推荐系统从经典架构走向生成式范式的完整脉络，这本书就是为你准备的。

---

## 内容地图

本书共 **12 个 Part** ，建议按顺序阅读；如果你已有推荐系统基础，也可以直接跳到感兴趣的主题。

| 篇章 | 主题 | 你将学到什么 |
| --- | --- | --- |
| Part 1 | 推荐系统全景 | 推荐系统基本问题、技术地图、特征与 Embedding 基础 |
| Part 2 | 快速候选召回 | 协同过滤、向量召回、双塔、序列召回、流式索引 |
| Part 3 | 精准偏好预测 | Wide&Deep、特征交叉、序列建模、多目标、多场景 |
| Part 4 | 重排多样性建模 | MMR、DPP、个性化重排与列表级优化 |
| Part 5 | 前沿趋势 | 去偏、冷启动、生成式推荐范式演进 |
| Part 6 | 生成式推荐基础 | 生成式范式、LLM 基础、Codebook、语义 ID |
| Part 7 | Scaling 生成式排序 | HSTU、生成式排序、MTGR、RankMixer、OneTrans |
| Part 8 | 端到端生成式应用 | 推荐、搜索、广告中的端到端生成式建模 |
| Part 9 | 推荐中的思考与推理 | 语义对齐、推理框架、自主推理探索 |
| Part 10 | 扩散模型推荐 | 扩散基础、数据增强、推荐应用 |
| Part 11 | 生成式推荐系统实战 | 系统架构、离线管线、在线管线、前端与部署 |
| Part 12 | 计算广告专题 | 拍卖机制、智能出价、合约与竞价广告、定向、数据交易、实验与反作弊 |

完整目录见 [SUMMARY.md](SUMMARY.md)。

---

### 1. 安装依赖

```bash
cargo install mdbook
cargo install mdbook-katex
```

### 2. 本地预览（中英双语一键启动）

```bash
./serve.sh
```

默认在 `http://localhost:3000` 启动，中文版在 `/zh/`、英文版在 `/en/`，根路径按浏览器语言自动跳转。

### 3. 构建静态站点

```bash
mdbook build                          # 中文版 → book/zh/
cp book-en.toml book.toml && mdbook build   # 英文版 → book/en/（构建后恢复 book.toml）
```

---

## 项目结构

本项目提供 **中英双语** 版本，结构与 [agent_learning](https://github.com/Haozhe-Xing/agent_learning) 一致：

```text
.
├── src/
│   ├── zh/                   # 中文源（当前目录）
│   │   ├── SUMMARY.md        # mdBook 目录结构
│   │   ├── part1-introduction/ … part11-project/
│   │   ├── appendix/         # 附录
│   │   ├── images/           # SVG 图表资源
│   │   └── viz/              # 交互式可视化资源
│   └── en/                   # 英文源（与 zh 1:1 对齐）
├── book.toml                 # 中文版配置 → book/zh/
├── book-en.toml              # 英文版配置 → book/en/
├── root-index.html           # 语言选择首页（按浏览器语言自动跳转）
├── serve.sh                  # 一键构建并本地预览
├── styles/                   # 自定义样式
└── scripts/                  # 自定义交互脚本
```

---

## 推荐阅读路径

### 初学者路径

适合刚接触推荐系统、希望建立完整知识框架的读者：

```text
Part 1 → Part 2 → Part 3 → Part 4 → Part 5
```

你会先掌握推荐系统的经典级联架构，再理解召回、排序、重排各模块如何协同工作。

### 进阶工程路径

适合已经做过推荐算法或推荐工程，希望补齐系统设计能力的读者：

```text
Part 2 → Part 3 → Part 4 → Part 11
```

你会重点理解工业推荐链路中的候选生成、偏好预测、列表优化和线上服务架构。

### 生成式推荐路径

适合关注大模型、生成式排序、语义 ID 与下一代推荐系统的读者：

```text
Part 5 → Part 6 → Part 7 → Part 8 → Part 9 → Part 10 → Part 11
```

你会从范式迁移开始，逐步进入生成式推荐的建模、推理和系统落地。

---

## 适合谁读？

- **推荐算法学习者** ：希望系统学习推荐系统核心模型与技术路线。
- **机器学习工程师** ：希望理解推荐系统从离线训练到在线服务的工程链路。
- **推荐系统从业者** ：希望补齐生成式推荐、语义 ID、端到端推荐等新方向。
- **技术面试准备者** ：希望建立清晰的推荐系统知识地图与表达框架。

---

## 写作约定

- 章节顶部使用徽章标注 **章节编号、预计阅读时间、难度级别**。
- 数学公式使用 `$行内公式$` 与 `$$独立公式$$`，由 `mdbook-katex` 渲染。
- 图表统一放在 `images/`，交互式可视化放在 `viz/`。
- 每章尽量包含 **常见错误、核心要点、FAQ、章节关联与分层练习**。

---

## 参与贡献

欢迎通过 Issue 或 Pull Request 参与改进：

- 修正错别字、公式、图表或链接问题。
- 补充推荐系统论文、工业案例或工程实践经验。
- 改进章节结构、示例代码、练习题与可视化内容。
- 提出你希望新增的推荐系统主题。

在提交内容时，建议保持：

- **术语统一** ：优先参考 [GLOSSARY.md](GLOSSARY.md)。
- **结构一致** ：遵循已有章节的组织方式。
- **解释清晰** ：优先说明直觉、边界条件和工程取舍。

---

## 致谢

本书重写自 Datawhale 开源项目 [fun-rec](https://github.com/datawhalechina/fun-rec)，感谢原项目作者与社区贡献者为推荐系统学习资料建设做出的贡献。

也感谢推荐系统、信息检索、机器学习与大模型社区中的研究者和工程师。本书中的许多内容都受益于公开论文、工业实践分享和开源社区讨论。

---

<div align="center">
  <strong>如果这本书对你有帮助，欢迎 Star、分享或参与共建。</strong>
</div>