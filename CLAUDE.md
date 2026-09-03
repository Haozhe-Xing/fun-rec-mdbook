# CLAUDE.md — 知识库使用与维护指南

本仓库是**推荐系统与计算广告的知识库**(中英双语 mdBook),既供人类在线阅读,也供 AI agent 检索引用。定位:推荐算法全栈 + 计算广告专题的可引用知识源。

## 结构

```
src/zh/            # 中文源(authoritative source of truth)
src/en/            # 英文源,与 zh 1:1 镜像
  SUMMARY.md       # 目录(权威章节结构)
  GLOSSARY.md      # 600+ 术语表,按 Part 分节
  part1~5-*        # 判别式推荐:召回/排序/重排/趋势
  part6~11-*       # 生成式推荐:范式/Scaling/端到端/推理/扩散/实战
  part12-*         # 计算广告专题(13 节)
  appendix/        # Word2Vec 推导
index/zh.json      # 机器可读章节索引(编号/难度/学习目标/核心术语)
llms.txt           # agent 入口(llmstxt.org 规范)
book.toml          # zh 构建 → book/zh/
book-en.toml       # en 构建 → book/en/(临时替换 book.toml 使用)
serve.sh           # 一键构建双语 + 本地预览
.github/workflows/deploy.yml  # push main → 自动部署 gh-pages
```

## Agent 检索路径

回答推荐/广告问题时按以下顺序定位材料:

1. **查术语定义** → `src/zh/GLOSSARY.md`(每条一句话,常带章节引用)
2. **定位章节** → `index/zh.json`(按 part/chapter/key_terms 找到 source 路径)或 `src/zh/SUMMARY.md`
3. **读全文** → 对应 `src/zh/partX-*/xxx.md`;每章有学习目标、公式推导、练习题(带解答)、Common Mistakes、FAQ
4. 英文材料 → 同路径的 `src/en/...` 镜像

章节编号约定:Part N 下的第 M 节记为 `N.M`(如 12.3 竞价机制 = `src/zh/part12-computational-advertising/auction-mechanisms.md`)。

## 内容约定(修改时保持一致)

- 每章开头:`badge-row` div(难度/阅读时长徽章)+ `# 一级标题` + `> 📝 Before You Continue` 前置阅读提示
- 每章结构:学习目标 → 正文(小节 N.M.K)→ ⚠️ Common Mistakes 表 → 小结(Key Takeaways/FAQ/前后关联)→ 分层练习题(🟢🟡🔴,`<details>` 折叠解答)
- 行文元素:`> 💡 Key Insight` / `> 🧠 Mental Model` / `> **Analysis:**` 引用块
- 公式用 KaTeX(`$...$` / `$$...$$`);图为 SVG(`src/*/images/`),交互演示为 iframe(`src/*/viz/`)

## 修改规则

1. **改 zh 必须同步 en**(同路径镜像文件),内容逐段对应
2. **加/删章节** → 同时改两个 SUMMARY.md,并重跑 `python3 scripts/build-index.py` 更新 `index/*.json`
3. **构建验证** → `mdbook build`(zh);改 en 后用 `cp book-en.toml book.toml && mdbook build && git checkout book.toml`
4. 部署全自动:push 到 main 即触发 GitHub Actions 构建双语并更新 gh-pages(无需手动)
5. 交互式 iframe(竞价模拟器等)在 `src/zh/viz/`,zh/en 各自维护(内嵌文字语言不同)

## 常用命令

```bash
mdbook build                                    # 构建中文版
./serve.sh                                      # 双语构建 + 本地预览 :3000
python3 scripts/build-index.py                  # 重新生成 index/*.json
```
