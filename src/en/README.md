<div align="center">
  <h1>RecSys Auto Research KB</h1>
  <p><strong>A knowledge base for recommender-system auto research · from cascading architectures to the generative paradigm</strong></p>

  <p><img alt="mdBook" src="https://img.shields.io/badge/Built%20with-mdBook-2f74c0?style=flat-square"><img alt="Language" src="https://img.shields.io/badge/Language-English-16a34a?style=flat-square"><img alt="Topic" src="https://img.shields.io/badge/Topic-Recommender%20Systems-f97316?style=flat-square"><img alt="Status" src="https://img.shields.io/badge/Status-In%20Progress-8b5cf6?style=flat-square"><a href="https://haozhe-xing.github.io/fun-rec-mdbook/"><img alt="Read Online" src="https://img.shields.io/badge/Read-Online-4A6CF7?style=flat-square"></a></p>

  <p>A knowledge base for recommender systems and computational advertising: discriminative recommendation, generative recommendation, and the advertising stack — structured, citable technical knowledge for auto research by humans and AI agents.</p>

  <p><a href="https://haozhe-xing.github.io/fun-rec-mdbook/"><strong>📖 Read Online</strong></a> · <a href="https://github.com/Haozhe-Xing/fun-rec-mdbook">GitHub Repository</a></p>
</div>

---

## What Is This?

**RecSys Auto Research KB** is a knowledge base for recommender-system auto research, rewritten and expanded from the Datawhale open-source project [fun-rec](https://github.com/datawhalechina/fun-rec).

> 📖 **Read online**: visit [https://haozhe-xing.github.io/fun-rec-mdbook/](https://haozhe-xing.github.io/fun-rec-mdbook/) for the latest version.

The book follows two main threads:

- **Discriminative recommendation**: retrieval, ranking, re-ranking, multi-task learning, multi-scenario modeling, debiasing, cold start, and other foundational capabilities of industrial recommender systems.
- **Generative recommendation**: semantic IDs, generative ranking, end-to-end recommendation, recommendation reasoning, diffusion models, and hands-on practice building a generative recommender.

If you want to move from "algorithm principles" to "system practice" and understand the full arc from the classic architecture to the generative paradigm, this book is for you.

---

## Content Map

The book has **11 Parts**, best read in order; if you already know the basics of recommender systems, feel free to jump straight to the topics you care about.

| Part | Theme | What You Will Learn |
| --- | --- | --- |
| Part 1 | Recommender Systems at a Glance | The basic problems of recommendation, a technical map, and feature & embedding fundamentals |
| Part 2 | Fast Candidate Retrieval | Collaborative filtering, vector retrieval, two-tower models, sequential retrieval, streaming indexes |
| Part 3 | Accurate Preference Prediction | Wide&Deep, feature crossing, sequence modeling, multi-task learning, multi-scenario modeling |
| Part 4 | Re-ranking for Diversity | MMR, DPP, personalized re-ranking, and list-level optimization |
| Part 5 | Frontier Trends | Debiasing, cold start, and the evolution toward generative recommendation |
| Part 6 | Foundations of Generative Recommendation | The generative paradigm, LLM basics, codebooks, semantic IDs |
| Part 7 | Scaling Generative Ranking | HSTU, generative ranking, MTGR, RankMixer, OneTrans |
| Part 8 | End-to-End Generative Applications | End-to-end generative modeling in recommendation, search, and advertising |
| Part 9 | Thinking and Reasoning in Recommendation | Semantic alignment, reasoning frameworks, autonomous reasoning exploration |
| Part 10 | Diffusion Models for Recommendation | Diffusion basics, data augmentation, recommendation applications |
| Part 11 | Building a Generative Recommender | System architecture, offline pipeline, online pipeline, frontend, and deployment |

See [SUMMARY.md](SUMMARY.md) for the full table of contents.

---

### 1. Install dependencies

```bash
cargo install mdbook
cargo install mdbook-katex
```

### 2. Local preview

```bash
./serve.sh
```

`./serve.sh` builds both language editions and starts a local preview server. Open the URL printed in the terminal to read the book — the language is auto-selected at the site root based on your browser settings.

### 3. Build static sites

```bash
mdbook build
```

Build artifacts are written to the `book/` directory (`book/zh/` and `book/en/`, one per edition; `./serve.sh` wraps the same build for both configs).

---

## Project Structure

This is a bilingual project: the Chinese and English editions share the same structure and are built separately.

```text
.
├── README.md                 # Home page and project intro
├── SUMMARY.md                # mdBook table of contents
├── book.toml                 # mdBook config (Chinese edition)
├── book-en.toml              # mdBook config (English edition)
├── serve.sh                  # Build/serve script for both editions
├── GLOSSARY.md               # Glossary
├── src/
│   ├── zh/                   # Chinese source (chapter sources + appendix)
│   ├── en/                   # English source (chapter sources + appendix)
│   ├── zh/images/            # SVG figures (Chinese edition)
│   ├── en/images/            # SVG figures (English edition)
│   ├── zh/viz/               # Interactive visualizations (Chinese edition)
│   └── en/viz/               # Interactive visualizations (English edition)
└── book/
    ├── zh/                   # Build output (Chinese edition)
    └── en/                   # Build output (English edition)
```

---

## Recommended Reading Paths

### Beginner path

For readers new to recommender systems who want to build a complete knowledge framework:

```text
Part 1 → Part 2 → Part 3 → Part 4 → Part 5
```

You will first master the classic cascading architecture of recommender systems, then understand how retrieval, ranking, and re-ranking work together.

### Advanced engineering path

For readers who have already worked on recommendation algorithms or recommendation engineering and want to strengthen their system-design skills:

```text
Part 2 → Part 3 → Part 4 → Part 11
```

You will focus on candidate generation, preference prediction, list-level optimization, and online serving architecture in the industrial recommendation pipeline.

### Generative recommendation path

For readers interested in LLMs, generative ranking, semantic IDs, and next-generation recommender systems:

```text
Part 5 → Part 6 → Part 7 → Part 8 → Part 9 → Part 10 → Part 11
```

You will start from the paradigm shift and work through the modeling, inference, and system deployment of generative recommendation step by step.

---

## Who Is This Book For?

- **Recommendation algorithm learners**: build a systematic grounding in the core models and technical roadmaps of recommender systems.
- **Machine learning engineers**: understand the full engineering loop of a recommender system, from offline training to online serving.
- **Recommender system practitioners**: catch up on new directions such as generative recommendation, semantic IDs, and end-to-end recommendation.
- **Interview candidates**: build a clear mental map and vocabulary for recommender-system topics.

---

## Writing Conventions

- Each chapter opens with badges for the **chapter number, estimated reading time, and difficulty level**.
- Math uses `$inline$` and `$$display$$` notation, rendered by `mdbook-katex`.
- Figures live in `images/`, and interactive visualizations live in `viz/`.
- Wherever possible, each chapter includes **common mistakes, key takeaways, an FAQ, chapter connections, and tiered practice problems**.

---

## Contributing

Issues and pull requests are welcome:

- Fix typos, formulas, figures, or broken links.
- Add recommender-system papers, industrial case studies, or engineering lessons.
- Improve chapter structure, example code, exercises, or visualizations.
- Propose new recommender-system topics you would like to see covered.

When submitting, please keep:

- **Consistent terminology**: follow [GLOSSARY.md](GLOSSARY.md) first.
- **Consistent structure**: mirror the organization of existing chapters.
- **Clear explanations**: favor intuition, boundary conditions, and engineering trade-offs.

---

## Acknowledgments

This book is rewritten from the Datawhale open-source project [fun-rec](https://github.com/datawhalechina/fun-rec). Thanks to the original authors and community contributors for building open learning materials on recommender systems.

Thanks also to the researchers and engineers in the recommender-system, information-retrieval, machine-learning, and LLM communities. Much of this book draws on public papers, industrial experience sharing, and open-source community discussions.

---

<div align="center">
  <strong>If this book helps you, consider starring the repository, sharing it, or contributing.</strong>
</div>
