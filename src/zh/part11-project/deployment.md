<div class="badge-row">
  <span style="background: #f0f4ff; color: #4A6CF7; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(74,108,247,0.2);">📖</span>
  <span style="background: #f0fdf4; color: #16a34a; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(22,163,74,0.2);">⏱️ ~24 min read</span>
  <span style="background: #fef9ec; color: #d97706; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(100,100,100,0.15);">🎯 Intermediate</span>
</div>

# 部署与运维

> 📝 **Before You Continue:** 请先读完 [11.5](./frontend.md) 与 [11.4](./online-pipeline.md)。本节把它们容器化编排成一个可一键启动、可监控、可排障的系统。

前面完成了离线、在线、前端的开发，可在本地跑通。但如何在其他机器快速复现？本节用 **Docker Compose** 做容器化部署。

读完本章，你将能够：

- 解释为何用 Docker Compose（环境一致/快速启动/隔离/易扩展）
- 读懂 `docker-compose.yaml` 中五个服务的配置，理解容器间用服务名 DNS 通信
- 描述前端多阶段构建（Node 构建 + Nginx 服务）如何减小镜像
- 执行完整启动流程：启基础设施 → 跑离线 → 导数据 → 建索引 → 访问
- 用 `docker compose ps`、健康检查、`redis-cli` 排查常见问题
- 完成 4 道分层练习题

---

## 11.6.0 为什么用 Docker Compose

本项目依赖五个服务：PostgreSQL（业务数据）、Redis（特征缓存）、Elasticsearch（搜索）、后端 API、前端应用。模型文件经共享目录在离/在线间传递。

手动部署需每台机装 PG/Redis/ES、配网络、处理版本兼容——繁琐易错、环境差异致各种问题。Docker Compose 用 **声明式 YAML** 描述所有服务及依赖，一条命令启动全系统。优势：

1. **环境一致性** ：容器含全部依赖，开发/测试/生产环境一致。
2. **快速启动** ：`docker compose up` 按依赖顺序自动起，免手动装配。
3. **隔离与安全** ：各服务独立容器，互不干扰。
4. **易于扩展** ：加服务只需改配置，不动现有。

---

## 11.6.1 Docker Compose 配置详解

`docker-compose.yaml` 定义六个服务（含后端构建）。逐一介绍。

**数据库 PostgreSQL** ：

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: funrec-postgres
    environment:
      POSTGRES_USER: funrec
      POSTGRES_PASSWORD: funrec123
      POSTGRES_DB: funrec_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data          # ← KEY LINE: 命名卷持久化数据
    networks:
      - funrec-network
```

`volumes` 命名卷 `postgres_data` 持久化数据，容器删数据仍在；`networks` 使其能与其他服务通信。

**缓存 Redis** ：

```yaml
  redis:
    image: redis:7-alpine
    container_name: funrec-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - funrec-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5                                       # ← KEY LINE: 连续失败才判不健康
```

`healthcheck` 定期 `redis-cli ping`，连续 5 次超时（每次 3s）判不健康，依赖它的服务可等其健康再起。

**搜索 Elasticsearch** ：

```yaml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:9.2.0
    container_name: funrec-elasticsearch
    environment:
      - discovery.type=single-node                     # ← KEY LINE: 单节点模式，适合开发
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"               # ← KEY LINE: 限制 JVM 堆，防开发机内存爆
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - funrec-network
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
```

**后端 FastAPI** ：

```yaml
  backend:
    build:
      context: ./backend
      dockerfile: dockerfile
    container_name: funrec-backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://funrec:funrec123@postgres:5432/funrec_db
      - REDIS_URL=redis://redis:6379/0
      - ELASTICSEARCH_URL=http://elasticsearch:9200
      - MODEL_DEPLOY_DIR=/app/tmp/web_project/deployed_models
    volumes:
      - ./backend:/app
      - ../tmp:/app/tmp
      - ${FUNREC_RAW_DATA_PATH}:/data
    depends_on:
      - postgres
      - elasticsearch
      - redis
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - funrec-network
```

注意数据库/Redis 地址用 **服务名** （如 `postgres`、`redis`）而非 `localhost`——容器间靠 Docker DNS 解析。后端 Dockerfile 分层构建（先装依赖再拷代码），改码不重装依赖：

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc postgresql-client curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
COPY pyproject.toml ./
RUN uv pip install --system -e .                  # ← KEY LINE: 先装依赖层，利用缓存
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**前端（多阶段构建）** ：先 Node 构建静态文件，再 Nginx 服务：

```yaml
  frontend:
    build:
      context: ./frontend
      dockerfile: dockerfile
    container_name: funrec-frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    networks:
      - funrec-network
```

Dockerfile 多阶段——最终镜像仅含构建产物 + Nginx，不含 Node/开发依赖：

```dockerfile
# 构建阶段
FROM node:22-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build                                # ← KEY LINE: 生成 dist 静态产物
# 生产阶段
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html   # ← KEY LINE: 仅拷产物，镜像更小
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**网络与数据** ：末尾定义共享网络与命名卷：

```yaml
volumes:
  postgres_data:
  redis_data:
  elasticsearch_data:
networks:
  funrec-network:
    driver: bridge                             # ← KEY LINE: 同桥接网络，服务名互通
```

![部署拓扑：五服务同处 funrec-network，共享卷持久化，模型经共享目录在离/在线传递](../images/part11-deployment-topology.svg)

---

## 11.6.2 环境准备与启动流程

**前置条件** ：安装 Docker、Docker Compose（Desktop 内置）、uv（`pip install uv`）。验证：

```bash
docker --version
docker compose version
uv --version
```

**数据准备** ：下载 `funrec-movielens-1m.zip` 解压，记录绝对路径（含 `movies.pkl`/`ratings.pkl`/`users.pkl`/`image/`）。

**获取代码** ：本项目全部代码位于 [datawhalechina/fun-rec](https://github.com/datawhalechina/fun-rec) 仓库的 `web_project/` 目录：

```bash
git clone https://github.com/datawhalechina/fun-rec.git
cd fun-rec/web_project
```

**环境变量** ：复制 `.env.example` 为 `.env` 并设数据路径：

```bash
cd web_project
cp .env.example .env
# 编辑 .env：
# FUNREC_RAW_DATA_PATH=/path/to/funrec-movielens-1m
# FUNREC_PROCESSED_DATA_PATH=/path/to/funrec-processed
```

`FUNREC_PROCESSED_DATA_PATH` 存特征工程与训练中间产物，需可写。

**启动基础设施** ：

```bash
docker compose up --build                    # 首次构建镜像
docker compose up -d --build                 # 后台运行
docker compose logs -f backend               # 看后端日志
```

**运行离线流程** （训练模型、初始化数据）：

```bash
cd backend
uv sync
make run-offline-pipeline
```

依次执行：特征工程 → 训练 YoutubeDNN/DeepFM → 特征上线 Redis → 模型部署共享目录（约 10–20 分钟）。

**加载数据到数据库** ：

```bash
make ingest-data-to-database                 # 建表 + 导入用户/电影/评分 + 建测试用户
```

**索引电影到 Elasticsearch** ：

```bash
make index-movies-to-elasticsearch           # 标题/类型/演员可搜索
```

**访问应用** ：

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端 | http://localhost:3000 | 用户界面 |
| 后端 API | http://localhost:8000 | API 服务 |
| API 文档 | http://localhost:8000/docs | Swagger |
| Elasticsearch | http://localhost:9200 | 搜索服务 |

测试账号：`test@funrec.com` / `test123456`。登录后可见个性化推荐、搜索、详情、评分。

---

## 11.6.3 服务健康检查与调试

**检查状态** ：

```bash
docker compose ps
# NAME  STATUS  PORTS ... 全部应为 Up
```

某服务 `Exited`/`Restarting` 即启动失败，查日志。

**验证各服务** ：

```bash
curl http://localhost:8000/health           # 后端 → {"status": "healthy"}
docker exec -it funrec-postgres pg_isready -U funrec   # PG → accepting connections
docker exec -it funrec-redis redis-cli ping            # Redis → PONG
curl http://localhost:9200                          # ES → 版本信息
```

**查 Redis 数据** （验证特征上线）：

```bash
docker exec -it funrec-redis redis-cli hget user:6041:profile frequent_genres
docker exec -it funrec-redis redis-cli llen user:6041:history
```

**常见问题排查** ：

- **容器启动失败** → `docker compose logs backend`；查 .env 路径、端口占用、依赖未就绪。
- **数据库连接失败** → `docker compose logs postgres`，看 `ready to accept connections`。
- **模型加载失败** → `ls ${FUNREC_PROCESSED_DATA_PATH}/web_project/deployed_models/`；无文件则重跑离线流程。
- **搜索无结果** → `curl http://localhost:9200/_cat/indices`；无 `movies` 索引则重跑索引命令。

> **Analysis:** 部署的难点不在「写配置」，而在「让五服务按依赖顺序健康起来、并能快速定位故障」。健康检查（healthcheck）、命名卷、服务名 DNS、日志与 `redis-cli` 探查，共同构成可观测、可恢复的交付基线。

---

## ⚠️ Common Mistakes in 11.6

| # | Mistake | Example | Why It's Wrong | Fix |
|---|---------|---------|---------------|-----|
| 1 | 容器间用 localhost | backend 连 localhost:5432 | 容器内 localhost 是自身，非 PG | 用服务名 postgres |
| 2 | 不挂持久卷 | 容器删数据丢 | 命名卷才持久化 | 挂 postgres_data 等卷 |
| 3 | ES 不限内存 | 默认堆吃满开发机 | 卡顿/OOM | ES_JAVA_OPTS 限 512m |
| 4 | 忘跑离线流程 | 直接访问推荐 → 空 | 无模型/特征 | 先 make run-offline-pipeline |
| 5 | 前端不构建 | 只拷源码不 npm run build | Nginx 无 dist | 多阶段构建生成 dist |

---

## 本章小结

### 📌 Key Takeaways

| Concept | Key Points | Why It Matters |
|---------|-----------|----------------|
| Compose 价值 | 一致/快启/隔离/易扩 | 一键复现多服务系统 |
| 服务名 DNS | postgres/redis 互访 | 容器间通信基础 |
| 持久卷 | 命名卷存数据 | 容器删数据不丢 |
| 多阶段构建 | Node 构建 + Nginx 服务 | 前端镜像最小 |
| 启动流程 | 启设施→离线→导数→索引→访问 | 顺序不可乱 |
| 健康与排障 | healthcheck + logs + cli | 可观测可恢复 |

### ❓ FAQ

**Q1: 为什么后端用服务名而不是宿主机 IP？**
> A: 同一 Docker 网络内，Compose 内置 DNS 把服务名解析到容器 IP。用 `localhost` 在容器内指向自身而非 PG，必然连不上。服务名是容器间通信的正确方式。

**Q2: 模型文件怎么从离线容器到在线容器？**
> A: 经共享目录（卷挂载）：离线 `deploy_local` 写 `deployed_models/`，在线 `RecallResourceManager` 从同挂载路径读，靠 `active.json` 指版本。本质是「文件传递」而非「网络调用」。

**Q3: 前端多阶段构建省了什么？**
> A: 最终镜像只含 `dist/` + Nginx，不含 Node.js 与 `node_modules` 等开发依赖，镜像体积与攻击面都小，生产更安全更快。

### 🔗 前后关联

- **11.3** 的 `make run-offline-pipeline` 即本节离线步骤的入口。
- **11.4** 在线服务经 `MODEL_DEPLOY_DIR` 卷加载本节部署的模型。
- **11.5** 前端经本节 Nginx 多阶段构建提供静态服务。
- **11.1** 的技术选型（PG/Redis/ES/Compose）在此落成运维配置。

---

## Practice Problems

Work through all problems in order — they get progressively harder. Each has a complete solution you can reveal after trying it yourself.

---

**Problem 11.6.1 — 容器通信** 🟢 Easy

后端 `DATABASE_URL` 写 `postgresql://funrec:funrec123@localhost:5432/funrec_db` 会怎样？正确写法是什么？

<details>
<summary>💡 Solution (click to reveal)</summary>

**答：** 容器内 `localhost` 指向后端自身，连不到 PostgreSQL，启动报连接拒绝。正确用服务名：`@postgres:5432`（Compose DNS 解析到 PG 容器）。

**Key points:**
- 容器网络内用服务名互访。
- localhost 在容器里是「自己」。

</details>

---

**Problem 11.6.2 — 数据持久化** 🟢 Easy

若不挂 `postgres_data` 命名卷，容器 `docker compose down` 后数据会怎样？挂卷后呢？

<details>
<summary>💡 Solution (click to reveal)</summary>

**答：** 不挂卷：容器文件系统随删而失，用户/电影/评分全丢，下次需重新 `ingest-data-to-database`。挂命名卷：数据存宿主机卷，容器删重建后数据仍在。

**Key points:**
- 有状态服务必须挂持久卷。
- 卷与容器生命周期解耦。

</details>

---

**Problem 11.6.3 — 启动顺序** 🟡 Medium

若跳过 `make run-offline-pipeline` 直接访问前端首页推荐，会发生什么？给出根因与最小修复步骤。

<details>
<summary>💡 Solution (click to reveal)</summary>

**答：** 后端 `/health` 可能 healthy（服务起了），但推荐 API 加载不到模型/物品向量 → `RecallResourceManager` 资源缺失，召回失败或返回空。根因：在线依赖离线产出的模型与特征。修复：`cd backend && uv sync && make run-offline-pipeline`，再 `make ingest-data-to-database` 与 `make index-movies-to-elasticsearch`。

**Key points:**
- 离线是「生产」、在线是「消费」，顺序不能反。
- 健康 ≠ 功能就绪，需验证资源存在。

</details>

---

**🏆 Challenge: 加一个缓存预热** 🔴 Hard

生产希望后端启动时主动把热门电影向量与高频用户画像预热进 Redis，减少首屏冷请求延迟。请基于本章组件，指出这一改动要动哪一层、需注意什么（150 字内）。

<details>
<summary>💡 Hint</summary>

改动在线服务启动钩子（如 `RecallResourceManager._ensure_resources_loaded` 后）：批量从 PG 读高频用户画像/历史写入 Redis，热门电影向量本就在 item_embeddings.npy 直接加载。注意：预热要在依赖的 PG/Redis 健康后做（depends_on + 重试），且只预热热点避免 Redis 膨胀；可用后台任务不阻塞启动。

</details>
