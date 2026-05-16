# 项目约定

## 项目概况

表征学习（Representation Learning）方向的学术研究项目。
语言风格：中文描述为主，术语用英文。

## 目录结构

- `literature/` — 文献阅读笔记
- `experiments/` — 实验计划与结果
- `deepresearch/` — deep research 的 prompt、response 与整理
- `src/` — 实验代码（Python + PyTorch）
- `PLAN.md` — 研究计划
- `PROGRESS.md` — 进展记录

## 文献笔记

- 每篇文献一个 Markdown 文件，放在 `literature/` 下
- 文件名格式：`作者-年份-关键词.md`
  - 例：`bengio-2013-representation.md`
- 内容应包含：论文核心思想、方法要点、与本研究的关联

## 实验流程

1. 先写 plan，再跑实验，跑完写 result
2. 实验在 tmux 中运行

### 复杂实验的委派

对于复杂实验，可以新开一个 tmux session，在里面启动 `copilot --allow-all` 并发送 prompt 让它执行。流程：
1. 草拟要发送的 prompt
2. **先展示给用户审核**，确认后才发送
3. prompt 本身不需要存档

### 目录结构

每次实验一个子目录，放在 `experiments/` 下：

```
experiments/exp-001-baseline/
├── plan.md          ← 目的、假设、方法/配置
├── result.md        ← 结果、分析
└── scripts/         ← 该实验专用的代码
```

序号三位数补零。

### 代码划分

- `src/` — 公用代码（模型定义、数据处理、工具函数等）
- `experiments/exp-NNN/scripts/` — 该实验专用的脚本（训练、评估等）
- 保持可复现：关键参数写入配置而非硬编码

## Git

- 及时 commit，不要攒一大堆改动
- 每个 agent 只 `git add` 并 commit 自己负责的文件，避免冲突
- 修改项目规范时先沟通，达成一致后再更新此文件

## Deep Research

每次 deep research 一个子目录，放在 `deepresearch/` 下：

```
deepresearch/dr-001-xxx/
├── prompt.md        ← 发出的 prompt
└── response.md      ← 收到的 response
```

职责：
- 关注和整理每次 deep research 的内容，提炼关键信息
- 需要时协助撰写 deep research 的 prompt

## 服务器

### 集群概况

8 台共享存储的 GPU 节点（jiagpu1-8），可用节点为 **jiagpu4-8**（1-3 无 SSH 权限）。

| 节点 | GPU | 显存/卡 |
|------|-----|---------|
| jiagpu4 | 8× NVIDIA A40 | 45GB |
| jiagpu5 | 8× NVIDIA A40 | 45GB |
| jiagpu6 | 8× NVIDIA A40 | 48GB |
| jiagpu7 | 8× NVIDIA A40 | 48GB |
| jiagpu8 | 8× NVIDIA A40 | 48GB |

- 当前工作节点：jiagpu6
- SSH 到其他节点：`ssh jiagpu{4,5,7,8}`
- 可以在空闲节点上跑实验

### 存储

- **beegfs**（`/beegfs_hdd/`）：共享存储，684T，适合大文件和数据同步，**不要放大量小文件**
- **NVMe SSD**（`/nvmessd/`）：本地高速存储，14T，适合放小文件、venv、缓存等
  - 项目 SSD 目录：`/nvmessd/lifanhong/`
- 实验代码和数据放 beegfs 同步，运行时的临时文件/缓存放 SSD

### 环境

- jiagpu4 Python venv: `/nvmessd/lifanhong/LR-env/venv`
  - 激活: `source /nvmessd/lifanhong/LR-env/venv/bin/activate`
  - 包含: torch 2.12, transformers, sae-lens, transformer-lens, hqq
  - HF 模型缓存: `/nvmessd/lifanhong/LR-env/cache/hf`
  - HuggingFace 已登录（支持 gated model 如 Gemma-2）
- 其他节点需要时再搭建环境

### 实验运行

- 实验在 tmux 中运行，session 名用实验名（如 `exp002`）
- 可并行：用 `CUDA_VISIBLE_DEVICES` 指定不同 GPU 对，同时跑多个 bit-width/配置
- 日志用 `tee` 写到 SSD：`/nvmessd/lifanhong/LR-env/`

## 文献下载

当被要求下载论文时：
1. 从 arXiv 下载 TeX 源码（`https://arxiv.org/e-print/{id}`）
2. 解压到 `literature/作者-年份-关键词/` 子目录下
3. 会议官网论文优先找对应的 arXiv 版本
