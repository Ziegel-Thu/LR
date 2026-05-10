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

- GPU 服务器：`ssh jiagpu45678`
- 大规模训练相关约定待后续确定，目前不要连接服务器

## 文献下载

当被要求下载论文时：
1. 从 arXiv 下载 TeX 源码（`https://arxiv.org/e-print/{id}`）
2. 解压到 `literature/作者-年份-关键词/` 子目录下
3. 会议官网论文优先找对应的 arXiv 版本
