# Future Attention and Expected Free Energy

## Chinese

### 项目简介

本仓库研究一个收缩后的核心问题：

在一个固定的 Transformer world model 中，是否可以用未来自注意力定义一个 Future Attention (FA) 信号，并检验它与 Expected Free Energy (EFE) 在动作排序上的关系。

当前版本不主张 `FA = EFE`。更准确的目标是：

- 用真实 world model 的未来 self-attention 定义 `oracle FA`
- 训练一个轻量 `FA predictor` 去近似该信号
- 在可精确计算 `Exact EFE` 的小型 POMDP 中比较两者的动作选择
- 研究闭环更新下，FA 是否能形成有功能的行为，以及何时会发生自我带偏

### 当前主实验

主实验脚本：

- `real_attention_fa_experiment.py`

该脚本包含：

- 固定的 causal Transformer world model
- 从真实未来自注意力中抽取 `oracle FA`
- 学习型 `FA predictor`
- `Exact EFE` 基线
- 闭环实验：`policy -> data -> oracle FA -> predictor -> new policy`

### 实验环境

实验不是语言模型任务，而是一个小型稳态 POMDP：

- 隐状态：资源点 `rich / barren`
- 观测：`(energy, cue)`
- 动作：`eat / inspect / move / wait`
- 偏好：维持能量处于安全区间
- 可见性：`full / partial / hard_partial`

设计这个环境的原因是：

- 可以精确计算 `Exact EFE`
- 可以显式控制部分可观测性
- 可以分析信息采集动作 `inspect` 的作用

### 当前结论边界

当前代码和结果支持以下较稳妥结论：

- 在固定小型 Transformer world model 上，未来 self-attention 可以定义 `oracle FA`
- `FA predictor` 可以学习该信号
- 在全可见条件下，FA 与 `Exact EFE` 存在中等对齐
- 在部分可观测条件下，这种对齐会显著受 rollout policy 和闭环分布漂移影响

当前代码不支持以下强结论：

- `FA = EFE`
- 真实大模型中的 future self-attention 必然等价于自由能优化
- 闭环更新一定形成稳定功能

### 当前结果摘要

当前这版主结果见：

- `full_current_rollout_2seeds.md`
- `partial_current_rollout_2seeds.md`
- `partial_entropy_rollout_2seeds.md`

代表性结果：

- `full` 条件下，`oracle FA vs Exact EFE` 约为 `70.31±8.85%`
- `partial` 条件下，`oracle FA vs Exact EFE` 约为 `54.17±7.29%`
- 在 `partial` 下改用固定 entropy-aware rollout policy，`predictor FA vs Exact EFE` 可到 `57.29±8.33%`，但行为回报下降

这意味着：

- FA 与 EFE 有非平凡对齐
- 但对齐并不稳健
- 闭环 rollout policy 会影响 oracle FA 本身

### 主要文件

- `real_attention_fa_experiment.py`: 真实 attention 主实验
- `FA_PAPER_EXPERIMENT.md`: 数学定义、论文边界、命题与实验方案
- `ATTENTION_FA_EXPERIMENT_GUIDE.md`: 运行说明
- `full_current_rollout_2seeds.json/.md`: 全可见主结果
- `partial_current_rollout_2seeds.json/.md`: 部分可观测主结果
- `partial_entropy_rollout_2seeds.json/.md`: 部分可观测 + 固定 entropy rollout 对照

### 运行方式

安装依赖：

```bash
pip install torch numpy
```

运行一个默认实验：

```bash
python real_attention_fa_experiment.py --visibility full --seeds 2 --rounds 2
```

运行部分可观测实验：

```bash
python real_attention_fa_experiment.py --visibility partial --seeds 2 --rounds 2 --oracle-rollout-policy current
```

运行固定 entropy rollout 对照：

```bash
python real_attention_fa_experiment.py --visibility partial --seeds 2 --rounds 2 --oracle-rollout-policy entropy
```

### 论文定位建议

基于当前结果，更合适的论文定位是：

- 机制验证型论文
- workshop / short paper / arXiv 预印本

更稳妥的主张应写成：

`attention-defined FA` 是一个由固定 world model 内部未来自注意力定义的内生信号；它可被学习，并在部分条件下与 `Exact EFE` 呈现有限但非平凡的动作排序对齐。

## English

### Overview

This repository studies a narrower and more defensible question:

Given a fixed Transformer world model, can future self-attention define a Future Attention (FA) signal, and how does that signal relate to Expected Free Energy (EFE) in action ranking?

The current project does not claim `FA = EFE`. Its actual goal is to:

- define `oracle FA` from real future self-attention in a world model
- train a lightweight `FA predictor` to approximate that signal
- compare FA-based action selection with `Exact EFE` in a small POMDP where EFE is tractable
- analyze whether FA yields useful behavior in closed loop, and when self-confirming drift appears

### Main Experiment

Main script:

- `real_attention_fa_experiment.py`

The script includes:

- a fixed causal Transformer world model
- extraction of `oracle FA` from real future self-attention
- a learned `FA predictor`
- an `Exact EFE` baseline
- a closed loop: `policy -> data -> oracle FA -> predictor -> new policy`

### Environment

The benchmark is not a language-model task. It is a small homeostatic POMDP with:

- hidden resource state: `rich / barren`
- observation: `(energy, cue)`
- actions: `eat / inspect / move / wait`
- preference structure: maintain energy within a safe regime
- visibility conditions: `full / partial / hard_partial`

This environment is used because it allows:

- exact EFE planning
- explicit control over partial observability
- direct analysis of epistemic actions such as `inspect`

### What the Current Results Support

The current code and results support the following claims:

- future self-attention in a fixed small Transformer world model can define `oracle FA`
- a learned `FA predictor` can approximate that signal
- under full observability, FA shows moderate alignment with `Exact EFE`
- under partial observability, alignment becomes strongly dependent on rollout policy and closed-loop distribution shift

The current results do not support:

- `FA = EFE`
- equivalence between future self-attention and free-energy optimization in general large models
- guaranteed stable function formation in closed loop

### Current Result Snapshot

Main result files:

- `full_current_rollout_2seeds.md`
- `partial_current_rollout_2seeds.md`
- `partial_entropy_rollout_2seeds.md`

Representative numbers:

- in `full`, `oracle FA vs Exact EFE` is about `70.31±8.85%`
- in `partial`, `oracle FA vs Exact EFE` is about `54.17±7.29%`
- in `partial` with a fixed entropy-aware rollout policy, `predictor FA vs Exact EFE` reaches `57.29±8.33%`, but behavioral return drops

Interpretation:

- FA is non-trivially aligned with EFE
- the alignment is not robust enough to justify an equivalence claim
- rollout policy affects oracle FA itself under partial observability

### Key Files

- `real_attention_fa_experiment.py`: real-attention main experiment
- `FA_PAPER_EXPERIMENT.md`: formal definitions, theory boundary, propositions, and paper framing
- `ATTENTION_FA_EXPERIMENT_GUIDE.md`: run guide
- `full_current_rollout_2seeds.json/.md`: full-observability results
- `partial_current_rollout_2seeds.json/.md`: partial-observability results
- `partial_entropy_rollout_2seeds.json/.md`: partial-observability entropy-rollout control

### Quick Start

Install dependencies:

```bash
pip install torch numpy
```

Run a default full-observability experiment:

```bash
python real_attention_fa_experiment.py --visibility full --seeds 2 --rounds 2
```

Run a partial-observability experiment:

```bash
python real_attention_fa_experiment.py --visibility partial --seeds 2 --rounds 2 --oracle-rollout-policy current
```

Run the entropy-aware rollout control:

```bash
python real_attention_fa_experiment.py --visibility partial --seeds 2 --rounds 2 --oracle-rollout-policy entropy
```

### Suggested Paper Positioning

Given the current evidence, the most defensible publication format is:

- a mechanism-validation paper
- a workshop paper, short paper, or arXiv preprint

The safest core claim is:

`attention-defined FA` is an endogenous signal induced by future self-attention in a fixed world model; it is learnable and shows limited but non-trivial alignment with `Exact EFE` under some conditions.
