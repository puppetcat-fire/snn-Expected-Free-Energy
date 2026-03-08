# Technical Retrospective: From FA/EFE To Open Sparse Pulse-SNN Control

## Why This Document Exists

This document is a long-form technical整理，目的不是写一篇论文摘要，而是把这条技术线真正发生过的事情讲清楚：

- 我们最初想证明什么
- 为什么原来的 `FA ~= EFE` 叙事站不稳
- 每次代码改动到底是为了解决什么问题
- 是受了什么约束，才不得不这样改
- 每个版本的实验结果说明了什么
- 现在这套 SNN 脉冲开放版到底做到了哪一步，没做到哪一步

这份文档面向的读者是假设他没有参与过之前讨论，但需要快速看懂：

1. 这不是一堆零散 demo，而是一条连续演化的技术路线。
2. 当前最可信的主线已经从 `FA = EFE` 转成了 `signed predictive contribution for next-step required inputs`。
3. 代码里哪些东西是已经实现的，哪些还是理论目标，必须严格区分。

---

## 0. One-Page Executive Summary

如果只看一句话，这个项目现在最准确的定位是：

> 它不是一个“纯 future attention 自动实现 EFE”的系统，而是一个“稀疏、可重连、部分开放的脉冲式 SNN world model”，用对下一刻实际需要输入的预测作为基础目标，再用动作对未来正负需求节点的 signed contribution 来选择动作。

这条路线里已经走通的关键点有：

- 纯 `FA` 不足以支撑“存在控制”。
- 只奖励上升会学歪，必须同时建模正负贡献。
- 大模型、更长前瞻、RWKV 内状态，本身不会自动把“朝目标偏”变成“稳定闭环控制”。
- 回到 SNN 后，如果把输入拆成脉冲节点、把动作做成双指数时程、把有环贡献定义成有限时间展开后的反事实删去差值，那么这条路线开始出现比较稳定的正信号。
- 节点可以不再是固定全活跃；当前代码已经实现了“固定上限 + 动态激活/回收”的开放 hidden 池。
- 新增节点不再“旱涝保收”，而是必须通过对下一刻预测需求的重要性来维持自身存在。

但目前仍然不能主张的东西也很明确：

- 不能说纯 self-attention 本身就足以指导存在。
- 不能说这套东西已经等价于 EFE。
- 不能说已经优于标准 RL/MPC。
- 不能说已经实现真正无限开放结构。

当前最稳的 claim 是：

> 在一个小型、延迟动作、持续扰动的小棍高度维持任务中，开放稀疏脉冲 SNN 可以通过 signed predictive contribution 形成可工作的控制偏置；节点开放和两跳归因都开始表现出机制性价值，但系统仍是原型级验证，而不是定论级系统。

---

## 1. The Original Idea And Where It Broke

### 1.1 Original Intuition

最初的核心直觉不是“我要做一个 RL 系统”，也不是“我要直接算 EFE”，而是：

- 有一组计算单元负责拟合反复出现的输入模式。
- 每个节点只能使用过去已经激活过的节点信息。
- 行动节点不受普通输入节点的约束，它们代表系统可以主动施加的干预。
- 一个行动节点是否该被选，不看外部 reward，而看它对后续已知输入节点的预测到底贡献了多少。
- 为了避免病态自激活，必须同时有“只注意上升/贴近”的节点和“只注意下降/远离”的节点。
- 整个图应该是稀疏的，但结构空间必须保持开放。

这个想法在 `main` 分支对应的原始雏形其实已经在：

- `main92.py`
- `node.py`
- `doNode.py`

里出现了。

### 1.2 Why `FA = EFE` Was Too Strong

后面一度把这条线表述成“future attention 近似 EFE”，这是因为当时想借一个更成熟的外部框架去解释结果。但代码和实验很快证明，这样写不稳。

最典型的问题在 `real_attention_fa_experiment.py` 那条线：

- `oracle FA` 不是纯 attention，而是 `attention * gate`
- `gate` 里混入了外部偏好和信息增益
- world model 的离线数据里还混入了 exact/EFE 风格采样
- 部署策略里还显式加了 immediate utility

所以当时最重要的工作，不是继续吹“FA 就是 EFE”，而是做消融，把问题拆干净。

---

## 2. Chronology: What Was Changed, Why, And What We Learned

这一节按时间顺序，把主要实验线和代码改动讲清楚。

## 2.1 `real_attention_fa_experiment.py`: First Serious FA Dissection

### Why It Was Modified

这条线当时的真实目的，是回答一个非常尖锐的问题：

> 现在代码里所谓的 FA，到底是不是纯从模型内部拿出来的，还是其实混了外部 EFE 风格东西？

于是我们加了三类关键开关：

- `--oracle-gate-mode`
  用来把 `full / attention_only / preference_only / info_only` 拆开
- `--offline-policy`
  用来控制 world model 离线数据是否含有 exact/EFE 行为
- `--policy-base-weight`
  用来关掉动作选择里的 immediate utility，做 pure-FA policy

### Constraint That Forced The Change

约束不是工程上的，而是论文上的：

- 如果主张“FA 解决复杂环境里 EFE 难算的问题”
- 那训练数据、标签构造、部署策略里都不能偷偷吃 EFE

否则结论会塌。

### What The Code Change Did

这次改动的本质不是“提升性能”，而是“把污染源隔离出来”。

它回答了三个问题：

1. 纯 attention 标签还剩多少信号？
2. 去掉 EFE 风格离线数据后效果还在不在？
3. 纯 FA policy 是否能单独站住？

### What The Results Showed

关键结果文件：

- `partial_current_rollout_2seeds.json`
- `partial_pure_fa_clean_2seeds.json`
- `partial_attention_only_labels_2seeds.json`
- `partial_randomdata_fullgate_2seeds.json`

代表性结果：

- 原始 partial 主结果：
  - `avg_return = 2.45`
  - `survival = 25.83%`
  - `predictor_vs_exact = 53.13%`
- pure FA clean：
  - `avg_return = -1.96`
  - `survival = 3.33%`
  - `predictor_vs_exact = 38.54%`
  - `predictor_vs_oracle = 63.54%`

这里最关键的解释不是“纯 FA 完全没学会”，而是：

- predictor 其实学到了某种内部自洽的 attention-like 信号
- 但这个信号并不够任务相关
- 所以它不能单独支撑有效控制

### What We Learned

这一轮最大的结论是：

> pure attention 里可能有可学习信号，但这个信号本身不是价值，也不是存在控制。

也正因为这个结论，后面才会转向 signed homeostatic / signed predictive 这条线。

---

## 2.2 `signed_homeostatic_attention_demo.py`: Why Signedness Had To Be Added

### Why It Was Modified

前一轮已经暴露出一个根问题：

- 只看“未来是否更强影响当前动作”
- 不足以区分“这个影响是把系统带向可存活区，还是带离可存活区”

所以我们需要一个极简 demo，把这个逻辑单独钉住。

### Constraint That Forced The Change

当时有一个很重要的约束：

> 不能重新掉回复杂 reward shaping。

所以我们没有去写一堆手工奖励，而是只定义一个安全区间，然后问：

- 更接近区间是正
- 更远离区间是负

### What The Code Change Did

`signed_homeostatic_attention_demo.py` 做了三种控制规则：

- `rise_only_attention`
- `signed_homeostatic_attention`
- `exact_homeostatic_planner`

这里的 attention 还是合成的，不是学出来的 Transformer/SNN attention。它的作用是纯机制验证：

- 如果只有“上升”这个正项，会怎样？
- 如果改成“朝安全区靠近为正，偏离为负”，会怎样？

### What The Results Showed

结果文件：

- `signed_homeostatic_attention_demo_20s_1000e.json`

核心结果：

- `rise_only_attention`
  - `avg_return = -11.923`
  - `survival = 0%`
  - `safe_step_rate = 17.85%`
  - `agreement_with_exact = 33.33%`
- `signed_homeostatic_attention`
  - `avg_return = 24.514`
  - `survival = 100%`
  - `safe_step_rate = 86.19%`
  - `agreement_with_exact = 100%`

代表状态也很直观：

- `glucose = 3` 时都选 `eat`
- `glucose = 5` 时 rise-only 还选 `eat`，signed 选 `wait`
- `glucose = 7` 时 rise-only 还选 `eat`，signed 选 `insulin`

### What We Learned

这一轮几乎把后面的主线定死了：

> 不带负号的前瞻偏置会学歪；必须同时表示“支持存在”和“破坏存在”的贡献。

这也是后面所有 `signed_*` 版本的理论起点。

---

## 2.3 `stick_height_signed_fa_experiment.py`: From Toy Homeostasis To Learned Continuous Control

### Why It Was Modified

签名原则在血糖 toy demo 上成立，不够。接下来要回答的是：

> 如果换成连续控制，而且用学出来的 world model，这个 signed 原理还成立吗？

所以我们做了“小棍高度维持”环境。

### Constraint That Forced The Change

这里有两个约束：

1. 不能直接回到 reward engineering
2. 不能只在离散 toy POMDP 里讲故事

于是环境改成：

- 连续状态
- 左右动作
- 动作有延迟
- 目标是把高度维持在目标带附近

### What The Code Change Did

这个版本的核心设计是：

- 用 Transformer world model 预测未来状态
- 从 imagined future 里读取 future attention
- 对每个候选动作算 signed FA

当时最强的一版结果在：

- `stick_height_signed_fa_repeat_run1.json`

### What The Results Showed

这版结果其实很强，但也暴露了问题：

- nominal
  - `prediction_only in-band = 55.32%`
  - `signed_fa = 78.48%`
- stress
  - `prediction_only in-band = 39.58%`
  - `signed_fa = 67.45%`

但同时：

- `signed_fa` 的 survival 不如 planner
- 有时甚至低于 `prediction_only`

### What We Learned

这一轮最重要的教训是：

> signed FA 能显著提高“把高度打进目标带”的能力，但这还不是完整 viability。

也就是说，它更像：

- “朝目标偏”

而不是：

- “在目标附近稳定存在”

这会成为后面持续困扰模型的一个核心问题。

---

## 2.4 `stick_height_softmax_semantic_fa_experiment.py`: Why Toward/Away Tokens And Softmax Were Added

### Why It Was Modified

这里的修改直接来自两个理论要求：

1. 单次真实交互里，环境只会给出一个动作的真实结果
2. 行动必须保留自由度，不能总是 `argmax`

所以要把控制结构改成：

- 对每个候选动作单独做 imagined branch
- 分别计算分数
- 再用 `softmax` 采样动作

### Constraint That Forced The Change

这个修改受的不是性能约束，而是建模约束：

- 如果生命是自由的
- 那它不应该永远只选确定性最大值
- 单步环境又只会反馈一个动作分支

于是正确结构必须是：

- 未来分支分开算
- 最后动作分布再归一化

### What The Code Change Did

这版代码做了三件事：

1. 候选动作分支化
2. 显式引入 `toward-target` / `away-from-target` semantic token
3. 用 `softmax(beta * score)` 而不是纯 `argmax`

结果文件：

- `stick_height_softmax_semantic_fa_results.json`

### What The Results Showed

结构上它更贴近理论，但性能反而下降了。

这不是坏事，反而说明：

- 理论更接近原始设计
- 但控制器本身还不够强

换句话说，这版更“像你想做的东西”，但还不够“会做”。

### What We Learned

这一轮的核心结论是：

> 正确的结构形式和强控制性能，不会自动同时出现。

从此以后，我们开始更认真区分：

- “理论上更纯”
- “经验上更稳”

这两个目标。

---

## 2.5 CartPole Standardization: Why We Switched To A Standard Physics Benchmark

### Why It Was Modified

前面的 stick 环境虽然有用，但仍然是自定义环境。为了避免总被质疑“是不是环境太 toy”，我们把相同思路搬到了标准 CartPole 物理上：

- `cartpole_height_softmax_semantic_fa_experiment.py`
- `cartpole_centered_height_semantic_fa_experiment.py`

### Constraint That Forced The Change

这里的约束是“可比较性”：

- 需要一个别人更熟悉的物理系统
- 同时还要符合用户提出的视觉和任务偏好：
  - 小车始终在镜头中央
  - 环境持续运行
  - 而不是一越界就终止

后面又进一步把扰动从小高斯噪声改成“偶尔的大冲击”，因为这更直观：

- 平时系统在跑
- 偶尔被狠狠推一下
- 看它能不能把杆高拉回来

### What The Code Change Did

这条线后来发展出多个版本：

- 标准语义 token 版
- centered continuing 版
- impulse disturbance 版
- larger model sweep
- horizon sweep
- uncertainty-gated 版

### What The Results Showed

真正重要的结果不是“某次偶然赢了”，而是这些 sweep 的负结果：

1. horizon 从 `4 -> 8 -> 12`
   没有解决守不住高点的问题
2. 模型从 `64x2 -> 128x4`
   没有自动把控制做好
3. uncertainty-gated FA
   几乎没有效果，因为 base logits 的熵长期接近最大，gate 基本全开

其中一个典型文件：

- `cartpole_centered_height_semantic_fa_impulse_h4.json`

代表性结果：

- nominal
  - `prediction_only = 9.50%`
  - `signed_semantic_fa = 10.42%`
- stress
  - `prediction_only = 12.58%`
  - `signed_semantic_fa = 12.67%`

### What We Learned

这一轮最重要的结论非常关键：

> 仅仅增加模型能力、前瞻长度或 stochastic gating，不会自动产生稳定闭环控制。

这直接推翻了一个常见直觉：

> “只要模型越来越准，控制精度自然会无限提高。”

实验告诉我们，不会。因为：

- 预测更准，不等于控制律就自动出现
- “更接近目标”不等于“围绕目标稳定存在”

---

## 2.6 RWKV Attempt: Why Internal Recurrence Alone Was Not Enough

### Why It Was Modified

既然问题看起来像“缺内部状态”，那很自然会想到：

- 要不要试试 RWKV 这种带递推内状态的结构？

于是做了：

- `cartpole_centered_height_semantic_rwkv_experiment.py`

并且还把 RWKV 里的 influence 统计从粗糙 proxy 改成了更精确的 action-specific recurrence。

### Constraint That Forced The Change

这个修改的真正约束是理论检验：

> 如果问题主要是 Transformer 没有内状态，那么换成 RWKV 之后，控制应该明显改善。

### What The Code Change Did

做了两层改动：

1. 把 world model 改成 RWKV-style time-mix / channel-mix 递推结构
2. 把“过去动作被未来读了多少”改成 RWKV-native recurrence 统计，而不是简单 proxy

### What The Results Showed

关键文件：

- `cartpole_centered_height_semantic_rwkv_run2_exactinfluence.json`

结果非常直接：

- nominal
  - `prediction_only = 5.00%`
  - `signed_semantic_fa = 5.00%`
- stress
  - `prediction_only = 12.08%`
  - `signed_semantic_fa = 12.08%`

也就是说，RWKV 版里 signed FA 几乎不起额外作用。

### What We Learned

这一轮的结论非常重要：

> 内部递推状态本身不是病根的解药。

问题不是简单的“缺 RNN”，而是：

- 控制语义本身不对
- 闭环结构不够
- 贡献定义不够贴近“下一刻实际需要”

这也是为什么后来决定真正回到 SNN 本身，而不是继续在 Transformer/RWKV 外壳里抠细节。

---

## 2.7 Why We Returned To SNN

回到 SNN 不是“回退”，而是承认：

- 你真正要研究的东西，本来就更像 SNN
- 不是像 Transformer token 那样的显式 attention map
- 而是像循环脉冲系统那样的局部递推、局部竞争、局部可塑

更具体地说，SNN 更自然地满足：

- 内部状态真实存在
- 行动可以是时间展开的，不是点事件
- 有环是天然的，不是特例
- 节点和边可以保持稀疏
- 结构可以逐渐长出和回收

这一点在原始 `main92.py`、`node.py`、`doNode.py` 就已经埋下了。

---

## 3. Current SNN Line: What The Code Actually Implements

这一节讲当前真正该看的主线：

- `snn_cyclic_stick_height_control.py`
- `snn_pulse_stick_height_control.py`

## 3.1 `snn_cyclic_stick_height_control.py`: First Cyclic SNN Prototype

### Why It Was Created

这一步的目的，是把前面几条经验真正融合：

- 动作必须有双指数时程
- 图里有环不是问题
- 贡献要用 finite-horizon ablation 定义
- 控制要基于 signed prospective contribution

### Constraint That Forced The Change

最大约束有三个：

1. 目标不能继续定义成复杂 reward
2. 图里有环，不能用静态路径计数算贡献
3. 行动是时间展开的，所以不能把动作节点当成普通点输入

### What The Code Did

核心设计：

- 环境：小棍状态 + 左右动作双指数痕迹
- 模型：稀疏有环 SNN world model
- 输入：向量化 required observation
- 控制：对 `left/right` 做 imagined rollout
- 评分：用 signed prospective contribution
- 有环贡献：用“有限时间展开后的反事实删去差值”

### What The Results Showed

关键文件：

- `snn_cyclic_stick_height_control_run1.json`
- `snn_cyclic_stick_height_control_run2.json`

代表性结果：

- world model
  - `val_loss ≈ 0.0177`
  - `val_height_mae ≈ 0.0912`
- 控制
  - nominal：`prediction_only 19.06% -> signed_prospective 24.06%`
  - stress：`prediction_only 13.75% -> signed_prospective 18.75%`

可视化选出的单场景更明显：

- `prediction_only in-band = 19.44%`
- `signed_prospective = 43.06%`
- `planner = 73.61%`

### What We Learned

这一版说明：

> 一旦贡献定义从“未来 attention”转成“有环展开后的反事实 signed contribution”，SNN 版本开始出现稳定正信号。

但它还有两个明显不足：

- 输入还是向量，不是脉冲节点
- 双跳 latent 贡献还没接

所以还不够像原始设计。

---

## 3.2 `snn_pulse_stick_height_control.py`: Turning Inputs Into Actual Pulse Nodes

### Why It Was Modified

用户后来明确提出：

> 当前输入信息流应该拆成脉冲版本，而不是只是一坨连续向量。

这一步非常关键，因为它把“下一刻实际需要输入”从抽象向量改成了可解释的节点集合。

### Constraint That Forced The Change

约束很清楚：

- 既然理论主轴是节点、脉冲、局部贡献
- 输入就不应该继续是黑盒连续向量

### What The Code Change Did

当前脉冲编码把小棍状态编码成 20 维多热脉冲：

- 左/右侧
- 角度区间
- 角速度区间
- 左右动作 alpha 是否活跃/强
- 高度相对目标带的位置
- 一个 `stable` 节点

关键函数：

- `pulse_from_state`
- `positive_mass`
- `negative_mass`

当前 `positive_mass` 是：

- `in_band`
- `stable`
- `angle_target`
- `omega_zero`

当前 `negative_mass` 是：

- `far_low`
- `low_near`
- `high_near`
- `far_high`
- 极端角速度
- `angle_high`

所以现在的“正负约束”已经不是口头上的，而是编码成了具体的 required pulse 群。

### What The Results Showed

关键文件：

- `snn_pulse_stick_height_control_twohop_tuned.json`

结果：

- nominal
  - `prediction_only = 17.62%`
  - `signed_pulse = 27.17%`
  - `signed_twohop_pulse = 25.61%`
- stress
  - `prediction_only = 20.05%`
  - `signed_pulse = 25.26%`
  - `signed_twohop_pulse = 26.30%`

### What We Learned

这一轮说明：

> 把 required signal 显式拆成 pulse node，比向量版更贴近原始理论，而且控制上仍然保持正信号。

---

## 3.3 Two-Hop Credit: How Multi-Hop Was Added

### Why It Was Modified

用户后面明确提出：

> 双跳也应该通，而且理论上它应该沿用同一套无约束行为节点的训练逻辑。

### Constraint That Forced The Change

这里的约束是理论统一性：

- one-hop 和 two-hop 不能是两套不同算法
- 应该都是“未来 required prediction 对某个节点的依赖程度”

### What The Code Change Did

当前 `signed_twohop_pulse` 的实现是一个近似 two-hop 方案：

1. 先对当前动作做第一步 rollout
2. 找出这个动作最先激活出的 top-k latent 节点
3. 在 imagined rollout 里把这些 latent 节点静默
4. 比较未来 required pulses 少了多少正质量、多了多少负质量

也就是说，two-hop 不是单独的头，而是：

- 先找动作激活出的中介节点
- 再算这些中介节点对未来需求的额外贡献

### What The Results Showed

固定池版本：

- nominal：
  - `signed_pulse = 27.17%`
  - `signed_twohop_pulse = 25.61%`
- stress：
  - `signed_pulse = 25.26%`
  - `signed_twohop_pulse = 26.30%`

### What We Learned

two-hop 已经不是空想，代码里真的能算，也开始有用。  
但它目前还只是“保守增益”，不是大突破。

这也符合预期：

- 双跳本来就比一步更难估
- 现在只是 top-k latent ablation 的近似
- 还不是全图多跳归因

---

## 3.4 Open Nodes: Why Fixed Hidden Count Was No Longer Enough

### Why It Was Modified

后来我们又碰到一个更结构性的质疑：

> 如果节点数完全固定，那不就把表达上限也写死了吗？

这其实是个很大的哲学和工程问题：

- 生命式系统不应该是一开始就把所有结构都给全
- 也不应该新增节点之后永远不死

### Constraint That Forced The Change

约束有两个：

1. 不能做 `O(N^2)` 全连接开放图
2. 新节点必须靠预测重要性维持自身存在，而不是“生出来就旱涝保收”

### What The Code Change Did

当前开放机制是：

- hidden 上限 `72`
- 初始只激活 `40`
- 其余节点作为 reserve pool 休眠
- 每隔若干轮，根据验证损失和节点活跃度，做 birth/death

也就是说，它不是“真正无限建新参数”，而是：

> 固定上限 + 动态激活/回收

这是一个很重要的工程折中。

### First Open Version Results

文件：

- `snn_pulse_stick_height_control_open.json`

结果：

- 初始 active hidden：`40`
- 最终 active hidden：`44`
- `births = 8`
- `deaths = 4`

控制上：

- nominal
  - `prediction_only = 17.19%`
  - `signed_pulse = 23.87%`
  - `signed_twohop_pulse = 26.30%`
- stress
  - `prediction_only = 19.53%`
  - `signed_pulse = 27.86%`
  - `signed_twohop_pulse = 24.57%`

### What We Learned

这一版说明：

> 开放节点机制是可行的，而且不会自动毁掉控制。

但还差一步：

- 新生节点需要“靠贡献活下去”
- 不能只靠被激活过就混成永久节点

---

## 3.5 New Nodes Must Earn Survival: Importance-Based Birth/Death

### Why It Was Modified

这一步直接来自用户的一个非常关键的要求：

> 新的节点是为了越来越精确的预测被需要的，它必须提供一定的预测重要性，才能维持自身存在。不像初始节点一样旱涝保收。

这是目前整个开放机制里最接近“生命味道”的部分。

### Constraint That Forced The Change

这里的约束不是性能，而是机制正确性：

- 如果新节点不需要靠贡献活着
- 那开放结构就退化成“随便点亮一些多余单元”

这不符合最初设计。

### What The Code Change Did

当前实现里，每个节点新增了：

- `node_importance`
- `node_age`
- `NEW_NODE_PROBATION_EVENTS`

训练时，用：

- 节点的平均 trace 活跃度
- 乘以 decoder 对关键 required pulses 的读出强度

来更新 `node_importance`。

可以把它理解成：

> 这个节点的活动，究竟有没有真的进入“下一刻需要被预测好”的那部分读出。

之后的生死规则是：

- 新节点出生后有 probation
- probation 结束时，如果重要性还太低，就可以被回收
- 初始节点和新节点待遇不同

这正是“新节点不能白拿存在权”的实现。

### What The Results Showed

文件：

- `snn_pulse_stick_height_control_open_importance_small_v2.json`
- `snn_pulse_prediction_accuracy.json`
- `snn_pulse_stick_growth_visual_importance_small_v2.json`

这一版的关键数字：

- 初始 active hidden：`40`
- 最终 active hidden：`47`
- `births = 8`
- `deaths = 1`
- `mean_new_node_importance = 1.1656`

并且不是所有新节点都零贡献。后期确实出现了有明显 importance 的新增节点，例如：

- `0.5407`
- `1.4057`
- `1.5055`

这说明它们不是装饰品，而是在为预测服务。

### What The Results Showed On Prediction

同一版验证集上：

- 整体 pulse bit accuracy：`91.75%`
- 高度五分类 bucket accuracy：`70.05%`
- `in_band` 关键节点 bit accuracy：`82.32%`
- toward-related 节点准确率：`87.25%`

这意味着：

- 它对“下一刻哪些需求脉冲会出现”的预测已经不低
- 高度这件事上，它更会预测“在哪个区间”和“是否进带”
- 而不是精确连续高度

### What The Results Showed On Control

这一版更偏机制验证，不是当前最强控制版。但 stress 下仍有正信号：

- nominal
  - `prediction_only = 22.14%`
  - `signed_pulse = 23.21%`
  - `signed_twohop_pulse = 20.12%`
- stress
  - `prediction_only = 16.31%`
  - `signed_pulse = 22.74%`
  - `signed_twohop_pulse = 24.64%`

### What We Learned

这一轮的真正价值不是“绝对分数最高”，而是：

> 开放节点、importance 生存权、signed predictive contribution，这三者第一次被放到了同一个工作版本里。

---

## 4. Current Information Flow: How The System Actually Works

这是当前最值得别人真正看懂的部分。

## 4.1 Environment

当前环境不是标准 CartPole，而是小棍高度维持环境。

它有几个关键性质：

- 动作是 `left / right`
- 动作不是点作用，而是先通过双指数 alpha 痕迹展开
- 环境里有随机冲击，通常是偶尔出现的 impulse
- 目标是把高度维持在目标带附近

这里动作之所以必须保留“先增后减”的双指数形状，是因为：

- 对函数节点来说，只要能达到阈值就够
- 但对行动节点来说，它代表的是一个时间展开的干预

这也是为什么 action node 和普通 function/predictive node 不该用完全相同的动态核。

## 4.2 Pulse Encoding

当前 `pulse_from_state` 把连续状态编码成多热脉冲。

它编码的不是原始每个实数值，而是更接近“实际需要”的离散节点：

- 我当前在目标左边还是右边
- 角度偏低、目标附近还是偏高
- 角速度向左快、向左慢、接近零、向右慢、向右快
- 左右 alpha 痕迹是否活跃
- 高度是远低、近低、在带内、近高、远高
- 是否稳定

这一步非常重要，因为它把“下一刻实际需要预测的是什么”变成了一组可解释节点，而不是黑盒向量。

## 4.3 World Model

当前主模型 `PulseCyclicSNN` 是：

- 稀疏
- 有环
- 带内部状态
- 支持节点活跃集合变化

它的 hidden 里有：

- membrane
- spikes
- trace
- adapt

当前 required 目标是：

- 预测下一刻这些 pulse 节点哪些会激活

这里要特别说明一个现实约束：

> 当前代码还不是完全按“每个节点只用本地规则训练自己参数”的版本。

它现在仍然使用：

- `Adam`
- `loss.backward()`
- 通过时间展开的全局反向传播

去训练 world model 的权重。

也就是说：

- 理论上，我们想要更局部、更节点式的训练规则
- 但当前工作版为了先把机制跑通，仍然用了端到端 PyTorch 训练

这点一定要讲清楚，否则读者会误以为当前已经完全实现了“每个节点各训各的本地规则”。

## 4.4 Signed Predictive Contribution

当前动作评分不是 reward，也不是 EFE，而是：

```text
Score(a)
  = immediate pulse score
  + prospective positive mass
  - weighted prospective negative mass
  + optional latent two-hop contribution
```

更具体说，当前代码里有三层：

1. `branch_pos - neg * branch_neg`
   未来整条 imagined branch 本身的 signed 质量
2. `delta_pos - neg * delta_neg`
   “有这个动作”和“没这个动作”相比，多带来的 signed 质量
3. `latent_pos - neg * latent_neg`
   如果把该动作激活出的中介 latent 节点静默掉，会损失多少 signed 质量

所以现在的控制逻辑已经不是“谁的 attention 大”，而是：

> 谁能让未来 required pulse 预测更向正需求偏、远离负需求偏。

## 4.5 Beta As Accuracy-Controlled Inverse Temperature

这里有一个容易误会的点。

当前系统不是“什么时候让前瞻干预更强”，而是：

- 前瞻分数本来就是动作分数本体的一部分
- 真正变化的是 softmax 的锐度

当前实现里：

```text
pi(a|h) = softmax(beta_t * Score(a))
```

而 `beta_t` 是由近期预测误差控制的。

概念上应理解成：

- 预测越准 -> inverse temperature 越高 -> 动作更果断
- 预测越不准 -> inverse temperature 越低 -> 分布更平 -> 更探索

这和普通“reward exploration”不一样，它更像：

> 对未来需求贡献还没摸清楚时，就保持探索；摸清楚后，再更果断地执行。

## 4.6 Cycles: How Contribution Is Calculated In A Loopy Graph

这是整个系统里最容易卡住的点之一。

问题是：

> 图里有环，那某个动作或节点对未来到底贡献了几次？怎么记账？

当前回答非常明确：

- 不做静态路径计数
- 不问这条边在图里绕了几圈
- 只看有限时间展开后的反事实差值

也就是说，对当前动作或节点 `z_t`：

1. 跑一遍完整 rollout
2. 再跑一遍把 `z_t` 拿掉/静默后的 rollout
3. 看未来 `H` 步 required pulse 预测差了多少

公式上就是：

```text
Contrib(z_t; H) = NeedLoss(mask z_t) - NeedLoss(full)
```

这个定义直接把环吃进时间展开里，所以：

> “图里有环”不会让归因失效，只是意味着归因必须定义在展开轨迹上，而不是静态拓扑上。

## 4.7 Sparse But Open

当前系统的结构原则可以压成一句：

> computation is sparse, structure remains open

具体来说：

- 不是全连接
- 不是所有 hidden 节点都一直活着
- 不是所有 reserve 节点都一开始就开
- 会 prune 弱边
- 会 grow 新边
- 会 birth 节点
- 会 death 节点

但又不是完全无限开放：

- 当前 hidden 参数上限是固定的
- 新节点来自 reserve pool
- 不是运行时真正 malloc 新参数

所以当前最准确的表述是：

> 这是一个“固定上限但开放激活集合”的结构，不是真正无上限的生长系统。

---

## 5. What Each Major Code File Was For

下面这张表，是为了让别人快速对齐每个文件的角色。

| File | Why It Was Added/Modified | Constraint | Main Design Change | Main Lesson |
| --- | --- | --- | --- | --- |
| `real_attention_fa_experiment.py` | 拆纯 FA 与外部门控/EFE 污染 | 论文不能把混合系统写成 pure FA | 增加 `oracle_gate_mode`、`offline_policy`、`policy_base_weight` 消融 | pure FA 信号可学，但不够控制 |
| `signed_homeostatic_attention_demo.py` | 单独验证 signed 原理 | 不能掉回复杂 reward shaping | 只定义安全区间，比较 rise-only vs signed | 只奖励上升一定学歪 |
| `stick_height_signed_fa_experiment.py` | 把 signed 原理搬到 learned continuous control | 需要连续控制验证 | Transformer world model + signed FA | 会“朝目标偏”，但未必“稳定存在” |
| `stick_height_softmax_semantic_fa_experiment.py` | 引入动作自由与概率选择 | 单次只执行一个动作，需 softmax 探索 | 候选动作分支 + toward/away token + softmax | 结构更纯，但控制还弱 |
| `cartpole_*_semantic_fa_*.py` | 放到标准物理并做更直观扰动 | 需要更标准、更直观 benchmark | centered continuing + impulse disturbance | 大模型/长前瞻不自动解决控制 |
| `cartpole_centered_height_semantic_rwkv_experiment.py` | 测试递推内状态是否是关键缺口 | 如果缺内状态，RWKV 应显著改善 | RWKV-style world model + exact influence recurrence | 内状态本身不是解药 |
| `snn_cyclic_stick_height_control.py` | 回到 SNN，处理双指数动作和有环归因 | 图有环，动作是时程干预 | 稀疏循环 SNN + finite-horizon ablation contribution | 这条路开始出现稳定正信号 |
| `snn_pulse_stick_height_control.py` | 把输入拆成真正 pulse node，并支持 two-hop/open nodes | 理论上需要节点级 required input 和开放结构 | pulse encoding + signed pulse mass + two-hop + active hidden pool | 最接近原始设计的工作版 |
| `snn_pulse_stick_growth_visualize.py` | 解释开放节点到底发生了什么 | 不只是说会长会死，要看历史 | 画 val loss、active nodes、birth/death、importance | 新节点不是全都无意义 |
| `snn_pulse_stick_control_visualize.py` | 解释控制行为而不是只看表格 | 需要直观案例 | 同场景对比 prediction/signed/twohop/planner | signed two-hop 在选中场景里确实更稳 |

---

## 6. Quantitative Summary Across The Whole Project

为了方便别人快速抓住关键趋势，这里只保留最重要的数字。

## 6.1 Pure FA Dissection

文件：

- `partial_current_rollout_2seeds.json`
- `partial_pure_fa_clean_2seeds.json`

| Setting | Avg Return | Survival | Predictor vs Exact |
| --- | ---: | ---: | ---: |
| Original partial FA | 2.45 | 25.83% | 53.13% |
| Pure FA clean | -1.96 | 3.33% | 38.54% |

解释：

- 纯 FA 不是完全无信号
- 但不能单独支撑控制

## 6.2 Signed Homeostatic Principle

文件：

- `signed_homeostatic_attention_demo_20s_1000e.json`

| Setting | Avg Return | Survival | Safe-Step | Agreement With Exact |
| --- | ---: | ---: | ---: | ---: |
| Rise-only | -11.923 | 0% | 17.85% | 33.33% |
| Signed homeostatic | 24.514 | 100% | 86.19% | 100% |

解释：

- 正负号不是锦上添花，而是必要条件

## 6.3 Learned Stick Control With Transformer

文件：

- `stick_height_signed_fa_repeat_run1.json`

| Setting | Nominal In-Band | Stress In-Band |
| --- | ---: | ---: |
| Prediction only | 55.32% | 39.58% |
| Signed FA | 78.48% | 67.45% |

解释：

- 会“朝目标偏”
- 但还没把整体 viability 一起学进去

## 6.4 CartPole Standardization

文件：

- `cartpole_centered_height_semantic_fa_impulse_h4.json`
- `cartpole_centered_height_semantic_fa_impulse_big128x4.json`
- `cartpole_centered_height_semantic_rwkv_run2_exactinfluence.json`

结论不在于某个数字高低，而在于三个负结论：

1. horizon 拉长没解决问题
2. 模型放大没解决问题
3. RWKV 内状态没解决问题

解释：

- 问题的根不是“模型太小”
- 而是控制语义和闭环形式不对

## 6.5 Cyclic SNN Vector Prototype

文件：

- `snn_cyclic_stick_height_control_run2.json`

| Setting | Nominal In-Band | Stress In-Band |
| --- | ---: | ---: |
| Prediction only | 19.06% | 13.75% |
| Signed prospective | 24.06% | 18.75% |

解释：

- SNN + cyclic ablation contribution 已经能带来稳定正增益

## 6.6 Pulse SNN With Two-Hop

文件：

- `snn_pulse_stick_height_control_twohop_tuned.json`

| Setting | Nominal In-Band | Stress In-Band |
| --- | ---: | ---: |
| Prediction only | 17.62% | 20.05% |
| Signed pulse | 27.17% | 25.26% |
| Signed two-hop pulse | 25.61% | 26.30% |

解释：

- 脉冲 required node 表达是通的
- two-hop 已经开始有用

## 6.7 Open Nodes + Importance Survival

文件：

- `snn_pulse_stick_height_control_open_importance_small_v2.json`
- `snn_pulse_prediction_accuracy.json`

结构结果：

- initial active hidden = `40`
- final active hidden = `47`
- births = `8`
- deaths = `1`

预测结果：

- pulse bit accuracy = `91.75%`
- height bucket accuracy = `70.05%`
- in-band bit accuracy = `82.32%`

stress 控制结果：

- prediction_only = `16.31%`
- signed_pulse = `22.74%`
- signed_twohop_pulse = `24.64%`

解释：

- 开放节点不是空话，真的发生了
- 新生节点开始靠 predictive importance 维持存在

---

## 7. What The Current Design Supports, And What It Does Not

## 7.1 What It Supports

当前最稳的结论是：

1. 只用“未来影响”是不够的，signed positive/negative split 是必要的。
2. 在有环图里，动作和节点的贡献可以通过 finite-horizon ablation 定义，而不需要静态路径计数。
3. 将输入拆成 pulse node 后，`next actual need` 变成了更可解释的 required-node prediction 问题。
4. 稀疏循环 SNN 能把 signed predictive contribution 变成可工作的控制偏置。
5. 两跳 latent 贡献已经可以用同一原则近似计算。
6. 节点开放机制已经从“口头设想”变成“能 birth / death / importance probation 的实现”。

## 7.2 What It Does Not Support

当前还不能严肃主张：

1. 纯 FA/self-attention 本身足以指导存在。
2. 当前系统已经等价于 EFE。
3. 当前系统已经优于 RL/MPC。
4. 当前系统已经实现真正无限开放结构。
5. 当前系统已经实现完全局部、每节点独立的训练规则。

最后这一点尤其重要。

当前工作版虽然结构上更像“每个节点各训各的”，但实际权重训练仍然是：

- 端到端
- BPTT
- Adam

所以更准确的说法是：

> 当前已经实现了“结构开放、脉冲表征、signed contribution、有限时间归因”的原型，但还没有实现完全局部的生命式训练法则。

---

## 8. The Most Important Design Distinctions To Remember

如果以后别人接手这个项目，我最建议他先记住以下区分。

### 8.1 Attention/Contribution Is Not Value

未来会回过头来“用到你”，不等于“这件事就是好的”。

所以必须有：

- 正贡献
- 负贡献

而不能只有一个单向“谁影响大谁赢”。

### 8.2 Predictive Accuracy Is Not Control Automatically

世界模型更准，不代表稳定控制律就自动出现。

必须同时有：

- 正确的 required signal 定义
- 正确的 signed contribution 定义
- 正确的 action competition / softmax 结构

### 8.3 Action Nodes And Function Nodes Are Not The Same

函数节点可以只是证据积累器。  
动作节点不行。

动作节点必须保留：

- rise
- peak
- decay

也就是双指数/alpha 核式的时程。

### 8.4 Cycles Are Not The Enemy

图里有环不是 bug。  
真正的难点只是：

- 不要用静态路径计数归因
- 要用时间展开的反事实归因

### 8.5 Open Structure Does Not Mean Dense Structure

这套系统的正确方向不是全连接，而是：

- 当前计算稀疏
- 当前前沿稀疏
- 长期结构开放

也就是：

> 稀疏运行，开放生长

---

## 9. What The Current Best Reading Of The Project Is

如果现在要给别人一句最稳的项目描述，我会写成：

> This project studies a sparse, cyclic, partially open pulse-node SNN that learns to predict next-step required inputs and chooses actions by their signed contribution to future required-node prediction. The current code shows that signed predictive contribution, two-hop latent credit, and importance-based node survival can all produce useful control bias in a delayed stick-height maintenance task, but the system remains a prototype and has not yet reached full locally trained open-ended viability control.

翻成中文就是：

> 这个项目研究的是一种稀疏、有环、部分开放的脉冲节点 SNN。它把“预测下一刻实际需要输入”作为基础目标，再按动作对未来 required 节点预测的 signed contribution 来选动作。当前代码已经证明 signed predictive contribution、两跳潜节点信用和基于预测重要性的节点生存机制都能在延迟小棍高度维持任务中产生有用控制偏置，但系统仍然是原型，而不是完整的局部训练开放式存在控制器。

---

## 10. Recommended Reading Order For New Contributors

如果别人第一次接这个仓库，我建议按这个顺序看：

1. `SNN_PREDICTIVE_ACTION_ARCHITECTURE.md`
   先把理论主轴看懂
2. `snn_pulse_stick_height_control.py`
   看当前主实现
3. `snn_pulse_prediction_accuracy.json`
   看当前 world model 到底学到了什么
4. `snn_pulse_stick_control_visual.json`
   看控制效果而不是只看 aggregate
5. `snn_pulse_stick_growth_visual_importance_small_v2.json`
   看开放节点和 importance 生死逻辑
6. `real_attention_fa_experiment.py` 和对应结果
   再回头理解为什么早期 `FA ~= EFE` 叙事不够稳

---

## 11. Final Bottom Line

这条项目线真正成熟起来，是从它不再硬说“FA 就是 EFE”开始的。

它现在最值得继续推进的，不是继续把模型做大，而是继续把这几个机制做实：

- required-node definition
- signed contribution
- cyclic ablation credit
- two-hop latent credit
- importance-based node survival
- 更局部化的训练法则

也就是说，真正的主线已经不是“更像大模型”，而是：

> 更像一个稀疏、开放、能靠预测需求维持自身结构与行动的脉冲系统。

