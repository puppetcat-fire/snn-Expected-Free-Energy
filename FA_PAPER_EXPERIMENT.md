# FA 与 EFE 的关系、论文边界与可发表实验方案

## 1. 先给结论

目前仓库里的 [fa_pomdp_validation.py](C:\Users\xiaob\Documents\snn-Expected-Free-Energy\fa_pomdp_validation.py) **不能直接作为正式论文主实验**，原因不是结果没信息量，而是它验证的对象与您真正定义的 FA 不完全一致。

更准确地说：

- **FA 不是拟合 EFE 的模型**。
- **FA 是对“未来自注意力如何回指当前行动/节点”的预测模型**。
- EFE 在论文里应该作为一个**外部理论基准/对照准则**，而不是 FA 的监督标签。

因此，论文主命题不能写成：

> FA 学会了 EFE。

更稳妥、也更容易被审稿人接受的命题是：

> FA 是一个由世界模型内部未来自注意力定义的内生量；在某些结构条件下，它与 EFE 的动作排序呈现可证明的一致性边界，因此可以作为低计算成本的工程近似启发式。

---

## 2. 当前实验能支持什么，不能支持什么

### 2.1 能支持的

当前结果可以支持以下较弱结论：

1. 在小型可控 POMDP 中，某类“前瞻式代理分数”可以在动作选择上接近 Exact EFE。
2. 在全可见或中等部分可观测条件下，低成本代理在行为上可能逼近 Exact EFE。
3. 与显式前向规划相比，这种代理信号在工程上有明显复杂度优势。

### 2.2 不能支持的

当前结果**不能**直接支持以下强结论：

1. FA 本身就是 EFE 的近似计算。
2. 未来自注意力在一般情形下等价于主动推理中的期望自由能。
3. 现有脚本已经严格验证了“FA 来源于世界模型未来自注意力”这一点。

核心原因有三条：

1. 当前脚本中的 `FA` 更像手工设计的启发式代理，而不是由世界模型未来注意力直接定义的量。
2. 还没有把“FA 的监督目标”严格限定为未来自注意力标签。
3. 还没有给出“FA 与 EFE 一致时的充分条件、不一致时的偏差来源和误差界”。

---

## 3. 论文里应该如何正确定义 FA

设基础世界模型为 \(M_\theta\)。在决策时刻 \(t\)，当前上下文/隐状态记为 \(h_t\)，候选行动为 \(a\in\mathcal A\)。

令

- \(\tau_{t+1:t+H}\sim q_\theta(\tau\mid h_t,a)\) 表示在固定当前状态并采取候选行动 \(a\) 后，世界模型对未来 \(H\) 步轨迹的预测分布。
- \(\alpha^{(\ell,h)}_{u\leftarrow v}(\tau)\) 表示轨迹 \(\tau\) 下，第 \(\ell\) 层第 \(h\) 个注意力头中，位置 \(u\) 对位置 \(v\) 的注意力权重。
- \(c(x_u)\in[0,1]\) 表示位置 \(u\) 的未来状态/未来 token 是否属于偏好相关事件，或其偏好强度。

定义未来注意力汇总核：

\[
\bar\alpha_{t+k\rightarrow t}(\tau)
= \sum_{\ell=1}^{L_{att}}\sum_{h=1}^{H_{att}} w_{\ell h}\,\alpha^{(\ell,h)}_{t+k\leftarrow t}(\tau),
\qquad
w_{\ell h}\ge 0,\ \sum_{\ell,h}w_{\ell h}=1.
\]

则 **FA 的原始定义** 应为：

\[
F_H(h_t,a)
=
\mathbb E_{q_\theta(\tau\mid h_t,a)}
\Big[
\sum_{k=1}^{H} \beta^{k-1}\, c(x_{t+k})\, \bar\alpha_{t+k\rightarrow t}(\tau)
\Big].
\]

这里：

- \(c(x_{t+k})\) 控制“未来什么事件值得被算作偏好相关”；
- \(\bar\alpha_{t+k\rightarrow t}\) 则刻画“这些未来事件在多大程度上回指当前行动节点”；
- \(\beta\in(0,1]\) 是时域折扣。

这一定义非常重要，因为它明确表明：

- FA 是由**世界模型内部的未来自注意力结构**决定的；
- 它不是从 EFE 公式直接回归出来的；
- 它本质上是一个 **preference-weighted future influence functional**。

---

## 4. EFE 作为外部比较基准的定义

在离散 POMDP 中，记隐藏状态为 \(s_t\)，观测为 \(o_t\)，belief 为 \(b_t(s)=q(s_t=s)\)。

令偏好分布为 \(p_C(o)\)。有限视野 \(H\) 下，可将截断 EFE 写为：

\[
G_H(b_t,a)
=
\mathbb E_{q(o_{t+1}\mid b_t,a)}[-\log p_C(o_{t+1})]
- \lambda_{ep}\, I_q(s_{t+1};o_{t+1}\mid b_t,a)
+ \gamma\, \mathbb E_{q(o_{t+1}\mid b_t,a)}V_{H-1}(b_{t+1}),
\]

其中

\[
V_H(b)=\min_{a\in\mathcal A} G_H(b,a).
\]

这一定义说明：

- EFE 同时包含 **pragmatic / risk** 项，和 **epistemic / information gain** 项；
- FA 只要是“未来注意力函数”，它天然更接近某种 **偏好加权影响量**；
- 因此，FA 最多在某些条件下与 EFE 的一部分排序一致，而不能先验等同于完整 EFE。

---

## 5. 论文里最关键的数学命题

下面给出一个适合写进论文主文或附录的命题。这个命题不宣称 FA = EFE，而是给出“何时两者动作排序会一致”的严格条件。

### 命题 1：FA 与 EFE 的动作一致性界

设在某一决策时刻 \(t\)，对所有动作 \(a\in\mathcal A\)，存在常数 \(\mu\in\mathbb R\)、\(\kappa>0\)、\(\varepsilon_{align}\ge 0\)，使得

\[
\big|G_H(b_t,a) - (\mu - \kappa F_H(h_t,a))\big| \le \varepsilon_{align}.
\]

再设一个预测器 \(\hat F_\phi(h_t,a)\) 满足统一逼近误差

\[
\big|\hat F_\phi(h_t,a)-F_H(h_t,a)\big|\le \varepsilon_{pred}, \qquad \forall a\in\mathcal A.
\]

记 EFE 最优动作为

\[
a^* = \arg\min_{a\in\mathcal A} G_H(b_t,a),
\]

并记 EFE 的最优间隔为

\[
\Delta_t
= 
\min_{a\neq a^*}\big(G_H(b_t,a)-G_H(b_t,a^*)\big).
\]

若

\[
\Delta_t > 2\varepsilon_{align} + 2\kappa\varepsilon_{pred},
\]

则由预测器诱导的 FA 动作

\[
\hat a = \arg\max_{a\in\mathcal A}\hat F_\phi(h_t,a)
\]

满足

\[
\hat a = a^*.
\]

### 证明

由对齐假设，对任意动作 \(a\)，有

\[
G_H(b_t,a) \ge \mu - \kappa F_H(h_t,a) - \varepsilon_{align},
\]

以及

\[
G_H(b_t,a) \le \mu - \kappa F_H(h_t,a) + \varepsilon_{align}.
\]

又由预测误差界，对任意动作 \(a\)，有

\[
F_H(h_t,a) \ge \hat F_\phi(h_t,a) - \varepsilon_{pred},
\qquad
F_H(h_t,a) \le \hat F_\phi(h_t,a) + \varepsilon_{pred}.
\]

因为 \(\hat a\) 最大化 \(\hat F_\phi\)，所以

\[
\hat F_\phi(h_t,\hat a) \ge \hat F_\phi(h_t,a^*).
\]

于是

\[
F_H(h_t,\hat a)
\ge
\hat F_\phi(h_t,\hat a)-\varepsilon_{pred}
\ge
\hat F_\phi(h_t,a^*)-\varepsilon_{pred}
\ge
F_H(h_t,a^*)-2\varepsilon_{pred}.
\]

因此

\[
G_H(b_t,\hat a)
\le
\mu - \kappa F_H(h_t,\hat a) + \varepsilon_{align}
\le
\mu - \kappa F_H(h_t,a^*) + 2\kappa\varepsilon_{pred} + \varepsilon_{align}.
\]

另一方面，

\[
G_H(b_t,a^*) \ge \mu - \kappa F_H(h_t,a^*) - \varepsilon_{align}.
\]

两式相减，得

\[
G_H(b_t,\hat a)-G_H(b_t,a^*)
\le
2\varepsilon_{align}+2\kappa\varepsilon_{pred}.
\]

若 \(\Delta_t > 2\varepsilon_{align} + 2\kappa\varepsilon_{pred}\)，则任何非最优动作都不可能达到该上界，因此 \(\hat a\) 必须等于 \(a^*\)。证毕。

### 这个命题的意义

这个命题足够重要，因为它给了你论文里最核心的一句严谨表述：

> 我们不主张 FA 等于 EFE；我们主张当 FA 与 EFE 在动作层面满足近似仿射对齐，且预测误差足够小于 EFE 的动作间隔时，FA 所诱导的动作选择与 EFE 一致。

这句话是可发表的，也是可证明的。

---

## 6. 可发表实验应该怎么设计

如果要做成正式论文，我建议把实验拆成三层。

### 实验 A：定义正确性的合成实验

目标：验证你定义的 FA 确实是“未来自注意力”，而不是别的 hand-crafted proxy。

具体做法：

1. 构造一个小型、完全可控的 action-conditioned world model。
2. 这个模型必须显式产生未来自注意力矩阵。
3. 对每个当前动作 \(a\)，用 rollout 直接估计真实的 \(F_H(h_t,a)\)。
4. 用一个小预测器 \(\hat F_\phi\) 去预测这个 \(F_H\)。
5. 监督标签只能来自 **future self-attention target**，绝不能来自 EFE。

这组实验回答的问题是：

> FA 作为一个独立对象，能不能被稳定学习？

### 实验 B：与 Exact EFE 的动作对比实验

目标：在 exact planning 可算的小 POMDP 上，比较两者的动作排序关系。

具体做法：

1. 构造一族小型 POMDP，保证 Exact EFE 能通过动态规划精确计算。
2. 用同一环境生成轨迹，训练 world model 和 FA predictor。
3. 对同一组 belief state，分别计算：
   - Exact EFE 排序 \(G_H(b,a)\)
   - FA 排序 \(\hat F_\phi(h,a)\)
4. 比较：
   - Top-1 action agreement
   - Kendall's \(\tau\)
   - Pairwise ranking accuracy
   - Normalized EFE regret

这组实验回答的问题是：

> FA 作为未来注意力模型，与 EFE 的动作排序到底有多接近？

### 实验 C：工程价值实验

目标：验证 FA 的计算优势是否真有数量级收益。

具体做法：

1. 固定状态集合，逐渐增加：
   - horizon \(H\)
   - 动作数 \(|\mathcal A|\)
   - 观测分支数 \(|\mathcal O|\)
2. 分别测量：
   - Exact EFE 每次决策耗时
   - FA predictor 每次决策耗时
   - 两者内存占用
3. 报告同等任务上的 return / regret / agreement 与 runtime 的 tradeoff。

这组实验回答的问题是：

> 即使 FA 不是 EFE，它是否因为复杂度低很多而具有工程价值？

---

## 7. 最好发表的主张形式

如果您想要更容易发表，我建议不要把论文主张写成“主动推理新理论”，而应该写成下面这种形式：

> 我们提出一个由未来自注意力定义的前瞻式动作评分函数 FA。FA 不是 EFE 的监督拟合物，而是世界模型内部的内生量。我们在一族 exact-EFE 可求解的 POMDP 中发现，FA 在特定结构条件下与 EFE 呈现稳定的动作排序一致性，并能以远低于前向规划的计算成本实现相近的行为性能。

这个版本有几个好处：

1. 概念上是对的，不会被一句“你这根本不是 EFE”直接打穿。
2. 理论上是守住边界的，只宣称 ranking alignment，不宣称 identity。
3. 工程上是有卖点的，可以突出复杂度优势。

---

## 8. 复杂度分析应该怎么写

### Exact EFE

对于有限 horizon 的离散 POMDP，若显式枚举动作和观测分支，则 Exact EFE 的决策复杂度随 horizon 呈指数增长。粗略写法可表述为：

\[
T_{EFE}(H) = O\big((|\mathcal A||\mathcal O|)^H \cdot C_{belief}\big),
\]

其中 \(C_{belief}\) 是一次 belief 更新与状态转移的代价。

### FA predictor

若 FA predictor 是一个前馈网络或轻量头，其在线决策复杂度一般为

\[
T_{FA} = O(|\mathcal A|\cdot C_\phi).
\]

因此，对于固定模型大小，FA 在 horizon 增长时通常呈近似线性，而 Exact EFE 呈指数增长。

这正是你最有工程价值的卖点之一。

---

## 9. 一篇能投出去的最小实验包应该包含什么

如果要到“可以投稿”的最低标准，我建议至少包含：

1. **一个定义正确的 FA world model 实验**
2. **一个 exact-EFE 可算的小 POMDP 对比基准**
3. **一个多随机种子统计表（至少 10 seeds）**
4. **一个复杂度/耗时对比图**
5. **一个理论命题 + 证明**
6. **一个失败案例分析**

失败案例分析尤其重要。你应该主动展示：

- 在强部分可观测条件下，FA 为什么会偏离 EFE；
- 这种偏离究竟来自缺失 epistemic term，还是来自未来注意力本身无法表示某些 belief 更新。

这会让论文可信很多。

---

## 10. 对当前仓库工作的判断

当前仓库工作更适合作为：

- pilot study
- 附录中的 early evidence
- 用来帮你调出合适环境和指标的原型

但还不适合直接当主实验投稿。

真正能投稿的下一步，不是继续堆更多 heuristic 环境，而是：

1. 把 FA 的定义彻底独立出来；
2. 让监督目标只来自未来自注意力；
3. 把 EFE 放到 evaluation side 做 oracle baseline；
4. 用上面的命题把“为什么有时一致、有时不一致”说清楚。

---

## 11. 下一步建议

下一步最值得做的是：

1. 新建一个 **attention-defined FA** 的合成 world model 实验。
2. 在这个实验里，显式保存 rollout 产生的未来注意力标签。
3. 单独训练 \(\hat F_\phi\) 去预测这些标签。
4. 再把其动作选择与 Exact EFE 做对比。

如果这样做，论文结构会变得非常清楚：

- 第 1 部分：定义 FA
- 第 2 部分：证明 FA 与 EFE 对齐时的动作一致性界
- 第 3 部分：用 exact POMDP 做 ranking 验证
- 第 4 部分：做工程复杂度对比

这条线是可发表的。

---

## 12. 更深一层的问题：闭环迭代后会不会形成功能？

这个问题比“FA 是否接近 EFE”更根本。

因为一旦你把 FA 用于干预策略，系统就不再是静态比较，而变成一个**闭环迭代系统**：

1. 现有模型产生未来注意力分布；
2. 你用这些注意力训练或更新 FA predictor；
3. 你用 FA 去干预当前动作选择；
4. 干预后的策略会采集到新数据；
5. 新数据又会改变未来自注意力；
6. 于是目标本身也在变。

这说明问题已经不是“FA 是不是 EFE”，而是：

> 由未来注意力定义的内生引导信号，在闭环迭代下是否会收敛到一个自洽、稳定、功能性的策略/表示系统？

### 12.1 两种完全不同的“一致性”

必须区分两种一致性：

#### 一致性 A：静态动作排序一致性

在固定 world model、固定数据分布、固定 belief 的条件下，比较

- FA 的动作分数 \(F_H(h_t,a)\)
- Exact EFE 的动作分数 \(G_H(b_t,a)\)

这就是前面的命题 1 所讨论的问题。

#### 一致性 B：闭环自洽一致性

在干预-采样-再训练的闭环下，问系统是否会到达一个固定点：

- predictor 预测的 FA
- 实际 rollout 产生的 future attention
- 干预后的策略
- 由新策略诱导的数据分布

这四者是否会互相匹配。

论文里一定要把这两层分开。否则审稿人会立刻追问：

> 你证明的是单步 ranking alignment，还是整个闭环系统的收敛？

两者不是一回事。

### 12.2 闭环系统的形式化

记：

- \(M_{\theta_k}\)：第 \(k\) 轮的 world model
- \(\pi_k(a\mid h)\)：第 \(k\) 轮策略
- \(F_{\phi_k}(h,a)\)：第 \(k\) 轮 FA predictor
- \(d_{\pi_k}\)：策略 \(\pi_k\) 在环境中诱导的数据分布

定义四个算子：

1. **注意力标签生成算子**
\[
\mathcal A(M_{\theta_k},\pi_k)
\mapsto
Y_k,
\]
即从当前模型和当前策略生成 future-attention 标签。

2. **FA 拟合算子**
\[
\phi_{k+1} = \mathcal F(Y_k),
\]
即用当前标签训练得到新的 predictor。

3. **策略干预算子**
\[
\pi_{k+1}(a\mid h)
\propto
\pi_{\theta_k}(a\mid h)
\exp\big(\lambda \hat F_{\phi_{k+1}}(h,a)\big).
\]

4. **模型更新算子**
\[
\theta_{k+1} = \mathcal U(d_{\pi_{k+1}}).
\]

于是整个系统可写成：

\[
(M_{\theta_k}, \phi_k, \pi_k)
\xrightarrow{\mathcal A,\mathcal F,\mathcal I,\mathcal U}
(M_{\theta_{k+1}}, \phi_{k+1}, \pi_{k+1}).
\]

### 12.3 什么叫“形成功能”

如果存在一个固定点 \((\theta^*,\phi^*,\pi^*)\)，使得：

\[
Y^* = \mathcal A(M_{\theta^*},\pi^*),
\]
\[
\phi^* = \mathcal F(Y^*),
\]
\[
\pi^*(a\mid h) \propto \pi_{\theta^*}(a\mid h)\exp(\lambda \hat F_{\phi^*}(h,a)),
\]
\[
\theta^* = \mathcal U(d_{\pi^*}),
\]

并且该固定点在行为上持续维持某种偏好相关功能，那么就可以说：

> FA 闭环确实形成了功能性结构。

注意，这里的“功能”不是一句哲学话，而是一个可实验定义的对象：

- 稳定维持偏好变量
- 在新分布下仍保持行为效用
- 未来注意力与当前干预节点形成稳定回指模式

### 12.4 什么时候闭环会收敛，什么时候会坏掉

闭环不一定会变好，至少有三种可能：

#### 情形 1：收敛到稳定功能点

这是你最想要的情况。通常需要：

1. 干预强度 \(\lambda\) 不太大；
2. 每轮更新造成的策略 KL 改变受控；
3. world model 在新分布上的泛化没有崩掉；
4. future attention 对偏好功能确实提供稳定信用分配。

#### 情形 2：正反馈自激，形成注意力回音室

这也是非常可能的失败模式：

- predictor 偏爱某些模式；
- 干预让这些模式更常出现；
- 新数据进一步强化这些模式的 future attention；
- 最终模型学到的是“自我强化的显著模式”，而不是真正的偏好维持功能。

这相当于闭环里出现了 self-confirming bias。

#### 情形 3：部分可观测下陷入局部策略

当环境存在隐藏状态、需要信息增益时，future self-attention 可能主要刻画“未来会不会继续引用当前节点”，但并不显式包含 belief reduction 或 epistemic value。

这会导致：

- FA 闭环可能学会某种稳定习惯；
- 但这种习惯未必是主动推理意义上的最优探索行为；
- 于是它可能有功能，却不是 EFE 最优功能。

### 12.5 论文上最安全的理论说法

因此，在理论上最安全的表述不是：

> FA 闭环会自动实现主动推理。

而是：

> FA 闭环定义了一个由未来注意力驱动的自洽策略更新系统；其是否与 EFE 对齐，取决于未来注意力是否在该任务中充当了偏好维持与信息收集的有效信用分配信号。

这句话非常关键，因为它把问题从“理论等价”改成了“结构条件下的对齐与收敛”。

---

## 13. 这部分应该怎么做成论文实验

如果要把“迭代后是否形成功能”做成可发表实验，我建议新增一组 **closed-loop FA formation** 实验。

### 13.1 闭环实验协议

每一轮 \(k=0,1,2,\dots,K\)：

1. 用当前策略 \(\pi_k\) 采样新轨迹；
2. 用当前 world model 或重训后的 world model 计算 future-attention 标签；
3. 更新 FA predictor \(\phi_{k+1}\)；
4. 用 \(\phi_{k+1}\) 干预策略得到 \(\pi_{k+1}\)；
5. 测试行为功能指标。

### 13.2 需要报告的量

每轮都报告：

1. **FA 标签拟合误差**
2. **policy KL**：\(D_{KL}(\pi_{k+1}\|\pi_k)\)
3. **future attention 分布漂移**
4. **任务功能指标**：例如生存率、稳态维持率、延迟奖励成功率
5. **与 Exact EFE 的动作一致率**
6. **是否出现塌缩**：例如单一动作占比、注意力熵坍缩、观测多样性下降

### 13.3 这组实验真正回答的问题

这组实验不是回答“FA 是否等于 EFE”，而是回答：

1. FA 闭环是否收敛？
2. 收敛到的是否是功能性策略？
3. 这种功能性策略与 EFE 最优策略的关系是什么？

这是更强、也更有研究价值的问题。

---

## 14. 你现在最值得写进论文的一句判断

如果用一句最准确的话概括这个新问题，我建议写成：

> FA 不是 EFE 的监督拟合物，而是由世界模型未来自注意力定义的内生前瞻信号；在闭环干预下，它诱导出一个非静态的自举系统。该系统是否形成稳定功能，以及该功能是否与 EFE 对齐，必须通过固定点、稳定性和行为后果三个层面分别验证。

这句话可以直接决定整篇论文的理论气质。
