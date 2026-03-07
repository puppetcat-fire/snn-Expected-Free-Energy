# Future Attention as a Learned Prospective Signal:
# Real Self-Attention, Expected Free Energy, and Closed-Loop Drift

Author note:

- This is a first manuscript draft for workshop / ALIFE-style submission.
- Citation placeholders should be replaced with final bibliography entries before submission.
- Claims are intentionally narrower than `FA = EFE`.

## Abstract

We study the relationship between Future Attention (FA) and Expected Free Energy (EFE) without assuming that the two are equivalent. We consider a fixed causal Transformer world model in a small homeostatic partially observable Markov decision process (POMDP) where Exact EFE is tractable. In this setting, we define `oracle FA` as future self-attention assigned back to a current candidate action during imagined rollouts, and train a lightweight predictor to approximate this signal for online control. We then compare action rankings induced by `oracle FA`, learned `predictor FA`, and `Exact EFE`, and analyze the closed-loop interaction among policy, data, attention targets, and learned control. The results show moderate action-ranking alignment between attention-defined FA and Exact EFE under full observability, but substantially weaker alignment under partial observability. Under partial observability, rollout policy and distribution shift become central factors: a fixed information-sensitive rollout policy can slightly improve FA-EFE ranking agreement, but often at the cost of behavioral return. These findings support a narrower conclusion than equivalence: future self-attention can induce a learnable and behaviorally useful prospective signal in a fixed world model, but its relationship to EFE is conditional and approximate rather than general or identity-like.

## 1. Introduction

Expected Free Energy (EFE) occupies a central role in active inference as a quantity that combines preference-seeking and information-seeking pressures into a single action-evaluation objective `[REF-active-inference-1; REF-active-inference-2]`. In parallel, modern sequence models allocate internal representational resources through self-attention, and these attention patterns can be interpreted as structured influence maps over future predictions `[REF-transformer]`. This raises a natural question: can future self-attention define an endogenous control-relevant signal that partially recovers the action preferences associated with EFE?

We call this signal Future Attention (FA). The motivating idea is not that FA is supervised to imitate EFE, nor that FA is assumed to be identical to EFE. Instead, the proposal is that a world model's own imagined future may assign attention back to currently available actions, and that this future-to-present credit structure may itself be useful for control. If such a signal exists, it would offer a computationally different route to prospective decision making: rather than explicitly solving an EFE planning problem online, an agent could learn to predict an internally generated future-attention quantity.

However, this idea immediately creates two problems. First, if FA is derived from a model's own future self-attention, when should it align with EFE at all? Second, once FA is used to intervene on behavior, the resulting policy changes the data distribution that will train future predictors, potentially altering future attention targets in a self-confirming loop. The relevant question is therefore not only whether FA correlates with EFE in a static comparison, but whether FA remains useful or drifts under closed-loop updating.

This paper addresses both questions in a deliberately small and controlled setting. We construct a homeostatic POMDP in which Exact EFE remains tractable. We then train a fixed causal Transformer world model over offline trajectories, extract `oracle FA` from real future self-attention during imagined rollouts, train a lightweight `FA predictor`, and compare FA-induced action rankings to Exact EFE. We also evaluate a closed loop in which only the FA predictor and policy are updated while the world model remains fixed.

Our contribution is not an equivalence proof. Instead, the paper makes three narrower claims:

1. A fixed Transformer world model can define an `oracle FA` signal from real future self-attention.
2. A lightweight predictor can learn this signal well enough to influence control.
3. FA and Exact EFE exhibit limited but non-trivial action-ranking alignment under some observability conditions, while partial observability reveals a clear closed-loop drift problem.

## 2. Related Framing

This work sits between active inference, world-model-based control, and mechanistic interpretations of self-attention.

From the active inference side, EFE combines pragmatic preference realization with epistemic information gain `[REF-active-inference-1; REF-active-inference-2]`. It therefore provides a strong normative baseline in small POMDPs, especially when exact computation is feasible.

From the sequence-model side, self-attention defines a structured dependency map between positions in a trajectory-like representation. Although attention should not be naively equated with causal explanation, future attention back to a present action can still be treated as an operational measure of how strongly imagined future predictions rely on that action token `[REF-transformer; REF-attention-interpretability]`.

The present work differs from direct policy imitation or value distillation. We do not train FA on EFE labels. Instead, EFE remains an external comparator, while FA is defined intrinsically from the world model's own future self-attention. In this sense, the paper asks whether a model-internal prospective signal can partially recover properties usually associated with explicit free-energy-style evaluation.

## 3. Problem Setting

### 3.1 Environment

We study a small homeostatic POMDP with:

- a hidden binary resource state: `rich` or `barren`
- an observable energy variable with values `0..4`
- a noisy cue about resource quality
- four actions: `eat`, `inspect`, `move`, and `wait`

The task combines preference maintenance and information gathering. `eat` can improve energy when the site is rich, but can be costly when the site is barren. `inspect` improves information quality. `move` changes the latent resource distribution. `wait` provides a passive baseline.

The environment is evaluated under three visibility regimes:

- `full`: passive and active observation are fully accurate
- `partial`: passive observation is noisy, inspection is more reliable
- `hard_partial`: passive observation is strongly degraded

This design is useful because it preserves a nontrivial epistemic structure while remaining small enough for Exact EFE planning.

### 3.2 World Model

The world model is a fixed causal Transformer over discrete trajectory tokens. The token vocabulary contains:

- `BOS`
- observation tokens representing `(energy, cue)`
- action tokens representing the four available actions
- `EOS`

The default model configuration uses:

- `d_model = 96`
- `nhead = 4`
- `num_layers = 3`
- `dim_ff = 192`

The model is trained offline to predict future observation tokens and trajectory termination, while action tokens are supplied as conditioning inputs rather than supervised targets. After training, the world model is frozen for all FA experiments.

### 3.3 Exact EFE Baseline

Because the environment is small, we can compute a finite-horizon Exact EFE baseline by belief-state planning. This baseline includes:

- a preference-risk term over future observations
- an information-gain term
- recursive future value over belief updates

Exact EFE is used only as an external baseline. It is never used as a supervision target for FA.

## 4. Future Attention

### 4.1 Oracle FA from Real Future Self-Attention

Let `h_t` denote the current history and `a` a candidate action. We append `a` to the current history and use the frozen world model to imagine future rollouts. During each imagined rollout, the model generates future observation tokens autoregressively. After each generated future observation token, we read the world model's last-layer self-attention and measure the attention assigned from that future token back to the current candidate action token.

For a single imagined rollout, oracle FA is constructed as a discounted sum of these future-to-present attention values, weighted by a gate that combines:

- preference utility of the imagined next energy state
- information gain implied by the imagined observation

The full `oracle FA` for a candidate action is the Monte Carlo average over imagined rollouts. In other words, oracle FA is not a hand-designed attention operator. It is a quantity extracted from the real self-attention tensors of the trained world model.

### 4.2 Learned FA Predictor

Computing oracle FA online is expensive even in this small setting, because it requires multiple imagined rollouts and attention extraction. We therefore train a lightweight predictor that takes:

- the current world-model hidden state
- the embedding of a candidate action

and outputs a scalar FA score. This predictor is trained on oracle FA labels collected under the current experimental rollout policy.

### 4.3 Policy

The FA policy mixes:

- immediate expected utility in the current belief state
- the standardized FA predictor score

The policy is stochastic during data collection and deterministic at evaluation time. Crucially, only the FA predictor and the induced policy are updated in closed loop. The world model remains fixed.

## 5. Closed-Loop Setup

We evaluate a closed loop of the form:

`policy -> data -> oracle FA labels -> predictor update -> new policy`

This loop matters because FA is not a passive diagnostic. Once the predictor influences behavior, the policy changes the trajectories on which future oracle FA labels are defined. Under partial observability this creates the possibility of self-confirming drift, where the system increasingly reinforces its own rollout-induced attention structure rather than aligning with EFE.

To probe this issue, we compare at least two oracle rollout policies under partial observability:

- `current`: oracle FA labels are generated under the current FA-induced policy
- `entropy`: oracle FA labels are generated under a fixed information-sensitive rollout policy

The second condition is intended to reduce closed-loop collapse by decoupling oracle label generation from the current learned policy.

## 6. Experimental Design

### 6.1 Main Questions

We ask four empirical questions:

1. Can future self-attention define a stable oracle FA signal in a fixed world model?
2. Can a lightweight predictor learn oracle FA?
3. To what extent do oracle FA and predictor FA agree with Exact EFE action rankings?
4. How strongly do observability and rollout policy affect this alignment?

### 6.2 Metrics

We report:

- world-model observation-token accuracy
- predictor-vs-oracle agreement
- oracle-vs-Exact-EFE agreement
- predictor-vs-Exact-EFE agreement
- predictor exact regret
- average return
- survival rate
- safe-step rate

The ranking metrics are the core evidence for the FA-EFE relationship. Behavioral metrics test whether the closed loop yields useful control.

### 6.3 Configurations Reported Here

The current draft focuses on three result sets:

- `full`, current rollout policy, 2 seeds
- `partial`, current rollout policy, 2 seeds
- `partial`, entropy-aware rollout policy, 2 seeds

These are not yet sufficient for a strong general claim, but they are sufficient for a mechanism-validation argument.

## 7. Results

### 7.1 Full Observability

Under full observability, the trained world model reaches observation-token accuracy of `66.98 ± 0.58%`. The final FA loop obtains:

- average return: `6.044 ± 0.999`
- survival rate: `42.92 ± 7.08%`
- safe-step rate: `80.71 ± 1.46%`
- predictor-vs-oracle agreement: `64.06 ± 6.77%`
- oracle-vs-Exact-EFE agreement: `70.31 ± 8.85%`
- predictor-vs-Exact-EFE agreement: `59.90 ± 5.73%`
- predictor regret: `0.361 ± 0.018`

These results indicate moderate but clearly non-random alignment between attention-defined FA and Exact EFE when the task is fully observable. At the same time, FA remains behaviorally weaker than Exact EFE, which achieves much higher return and survival.

### 7.2 Partial Observability with Current Rollout Policy

Under partial observability, the world model reaches observation-token accuracy of `46.15 ± 0.13%`. The final FA loop under the current-policy rollout condition obtains:

- average return: `2.450 ± 0.987`
- survival rate: `25.83 ± 4.17%`
- safe-step rate: `72.22 ± 2.75%`
- predictor-vs-oracle agreement: `66.67 ± 6.25%`
- oracle-vs-Exact-EFE agreement: `54.17 ± 7.29%`
- predictor-vs-Exact-EFE agreement: `53.12 ± 11.46%`
- predictor regret: `0.339 ± 0.082`

The most important point is that predictor learning remains viable, but FA-EFE alignment drops substantially compared with the full-observability regime. This supports the claim that partial observability is not a minor nuisance variable but a core determinant of whether attention-defined FA tracks Exact EFE.

### 7.3 Partial Observability with Entropy-Aware Oracle Rollouts

When oracle labels are generated by a fixed entropy-aware rollout policy rather than the current FA-induced policy, the final FA loop yields:

- average return: `0.633 ± 0.761`
- survival rate: `12.08 ± 5.42%`
- safe-step rate: `67.08 ± 2.64%`
- predictor-vs-oracle agreement: `78.65 ± 1.56%`
- oracle-vs-Exact-EFE agreement: `55.73 ± 7.81%`
- predictor-vs-Exact-EFE agreement: `57.29 ± 8.33%`
- predictor regret: `0.275 ± 0.066`

This condition shows an important tradeoff. A fixed information-sensitive rollout policy modestly improves FA-EFE ranking alignment and reduces regret, but it lowers behavioral return. In other words, reducing self-confirming drift in oracle labels can help ranking fidelity while hurting closed-loop task performance.

### 7.4 Result Summary

Taken together, the experiments suggest:

- oracle FA is learnable in all reported settings
- full observability yields the strongest FA-EFE alignment
- partial observability weakens alignment substantially
- rollout policy changes the oracle labels themselves, not only the learned predictor

The last point is especially important. It means the FA problem is intrinsically closed-loop: the signal being learned depends on the behavior induced by previous versions of the learner.

## 8. Discussion

### 8.1 What Has Been Shown

The present experiments justify a restrained claim: future self-attention in a fixed world model can induce an endogenous prospective signal that is learnable and partially aligned with Exact EFE.

This is already stronger than a synthetic-attention toy result, because oracle FA is extracted from real self-attention tensors in a trained Transformer world model.

### 8.2 What Has Not Been Shown

The experiments do not justify the stronger claim that FA is identical to EFE. Several reasons make such a conclusion premature:

- alignment remains moderate rather than near-perfect
- partial observability produces clear degradation
- rollout policy changes oracle FA itself
- the environment is still a small toy POMDP

The correct interpretation is therefore conditional approximation, not equivalence.

### 8.3 Closed-Loop Drift as a Core Finding

An important outcome of the partial-observability experiments is that the FA question cannot be treated as static label fitting alone. Because oracle FA is defined through imagined rollouts, any learned policy that changes the state-action visitation distribution can reshape the future attention targets used in later rounds. This makes closed-loop drift a first-class object of study rather than a nuisance artifact.

From a broader artificial-life perspective, this is arguably one of the most interesting findings in the paper. A system can learn to act using an endogenous prospective signal, but the meaning of that signal depends on the loop through which behavior, data, and internal attention co-determine one another.

## 9. Limitations

This draft has several important limitations.

First, the environment is small and intentionally simplified. This is necessary for Exact EFE comparison, but it limits external validity.

Second, the current result tables use only two seeds for the main reported conditions. This is sufficient for a first draft but not for a strong final submission.

Third, the current work uses a fixed world model. This is a feature for mechanistic clarity, but it leaves open the question of how FA behaves when the underlying world model also updates over time.

Fourth, the reported comparison focuses on Exact EFE and a small number of heuristic rollout controls. A stronger final paper should include broader baselines and more systematic ablations.

## 10. Conclusion

We introduced an experimental framework in which Future Attention is defined from real future self-attention in a fixed Transformer world model and compared against Exact Expected Free Energy in a tractable POMDP. The results support a narrow but substantive conclusion: attention-defined FA can be learned and can display non-trivial alignment with EFE, especially under full observability, but this relationship is conditional and unstable under partial observability. In particular, partial observability reveals a closed-loop drift problem in which rollout policy affects oracle FA itself. The main theoretical implication is that FA should not be treated as an a priori approximation to EFE. The main engineering implication is that model-internal future attention can still provide a usable prospective control signal, provided that its closed-loop formation dynamics are treated as part of the object of study.

## 11. Future Work

The next version of this paper should add:

- more seeds and significance testing
- the `hard_partial` condition in the final result table
- fixed-context evaluation to separate oracle mismatch from policy-induced distribution shift
- an additional environment beyond the current homeostatic POMDP
- a final comparison section on when FA is behaviorally useful even when it is not strongly EFE-aligned

## AI Use Disclosure

Generative AI tools were used to assist with code drafting, experiment scripting, manuscript structuring, and language editing. All experimental design decisions, mathematical claims, result verification, interpretation, and final manuscript content were reviewed and validated by the human author. No AI system is listed as an author.

## References to Complete

- `[REF-active-inference-1]` Foundational active inference / EFE reference
- `[REF-active-inference-2]` Additional active inference / planning reference
- `[REF-transformer]` Transformer reference
- `[REF-attention-interpretability]` Attention analysis / interpretability reference
