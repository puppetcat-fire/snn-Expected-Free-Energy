# SNN Predictive Action Architecture

## Core Claim

The system is not defined by external reward maximization.
It is defined by one base requirement:

`predict the actual required inputs of the next moment as well as possible`

The network is a plastic spiking system with three coupled functions:

1. fit recurring input patterns using historical activations
2. allow unconstrained action nodes to intervene
3. score actions by how much they contribute to predicting future required nodes

This makes action selection endogenous: actions are chosen because they improve the network's own next-step predictive adequacy, not because of an externally attached scalar reward.

## Existing Intuition In Main Branch

The current main-branch prototypes already contain the core pieces:

- [main92.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/main92.py): single-node timing-fit prototype
- [node.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/node.py): double-exponential connection kernel with delayed credit update
- [doNode.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/doNode.py): unconstrained action-like node that samples a future activation time from an energy distribution and receives delayed feedback

The conceptual path is already connected.
What is missing is a clean unified definition of:

- what counts as "actual need"
- how action contribution is measured
- how positive and negative future effects are separated
- how one-hop and two-hop credit use the same rule

## Minimal System Definition

### 1. Node Types

Use four node classes.

- `required nodes`
  Nodes whose next-step activation must be predicted because they correspond to actual incoming needs or boundary-relevant variables.
- `latent compute nodes`
  Internal nodes that help model recurring patterns.
- `action nodes`
  Unconstrained nodes that can intervene and alter future node activations.
- `constraint nodes`
  Positive and negative evaluators:
  - `rise / toward` nodes
  - `fall / away` nodes

These node classes should not share exactly the same temporal kernel.

- `required / latent compute nodes`
  may use simpler activation or accumulation dynamics if they only need to become active when enough predictive evidence arrives.
- `action nodes`
  should keep an `alpha`-like or double-exponential temporal kernel with a clear rise-then-fall profile.

This distinction is important:

- a function node only needs to cross an activation condition
- an action node represents an intervention whose influence unfolds over time

So for action nodes, "first increase, then decay" is not optional.
It is part of what defines an action.

### 2. Base Objective

For each required node `r`:

```text
predict p(r_{t+1} | history)
```

All plasticity is anchored to improving the prediction of required nodes at the next moment.

### 3. Historical Fitting

Each node only uses previously activated nodes.
A recurring pattern is therefore learned from repeated historical co-activation and relative delay structure.

This is already the spirit of [main92.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/main92.py) and [node.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/node.py):

- repeated input activations
- delayed energy accumulation
- threshold crossing
- parameter correction from timing mismatch

### 4. Action Contribution

Action nodes are not scored by direct target labels.
They are scored by their contribution to future predictive success.

For an action node `a_t`, define its contribution to required-node prediction as:

```text
Contrib(a_t) = PredErr(without a_t) - PredErr(with a_t)
```

Equivalent share form:

```text
ContribShare(a_t) =
sum over future required nodes of
the proportion of predictive activation trace attributable to a_t
```

This is the formal version of:

"count how much the action contributes to predicting later known-node activations"

### 5. Positive and Negative Constraint

The system must not only reinforce "rise" or "toward target" effects.
It must simultaneously represent:

- positive contribution
- negative contribution

Define:

```text
Pos(a_t) = contribution of a_t to future rise / toward / support nodes
Neg(a_t) = contribution of a_t to future fall / away / destabilizing nodes
```

Then:

```text
Score(a_t) = Pos(a_t) - lambda_neg * Neg(a_t)
```

Without this signed split, the system will learn strong-impact but pathological strategies.

### 6. Alpha Kernel And Path Replacement

The alpha-like connection kernel is not only a time delay device.
It provides a mechanism for path competition.

If two upstream paths can predict the same required future activation:

- the path with better timing and stronger predictive contribution gains more share
- the weaker path gradually loses share or is removed

This gives "advantage path replacement".

Formally, each connection maintains a temporal kernel:

```text
K_ij(tau)
```

and future predictive credit is weighted by:

```text
Credit_ij(tau) propto K_ij(tau) * contribution_at_tau
```

For action nodes, this kernel should be explicitly asymmetric in time:

```text
K_action(tau) = exp(-tau / tau_decay) - exp(-tau / tau_rise)
```

with:

- early increase
- later decay

This allows an action to be represented as a temporally extended intervention rather than a point event.

For purely functional prediction nodes, this full rise-then-fall shape may be unnecessary.
Those nodes can use simpler thresholded accumulation if their role is only:

- gather evidence
- activate when enough support exists
- pass predictive demand onward

So the architecture should treat `alpha kernels` as mandatory for action-like interventions, not mandatory for every node.

### 7. Softmax As Exploration And Information Collection

Action selection should remain probabilistic.

Use:

```text
pi(a_t | h_t) = softmax(beta_t * Score(a_t))
```

where `beta_t` is an inverse temperature controlled by how accurate recent
downstream demand prediction has been.

So:

- low predictive accuracy -> lower `beta_t` -> flatter distribution -> more exploration
- high predictive accuracy -> higher `beta_t` -> sharper distribution -> more decisive action

This is not ordinary reward exploration.
It is contribution-discovery under uncertainty.

If standard temperature notation is preferred, then:

```text
T_t = 1 / beta_t
```

and therefore higher accuracy should correspond to lower `T_t`, not higher.

### 8. Contribution In Cyclic Graphs

Cycles are not a special failure case.
They should be handled by finite time-unrolling.

For any currently active node or action candidate `z_t`, define contribution over a
finite horizon `H` by comparing:

- the full recurrent rollout
- a counterfactual rollout in which `z_t` is clamped away or masked at the current step

That is:

```text
Contrib(z_t; H) = NeedLoss(mask z_t, t:t+H) - NeedLoss(full, t:t+H)
```

For signed prospective accounting, use:

```text
PosContrib(z_t; H)
  = sum over future positive required nodes of
    [Act_full - Act_masked]

NegContrib(z_t; H)
  = sum over future negative required nodes of
    [Act_full - Act_masked]
```

Then:

```text
TotalScore(z_t)
  = BaseNeedPredictionGain(z_t)
  + beta_pos * PosContrib(z_t; H)
  - beta_neg * NegContrib(z_t; H)
```

This definition already includes all recurrent loops that occur within the
unrolled horizon.

So "the graph has cycles" does not break contribution accounting.
It only means contribution must be defined on the unfolded recurrent trajectory
rather than by static path counting.

### 9. Sparse But Open Structure

The network should not be fully connected.

At any moment:

- only a sparse active frontier should participate in computation
- each node should retain only a small top-k set of strong incoming paths
- connection updates should be local to recently active and recently useful nodes

But across time:

- new nodes can appear
- new edges can grow
- weak or redundant paths can disappear

So the intended structural principle is:

```text
computation is sparse
structure remains open
```

This avoids `O(N^2)` all-to-all matching while preserving unbounded structural
possibility.

### 10. Two-Hop And Multi-Hop Extension

Two-hop learning does not require a separate principle.
Internal unconstrained nodes can be trained with the same logic as action nodes.

That is:

- an internal node is treated as a provisional intervention node
- its score is defined by how much future required-node prediction depends on it
- therefore one-hop and two-hop credit share the same rule

So multi-hop extension is not a new algorithm.
It is repeated application of the same contribution accounting over deeper causal paths.

## Unified Training Rule

For each active node or action candidate `z_t`:

```text
TotalScore(z_t) =
  BaseNeedPredictionGain(z_t)
  + beta_pos * PosContrib(z_t)
  - beta_neg * NegContrib(z_t)
```

Then:

- if `z_t` is a required or latent node:
  update incoming structure to better fit repeated predictive demand
- if `z_t` is an action or unconstrained node:
  sample or select according to softmax over `TotalScore`

## Structural Plasticity

Connection weights alone are not enough.
The architecture should allow:

- connection strengthening
- connection weakening
- connection removal
- new connection growth between repeatedly co-contributing nodes

Minimal structural rule:

```text
Grow(i, j) if repeated predictive co-contribution exceeds threshold
Prune(i, j) if long-run signed contribution magnitude stays below threshold
```

This matches the intended principle:

"nodes and connections both increase and decrease, all in service of predicting next actual need"

## Mapping To Current Code

### [main92.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/main92.py)

What it already gives:

- history-based temporal accumulation
- timing mismatch update
- threshold-based activation

What it lacks:

- explicit action contribution accounting
- positive/negative future split
- structural growth and pruning
- multi-node predictive objective

### [node.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/node.py)

What it already gives:

- alpha-like double exponential kernels
- delayed credit update from full energy trajectory
- better temporal path representation than `main92.py`

What it lacks:

- explicit distinction between required nodes and action nodes
- signed future contribution decomposition

This file is much closer to the correct temporal form for action nodes than [main92.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/main92.py).

### [doNode.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/doNode.py)

What it already gives:

- unconstrained action-like sampling
- future-timing distribution
- delayed feedback to action-like node

What it should become:

- the canonical action-node implementation
- extended with positive/negative contribution channels
- reused for two-hop latent intervention nodes

Its double-exponential action timing should be preserved.
That rise-then-fall profile is part of why it is suitable for actions rather than ordinary function nodes.

## Recommended Next Implementation

### Phase 1: Clean SNN Core

Refactor into:

- `RequiredNode`
- `LatentNode`
- `ActionNode`
- `ConstraintNode`
- `AlphaConn`

### Phase 2: Signed Contribution

Add two future evaluators:

- `TowardNode`
- `AwayNode`

and define:

```text
ProspectiveScore(a) = TowardCredit(a) - lambda * AwayCredit(a)
```

### Phase 3: Contribution Accounting

Implement one of two estimators:

- trace share estimator
- ablation estimator

The ablation estimator is slower but cleaner:

```text
Contrib(a) = PredErr(masked action a) - PredErr(full)
```

In cyclic graphs, this estimator should be computed on a finite unrolled horizon.

### Phase 4: Accuracy-Controlled Softmax

Use recent downstream prediction accuracy to control `beta_t` in:

```text
pi(a_t | h_t) = softmax(beta_t * Score(a_t))
```

### Phase 5: Structural Plasticity

Add:

- grow candidate connection when repeated delayed co-contribution appears
- prune when signed contribution stays weak

## Safe Paper Claim

The strongest defensible claim is:

"An SNN with plastic temporal kernels and unconstrained action nodes can treat action selection as contribution to future required-node prediction, and can be extended with signed prospective constraints, cyclic ablation-based contribution accounting, and accuracy-controlled softmax selection to form a resource-limited existence-oriented controller."

Do not claim:

- equivalence to EFE
- guaranteed optimal control
- that pure positive prospective influence is sufficient

## Immediate Practical Direction

If time is limited, do not keep expanding the Transformer line first.
Use the current `main`-branch intuition as the main track:

1. keep [node.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/node.py) kernel logic
2. treat [doNode.py](C:/Users/xiaob/Documents/snn-Expected-Free-Energy/doNode.py) as the action-node prototype
3. add signed future contribution
4. add cyclic ablation-based contribution accounting
5. add connection growth/pruning only after the signed version works

That is the shortest route to a coherent system that still matches the original theory.
