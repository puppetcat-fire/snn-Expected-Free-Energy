#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signed homeostatic attention demo.

This demo is intentionally small and fast. It does not try to show
"attention = EFE". Instead, it shows a narrower point:

1. If a controller scores actions only by how much they increase a sensor
   variable (for example glucose), it can learn a pathological policy.
2. If the same future-attention structure is signed by homeostatic value,
   increases and decreases are treated differently depending on whether they
   move the organism toward or away from a survivable range.
3. A signed prospective-attention heuristic can recover the exact
   homeostatic planner's action choices in this toy setting.

Environment
-----------
- Sensor variable: glucose in [0, 10]
- Safe range: [4, 6]
- Actions:
  - eat: immediate +2 glucose, plus +1 carry to the next step
  - insulin: immediate -2 glucose, plus -1 carry to the next step
  - wait: no direct intervention
- External disturbance each step: -1 / 0 / +1 with fixed probabilities

Policies compared
-----------------
- rise_only_attention:
  score actions by positive future glucose increases only
- signed_homeostatic_attention:
  score actions by future homeostatic value change, with positive and
  negative contributions
- exact_homeostatic_planner:
  dynamic-programming baseline that maximizes expected cumulative
  homeostatic value

No third-party dependencies are required.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

ACTIONS: Tuple[str, ...] = ("eat", "insulin", "wait")
DISTURBANCE_BRANCHES: Tuple[Tuple[int, float], ...] = ((-1, 0.30), (0, 0.40), (1, 0.30))

MAX_GLUCOSE = 10
SAFE_LOW = 4
SAFE_HIGH = 6
EPISODE_HORIZON = 18
PLAN_DEPTH = 4
ATTN_HORIZON = 3
DISCOUNT = 0.92

ACTION_IMMEDIATE = {
    "eat": 2,
    "insulin": -2,
    "wait": 0,
}

ACTION_CARRY = {
    "eat": 1,
    "insulin": -1,
    "wait": 0,
}

ACTION_ATTENTION_PROFILE = {
    "eat": (2.0, 1.0),
    "insulin": (2.0, 1.0),
    "wait": (1.0,),
}


@dataclass(frozen=True)
class EvalMetrics:
    avg_return: float
    survival_rate: float
    safe_step_rate: float
    agreement_with_exact: float


@dataclass(frozen=True)
class StateExample:
    glucose: int
    carry: int
    rise_only_action: str
    rise_only_scores: Dict[str, float]
    signed_action: str
    signed_scores: Dict[str, float]
    exact_action: str
    exact_scores: Dict[str, float]


def clamp_glucose(value: int) -> int:
    return max(0, min(MAX_GLUCOSE, value))


def glucose_deviation(glucose: float) -> float:
    if SAFE_LOW <= glucose <= SAFE_HIGH:
        return 0.0
    if glucose < SAFE_LOW:
        return float(SAFE_LOW - glucose)
    return float(glucose - SAFE_HIGH)


def homeostatic_value(glucose: float) -> float:
    if glucose <= 0 or glucose >= MAX_GLUCOSE:
        return -12.0
    return 1.5 - glucose_deviation(glucose)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def transition(state: Tuple[int, int], action: str, disturbance: int) -> Tuple[int, int]:
    glucose, carry = state
    next_glucose = clamp_glucose(glucose + carry + ACTION_IMMEDIATE[action] + disturbance)
    return next_glucose, ACTION_CARRY[action]


@lru_cache(maxsize=None)
def exact_homeostatic_value(state: Tuple[int, int], depth: int) -> float:
    if depth <= 0 or state[0] <= 0 or state[0] >= MAX_GLUCOSE:
        return 0.0
    best = float("-inf")
    for action in ACTIONS:
        total = 0.0
        for disturbance, prob in DISTURBANCE_BRANCHES:
            next_state = transition(state, action, disturbance)
            total += prob * (
                homeostatic_value(next_state[0]) + DISCOUNT * exact_homeostatic_value(next_state, depth - 1)
            )
        if total > best:
            best = total
    return best


def exact_homeostatic_action_scores(state: Tuple[int, int], depth: int = PLAN_DEPTH) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for action in ACTIONS:
        total = 0.0
        for disturbance, prob in DISTURBANCE_BRANCHES:
            next_state = transition(state, action, disturbance)
            total += prob * (
                homeostatic_value(next_state[0]) + DISCOUNT * exact_homeostatic_value(next_state, depth - 1)
            )
        scores[action] = total
    return scores


def argmax_action(scores: Dict[str, float]) -> str:
    return max(ACTIONS, key=lambda action: (scores[action], action))


def prospective_attention_weights(action: str, horizon: int = ATTN_HORIZON) -> List[float]:
    profile = list(ACTION_ATTENTION_PROFILE[action])[:horizon]
    if len(profile) < horizon:
        profile.extend([0.0] * (horizon - len(profile)))
    total = sum(profile)
    if total <= 1e-12:
        return [1.0 / float(horizon) for _ in range(horizon)]
    return [value / total for value in profile]


def expected_future_glucose_sequence(
    state: Tuple[int, int],
    action: str,
    horizon: int = ATTN_HORIZON,
) -> List[float]:
    current: Dict[Tuple[int, int], float] = {}
    for disturbance, prob in DISTURBANCE_BRANCHES:
        next_state = transition(state, action, disturbance)
        current[next_state] = current.get(next_state, 0.0) + prob

    expectations = [sum(next_state[0] * prob for next_state, prob in current.items())]

    for _ in range(1, horizon):
        next_dist: Dict[Tuple[int, int], float] = {}
        for partial_state, partial_prob in current.items():
            for disturbance, prob in DISTURBANCE_BRANCHES:
                next_state = transition(partial_state, "wait", disturbance)
                next_dist[next_state] = next_dist.get(next_state, 0.0) + partial_prob * prob
        current = next_dist
        expectations.append(sum(next_state[0] * prob for next_state, prob in current.items()))

    return expectations


def rise_only_attention_scores(state: Tuple[int, int], horizon: int = ATTN_HORIZON) -> Dict[str, float]:
    glucose_now = float(state[0])
    scores: Dict[str, float] = {}
    for action in ACTIONS:
        weights = prospective_attention_weights(action, horizon=horizon)
        future_glucose = expected_future_glucose_sequence(state, action, horizon=horizon)
        scores[action] = sum(weight * max(glucose - glucose_now, 0.0) for weight, glucose in zip(weights, future_glucose))
    return scores


def signed_homeostatic_attention_scores(state: Tuple[int, int], horizon: int = ATTN_HORIZON) -> Dict[str, float]:
    value_now = homeostatic_value(float(state[0]))
    scores: Dict[str, float] = {}
    for action in ACTIONS:
        weights = prospective_attention_weights(action, horizon=horizon)
        future_glucose = expected_future_glucose_sequence(state, action, horizon=horizon)
        scores[action] = sum(weight * (homeostatic_value(glucose) - value_now) for weight, glucose in zip(weights, future_glucose))
    return scores


def policy_action(policy_name: str, state: Tuple[int, int]) -> str:
    if policy_name == "rise_only_attention":
        return argmax_action(rise_only_attention_scores(state))
    if policy_name == "signed_homeostatic_attention":
        return argmax_action(signed_homeostatic_attention_scores(state))
    if policy_name == "exact_homeostatic_planner":
        return argmax_action(exact_homeostatic_action_scores(state))
    raise ValueError(f"Unknown policy: {policy_name}")


def exact_agreement(policy_name: str) -> float:
    total = 0
    matches = 0
    for glucose in range(1, MAX_GLUCOSE):
        for carry in (-1, 0, 1):
            state = (glucose, carry)
            heuristic_action = policy_action(policy_name, state)
            exact_action = policy_action("exact_homeostatic_planner", state)
            total += 1
            matches += int(heuristic_action == exact_action)
    return matches / float(max(1, total))


def simulate_episode(policy_name: str, seed: int) -> Tuple[float, int, int, int]:
    rng = random.Random(seed)
    state = (rng.choice([3, 4, 5, 6, 7]), 0)
    total_return = 0.0
    safe_steps = 0
    total_steps = 0

    for _ in range(EPISODE_HORIZON):
        glucose, _carry = state
        if glucose <= 0 or glucose >= MAX_GLUCOSE:
            break
        action = policy_action(policy_name, state)
        disturbance = rng.choices([branch[0] for branch in DISTURBANCE_BRANCHES], weights=[branch[1] for branch in DISTURBANCE_BRANCHES], k=1)[0]
        state = transition(state, action, disturbance)
        total_return += homeostatic_value(float(state[0]))
        safe_steps += int(SAFE_LOW <= state[0] <= SAFE_HIGH)
        total_steps += 1

    survived = int(0 < state[0] < MAX_GLUCOSE)
    return total_return, survived, safe_steps, total_steps


def evaluate_policy(policy_name: str, episodes: int, seed: int) -> EvalMetrics:
    total_return = 0.0
    survived = 0
    safe_steps = 0
    total_steps = 0
    for episode_idx in range(episodes):
        episode_return, episode_survived, episode_safe, episode_steps = simulate_episode(policy_name, seed + 1000 * episode_idx)
        total_return += episode_return
        survived += episode_survived
        safe_steps += episode_safe
        total_steps += episode_steps
    return EvalMetrics(
        avg_return=total_return / float(max(1, episodes)),
        survival_rate=survived / float(max(1, episodes)),
        safe_step_rate=safe_steps / float(max(1, total_steps)),
        agreement_with_exact=exact_agreement(policy_name),
    )


def representative_examples() -> List[StateExample]:
    examples: List[StateExample] = []
    for glucose in (3, 5, 7):
        state = (glucose, 0)
        rise_scores = rise_only_attention_scores(state)
        signed_scores = signed_homeostatic_attention_scores(state)
        exact_scores = exact_homeostatic_action_scores(state)
        examples.append(
            StateExample(
                glucose=glucose,
                carry=0,
                rise_only_action=argmax_action(rise_scores),
                rise_only_scores={action: round(rise_scores[action], 6) for action in ACTIONS},
                signed_action=argmax_action(signed_scores),
                signed_scores={action: round(signed_scores[action], 6) for action in ACTIONS},
                exact_action=argmax_action(exact_scores),
                exact_scores={action: round(exact_scores[action], 6) for action in ACTIONS},
            )
        )
    return examples


def aggregate_over_seeds(policy_name: str, episodes: int, seeds: int) -> Dict[str, Dict[str, float]]:
    metrics = [evaluate_policy(policy_name, episodes=episodes, seed=202 + 17 * seed_idx) for seed_idx in range(seeds)]
    avg_return = [item.avg_return for item in metrics]
    survival_rate = [item.survival_rate for item in metrics]
    safe_step_rate = [item.safe_step_rate for item in metrics]
    agreement = [item.agreement_with_exact for item in metrics]

    def pack(values: Sequence[float]) -> Dict[str, float]:
        mean_value, std_value = mean_std(values)
        return {"mean": mean_value, "std": std_value}

    return {
        "avg_return": pack(avg_return),
        "survival_rate": pack(survival_rate),
        "safe_step_rate": pack(safe_step_rate),
        "agreement_with_exact": pack(agreement),
    }


def print_summary(summary: Dict[str, Dict[str, Dict[str, float]]], examples: Sequence[StateExample]) -> None:
    def fmt(entry: Dict[str, float], pct: bool = False) -> str:
        scale = 100.0 if pct else 1.0
        suffix = "%" if pct else ""
        return f"{entry['mean'] * scale:.2f}±{entry['std'] * scale:.2f}{suffix}"

    print("=" * 108)
    print("Signed homeostatic attention demo")
    print("Interpretation: positive-only glucose rise can be pathological; signed homeostatic attention restores bounded control.")
    print("=" * 108)
    print(f"{'Policy':28s} | {'Return':>12s} | {'Survival':>11s} | {'SafeRate':>11s} | {'Agree/Exact':>12s}")
    print("-" * 108)
    for policy_name, label in (
        ("rise_only_attention", "Rise-only attention"),
        ("signed_homeostatic_attention", "Signed homeostatic attention"),
        ("exact_homeostatic_planner", "Exact homeostatic planner"),
    ):
        block = summary[policy_name]
        print(
            f"{label:28s} | {fmt(block['avg_return']):>12s} | {fmt(block['survival_rate'], pct=True):>11s} | "
            f"{fmt(block['safe_step_rate'], pct=True):>11s} | {fmt(block['agreement_with_exact'], pct=True):>12s}"
        )

    print("\nRepresentative states")
    for example in examples:
        print(
            f"glucose={example.glucose}, carry={example.carry} | "
            f"rise_only={example.rise_only_action:7s} | signed={example.signed_action:7s} | exact={example.exact_action:7s}"
        )


def write_markdown_summary(path: str, summary: Dict[str, Dict[str, Dict[str, float]]], examples: Sequence[StateExample]) -> None:
    def render(entry: Dict[str, float], pct: bool = False) -> str:
        scale = 100.0 if pct else 1.0
        suffix = "%" if pct else ""
        return f"{entry['mean'] * scale:.2f}±{entry['std'] * scale:.2f}{suffix}"

    lines = [
        "# Signed Homeostatic Attention Demo",
        "",
        "- `rise_only_attention`: reward only future glucose increases.",
        "- `signed_homeostatic_attention`: reward future movement toward the safe range, penalize movement away from it.",
        "- `exact_homeostatic_planner`: expected-value dynamic-programming baseline.",
        "",
        "| Policy | Avg return | Survival rate | Safe-step rate | Agreement with exact |",
        "|---|---:|---:|---:|---:|",
    ]

    for policy_name, label in (
        ("rise_only_attention", "Rise-only attention"),
        ("signed_homeostatic_attention", "Signed homeostatic attention"),
        ("exact_homeostatic_planner", "Exact homeostatic planner"),
    ):
        block = summary[policy_name]
        lines.append(
            f"| {label} | {render(block['avg_return'])} | {render(block['survival_rate'], pct=True)} | "
            f"{render(block['safe_step_rate'], pct=True)} | {render(block['agreement_with_exact'], pct=True)} |"
        )

    lines.extend(
        [
            "",
            "## Representative States",
            "",
            "| Glucose | Carry | Rise-only action | Signed action | Exact action |",
            "|---:|---:|---|---|---|",
        ]
    )

    for example in examples:
        lines.append(
            f"| {example.glucose} | {example.carry} | {example.rise_only_action} | {example.signed_action} | {example.exact_action} |"
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo of signed homeostatic attention in a bounded glucose-control task.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--json", type=str, default="signed_homeostatic_attention_demo.json")
    parser.add_argument("--md", type=str, default="signed_homeostatic_attention_demo.md")
    args = parser.parse_args()

    exact_homeostatic_value.cache_clear()

    summary = {
        "rise_only_attention": aggregate_over_seeds("rise_only_attention", episodes=args.episodes, seeds=args.seeds),
        "signed_homeostatic_attention": aggregate_over_seeds("signed_homeostatic_attention", episodes=args.episodes, seeds=args.seeds),
        "exact_homeostatic_planner": aggregate_over_seeds("exact_homeostatic_planner", episodes=args.episodes, seeds=args.seeds),
    }
    examples = representative_examples()

    print_summary(summary, examples)
    payload = {
        "config": {
            "episodes": args.episodes,
            "seeds": args.seeds,
            "episode_horizon": EPISODE_HORIZON,
            "plan_depth": PLAN_DEPTH,
            "attention_horizon": ATTN_HORIZON,
            "safe_range": [SAFE_LOW, SAFE_HIGH],
            "actions": list(ACTIONS),
        },
        "summary": summary,
        "representative_examples": [asdict(example) for example in examples],
    }
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_markdown_summary(args.md, summary, examples)
    print(f"\nRaw results written to {args.json}")
    print(f"Markdown summary written to {args.md}")


if __name__ == "__main__":
    main()
