#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize CartPole target-height semantic future-attention behavior.

This script trains the current CartPole semantic-FA model, searches for a
scenario where signed semantic FA diverges from prediction-only control, and
produces:

1. A trajectory plot (height and cart position over time)
2. A side-by-side animation GIF
3. A small JSON summary for the selected scenario
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

import cartpole_height_softmax_semantic_fa_experiment as exp


@dataclass
class Trajectory:
    policy: str
    states: List[exp.CartPoleState]
    actions: List[str]
    heights: List[float]
    xs: List[float]
    entropies: List[float]
    survived: bool
    in_band_rate: float


def train_model(
    offline_episodes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    offline_noise_mode: str,
    offline_high_noise_scale: float,
) -> tuple[exp.CartPoleWorldModel, exp.WorldModelMetrics]:
    torch_seed = seed
    import torch

    torch.manual_seed(torch_seed)
    random.seed(seed)
    episodes = exp.collect_offline_episodes(
        seed=seed,
        episodes=offline_episodes,
        horizon=exp.EPISODE_HORIZON,
        noise_mode=offline_noise_mode,
        nominal_noise_std=exp.NOISE_STD,
        high_noise_std=exp.NOISE_STD * offline_high_noise_scale,
    )
    split = max(1, int(len(episodes) * 0.85))
    train_episodes = episodes[:split]
    val_episodes = episodes[split:]
    train_x, train_y = exp.build_samples(train_episodes, history_steps=exp.HISTORY_STEPS)
    val_x, val_y = exp.build_samples(val_episodes or train_episodes[:1], history_steps=exp.HISTORY_STEPS)
    return exp.train_world_model(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_ff=128,
        dropout=0.05,
        seed=seed,
    )


def rollout_scenario(
    policy_name: str,
    model: exp.CartPoleWorldModel,
    scenario: exp.EvalScenario,
    fa_weight: float,
    rollout_mode: str,
    temperature: float,
) -> Trajectory:
    history_states = list(scenario.history_states)
    history_actions = list(scenario.history_actions)
    state = history_states[-1]
    states = [state]
    actions: List[str] = []
    heights = [exp.state_height(state)]
    xs = [state.x]
    entropies: List[float] = []
    survived = True

    for disturbance, draw in zip(scenario.disturbances, scenario.action_draws):
        if exp.failed(state):
            survived = False
            break
        action, entropy = exp.choose_action(
            policy_name=policy_name,
            model=model,
            history_states=history_states,
            history_actions=history_actions,
            fa_weight=fa_weight,
            rollout_mode=rollout_mode,
            temperature=temperature,
            draw=draw,
        )
        entropies.append(entropy)
        actions.append(action)
        state = exp.transition_dynamics(state, action, disturbance=disturbance)
        states.append(state)
        heights.append(exp.state_height(state))
        xs.append(state.x)
        history_states = history_states[1:] + [state]
        history_actions = history_actions[1:] + [action]

    if exp.failed(states[-1]):
        survived = False

    controlled_heights = heights[1:] if len(heights) > 1 else heights
    in_band_hits = sum(1 for h in controlled_heights if exp.band_distance(h) <= 1e-8)
    in_band_rate = in_band_hits / float(max(1, len(controlled_heights)))
    return Trajectory(
        policy=policy_name,
        states=states,
        actions=actions,
        heights=heights,
        xs=xs,
        entropies=entropies,
        survived=survived,
        in_band_rate=in_band_rate,
    )


def scenario_advantage(signed: Trajectory, prediction: Trajectory, positive: Trajectory) -> float:
    return (
        2.0 * (signed.in_band_rate - prediction.in_band_rate)
        + 1.0 * (float(signed.survived) - float(prediction.survived))
        + 0.5 * (signed.in_band_rate - positive.in_band_rate)
    )


def select_scenario(
    model: exp.CartPoleWorldModel,
    scenarios: Sequence[exp.EvalScenario],
    fa_weight: float,
    rollout_mode: str,
    temperature: float,
) -> tuple[exp.EvalScenario, Dict[str, Trajectory], float]:
    best_score = float("-inf")
    best_scenario = scenarios[0]
    best_rollouts: Dict[str, Trajectory] = {}
    for scenario in scenarios:
        rollouts = {
            name: rollout_scenario(name, model, scenario, fa_weight, rollout_mode, temperature)
            for name in ("prediction_only", "positive_fa", "signed_semantic_fa", "planner")
        }
        score = scenario_advantage(
            signed=rollouts["signed_semantic_fa"],
            prediction=rollouts["prediction_only"],
            positive=rollouts["positive_fa"],
        )
        if score > best_score:
            best_score = score
            best_scenario = scenario
            best_rollouts = rollouts
    return best_scenario, best_rollouts, best_score


def plot_trajectories(out_path: Path, regime: str, rollouts: Dict[str, Trajectory]) -> None:
    colors = {
        "prediction_only": "#4e79a7",
        "positive_fa": "#f28e2b",
        "signed_semantic_fa": "#e15759",
        "planner": "#59a14f",
    }
    labels = {
        "prediction_only": "Prediction only",
        "positive_fa": "Positive-only FA",
        "signed_semantic_fa": "Signed semantic FA",
        "planner": "Planner",
    }
    width = 1200
    height = 760
    margin = 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    top_box = (margin, 80, width - margin, 350)
    bottom_box = (margin, 430, width - margin, 700)

    def draw_axes(box: tuple[int, int, int, int], title: str, y_min: float, y_max: float, y_label: str) -> None:
        left, top, right, bottom = box
        draw.rectangle(box, outline="#cccccc", width=1)
        draw.text((left, top - 28), title, fill="black")
        draw.text((10, (top + bottom) // 2), y_label, fill="#444444")
        draw.text((left, bottom + 10), "Control step", fill="#444444")
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - frac * (bottom - top)
            draw.line((left, y, right, y), fill="#efefef", width=1)
            value = y_min + frac * (y_max - y_min)
            draw.text((left - 50, y - 8), f"{value:.2f}", fill="#666666")

    def map_point(box: tuple[int, int, int, int], step: int, max_step: int, value: float, y_min: float, y_max: float) -> tuple[float, float]:
        left, top, right, bottom = box
        x = left + (step / max(1, max_step)) * (right - left)
        y = bottom - ((value - y_min) / max(1e-6, (y_max - y_min))) * (bottom - top)
        return x, y

    max_step = max(len(traj.heights) for traj in rollouts.values()) - 1
    draw.text((margin, 20), f"{regime} scenario: target-height trajectories", fill="black")

    draw_axes(top_box, "Pole-tip height", -0.05, 1.05, "height")
    draw_axes(bottom_box, "Cart position x", -exp.X_THRESHOLD * 1.1, exp.X_THRESHOLD * 1.1, "x")

    band_top_y = map_point(top_box, 0, max_step, exp.TARGET_HEIGHT + exp.TARGET_BAND, -0.05, 1.05)[1]
    band_bottom_y = map_point(top_box, 0, max_step, exp.TARGET_HEIGHT - exp.TARGET_BAND, -0.05, 1.05)[1]
    draw.rectangle((top_box[0], band_top_y, top_box[2], band_bottom_y), fill="#e6f6e6", outline=None)
    target_y = map_point(top_box, 0, max_step, exp.TARGET_HEIGHT, -0.05, 1.05)[1]
    draw.line((top_box[0], target_y, top_box[2], target_y), fill="#5b9e5b", width=2)

    for threshold in (-exp.X_THRESHOLD, exp.X_THRESHOLD):
        y = map_point(bottom_box, 0, max_step, threshold, -exp.X_THRESHOLD * 1.1, exp.X_THRESHOLD * 1.1)[1]
        draw.line((bottom_box[0], y, bottom_box[2], y), fill="#999999", width=1)

    legend_x = width - 320
    legend_y = 30
    for idx, (name, traj) in enumerate(rollouts.items()):
        color = colors[name]
        label = f"{labels[name]} | in-band {traj.in_band_rate*100:.1f}% | survive {int(traj.survived)}"
        y = legend_y + idx * 20
        draw.line((legend_x, y + 8, legend_x + 22, y + 8), fill=color, width=3)
        draw.text((legend_x + 28, y), label, fill="black")

        for step in range(1, len(traj.heights)):
            p0 = map_point(top_box, step - 1, max_step, traj.heights[step - 1], -0.05, 1.05)
            p1 = map_point(top_box, step, max_step, traj.heights[step], -0.05, 1.05)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=3)
        for step in range(1, len(traj.xs)):
            p0 = map_point(bottom_box, step - 1, max_step, traj.xs[step - 1], -exp.X_THRESHOLD * 1.1, exp.X_THRESHOLD * 1.1)
            p1 = map_point(bottom_box, step, max_step, traj.xs[step], -exp.X_THRESHOLD * 1.1, exp.X_THRESHOLD * 1.1)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=3)

    image.save(out_path)


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, state: exp.CartPoleState, step: int) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    ground_y = top + int(height * 0.78)
    cart_y = ground_y - 12
    track_margin = 26
    pole_len = int(height * 0.34)

    draw.rectangle(box, fill="white", outline="#cccccc")
    draw.text((left + 8, top + 6), title, fill="black")
    draw.text((left + 8, top + 24), f"step={step:03d}  h={exp.state_height(state):.2f}", fill="#444444")

    band_top = cart_y - int(pole_len * (exp.TARGET_HEIGHT + exp.TARGET_BAND))
    band_bottom = cart_y - int(pole_len * (exp.TARGET_HEIGHT - exp.TARGET_BAND))
    draw.rectangle((left + 10, band_top, right - 10, band_bottom), fill="#e6f6e6", outline=None)
    draw.line((left + 10, cart_y - int(pole_len * exp.TARGET_HEIGHT), right - 10, cart_y - int(pole_len * exp.TARGET_HEIGHT)), fill="#5b9e5b", width=1)

    draw.line((left + track_margin, ground_y, right - track_margin, ground_y), fill="#555555", width=2)
    cart_x = left + track_margin + (state.x + exp.X_THRESHOLD) / (2.0 * exp.X_THRESHOLD) * (width - 2 * track_margin)
    cart_w = 36
    cart_h = 18
    draw.rectangle((cart_x - cart_w / 2, cart_y - cart_h / 2, cart_x + cart_w / 2, cart_y + cart_h / 2), fill="#4e79a7", outline="black")

    pivot_x = cart_x
    pivot_y = cart_y - cart_h / 2
    tip_x = pivot_x + pole_len * math.sin(state.theta)
    tip_y = pivot_y - pole_len * math.cos(state.theta)
    draw.line((pivot_x, pivot_y, tip_x, tip_y), fill="#d62728", width=4)
    draw.ellipse((pivot_x - 3, pivot_y - 3, pivot_x + 3, pivot_y + 3), fill="black")


def make_animation(
    out_path: Path,
    rollouts: Dict[str, Trajectory],
    frame_duration_ms: int,
    final_hold_frames: int,
) -> None:
    panel_w = 320
    panel_h = 220
    header_h = 24
    width = panel_w * 3
    height = panel_h + header_h
    policies = ("prediction_only", "signed_semantic_fa", "planner")
    titles = {
        "prediction_only": "Prediction only",
        "signed_semantic_fa": "Signed semantic FA",
        "planner": "Planner",
    }
    max_len = max(len(rollouts[name].states) for name in policies)
    frames: List[Image.Image] = []
    for step in range(max_len):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 4), "CartPole target-height control", fill="black")
        for idx, name in enumerate(policies):
            traj = rollouts[name]
            state = traj.states[min(step, len(traj.states) - 1)]
            left = idx * panel_w
            draw_panel(draw, (left, header_h, left + panel_w, header_h + panel_h), titles[name], state, step)
        frames.append(image)
    if frames:
        frames.extend([frames[-1].copy() for _ in range(max(0, final_hold_frames))])
        durations = [max(20, frame_duration_ms)] * max(1, len(frames))
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CartPole semantic-FA rollouts.")
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--offline-episodes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--fa-weight", type=float, default=0.7)
    parser.add_argument("--policy-temperature", type=float, default=0.08)
    parser.add_argument("--offline-noise-mode", type=str, default="mixed", choices=("nominal", "high", "mixed"))
    parser.add_argument("--offline-high-noise-scale", type=float, default=1.6)
    parser.add_argument("--rollout-mode", type=str, default="repeat", choices=("repeat", "greedy_prediction"))
    parser.add_argument("--search-scenarios", type=int, default=24)
    parser.add_argument("--regime", type=str, default="stress", choices=("nominal", "stress"))
    parser.add_argument("--frame-duration-ms", type=int, default=70)
    parser.add_argument("--final-hold-frames", type=int, default=10)
    parser.add_argument("--png", type=str, default="cartpole_height_semantic_fa_visual.png")
    parser.add_argument("--gif", type=str, default="cartpole_height_semantic_fa_visual.gif")
    parser.add_argument("--json", type=str, default="cartpole_height_semantic_fa_visual.json")
    args = parser.parse_args()

    model, wm_metrics = train_model(
        offline_episodes=args.offline_episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        offline_noise_mode=args.offline_noise_mode,
        offline_high_noise_scale=args.offline_high_noise_scale,
    )

    noise_std = exp.NOISE_STD if args.regime == "nominal" else exp.NOISE_STD * 1.8
    scenarios = exp.build_eval_scenarios(
        seed=args.seed + (3000 if args.regime == "nominal" else 4000),
        episodes=args.search_scenarios,
        history_steps=exp.HISTORY_STEPS,
        horizon=exp.EPISODE_HORIZON,
        noise_std=noise_std,
    )
    scenario, rollouts, score = select_scenario(
        model=model,
        scenarios=scenarios,
        fa_weight=args.fa_weight,
        rollout_mode=args.rollout_mode,
        temperature=args.policy_temperature,
    )

    plot_trajectories(Path(args.png), args.regime, rollouts)
    make_animation(
        Path(args.gif),
        rollouts,
        frame_duration_ms=args.frame_duration_ms,
        final_hold_frames=args.final_hold_frames,
    )

    summary = {
        "config": {
            "seed": args.seed,
            "regime": args.regime,
            "offline_episodes": args.offline_episodes,
            "epochs": args.epochs,
            "fa_weight": args.fa_weight,
            "policy_temperature": args.policy_temperature,
            "rollout_mode": args.rollout_mode,
            "search_scenarios": args.search_scenarios,
            "frame_duration_ms": args.frame_duration_ms,
            "final_hold_frames": args.final_hold_frames,
        },
        "world_model": asdict(wm_metrics),
        "selection_score": score,
        "selected_scenario": {
            "history_states": [asdict(s) for s in scenario.history_states],
            "history_actions": list(scenario.history_actions),
            "disturbances": list(scenario.disturbances),
        },
        "rollouts": {
            name: {
                "survived": traj.survived,
                "in_band_rate": traj.in_band_rate,
                "steps": len(traj.actions),
                "actions": traj.actions,
            }
            for name, traj in rollouts.items()
        },
        "artifacts": {
            "png": str(Path(args.png).resolve()),
            "gif": str(Path(args.gif).resolve()),
        },
    }
    Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Trajectory plot written to {Path(args.png).resolve()}")
    print(f"Animation written to {Path(args.gif).resolve()}")
    print(f"Scenario summary written to {Path(args.json).resolve()}")


if __name__ == "__main__":
    main()
