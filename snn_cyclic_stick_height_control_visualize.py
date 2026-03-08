#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize sparse cyclic SNN stick-height control.

This script trains the current cyclic-SNN world model, searches for a scenario
where signed prospective control differs from prediction-only control, then
produces:

1. A static trajectory plot
2. A side-by-side GIF
3. A JSON summary for the selected scenario
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw
import torch

import snn_cyclic_stick_height_control as exp


@dataclass(frozen=True)
class Scenario:
    initial_state: exp.StickState
    disturbances: Tuple[float, ...]
    action_seed: int


@dataclass
class Trajectory:
    policy: str
    states: List[exp.StickState]
    actions: List[str]
    heights: List[float]
    torques: List[float]
    disturbances: List[float]
    betas: List[float]
    entropies: List[float]
    in_band_rate: float
    mean_abs_height_error: float


def train_model(seed: int, train_episodes: int, val_episodes: int, epochs: int) -> tuple[exp.SparseCyclicSNN, exp.WorldModelMetrics]:
    exp.TRAIN_EPISODES = train_episodes
    exp.VAL_EPISODES = val_episodes
    exp.MODEL_EPOCHS = epochs
    torch.manual_seed(seed)
    random.seed(seed)
    return exp.train_world_model(seed=seed)


def sample_scenarios(seed: int, count: int, horizon: int, disturbance_scale: float) -> List[Scenario]:
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    for idx in range(count):
        initial = exp.random_initial_state(rng)
        disturbances = exp.sample_disturbance_sequence(
            random.Random(seed * 1000 + idx),
            horizon,
            disturbance_scale,
            mode=exp.DEFAULT_DISTURBANCE_MODE,
        )
        scenarios.append(Scenario(initial_state=initial, disturbances=disturbances, action_seed=seed * 17 + idx))
    return scenarios


def state_torque(state: exp.StickState) -> float:
    return exp.right_alpha(state) - exp.left_alpha(state)


def rollout_policy(policy: str, model: exp.SparseCyclicSNN, scenario: Scenario) -> Trajectory:
    rng = random.Random(scenario.action_seed)
    state = scenario.initial_state
    states = [state]
    heights = [exp.height_from_theta(state.theta)]
    torques = [state_torque(state)]
    actions: List[str] = []
    disturbances = [0.0]
    betas: List[float] = [0.0]
    entropies: List[float] = [0.0]

    snn_state = model.zero_state(batch_size=1, device=torch.device("cpu"))
    error_ema = 0.08

    model.eval()
    with torch.no_grad():
        for disturbance in scenario.disturbances:
            if policy == "planner":
                action = exp.planner_action(state)
                beta = 0.0
                probs = [1.0 if action == exp.ACTIONS[0] else 0.0, 1.0 if action == exp.ACTIONS[1] else 0.0]
                pred_next = None
            else:
                obs_vec = torch.tensor(exp.state_to_obs(state), dtype=torch.float32)
                beta = exp.prediction_error_to_beta(error_ema)
                action, _, probs = exp.choose_action_with_policy(policy, model, obs_vec, snn_state, beta, rng)
                pred_next, updated_state = model.step(
                    obs_vec.unsqueeze(0),
                    torch.tensor([exp.action_pulse(action)], dtype=torch.float32),
                    snn_state.clone(),
                )
                target_next = exp.transition_dynamics(state, action, disturbance=disturbance)
                target_obs = torch.tensor(exp.state_to_obs(target_next), dtype=torch.float32)
                pred_error = float(exp.weighted_loss(pred_next[0], target_obs).item())
                error_ema = 0.94 * error_ema + 0.06 * pred_error
                snn_state = updated_state

            actions.append(action)
            state = exp.transition_dynamics(state, action, disturbance=disturbance)
            states.append(state)
            heights.append(exp.height_from_theta(state.theta))
            torques.append(state_torque(state))
            disturbances.append(disturbance)
            betas.append(beta)
            entropies.append(exp.action_entropy(probs))

    controlled_heights = heights[1:]
    in_band_hits = sum(1 for h in controlled_heights if exp.band_distance(h) <= 1e-8)
    mean_abs_error = sum(abs(h - exp.TARGET_HEIGHT) for h in controlled_heights) / float(max(1, len(controlled_heights)))
    return Trajectory(
        policy=policy,
        states=states,
        actions=actions,
        heights=heights,
        torques=torques,
        disturbances=disturbances,
        betas=betas,
        entropies=entropies,
        in_band_rate=in_band_hits / float(max(1, len(controlled_heights))),
        mean_abs_height_error=mean_abs_error,
    )


def scenario_advantage(rollouts: Dict[str, Trajectory]) -> float:
    signed = rollouts["signed_prospective"]
    prediction = rollouts["prediction_only"]
    planner = rollouts["planner"]
    return (
        2.2 * (signed.in_band_rate - prediction.in_band_rate)
        + 0.8 * (prediction.mean_abs_height_error - signed.mean_abs_height_error)
        + 0.5 * (signed.in_band_rate - planner.in_band_rate)
    )


def select_scenario(model: exp.SparseCyclicSNN, scenarios: Sequence[Scenario]) -> tuple[Scenario, Dict[str, Trajectory], float]:
    best = scenarios[0]
    best_rollouts: Dict[str, Trajectory] = {}
    best_score = float("-inf")
    for scenario in scenarios:
        rollouts = {
            "prediction_only": rollout_policy("prediction_only", model, scenario),
            "signed_prospective": rollout_policy("signed_prospective", model, scenario),
            "planner": rollout_policy("planner", model, scenario),
        }
        score = scenario_advantage(rollouts)
        if score > best_score:
            best = scenario
            best_rollouts = rollouts
            best_score = score
    return best, best_rollouts, best_score


def map_point(box: tuple[int, int, int, int], step: int, max_step: int, value: float, y_min: float, y_max: float) -> tuple[float, float]:
    left, top, right, bottom = box
    x = left + (step / max(1, max_step)) * (right - left)
    y = bottom - ((value - y_min) / max(1e-6, (y_max - y_min))) * (bottom - top)
    return x, y


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, y_min: float, y_max: float, y_label: str) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#cccccc", width=1)
    draw.text((left, top - 24), title, fill="black")
    draw.text((8, (top + bottom) // 2), y_label, fill="#555555")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - frac * (bottom - top)
        draw.line((left, y, right, y), fill="#efefef", width=1)
        value = y_min + frac * (y_max - y_min)
        draw.text((left - 44, y - 7), f"{value:.2f}", fill="#666666")


def plot_trajectories(out_path: Path, rollouts: Dict[str, Trajectory]) -> None:
    colors = {
        "prediction_only": "#4e79a7",
        "signed_prospective": "#e15759",
        "planner": "#59a14f",
    }
    labels = {
        "prediction_only": "Prediction only",
        "signed_prospective": "Signed prospective",
        "planner": "Planner",
    }
    width = 1280
    height = 860
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = 70

    top = (margin, 80, width - margin, 300)
    middle = (margin, 380, width - margin, 560)
    bottom = (margin, 640, width - margin, 810)

    max_step = max(len(traj.heights) for traj in rollouts.values()) - 1
    draw.text((margin, 22), "Sparse cyclic SNN stick-height control", fill="black")

    draw_axes(draw, top, "Height", -0.05, 1.05, "h")
    draw_axes(draw, middle, "Action alpha torque (right - left)", -exp.MAX_ACTION_ALPHA, exp.MAX_ACTION_ALPHA, "tau")
    draw_axes(draw, bottom, "Impulse disturbance", -1.2, 1.2, "dist")

    band_top = map_point(top, 0, max_step, exp.TARGET_HEIGHT + exp.TARGET_BAND, -0.05, 1.05)[1]
    band_bottom = map_point(top, 0, max_step, exp.TARGET_HEIGHT - exp.TARGET_BAND, -0.05, 1.05)[1]
    draw.rectangle((top[0], band_top, top[2], band_bottom), fill="#e6f6e6", outline=None)
    target_y = map_point(top, 0, max_step, exp.TARGET_HEIGHT, -0.05, 1.05)[1]
    draw.line((top[0], target_y, top[2], target_y), fill="#4e9d4e", width=2)

    legend_x = width - 360
    legend_y = 28
    for idx, name in enumerate(("prediction_only", "signed_prospective", "planner")):
        traj = rollouts[name]
        color = colors[name]
        y = legend_y + idx * 22
        label = f"{labels[name]} | in-band {traj.in_band_rate*100:.1f}% | mae {traj.mean_abs_height_error:.3f}"
        draw.line((legend_x, y + 8, legend_x + 22, y + 8), fill=color, width=3)
        draw.text((legend_x + 28, y), label, fill="black")

        for step in range(1, len(traj.heights)):
            p0 = map_point(top, step - 1, max_step, traj.heights[step - 1], -0.05, 1.05)
            p1 = map_point(top, step, max_step, traj.heights[step], -0.05, 1.05)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=3)
        for step in range(1, len(traj.torques)):
            p0 = map_point(middle, step - 1, max_step, traj.torques[step - 1], -exp.MAX_ACTION_ALPHA, exp.MAX_ACTION_ALPHA)
            p1 = map_point(middle, step, max_step, traj.torques[step], -exp.MAX_ACTION_ALPHA, exp.MAX_ACTION_ALPHA)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=3)

    disturbance = rollouts["signed_prospective"].disturbances
    for step in range(1, len(disturbance)):
        p0 = map_point(bottom, step - 1, max_step, disturbance[step - 1], -1.2, 1.2)
        p1 = map_point(bottom, step, max_step, disturbance[step], -1.2, 1.2)
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill="#333333", width=3)

    info_y = 330
    signed = rollouts["signed_prospective"]
    pred = rollouts["prediction_only"]
    draw.text((margin, info_y), f"Signed beta mean: {sum(signed.betas[1:]) / max(1, len(signed.betas) - 1):.3f}", fill="#444444")
    draw.text((margin + 280, info_y), f"Prediction entropy mean: {sum(pred.entropies[1:]) / max(1, len(pred.entropies) - 1):.3f}", fill="#444444")
    draw.text((margin + 620, info_y), f"Signed entropy mean: {sum(signed.entropies[1:]) / max(1, len(signed.entropies) - 1):.3f}", fill="#444444")

    image.save(out_path)


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, state: exp.StickState, action: str, disturbance: float, beta: float) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#cfcfcf", width=1)
    draw.text((left + 8, top + 6), title, fill="black")

    cx = (left + right) // 2
    base_y = bottom - 34
    draw.line((left + 12, base_y, right - 12, base_y), fill="#9a9a9a", width=2)

    length = min(right - left, bottom - top) * 0.34
    theta = state.theta
    tip_x = cx + length * math.sin(theta)
    tip_y = base_y - length * math.cos(theta)
    draw.line((cx, base_y, tip_x, tip_y), fill="#1f2d3d", width=5)
    draw.ellipse((tip_x - 6, tip_y - 6, tip_x + 6, tip_y + 6), fill="#d95f02")

    target_angle = math.acos(exp.clamp(exp.TARGET_HEIGHT, -1.0, 1.0))
    band_half = 0.12
    for sign in (-1.0, 1.0):
        low_angle = sign * max(0.0, target_angle - band_half)
        high_angle = sign * min(math.pi - 0.05, target_angle + band_half)
        low_x = cx + length * math.sin(low_angle)
        low_y = base_y - length * math.cos(low_angle)
        high_x = cx + length * math.sin(high_angle)
        high_y = base_y - length * math.cos(high_angle)
        draw.line((cx, base_y, low_x, low_y), fill="#d5ecd5", width=2)
        draw.line((cx, base_y, high_x, high_y), fill="#d5ecd5", width=2)

    height = exp.height_from_theta(state.theta)
    torque = state_torque(state)
    draw.text((left + 8, top + 28), f"h={height:.3f}", fill="#333333")
    draw.text((left + 8, top + 46), f"tau={torque:.3f}", fill="#333333")
    draw.text((left + 8, top + 64), f"a={action}", fill="#333333")
    draw.text((left + 8, top + 82), f"dist={disturbance:+.2f}", fill="#333333")
    draw.text((left + 8, top + 100), f"beta={beta:.2f}", fill="#333333")


def make_animation(out_path: Path, rollouts: Dict[str, Trajectory], frame_ms: int) -> None:
    ordered = ["prediction_only", "signed_prospective", "planner"]
    max_steps = max(len(rollouts[name].states) for name in ordered)
    frames: List[Image.Image] = []

    for step in range(max_steps):
        image = Image.new("RGB", (1260, 440), "white")
        draw = ImageDraw.Draw(image)
        draw.text((28, 18), f"Sparse cyclic SNN stick control | step {step}", fill="black")

        boxes = [
            (20, 56, 410, 414),
            (435, 56, 825, 414),
            (850, 56, 1240, 414),
        ]
        for box, name in zip(boxes, ordered):
            traj = rollouts[name]
            idx = min(step, len(traj.states) - 1)
            action = traj.actions[idx - 1] if idx > 0 and idx - 1 < len(traj.actions) else "-"
            disturbance = traj.disturbances[idx] if idx < len(traj.disturbances) else 0.0
            beta = traj.betas[idx] if idx < len(traj.betas) else 0.0
            title = name.replace("_", " ")
            draw_panel(draw, box, title, traj.states[idx], action, disturbance, beta)
        frames.append(image)

    if not frames:
        return
    frames += [frames[-1]] * 10
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sparse cyclic SNN stick-height control.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", type=int, default=18)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--disturbance-scale", type=float, default=0.72)
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--png", type=str, default="snn_cyclic_stick_height_control_visual.png")
    parser.add_argument("--gif", type=str, default="snn_cyclic_stick_height_control_visual.gif")
    parser.add_argument("--json", type=str, default="snn_cyclic_stick_height_control_visual.json")
    parser.add_argument("--frame-ms", type=int, default=130)
    args = parser.parse_args()

    model, metrics = train_model(
        seed=args.seed,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        epochs=args.epochs,
    )
    scenarios = sample_scenarios(
        seed=args.seed + 100,
        count=args.scenarios,
        horizon=args.horizon,
        disturbance_scale=args.disturbance_scale,
    )
    scenario, rollouts, score = select_scenario(model, scenarios)

    png_path = Path(args.png).resolve()
    gif_path = Path(args.gif).resolve()
    json_path = Path(args.json).resolve()

    plot_trajectories(png_path, rollouts)
    make_animation(gif_path, rollouts, frame_ms=args.frame_ms)

    summary = {
        "seed": args.seed,
        "selection_score": score,
        "world_model": asdict(metrics),
        "scenario": {
            "initial_state": asdict(scenario.initial_state),
            "disturbance_preview": list(scenario.disturbances[:16]),
            "action_seed": scenario.action_seed,
        },
        "rollouts": {
            name: {
                "in_band_rate": traj.in_band_rate,
                "mean_abs_height_error": traj.mean_abs_height_error,
                "mean_beta": sum(traj.betas[1:]) / max(1, len(traj.betas) - 1),
                "mean_entropy": sum(traj.entropies[1:]) / max(1, len(traj.entropies) - 1),
                "final_height": traj.heights[-1],
            }
            for name, traj in rollouts.items()
        },
        "png": str(png_path),
        "gif": str(gif_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Trajectory plot written to {png_path}")
    print(f"Animation written to {gif_path}")
    print(f"Scenario summary written to {json_path}")


if __name__ == "__main__":
    main()
