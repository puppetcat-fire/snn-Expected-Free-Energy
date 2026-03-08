#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the current open pulse-node sparse cyclic SNN stick controller.
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

import snn_pulse_stick_height_control as exp


@dataclass(frozen=True)
class Scenario:
    initial_state: exp.base.StickState
    disturbances: Tuple[float, ...]
    action_seed: int


@dataclass
class Trajectory:
    policy: str
    states: List[exp.base.StickState]
    actions: List[str]
    heights: List[float]
    torques: List[float]
    disturbances: List[float]
    betas: List[float]
    entropies: List[float]
    in_band_rate: float
    mean_abs_height_error: float


def train_model(seed: int, train_episodes: int, val_episodes: int, epochs: int) -> tuple[exp.PulseCyclicSNN, exp.WorldModelMetrics]:
    exp.TRAIN_EPISODES = train_episodes
    exp.VAL_EPISODES = val_episodes
    exp.MODEL_EPOCHS = epochs
    torch.manual_seed(seed)
    random.seed(seed)
    return exp.train_world_model(seed)


def sample_scenarios(seed: int, count: int, horizon: int, disturbance_scale: float) -> List[Scenario]:
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    for idx in range(count):
        initial = exp.base.random_initial_state(rng)
        disturbances = exp.base.sample_disturbance_sequence(
            random.Random(seed * 1000 + idx),
            horizon,
            disturbance_scale,
            mode=exp.base.DEFAULT_DISTURBANCE_MODE,
        )
        scenarios.append(Scenario(initial_state=initial, disturbances=disturbances, action_seed=seed * 37 + idx))
    return scenarios


def state_torque(state: exp.base.StickState) -> float:
    return exp.base.right_alpha(state) - exp.base.left_alpha(state)


def rollout_policy(policy: str, model: exp.PulseCyclicSNN, scenario: Scenario) -> Trajectory:
    rng = random.Random(scenario.action_seed)
    state = scenario.initial_state
    states = [state]
    heights = [exp.base.height_from_theta(state.theta)]
    torques = [state_torque(state)]
    actions: List[str] = []
    disturbances = [0.0]
    betas = [0.0]
    entropies = [0.0]

    pulse_state = model.zero_state(batch_size=1, device=torch.device("cpu"))
    error_ema = 0.10

    model.eval()
    with torch.no_grad():
        for disturbance in scenario.disturbances:
            if policy == "planner":
                action = exp.base.planner_action(state)
                beta = 0.0
                probs = [1.0 if action == exp.base.ACTIONS[0] else 0.0, 1.0 if action == exp.base.ACTIONS[1] else 0.0]
            else:
                pulse_vec = torch.tensor(exp.pulse_from_state(state), dtype=torch.float32)
                beta = exp.prediction_error_to_beta(error_ema)
                action, probs = exp.choose_action_with_policy(policy, model, pulse_vec, pulse_state, beta, rng)
                pred_next, next_pulse_state = model.step(
                    pulse_vec.unsqueeze(0),
                    torch.tensor([exp.base.action_pulse(action)], dtype=torch.float32),
                    pulse_state.clone(),
                )
                next_state = exp.base.transition_dynamics(state, action, disturbance=disturbance)
                target_pulses = torch.tensor(exp.pulse_from_state(next_state), dtype=torch.float32)
                pred_error = float(exp.weighted_bce(pred_next[0], target_pulses).item())
                error_ema = 0.94 * error_ema + 0.06 * pred_error
                pulse_state = next_pulse_state

            actions.append(action)
            state = exp.base.transition_dynamics(state, action, disturbance=disturbance)
            states.append(state)
            heights.append(exp.base.height_from_theta(state.theta))
            torques.append(state_torque(state))
            disturbances.append(disturbance)
            betas.append(beta)
            entropies.append(exp.action_entropy(probs))

    controlled_heights = heights[1:]
    in_band_hits = sum(1 for h in controlled_heights if exp.base.band_distance(h) <= 1e-8)
    mean_abs_error = sum(abs(h - exp.base.TARGET_HEIGHT) for h in controlled_heights) / float(max(1, len(controlled_heights)))
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


def scenario_score(rollouts: Dict[str, Trajectory]) -> float:
    prediction = rollouts["prediction_only"]
    signed = rollouts["signed_pulse"]
    twohop = rollouts["signed_twohop_pulse"]
    best = signed if signed.in_band_rate >= twohop.in_band_rate else twohop
    return (
        2.0 * (best.in_band_rate - prediction.in_band_rate)
        + 0.7 * (prediction.mean_abs_height_error - best.mean_abs_height_error)
    )


def select_scenario(model: exp.PulseCyclicSNN, scenarios: Sequence[Scenario]) -> tuple[Scenario, Dict[str, Trajectory], float]:
    best_scenario = scenarios[0]
    best_rollouts: Dict[str, Trajectory] = {}
    best_score = float("-inf")
    for scenario in scenarios:
        rollouts = {
            "prediction_only": rollout_policy("prediction_only", model, scenario),
            "signed_pulse": rollout_policy("signed_pulse", model, scenario),
            "signed_twohop_pulse": rollout_policy("signed_twohop_pulse", model, scenario),
            "planner": rollout_policy("planner", model, scenario),
        }
        score = scenario_score(rollouts)
        if score > best_score:
            best_score = score
            best_scenario = scenario
            best_rollouts = rollouts
    return best_scenario, best_rollouts, best_score


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
        draw.text((left - 48, y - 7), f"{value:.2f}", fill="#666666")


def plot_trajectories(out_path: Path, rollouts: Dict[str, Trajectory]) -> None:
    colors = {
        "prediction_only": "#4e79a7",
        "signed_pulse": "#e15759",
        "signed_twohop_pulse": "#f28e2b",
        "planner": "#59a14f",
    }
    labels = {
        "prediction_only": "Prediction only",
        "signed_pulse": "Signed pulse",
        "signed_twohop_pulse": "Signed two-hop",
        "planner": "Planner",
    }
    width = 1320
    height = 920
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = 72

    top = (margin, 80, width - margin, 300)
    mid = (margin, 380, width - margin, 580)
    bottom = (margin, 660, width - margin, 860)

    draw.text((margin, 20), "Open pulse-node SNN stick control", fill="black")
    max_step = max(len(traj.heights) for traj in rollouts.values()) - 1

    draw_axes(draw, top, "Height", -0.05, 1.05, "h")
    draw_axes(draw, mid, "Action alpha torque", -exp.base.MAX_ACTION_ALPHA, exp.base.MAX_ACTION_ALPHA, "tau")
    draw_axes(draw, bottom, "Impulse disturbance", -1.2, 1.2, "dist")

    band_top = map_point(top, 0, max_step, exp.base.TARGET_HEIGHT + exp.base.TARGET_BAND, -0.05, 1.05)[1]
    band_bottom = map_point(top, 0, max_step, exp.base.TARGET_HEIGHT - exp.base.TARGET_BAND, -0.05, 1.05)[1]
    draw.rectangle((top[0], band_top, top[2], band_bottom), fill="#e6f6e6", outline=None)
    target_y = map_point(top, 0, max_step, exp.base.TARGET_HEIGHT, -0.05, 1.05)[1]
    draw.line((top[0], target_y, top[2], target_y), fill="#4e9d4e", width=2)

    legend_x = width - 390
    legend_y = 20
    ordered = ("prediction_only", "signed_pulse", "signed_twohop_pulse", "planner")
    for idx, name in enumerate(ordered):
        traj = rollouts[name]
        color = colors[name]
        y = legend_y + idx * 20
        label = f"{labels[name]} | in-band {traj.in_band_rate*100:.1f}% | mae {traj.mean_abs_height_error:.3f}"
        draw.line((legend_x, y + 8, legend_x + 22, y + 8), fill=color, width=3)
        draw.text((legend_x + 28, y), label, fill="black")

        for step in range(1, len(traj.heights)):
            p0 = map_point(top, step - 1, max_step, traj.heights[step - 1], -0.05, 1.05)
            p1 = map_point(top, step, max_step, traj.heights[step], -0.05, 1.05)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=3)
        for step in range(1, len(traj.torques)):
            p0 = map_point(mid, step - 1, max_step, traj.torques[step - 1], -exp.base.MAX_ACTION_ALPHA, exp.base.MAX_ACTION_ALPHA)
            p1 = map_point(mid, step, max_step, traj.torques[step], -exp.base.MAX_ACTION_ALPHA, exp.base.MAX_ACTION_ALPHA)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=3)

    disturbance = rollouts["signed_pulse"].disturbances
    for step in range(1, len(disturbance)):
        p0 = map_point(bottom, step - 1, max_step, disturbance[step - 1], -1.2, 1.2)
        p1 = map_point(bottom, step, max_step, disturbance[step], -1.2, 1.2)
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill="#333333", width=3)

    image.save(out_path)


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, state: exp.base.StickState, action: str, disturbance: float, beta: float) -> None:
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

    target_angle = math.acos(exp.base.clamp(exp.base.TARGET_HEIGHT, -1.0, 1.0))
    band_half = 0.11
    for sign in (-1.0, 1.0):
        low_angle = sign * max(0.0, target_angle - band_half)
        high_angle = sign * min(math.pi - 0.05, target_angle + band_half)
        low_x = cx + length * math.sin(low_angle)
        low_y = base_y - length * math.cos(low_angle)
        high_x = cx + length * math.sin(high_angle)
        high_y = base_y - length * math.cos(high_angle)
        draw.line((cx, base_y, low_x, low_y), fill="#d5ecd5", width=2)
        draw.line((cx, base_y, high_x, high_y), fill="#d5ecd5", width=2)

    height = exp.base.height_from_theta(state.theta)
    torque = exp.base.right_alpha(state) - exp.base.left_alpha(state)
    draw.text((left + 8, top + 28), f"h={height:.3f}", fill="#333333")
    draw.text((left + 8, top + 46), f"tau={torque:.3f}", fill="#333333")
    draw.text((left + 8, top + 64), f"a={action}", fill="#333333")
    draw.text((left + 8, top + 82), f"dist={disturbance:+.2f}", fill="#333333")
    draw.text((left + 8, top + 100), f"beta={beta:.2f}", fill="#333333")


def make_animation(out_path: Path, rollouts: Dict[str, Trajectory], frame_ms: int) -> None:
    ordered = ["prediction_only", "signed_pulse", "signed_twohop_pulse", "planner"]
    max_steps = max(len(rollouts[name].states) for name in ordered)
    frames: List[Image.Image] = []

    for step in range(max_steps):
        image = Image.new("RGB", (1660, 450), "white")
        draw = ImageDraw.Draw(image)
        draw.text((28, 18), f"Open pulse-node SNN stick control | step {step}", fill="black")

        boxes = [
            (20, 56, 410, 414),
            (430, 56, 820, 414),
            (840, 56, 1230, 414),
            (1250, 56, 1640, 414),
        ]
        for box, name in zip(boxes, ordered):
            traj = rollouts[name]
            idx = min(step, len(traj.states) - 1)
            action = traj.actions[idx - 1] if idx > 0 and idx - 1 < len(traj.actions) else "-"
            disturbance = traj.disturbances[idx] if idx < len(traj.disturbances) else 0.0
            beta = traj.betas[idx] if idx < len(traj.betas) else 0.0
            draw_panel(draw, box, name.replace("_", " "), traj.states[idx], action, disturbance, beta)
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
    parser = argparse.ArgumentParser(description="Visualize open pulse-node SNN stick control.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=84)
    parser.add_argument("--disturbance-scale", type=float, default=0.72)
    parser.add_argument("--train-episodes", type=int, default=96)
    parser.add_argument("--val-episodes", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--png", type=str, default="snn_pulse_stick_control_visual.png")
    parser.add_argument("--gif", type=str, default="snn_pulse_stick_control_visual.gif")
    parser.add_argument("--json", type=str, default="snn_pulse_stick_control_visual.json")
    parser.add_argument("--frame-ms", type=int, default=130)
    args = parser.parse_args()

    model, metrics = train_model(args.seed, args.train_episodes, args.val_episodes, args.epochs)
    scenarios = sample_scenarios(args.seed + 100, args.scenarios, args.horizon, args.disturbance_scale)
    scenario, rollouts, score = select_scenario(model, scenarios)

    png_path = Path(args.png).resolve()
    gif_path = Path(args.gif).resolve()
    json_path = Path(args.json).resolve()

    plot_trajectories(png_path, rollouts)
    make_animation(gif_path, rollouts, frame_ms=args.frame_ms)

    payload = {
        "selection_score": score,
        "world_model": asdict(metrics),
        "scenario": {
            "initial_state": asdict(scenario.initial_state),
            "disturbance_preview": list(scenario.disturbances[:18]),
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
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Trajectory plot written to {png_path}")
    print(f"Animation written to {gif_path}")
    print(f"Scenario summary written to {json_path}")


if __name__ == "__main__":
    main()
