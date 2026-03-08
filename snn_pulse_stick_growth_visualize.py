#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize node opening / closing in the pulse-node sparse cyclic SNN.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw
import torch

import snn_pulse_stick_height_control as exp


def map_point(box: tuple[int, int, int, int], x_idx: int, x_max: int, value: float, y_min: float, y_max: float) -> tuple[float, float]:
    left, top, right, bottom = box
    x = left + (x_idx / max(1, x_max)) * (right - left)
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


def evaluate_brief(model: exp.PulseCyclicSNN, seed: int) -> Dict[str, Dict[str, float]]:
    return {
        "nominal_prediction_only": exp.evaluate_policy(model, seed + 101, "prediction_only", disturbance_scale=0.45).__dict__,
        "nominal_signed_pulse": exp.evaluate_policy(model, seed + 202, "signed_pulse", disturbance_scale=0.45).__dict__,
        "nominal_signed_twohop_pulse": exp.evaluate_policy(model, seed + 252, "signed_twohop_pulse", disturbance_scale=0.45).__dict__,
        "stress_prediction_only": exp.evaluate_policy(model, seed + 303, "prediction_only", disturbance_scale=0.72).__dict__,
        "stress_signed_pulse": exp.evaluate_policy(model, seed + 404, "signed_pulse", disturbance_scale=0.72).__dict__,
        "stress_signed_twohop_pulse": exp.evaluate_policy(model, seed + 454, "signed_twohop_pulse", disturbance_scale=0.72).__dict__,
    }


def plot_growth(out_path: Path, history: List[exp.GrowthSnapshot], summary_text: List[str]) -> None:
    width = 1400
    height = 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    margin = 80
    top = (margin, 90, width - margin, 280)
    mid = (margin, 360, width - margin, 560)
    bottom = (margin, 650, width - margin, 920)

    draw.text((margin, 22), "Pulse-node open sparse cyclic SNN growth", fill="black")

    epochs = [snap.epoch for snap in history]
    losses = [snap.val_loss for snap in history]
    active = [snap.active_nodes for snap in history]
    births = [snap.births for snap in history]
    deaths = [snap.deaths for snap in history]

    draw_axes(draw, top, "Validation loss", min(losses) * 0.95, max(max(losses), exp.GROWTH_LOSS_THRESHOLD) * 1.05, "loss")
    draw_axes(draw, mid, "Active nodes and birth/death events", 0, exp.MODEL_HIDDEN + 4, "nodes")
    draw.rectangle(bottom, outline="#cccccc", width=1)
    draw.text((bottom[0], bottom[1] - 24), "Active-mask timeline (green=active)", fill="black")

    x_max = len(history) - 1 if len(history) > 1 else 1
    for i in range(1, len(history)):
        p0 = map_point(top, i - 1, x_max, losses[i - 1], min(losses) * 0.95, max(max(losses), exp.GROWTH_LOSS_THRESHOLD) * 1.05)
        p1 = map_point(top, i, x_max, losses[i], min(losses) * 0.95, max(max(losses), exp.GROWTH_LOSS_THRESHOLD) * 1.05)
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill="#4e79a7", width=3)
        p0 = map_point(mid, i - 1, x_max, active[i - 1], 0, exp.MODEL_HIDDEN + 4)
        p1 = map_point(mid, i, x_max, active[i], 0, exp.MODEL_HIDDEN + 4)
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill="#59a14f", width=3)

    threshold_y = map_point(top, 0, x_max, exp.GROWTH_LOSS_THRESHOLD, min(losses) * 0.95, max(max(losses), exp.GROWTH_LOSS_THRESHOLD) * 1.05)[1]
    draw.line((top[0], threshold_y, top[2], threshold_y), fill="#e15759", width=2)
    draw.text((top[2] - 170, threshold_y - 18), f"grow threshold {exp.GROWTH_LOSS_THRESHOLD:.3f}", fill="#e15759")

    prev_birth = 0
    prev_death = 0
    for i, snap in enumerate(history):
        x = map_point(mid, i, x_max, 0, 0, exp.MODEL_HIDDEN + 4)[0]
        draw.text((x - 10, mid[3] + 10), f"{snap.epoch}", fill="#666666")
        birth_delta = snap.births - prev_birth
        death_delta = snap.deaths - prev_death
        prev_birth = snap.births
        prev_death = snap.deaths
        if birth_delta > 0:
            draw.line((x, mid[3] - 8, x, mid[3] - 8 - 18 * birth_delta), fill="#2ca02c", width=5)
            draw.text((x + 4, mid[3] - 22 - 18 * birth_delta), f"+{birth_delta}", fill="#2ca02c")
        if death_delta > 0:
            draw.line((x, mid[3] - 8, x, mid[3] - 8 - 18 * death_delta), fill="#d62728", width=5)
            draw.text((x + 4, mid[3] - 22 - 18 * death_delta), f"-{death_delta}", fill="#d62728")

    rows = exp.MODEL_HIDDEN
    cols = len(history)
    if cols == 0:
        image.save(out_path)
        return

    cell_w = max(4, int((bottom[2] - bottom[0] - 100) / max(1, cols)))
    cell_h = max(2, int((bottom[3] - bottom[1] - 30) / max(1, rows)))
    grid_left = bottom[0] + 60
    grid_top = bottom[1] + 10
    for row in range(rows):
        if row % 8 == 0:
            draw.text((bottom[0] + 4, grid_top + row * cell_h - 4), f"{row}", fill="#666666")
        for col, snap in enumerate(history):
            x0 = grid_left + col * cell_w
            y0 = grid_top + row * cell_h
            active_value = snap.active_mask[row]
            fill = "#4e9d4e" if active_value > 0 else "#f0f0f0"
            draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), fill=fill, outline=None)
    for col, snap in enumerate(history):
        x0 = grid_left + col * cell_w
        draw.text((x0, bottom[3] - 16), f"{snap.epoch}", fill="#666666")

    text_y = 295
    for line in summary_text:
        draw.text((margin, text_y), line, fill="#333333")
        text_y += 18

    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize node opening/closing in pulse-node SNN training.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--png", type=str, default="snn_pulse_stick_growth_visual.png")
    parser.add_argument("--json", type=str, default="snn_pulse_stick_growth_visual.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, metrics, history = exp.train_world_model_with_history(args.seed)
    evaluation = evaluate_brief(model, args.seed)

    png_path = Path(args.png).resolve()
    json_path = Path(args.json).resolve()

    summary_text = [
        f"final active nodes: {int(metrics.active_nodes)} / {exp.MODEL_HIDDEN}",
        f"births: {int(metrics.births)}   deaths: {int(metrics.deaths)}",
        f"val loss: {metrics.val_loss:.4f}   in-band BCE: {metrics.val_in_band_bce:.4f}",
        f"nominal signed_pulse in-band: {evaluation['nominal_signed_pulse']['in_band_rate']*100:.1f}%",
        f"nominal signed_twohop_pulse in-band: {evaluation['nominal_signed_twohop_pulse']['in_band_rate']*100:.1f}%",
        f"stress signed_pulse in-band: {evaluation['stress_signed_pulse']['in_band_rate']*100:.1f}%",
        f"stress signed_twohop_pulse in-band: {evaluation['stress_signed_twohop_pulse']['in_band_rate']*100:.1f}%",
    ]
    plot_growth(png_path, history, summary_text)

    payload = {
        "config": {
            "seed": args.seed,
            "hidden_dim": exp.MODEL_HIDDEN,
            "init_active_hidden": exp.INIT_ACTIVE_HIDDEN,
            "growth_loss_threshold": exp.GROWTH_LOSS_THRESHOLD,
            "latent_weight": exp.LATENT_WEIGHT,
            "latent_topk": exp.LATENT_TOPK,
        },
        "world_model": asdict(metrics),
        "history": [asdict(item) for item in history],
        "evaluation": evaluation,
        "png": str(png_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Growth plot written to {png_path}")
    print(f"Summary written to {json_path}")


if __name__ == "__main__":
    main()
