#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target-band stick-height control with signed future attention.

This script implements a small, self-contained experiment meant to test a
specific claim:

1. A controller can be given only a target height band and a left/right action
   space.
2. A transformer world model predicts the next physical state and height.
3. Real self-attention from imagined future state tokens back to the current
   action token can be used as a prospective bias.
4. A signed variant of that bias, based only on whether predicted future
   heights move toward or away from the target band, should stabilize control
   better than a positive-only variant.

The experiment deliberately avoids EFE. The external baseline is a small
deterministic short-horizon viability planner defined on the true dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: Tuple[str, ...] = ("left", "right")
ACTION_VALUES: Dict[str, float] = {"left": -1.0, "right": 1.0}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

MAX_OMEGA = 3.5
MAX_DRIVE = 2.5
FAIL_ANGLE = 1.55
DT = 0.22
DAMPING = 0.94
DRIVE_DECAY = 0.82
DRIVE_PUSH = 0.78
TORQUE_SCALE = 0.19
GRAVITY_SCALE = 0.48
NOISE_STD = 0.055

TARGET_HEIGHT = 0.72
TARGET_BAND = 0.08
EPISODE_HORIZON = 60

HISTORY_STEPS = 6
FA_HORIZON = 4
FA_DISCOUNT = 0.88
PLANNER_DEPTH = 5
PLANNER_DISCOUNT = 0.90

TOKEN_DIM = 6
STATE_DIM = 4


@dataclass(frozen=True)
class StickState:
    theta: float
    omega: float
    drive: float


@dataclass(frozen=True)
class EvalMetrics:
    in_band_rate: float
    survival_rate: float
    mean_abs_height_error: float
    planner_agreement: float
    mean_episode_length: float


@dataclass(frozen=True)
class WorldModelMetrics:
    val_loss: float
    val_height_mae: float


@dataclass(frozen=True)
class EvalScenario:
    history_states: Tuple[StickState, ...]
    history_actions: Tuple[str, ...]
    disturbances: Tuple[float, ...]
    random_action_seed: int


@dataclass(frozen=True)
class RepresentativeCase:
    theta: float
    omega: float
    drive: float
    height: float
    planner_action: str
    prediction_only_action: str
    positive_fa_action: str
    signed_fa_action: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def height_from_theta(theta: float) -> float:
    return math.cos(theta)


def state_height(state: StickState) -> float:
    return height_from_theta(state.theta)


def failed(state: StickState) -> bool:
    return abs(state.theta) >= FAIL_ANGLE


def band_distance(height: float) -> float:
    return max(0.0, abs(height - TARGET_HEIGHT) - TARGET_BAND)


def band_signal(height: float) -> float:
    return 1.0 if band_distance(height) <= 1e-8 else -1.0


def progress_signal(current_height: float, future_height: float) -> float:
    current_distance = band_distance(current_height)
    future_distance = band_distance(future_height)
    if current_distance <= 1e-8 and future_distance <= 1e-8:
        return 1.0
    if current_distance <= 1e-8 and future_distance > 1e-8:
        return -1.0
    if future_distance <= 1e-8 and current_distance > 1e-8:
        return 1.0
    improvement = current_distance - future_distance
    return clamp(improvement / max(TARGET_BAND, 1e-6), -1.0, 1.0)


def upward_signal(current_height: float, future_height: float) -> float:
    return max(future_height - current_height, 0.0)


def base_height_score(height: float) -> float:
    return -band_distance(height)


def state_to_vector(state: StickState) -> Tuple[float, float, float, float]:
    return (
        state.theta / math.pi,
        state.omega / MAX_OMEGA,
        state.drive / MAX_DRIVE,
        state_height(state),
    )


def vector_to_state(pred: Sequence[float]) -> StickState:
    theta = clamp(float(pred[0]), -1.0, 1.0) * math.pi
    omega = clamp(float(pred[1]), -1.0, 1.0) * MAX_OMEGA
    drive = clamp(float(pred[2]), -1.0, 1.0) * MAX_DRIVE
    return StickState(theta=theta, omega=omega, drive=drive)


def state_token(state: StickState) -> List[float]:
    theta_norm, omega_norm, drive_norm, height = state_to_vector(state)
    return [theta_norm, omega_norm, drive_norm, height, 0.0, 0.0]


def action_token(action: str) -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, ACTION_VALUES[action], 1.0]


def transition_dynamics(state: StickState, action: str, disturbance: float) -> StickState:
    torque = ACTION_VALUES[action]
    drive = clamp(DRIVE_DECAY * state.drive + DRIVE_PUSH * torque, -MAX_DRIVE, MAX_DRIVE)
    omega = DAMPING * state.omega + TORQUE_SCALE * drive - GRAVITY_SCALE * math.sin(state.theta) + disturbance
    omega = clamp(omega, -MAX_OMEGA, MAX_OMEGA)
    theta = wrap_angle(state.theta + DT * omega)
    return StickState(theta=theta, omega=omega, drive=drive)


def deterministic_step(state: StickState, action: str) -> StickState:
    return transition_dynamics(state, action, disturbance=0.0)


def stochastic_step(state: StickState, action: str, rng: random.Random, noise_std: float = NOISE_STD) -> StickState:
    return transition_dynamics(state, action, disturbance=rng.gauss(0.0, noise_std))


def random_initial_state(rng: random.Random) -> StickState:
    return StickState(
        theta=rng.uniform(-1.15, 1.15),
        omega=rng.uniform(-0.75, 0.75),
        drive=rng.uniform(-0.35, 0.35),
    )


def quantize_planner_state(state: StickState) -> Tuple[int, int, int]:
    theta_bin = int(round(((state.theta + FAIL_ANGLE) / (2.0 * FAIL_ANGLE)) * 40.0))
    omega_bin = int(round(((state.omega + MAX_OMEGA) / (2.0 * MAX_OMEGA)) * 30.0))
    drive_bin = int(round(((state.drive + MAX_DRIVE) / (2.0 * MAX_DRIVE)) * 18.0))
    return int(clamp(theta_bin, 0, 40)), int(clamp(omega_bin, 0, 30)), int(clamp(drive_bin, 0, 18))


def dequantize_planner_state(theta_bin: int, omega_bin: int, drive_bin: int) -> StickState:
    theta = -FAIL_ANGLE + (2.0 * FAIL_ANGLE) * (theta_bin / 40.0)
    omega = -MAX_OMEGA + (2.0 * MAX_OMEGA) * (omega_bin / 30.0)
    drive = -MAX_DRIVE + (2.0 * MAX_DRIVE) * (drive_bin / 18.0)
    return StickState(theta=theta, omega=omega, drive=drive)


class PlannerCache:
    def __init__(self) -> None:
        self.values: Dict[Tuple[int, int, int, int], float] = {}

    def clear(self) -> None:
        self.values.clear()


PLANNER_CACHE = PlannerCache()


def planner_value(state: StickState, depth: int) -> float:
    if depth <= 0:
        return 0.0
    if failed(state):
        return -4.0
    theta_bin, omega_bin, drive_bin = quantize_planner_state(state)
    key = (theta_bin, omega_bin, drive_bin, depth)
    if key in PLANNER_CACHE.values:
        return PLANNER_CACHE.values[key]

    best = float("-inf")
    for action in ACTIONS:
        next_state = deterministic_step(dequantize_planner_state(theta_bin, omega_bin, drive_bin), action)
        score = band_signal(state_height(next_state)) + PLANNER_DISCOUNT * planner_value(next_state, depth - 1)
        best = max(best, score)
    PLANNER_CACHE.values[key] = best
    return best


def planner_action_scores(state: StickState, depth: int = PLANNER_DEPTH) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for action in ACTIONS:
        next_state = deterministic_step(state, action)
        scores[action] = band_signal(state_height(next_state)) + PLANNER_DISCOUNT * planner_value(next_state, depth - 1)
    return scores


def argmax_action(scores: Dict[str, float]) -> str:
    return max(ACTIONS, key=lambda action: (scores[action], action))


def collect_episode(rng: random.Random, horizon: int, noise_std: float) -> Tuple[List[StickState], List[str]]:
    states = [random_initial_state(rng)]
    actions: List[str] = []
    current = states[0]
    for _ in range(horizon):
        action = rng.choice(ACTIONS)
        actions.append(action)
        current = stochastic_step(current, action, rng, noise_std=noise_std)
        states.append(current)
        if failed(current):
            break
    while len(actions) < horizon:
        actions.append(rng.choice(ACTIONS))
        states.append(states[-1])
    return states[: horizon + 1], actions[:horizon]


def collect_dataset(seed: int, episodes: int, horizon: int, noise_std: float) -> List[Tuple[List[StickState], List[str]]]:
    rng = random.Random(seed)
    return [collect_episode(rng, horizon, noise_std=noise_std) for _ in range(episodes)]


def build_samples(
    episodes: Sequence[Tuple[List[StickState], List[str]]],
    history_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs: List[List[List[float]]] = []
    targets: List[List[float]] = []
    for states, actions in episodes:
        usable_steps = min(len(actions), len(states) - 1)
        if usable_steps < history_steps:
            continue
        for step_idx in range(history_steps - 1, usable_steps):
            token_seq: List[List[float]] = []
            start_idx = step_idx - history_steps + 1
            for idx in range(start_idx, step_idx + 1):
                token_seq.append(state_token(states[idx]))
                token_seq.append(action_token(actions[idx]))
            inputs.append(token_seq)
            targets.append(list(state_to_vector(states[step_idx + 1])))
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.proj(out), attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn


class StickWorldModel(nn.Module):
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Linear(token_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, STATE_DIM)

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = tokens.size(1)
        x = self.token_proj(tokens) + self.pos_emb[:, :seq_len, :]
        x = self.dropout(x)
        last_attn = None
        for block in self.blocks:
            x, last_attn = block(x)
        assert last_attn is not None
        return self.ln_f(x), last_attn

    def predict_next_state(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encode(tokens)
        return self.head(hidden[:, -1, :])


def batch_iter(x: torch.Tensor, y: torch.Tensor, batch_size: int, rng: random.Random) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    indices = list(range(x.size(0)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        yield x[batch_ids].to(DEVICE), y[batch_ids].to(DEVICE)


def evaluate_world_model(model: StickWorldModel, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> WorldModelMetrics:
    model.eval()
    losses: List[float] = []
    height_errors: List[float] = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            batch_x = x[start : start + batch_size].to(DEVICE)
            batch_y = y[start : start + batch_size].to(DEVICE)
            pred = model.predict_next_state(batch_x)
            losses.append(float(F.mse_loss(pred, batch_y).item()))
            height_errors.extend(torch.abs(pred[:, 2] - batch_y[:, 2]).cpu().tolist())
    return WorldModelMetrics(
        val_loss=statistics.mean(losses) if losses else 0.0,
        val_height_mae=statistics.mean(height_errors) if height_errors else 0.0,
    )


def train_world_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_ff: int,
    dropout: float,
    seed: int,
) -> Tuple[StickWorldModel, WorldModelMetrics]:
    model = StickWorldModel(
        token_dim=TOKEN_DIM,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        max_seq_len=2 * HISTORY_STEPS + 2 * FA_HORIZON + 2,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    best_metric = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in batch_iter(train_x, train_y, batch_size=batch_size, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            pred = model.predict_next_state(batch_x)
            loss = F.mse_loss(pred, batch_y)
            total_loss = loss + 0.35 * F.l1_loss(pred[:, 2], batch_y[:, 2])
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        metrics = evaluate_world_model(model, val_x, val_y, batch_size=batch_size)
        if metrics.val_loss < best_metric:
            best_metric = metrics.val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate_world_model(model, val_x, val_y, batch_size=batch_size)


def history_to_tokens(states: Sequence[StickState], actions: Sequence[str]) -> List[List[float]]:
    assert len(states) == len(actions) + 1
    tokens: List[List[float]] = []
    for state, action in zip(states[:-1], actions):
        tokens.append(state_token(state))
        tokens.append(action_token(action))
    tokens.append(state_token(states[-1]))
    return tokens


def predict_state_from_tokens(model: StickWorldModel, tokens: Sequence[Sequence[float]]) -> StickState:
    batch = torch.tensor([tokens], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = model.predict_next_state(batch)[0].detach().cpu().tolist()
    return vector_to_state(pred)


def rollout_action_by_prediction(model: StickWorldModel, tokens: Sequence[Sequence[float]]) -> str:
    scores: Dict[str, float] = {}
    for action in ACTIONS:
        pred_state = predict_state_from_tokens(model, list(tokens) + [action_token(action)])
        scores[action] = base_height_score(state_height(pred_state))
    return argmax_action(scores)


def imagined_attention_scores(
    model: StickWorldModel,
    history_tokens_seq: Sequence[Sequence[float]],
    current_height: float,
    action: str,
    horizon: int,
    discount: float,
    rollout_mode: str,
) -> Dict[str, float]:
    context: List[List[float]] = [list(token) for token in history_tokens_seq]
    context.append(action_token(action))
    candidate_index = len(context) - 1

    first_pred_state = predict_state_from_tokens(model, context)
    base_score = base_height_score(state_height(first_pred_state))

    positive_fa = 0.0
    signed_fa = 0.0
    current_pred = first_pred_state

    for step_idx in range(horizon):
        context.append(state_token(current_pred))
        batch = torch.tensor([context], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            _, attn = model.encode(batch)
        attn_to_candidate = float(attn.mean(dim=1)[0, -1, candidate_index].item())
        future_height = state_height(current_pred)
        positive_fa += (discount**step_idx) * attn_to_candidate * upward_signal(current_height, future_height)
        signal = progress_signal(current_height, future_height)
        signed_fa += (discount**step_idx) * attn_to_candidate * signal

        if step_idx + 1 >= horizon:
            break
        if rollout_mode == "repeat":
            rollout_action = action
        elif rollout_mode == "prediction_only":
            rollout_action = rollout_action_by_prediction(model, context)
        else:
            raise ValueError(f"unknown rollout mode: {rollout_mode}")
        context.append(action_token(rollout_action))
        current_pred = predict_state_from_tokens(model, context)

    return {"base_score": base_score, "positive_fa": positive_fa, "signed_fa": signed_fa}


def choose_action(
    policy_name: str,
    model: StickWorldModel,
    history_states: Sequence[StickState],
    history_actions: Sequence[str],
    fa_weight: float,
    rollout_mode: str,
) -> str:
    if policy_name == "planner":
        return argmax_action(planner_action_scores(history_states[-1]))
    history_tokens_seq = history_to_tokens(history_states, history_actions)
    current_height = state_height(history_states[-1])
    scores: Dict[str, float] = {}
    for action in ACTIONS:
        imagined = imagined_attention_scores(
            model,
            history_tokens_seq,
            current_height=current_height,
            action=action,
            horizon=FA_HORIZON,
            discount=FA_DISCOUNT,
            rollout_mode=rollout_mode,
        )
        if policy_name == "prediction_only":
            scores[action] = imagined["base_score"]
        elif policy_name == "positive_fa":
            scores[action] = imagined["base_score"] + fa_weight * imagined["positive_fa"]
        elif policy_name == "signed_fa":
            scores[action] = imagined["base_score"] + fa_weight * imagined["signed_fa"]
        else:
            raise ValueError(f"unknown policy: {policy_name}")
    return argmax_action(scores)


def evaluate_policy(
    policy_name: str,
    model: StickWorldModel,
    horizon: int,
    fa_weight: float,
    scenarios: Sequence[EvalScenario],
    rollout_mode: str,
) -> EvalMetrics:
    total_band_hits = 0
    total_steps = 0
    survived = 0
    total_abs_height_error = 0.0
    planner_matches = 0
    planner_total = 0
    episode_lengths: List[int] = []

    for scenario in scenarios:
        history_states = list(scenario.history_states)
        history_actions = list(scenario.history_actions)
        state = history_states[-1]
        action_rng = random.Random(scenario.random_action_seed)

        steps_taken = 0
        failed_episode = False
        for disturbance in scenario.disturbances[:horizon]:
            if failed(state):
                failed_episode = True
                break
            if policy_name == "random":
                action = action_rng.choice(ACTIONS)
            else:
                action = choose_action(policy_name, model, history_states, history_actions, fa_weight, rollout_mode=rollout_mode)

            planner_action = argmax_action(planner_action_scores(state))
            planner_matches += int(action == planner_action)
            planner_total += 1

            state = transition_dynamics(state, action, disturbance=disturbance)
            height = state_height(state)
            total_band_hits += int(band_distance(height) <= 1e-8)
            total_abs_height_error += abs(height - TARGET_HEIGHT)
            total_steps += 1
            steps_taken += 1

            history_states = history_states[1:] + [state]
            history_actions = history_actions[1:] + [action]

        if not failed_episode and not failed(state):
            survived += 1
        episode_lengths.append(steps_taken)

    return EvalMetrics(
        in_band_rate=total_band_hits / float(max(1, total_steps)),
        survival_rate=survived / float(max(1, len(scenarios))),
        mean_abs_height_error=total_abs_height_error / float(max(1, total_steps)),
        planner_agreement=planner_matches / float(max(1, planner_total)),
        mean_episode_length=statistics.mean(episode_lengths) if episode_lengths else 0.0,
    )


def build_eval_scenarios(
    seed: int,
    episodes: int,
    history_steps: int,
    horizon: int,
    noise_std: float,
) -> List[EvalScenario]:
    rng = random.Random(seed)
    scenarios: List[EvalScenario] = []
    for episode_idx in range(episodes):
        del episode_idx
        initial_state = random_initial_state(rng)
        history_states = [initial_state]
        history_actions: List[str] = []
        current = initial_state
        for _ in range(history_steps - 1):
            action = rng.choice(ACTIONS)
            disturbance = rng.gauss(0.0, noise_std)
            history_actions.append(action)
            current = transition_dynamics(current, action, disturbance=disturbance)
            history_states.append(current)
        disturbances = tuple(rng.gauss(0.0, noise_std) for _ in range(horizon))
        scenarios.append(
            EvalScenario(
                history_states=tuple(history_states),
                history_actions=tuple(history_actions),
                disturbances=disturbances,
                random_action_seed=rng.randint(0, 10**9),
            )
        )
    return scenarios


def representative_cases(model: StickWorldModel, fa_weight: float, rollout_mode: str) -> List[RepresentativeCase]:
    cases = [
        StickState(theta=-0.95, omega=0.10, drive=-0.30),
        StickState(theta=-0.65, omega=0.60, drive=0.45),
        StickState(theta=0.30, omega=0.00, drive=0.10),
        StickState(theta=0.95, omega=-0.10, drive=0.30),
    ]
    results: List[RepresentativeCase] = []
    for state in cases:
        history_states = [state for _ in range(HISTORY_STEPS)]
        history_actions = ["left", "right", "left", "right", "left"]
        results.append(
            RepresentativeCase(
                theta=state.theta,
                omega=state.omega,
                drive=state.drive,
                height=state_height(state),
                planner_action=argmax_action(planner_action_scores(state)),
                prediction_only_action=choose_action("prediction_only", model, history_states, history_actions, fa_weight, rollout_mode),
                positive_fa_action=choose_action("positive_fa", model, history_states, history_actions, fa_weight, rollout_mode),
                signed_fa_action=choose_action("signed_fa", model, history_states, history_actions, fa_weight, rollout_mode),
            )
        )
    return results


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def pack_metric(values: Sequence[float]) -> Dict[str, float]:
    mean_value, std_value = mean_std(values)
    return {"mean": mean_value, "std": std_value}


def aggregate_metrics(metrics: Sequence[EvalMetrics]) -> Dict[str, Dict[str, float]]:
    return {
        "in_band_rate": pack_metric([item.in_band_rate for item in metrics]),
        "survival_rate": pack_metric([item.survival_rate for item in metrics]),
        "mean_abs_height_error": pack_metric([item.mean_abs_height_error for item in metrics]),
        "planner_agreement": pack_metric([item.planner_agreement for item in metrics]),
        "mean_episode_length": pack_metric([item.mean_episode_length for item in metrics]),
    }


def collect_offline_episodes(
    seed: int,
    episodes: int,
    horizon: int,
    noise_mode: str,
    nominal_noise_std: float,
    high_noise_std: float,
) -> List[Tuple[List[StickState], List[str]]]:
    if noise_mode == "nominal":
        return collect_dataset(seed=seed, episodes=episodes, horizon=horizon, noise_std=nominal_noise_std)
    if noise_mode == "high":
        return collect_dataset(seed=seed, episodes=episodes, horizon=horizon, noise_std=high_noise_std)
    if noise_mode == "mixed":
        first_count = episodes // 2
        second_count = episodes - first_count
        mixed = collect_dataset(seed=seed, episodes=first_count, horizon=horizon, noise_std=nominal_noise_std)
        mixed.extend(collect_dataset(seed=seed + 1, episodes=second_count, horizon=horizon, noise_std=high_noise_std))
        rng = random.Random(seed + 999)
        rng.shuffle(mixed)
        return mixed
    raise ValueError(f"unknown offline noise mode: {noise_mode}")


def format_metric(entry: Dict[str, float], pct: bool = False) -> str:
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    return f"{entry['mean'] * scale:.2f}±{entry['std'] * scale:.2f}{suffix}"


def write_markdown(path: str, summary: Dict[str, object], cases: Sequence[RepresentativeCase]) -> None:
    world_model = summary["world_model"]
    nominal_policies = summary["nominal_policies"]
    stress_policies = summary["stress_policies"]
    lines = [
        "# Stick Height Signed-FA Experiment",
        "",
        "Target-band control using a transformer world model and real self-attention.",
        "",
        f"- `target_height = {TARGET_HEIGHT:.2f}`",
        f"- `target_band = ±{TARGET_BAND:.2f}`",
        f"- `episode_horizon = {EPISODE_HORIZON}`",
        f"- `history_steps = {HISTORY_STEPS}`",
        f"- `fa_horizon = {FA_HORIZON}`",
        f"- `offline_noise_mode = {summary['config']['offline_noise_mode']}`",
        f"- `rollout_mode = {summary['config']['rollout_mode']}`",
        "",
        "## World Model",
        "",
        f"- `val_loss = {world_model['val_loss']['mean']:.4f} ± {world_model['val_loss']['std']:.4f}`",
        f"- `val_height_mae = {world_model['val_height_mae']['mean']:.4f} ± {world_model['val_height_mae']['std']:.4f}`",
        "",
        "## Nominal Policy Results",
        "",
        "| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy_name, label in (
        ("random", "Random"),
        ("prediction_only", "Prediction only"),
        ("positive_fa", "Positive-only FA"),
        ("signed_fa", "Signed FA"),
        ("planner", "Planner"),
    ):
        block = nominal_policies[policy_name]
        lines.append(
            f"| {label} | {format_metric(block['in_band_rate'], pct=True)} | "
            f"{format_metric(block['survival_rate'], pct=True)} | {format_metric(block['mean_abs_height_error'])} | "
            f"{format_metric(block['planner_agreement'], pct=True)} | {format_metric(block['mean_episode_length'])} |"
        )

    lines.extend(
        [
            "",
            "## Stress Policy Results",
            "",
            "| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_name, label in (
        ("random", "Random"),
        ("prediction_only", "Prediction only"),
        ("positive_fa", "Positive-only FA"),
        ("signed_fa", "Signed FA"),
        ("planner", "Planner"),
    ):
        block = stress_policies[policy_name]
        lines.append(
            f"| {label} | {format_metric(block['in_band_rate'], pct=True)} | "
            f"{format_metric(block['survival_rate'], pct=True)} | {format_metric(block['mean_abs_height_error'])} | "
            f"{format_metric(block['planner_agreement'], pct=True)} | {format_metric(block['mean_episode_length'])} |"
        )

    lines.extend(
        [
            "",
            "## Representative States",
            "",
            "| Theta | Omega | Drive | Height | Planner | Prediction | Positive FA | Signed FA |",
            "|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for case in cases:
        lines.append(
            f"| {case.theta:.2f} | {case.omega:.2f} | {case.drive:.2f} | {case.height:.2f} | {case.planner_action} | "
            f"{case.prediction_only_action} | {case.positive_fa_action} | {case.signed_fa_action} |"
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stick-height target-band control with signed future attention.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--offline-episodes", type=int, default=120)
    parser.add_argument("--offline-horizon", type=int, default=EPISODE_HORIZON)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--fa-weight", type=float, default=0.7)
    parser.add_argument("--offline-noise-mode", type=str, default="mixed", choices=("nominal", "high", "mixed"))
    parser.add_argument("--offline-high-noise-scale", type=float, default=1.6)
    parser.add_argument("--stress-noise-scale", type=float, default=1.8)
    parser.add_argument("--rollout-mode", type=str, default="repeat", choices=("repeat", "prediction_only"))
    parser.add_argument("--json", type=str, default="stick_height_signed_fa_results.json")
    parser.add_argument("--md", type=str, default="stick_height_signed_fa_results.md")
    args = parser.parse_args()

    world_model_metrics: List[WorldModelMetrics] = []
    nominal_policy_metrics: Dict[str, List[EvalMetrics]] = {
        name: [] for name in ("random", "prediction_only", "positive_fa", "signed_fa", "planner")
    }
    stress_policy_metrics: Dict[str, List[EvalMetrics]] = {
        name: [] for name in ("random", "prediction_only", "positive_fa", "signed_fa", "planner")
    }
    last_model: StickWorldModel | None = None

    for run_idx in range(args.runs):
        base_seed = 200 + 37 * run_idx
        torch.manual_seed(base_seed)
        random.seed(base_seed)
        PLANNER_CACHE.clear()

        episodes = collect_offline_episodes(
            seed=base_seed,
            episodes=args.offline_episodes,
            horizon=args.offline_horizon,
            noise_mode=args.offline_noise_mode,
            nominal_noise_std=NOISE_STD,
            high_noise_std=NOISE_STD * args.offline_high_noise_scale,
        )
        split = max(1, int(len(episodes) * 0.85))
        train_episodes = episodes[:split]
        val_episodes = episodes[split:]
        train_x, train_y = build_samples(train_episodes, history_steps=HISTORY_STEPS)
        val_x, val_y = build_samples(val_episodes or train_episodes[:1], history_steps=HISTORY_STEPS)

        model, wm_metrics = train_world_model(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            seed=base_seed,
        )
        last_model = model
        world_model_metrics.append(wm_metrics)
        nominal_scenarios = build_eval_scenarios(
            seed=base_seed + 1000,
            episodes=args.eval_episodes,
            history_steps=HISTORY_STEPS,
            horizon=EPISODE_HORIZON,
            noise_std=NOISE_STD,
        )
        stress_scenarios = build_eval_scenarios(
            seed=base_seed + 2000,
            episodes=args.eval_episodes,
            history_steps=HISTORY_STEPS,
            horizon=EPISODE_HORIZON,
            noise_std=NOISE_STD * args.stress_noise_scale,
        )

        for policy_name in nominal_policy_metrics.keys():
            nominal_policy_metrics[policy_name].append(
                evaluate_policy(
                    policy_name=policy_name,
                    model=model,
                    horizon=EPISODE_HORIZON,
                    fa_weight=args.fa_weight,
                    scenarios=nominal_scenarios,
                    rollout_mode=args.rollout_mode,
                )
            )
            stress_policy_metrics[policy_name].append(
                evaluate_policy(
                    policy_name=policy_name,
                    model=model,
                    horizon=EPISODE_HORIZON,
                    fa_weight=args.fa_weight,
                    scenarios=stress_scenarios,
                    rollout_mode=args.rollout_mode,
                )
            )

    assert last_model is not None
    cases = representative_cases(last_model, fa_weight=args.fa_weight, rollout_mode=args.rollout_mode)

    summary = {
        "config": {
            "runs": args.runs,
            "offline_episodes": args.offline_episodes,
            "offline_horizon": args.offline_horizon,
            "eval_episodes": args.eval_episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dim_ff": args.dim_ff,
            "dropout": args.dropout,
            "fa_weight": args.fa_weight,
            "offline_noise_mode": args.offline_noise_mode,
            "offline_high_noise_std": NOISE_STD * args.offline_high_noise_scale,
            "rollout_mode": args.rollout_mode,
            "target_height": TARGET_HEIGHT,
            "target_band": TARGET_BAND,
            "history_steps": HISTORY_STEPS,
            "fa_horizon": FA_HORIZON,
            "device": str(DEVICE),
            "nominal_noise_std": NOISE_STD,
            "stress_noise_std": NOISE_STD * args.stress_noise_scale,
        },
        "world_model": {
            "val_loss": pack_metric([item.val_loss for item in world_model_metrics]),
            "val_height_mae": pack_metric([item.val_height_mae for item in world_model_metrics]),
        },
        "nominal_policies": {name: aggregate_metrics(metrics) for name, metrics in nominal_policy_metrics.items()},
        "stress_policies": {name: aggregate_metrics(metrics) for name, metrics in stress_policy_metrics.items()},
        "representative_cases": [asdict(case) for case in cases],
    }

    print("=" * 118)
    print("Stick height signed-FA experiment")
    print("Interpretation: does signed future attention help keep the stick inside a target height band?")
    print("=" * 118)
    print(
        f"World model | val_loss={format_metric(summary['world_model']['val_loss'])} | "
        f"val_height_mae={format_metric(summary['world_model']['val_height_mae'])}"
    )
    print("-" * 118)
    print("Nominal noise")
    print(f"{'Policy':20s} | {'InBand':>12s} | {'Survival':>11s} | {'Height MAE':>12s} | {'PlanAgree':>11s} | {'EpLen':>8s}")
    print("-" * 118)
    for policy_name, label in (
        ("random", "Random"),
        ("prediction_only", "Prediction only"),
        ("positive_fa", "Positive-only FA"),
        ("signed_fa", "Signed FA"),
        ("planner", "Planner"),
    ):
        block = summary["nominal_policies"][policy_name]
        print(
            f"{label:20s} | {format_metric(block['in_band_rate'], pct=True):>12s} | "
            f"{format_metric(block['survival_rate'], pct=True):>11s} | "
            f"{format_metric(block['mean_abs_height_error']):>12s} | "
            f"{format_metric(block['planner_agreement'], pct=True):>11s} | "
            f"{format_metric(block['mean_episode_length']):>8s}"
        )

    print("-" * 118)
    print("Stress noise")
    print(f"{'Policy':20s} | {'InBand':>12s} | {'Survival':>11s} | {'Height MAE':>12s} | {'PlanAgree':>11s} | {'EpLen':>8s}")
    print("-" * 118)
    for policy_name, label in (
        ("random", "Random"),
        ("prediction_only", "Prediction only"),
        ("positive_fa", "Positive-only FA"),
        ("signed_fa", "Signed FA"),
        ("planner", "Planner"),
    ):
        block = summary["stress_policies"][policy_name]
        print(
            f"{label:20s} | {format_metric(block['in_band_rate'], pct=True):>12s} | "
            f"{format_metric(block['survival_rate'], pct=True):>11s} | "
            f"{format_metric(block['mean_abs_height_error']):>12s} | "
            f"{format_metric(block['planner_agreement'], pct=True):>11s} | "
            f"{format_metric(block['mean_episode_length']):>8s}"
        )

    print("\nRepresentative states")
    for case in cases:
        print(
            f"theta={case.theta:+.2f}, omega={case.omega:+.2f}, drive={case.drive:+.2f}, height={case.height:.2f} | "
            f"planner={case.planner_action:5s} | prediction={case.prediction_only_action:5s} | "
            f"positive={case.positive_fa_action:5s} | signed={case.signed_fa_action:5s}"
        )

    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown(args.md, summary, cases)
    print(f"\nRaw results written to {args.json}")
    print(f"Markdown summary written to {args.md}")


if __name__ == "__main__":
    main()
