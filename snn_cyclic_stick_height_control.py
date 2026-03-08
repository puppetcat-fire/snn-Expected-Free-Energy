"""
Sparse cyclic SNN stick-height maintenance.

This script implements a small proof-of-concept controller built around the
design developed in the main branch discussion:

1. A sparse recurrent SNN-like world model with cycles predicts next-step
   required observations.
2. Action nodes use double-exponential (alpha-like) traces so a left/right
   pulse unfolds over time instead of acting as a point impulse.
3. Action choice is based on signed prospective contribution:
   contribution is measured by finite-horizon counterfactual deletion on the
   time-unrolled recurrent graph, which remains well-defined even when the
   recurrent graph contains cycles.
4. Action sampling uses a softmax whose inverse-temperature depends on recent
   predictive accuracy.
5. The recurrent graph stays sparse, but supports limited grow/prune rewiring.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIONS: Tuple[str, ...] = ("left", "right")
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}

DT = 0.12
MAX_OMEGA = 3.25
THETA_WRAP = math.pi

TARGET_HEIGHT = 0.92
TARGET_BAND = 0.07
SEMANTIC_SCALE = 0.35

TORQUE_GAIN = 1.18
GRAVITY_GAIN = 0.94
OMEGA_DAMPING = 0.955

ACTION_PULSE = 1.0
ACTION_GAIN = 1.55
ACTION_RISE_TAU = 0.18
ACTION_DECAY_TAU = 0.58
ACTION_RISE_FACTOR = math.exp(-DT / ACTION_RISE_TAU)
ACTION_DECAY_FACTOR = math.exp(-DT / ACTION_DECAY_TAU)
MAX_ACTION_ALPHA = ACTION_GAIN * 2.4

DEFAULT_DISTURBANCE_MODE = "impulse"
IMPULSE_PROB = 0.045
IMPULSE_MIN_STEPS = 2
IMPULSE_MAX_STEPS = 5
IMPULSE_JITTER_RATIO = 0.22

OBS_DIM = 7
ACTION_DIM = 2

TRAIN_EPISODES = 160
TRAIN_HORIZON = 96
VAL_EPISODES = 32
SEQ_LEN = 16

MODEL_HIDDEN = 48
MODEL_IN_DEGREE = 6
MODEL_EPOCHS = 14
MODEL_LR = 2.2e-3
BATCH_SIZE = 32

FA_HORIZON = 8
FA_DISCOUNT = 0.86
NEG_WEIGHT = 1.0
BASE_WEIGHT = 0.72
PROSPECTIVE_WEIGHT = 1.0
DELTA_WEIGHT = 0.85

EVAL_EPISODES = 12
EVAL_HORIZON = 96

REWIRE_INTERVAL = 3
REWIRE_PRUNE_PER_NODE = 1
REWIRE_GROW_PER_NODE = 1

BETA_MIN = 1.25
BETA_MAX = 6.0
ACCURACY_SCALE = 0.09

PLANNER_DEPTH = 5
PLANNER_DISCOUNT = 0.91


@dataclass(frozen=True)
class StickState:
    theta: float
    omega: float
    left_rise: float
    left_decay: float
    right_rise: float
    right_decay: float


@dataclass(frozen=True)
class WorldModelMetrics:
    val_loss: float
    val_height_mae: float


@dataclass(frozen=True)
class EvalMetrics:
    in_band_rate: float
    mean_abs_height_error: float
    planner_agreement: float
    mean_beta: float
    mean_action_entropy: float


@dataclass(frozen=True)
class RunSummary:
    config: Dict[str, float | int | str]
    world_model: Dict[str, float]
    evaluation: Dict[str, Dict[str, float]]


class SNNState:
    def __init__(
        self,
        membrane: torch.Tensor,
        spikes: torch.Tensor,
        trace: torch.Tensor,
        adapt: torch.Tensor,
        action_rise: torch.Tensor,
        action_decay: torch.Tensor,
    ) -> None:
        self.membrane = membrane
        self.spikes = spikes
        self.trace = trace
        self.adapt = adapt
        self.action_rise = action_rise
        self.action_decay = action_decay

    def clone(self) -> "SNNState":
        return SNNState(
            membrane=self.membrane.clone(),
            spikes=self.spikes.clone(),
            trace=self.trace.clone(),
            adapt=self.adapt.clone(),
            action_rise=self.action_rise.clone(),
            action_decay=self.action_decay.clone(),
        )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(theta: float) -> float:
    return (theta + THETA_WRAP) % (2.0 * THETA_WRAP) - THETA_WRAP


def band_distance(height: float) -> float:
    return max(0.0, abs(height - TARGET_HEIGHT) - TARGET_BAND)


def toward_value(height: float) -> float:
    return clamp(1.0 - band_distance(height) / SEMANTIC_SCALE, 0.0, 1.0)


def away_value(height: float) -> float:
    return clamp(band_distance(height) / SEMANTIC_SCALE, 0.0, 1.0)


def height_from_theta(theta: float) -> float:
    return math.cos(theta)


def alpha_from_traces(rise: float, decay: float) -> float:
    return clamp(ACTION_GAIN * max(decay - rise, 0.0), 0.0, MAX_ACTION_ALPHA)


def left_alpha(state: StickState) -> float:
    return alpha_from_traces(state.left_rise, state.left_decay)


def right_alpha(state: StickState) -> float:
    return alpha_from_traces(state.right_rise, state.right_decay)


def state_to_obs(state: StickState) -> List[float]:
    height = height_from_theta(state.theta)
    left = left_alpha(state)
    right = right_alpha(state)
    return [
        state.theta / math.pi,
        clamp(state.omega / MAX_OMEGA, -1.0, 1.0),
        clamp(left / MAX_ACTION_ALPHA, 0.0, 1.0),
        clamp(right / MAX_ACTION_ALPHA, 0.0, 1.0),
        height,
        toward_value(height),
        away_value(height),
    ]


def obs_height(obs: Sequence[float]) -> float:
    return clamp(float(obs[4]), -1.0, 1.0)


def obs_value(obs: Sequence[float]) -> float:
    return toward_value(obs_height(obs)) - away_value(obs_height(obs))


def action_pulse(action: str | None) -> List[float]:
    if action is None:
        return [0.0, 0.0]
    pulse = [0.0, 0.0]
    pulse[ACTION_TO_INDEX[action]] = 1.0
    return pulse


def update_alpha_trace(rise: float, decay: float, pulse: float) -> Tuple[float, float]:
    rise = rise * ACTION_RISE_FACTOR + ACTION_PULSE * pulse
    decay = decay * ACTION_DECAY_FACTOR + ACTION_PULSE * pulse
    return rise, decay


def transition_dynamics(state: StickState, action: str | None, disturbance: float) -> StickState:
    left_pulse = 1.0 if action == "left" else 0.0
    right_pulse = 1.0 if action == "right" else 0.0

    left_rise, left_decay = update_alpha_trace(state.left_rise, state.left_decay, left_pulse)
    right_rise, right_decay = update_alpha_trace(state.right_rise, state.right_decay, right_pulse)

    torque = right_alpha(StickState(0.0, 0.0, 0.0, 0.0, right_rise, right_decay)) - left_alpha(
        StickState(0.0, 0.0, left_rise, left_decay, 0.0, 0.0)
    )
    torque += disturbance

    omega = OMEGA_DAMPING * state.omega + TORQUE_GAIN * torque - GRAVITY_GAIN * math.sin(state.theta)
    omega = clamp(omega, -MAX_OMEGA, MAX_OMEGA)
    theta = wrap_angle(state.theta + DT * omega)
    return StickState(
        theta=theta,
        omega=omega,
        left_rise=left_rise,
        left_decay=left_decay,
        right_rise=right_rise,
        right_decay=right_decay,
    )


def sample_disturbance_sequence(
    rng: random.Random,
    horizon: int,
    scale: float,
    mode: str = DEFAULT_DISTURBANCE_MODE,
) -> Tuple[float, ...]:
    if mode != "impulse":
        raise ValueError(f"unknown disturbance mode: {mode}")

    disturbances: List[float] = []
    remaining = 0
    current = 0.0
    for _ in range(horizon):
        if remaining <= 0 and rng.random() < IMPULSE_PROB:
            remaining = rng.randint(IMPULSE_MIN_STEPS, IMPULSE_MAX_STEPS)
            signed_scale = max(0.15 * scale, rng.gauss(scale, scale * IMPULSE_JITTER_RATIO))
            current = signed_scale if rng.random() < 0.5 else -signed_scale
        disturbances.append(current if remaining > 0 else 0.0)
        remaining = max(0, remaining - 1)
        if remaining <= 0:
            current = 0.0
    return tuple(disturbances)


def random_initial_state(rng: random.Random) -> StickState:
    return StickState(
        theta=rng.uniform(-1.2, 1.2),
        omega=rng.uniform(-0.9, 0.9),
        left_rise=rng.uniform(0.0, 0.2),
        left_decay=rng.uniform(0.0, 0.25),
        right_rise=rng.uniform(0.0, 0.2),
        right_decay=rng.uniform(0.0, 0.25),
    )


def heuristic_action(state: StickState, rng: random.Random) -> str:
    target_angle = math.acos(clamp(TARGET_HEIGHT, -1.0, 1.0))
    current = abs(state.theta)
    if current < target_angle - 0.06:
        return "right" if state.theta >= 0.0 else "left"
    if current > target_angle + 0.06:
        return "left" if state.theta >= 0.0 else "right"
    if abs(state.omega) > 0.25:
        return "left" if state.omega > 0.0 else "right"
    return rng.choice(ACTIONS)


def collect_episode(
    rng: random.Random,
    horizon: int,
    disturbance_scale: float,
    disturbance_mode: str,
    policy_mix: str,
) -> Tuple[List[StickState], List[str], Tuple[float, ...]]:
    state = random_initial_state(rng)
    states = [state]
    actions: List[str] = []
    disturbances = sample_disturbance_sequence(rng, horizon, disturbance_scale, mode=disturbance_mode)
    for disturbance in disturbances:
        if policy_mix == "random":
            action = rng.choice(ACTIONS)
        elif policy_mix == "heuristic":
            action = heuristic_action(state, rng)
        elif policy_mix == "mixed":
            action = heuristic_action(state, rng) if rng.random() < 0.55 else rng.choice(ACTIONS)
        else:
            raise ValueError(f"unknown policy_mix: {policy_mix}")
        actions.append(action)
        state = transition_dynamics(state, action, disturbance=disturbance)
        states.append(state)
    return states, actions, disturbances


def collect_dataset(
    seed: int,
    episodes: int,
    horizon: int,
    disturbance_scale: float,
    disturbance_mode: str,
    policy_mix: str,
) -> List[Tuple[List[StickState], List[str]]]:
    rng = random.Random(seed)
    dataset: List[Tuple[List[StickState], List[str]]] = []
    for _ in range(episodes):
        states, actions, _ = collect_episode(rng, horizon, disturbance_scale, disturbance_mode, policy_mix)
        dataset.append((states, actions))
    return dataset


def build_sequence_samples(
    episodes: Sequence[Tuple[List[StickState], List[str]]],
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_sequences: List[List[List[float]]] = []
    action_sequences: List[List[List[float]]] = []
    target_sequences: List[List[List[float]]] = []
    for states, actions in episodes:
        obs = [state_to_obs(state) for state in states]
        if len(actions) < seq_len:
            continue
        for start in range(0, len(actions) - seq_len + 1):
            obs_sequences.append(obs[start : start + seq_len])
            action_sequences.append([action_pulse(action) for action in actions[start : start + seq_len]])
            target_sequences.append(obs[start + 1 : start + seq_len + 1])
    return (
        torch.tensor(obs_sequences, dtype=torch.float32),
        torch.tensor(action_sequences, dtype=torch.float32),
        torch.tensor(target_sequences, dtype=torch.float32),
    )


def make_sparse_mask(hidden_dim: int, in_degree: int, rng: random.Random) -> torch.Tensor:
    mask = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32)
    for dst in range(hidden_dim):
        candidates = list(range(hidden_dim))
        rng.shuffle(candidates)
        chosen = candidates[:in_degree]
        for src in chosen:
            mask[dst, src] = 1.0
    return mask


class SparseCyclicSNN(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int, in_degree: int, seed: int) -> None:
        super().__init__()
        rng = random.Random(seed)
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim * 2, hidden_dim)
        self.decoder = nn.Linear(hidden_dim + action_dim, obs_dim)

        self.recurrent = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.12)
        self.register_buffer("rec_mask", make_sparse_mask(hidden_dim, in_degree, rng))
        self.register_buffer("coactivity", torch.zeros(hidden_dim, hidden_dim))
        self.register_buffer("usage", torch.zeros(hidden_dim, hidden_dim))

        self.membrane_decay = 0.86
        self.trace_decay = 0.90
        self.adapt_decay = 0.93
        self.adapt_strength = 0.24
        self.threshold = 0.18
        self.spike_temp = 0.28

        self.register_buffer("action_rise_decay", torch.full((action_dim,), ACTION_RISE_FACTOR))
        self.register_buffer("action_decay_decay", torch.full((action_dim,), ACTION_DECAY_FACTOR))

    def zero_state(self, batch_size: int, device: torch.device) -> SNNState:
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        action_zeros = torch.zeros(batch_size, self.action_dim, device=device)
        return SNNState(
            membrane=zeros.clone(),
            spikes=zeros.clone(),
            trace=zeros.clone(),
            adapt=zeros.clone(),
            action_rise=action_zeros.clone(),
            action_decay=action_zeros.clone(),
        )

    def masked_recurrent(self) -> torch.Tensor:
        return self.recurrent * self.rec_mask

    def step(self, obs: torch.Tensor, pulses: torch.Tensor, state: SNNState) -> Tuple[torch.Tensor, SNNState]:
        action_rise = state.action_rise * self.action_rise_decay + pulses
        action_decay = state.action_decay * self.action_decay_decay + pulses
        action_alpha = torch.relu(action_decay - action_rise) * ACTION_GAIN
        action_alpha_norm = torch.clamp(action_alpha / MAX_ACTION_ALPHA, 0.0, 1.0)

        rec_drive = F.linear(state.trace, self.masked_recurrent())
        total_drive = self.obs_proj(obs) + self.action_proj(torch.cat([pulses, action_alpha_norm], dim=-1)) + rec_drive

        membrane = self.membrane_decay * state.membrane + total_drive - self.adapt_strength * state.adapt
        spikes = torch.sigmoid((membrane - self.threshold) / self.spike_temp)
        adapt = self.adapt_decay * state.adapt + spikes
        trace = self.trace_decay * state.trace + spikes

        pred_raw = self.decoder(torch.cat([trace, action_alpha_norm], dim=-1))
        pred = pred_raw.clone()
        pred[:, 0] = torch.tanh(pred_raw[:, 0])
        pred[:, 1] = torch.tanh(pred_raw[:, 1])
        pred[:, 2] = torch.sigmoid(pred_raw[:, 2])
        pred[:, 3] = torch.sigmoid(pred_raw[:, 3])
        pred[:, 4] = torch.tanh(pred_raw[:, 4])
        pred[:, 5] = torch.sigmoid(pred_raw[:, 5])
        pred[:, 6] = torch.sigmoid(pred_raw[:, 6])

        next_state = SNNState(
            membrane=membrane,
            spikes=spikes,
            trace=trace,
            adapt=adapt,
            action_rise=action_rise,
            action_decay=action_decay,
        )
        return pred, next_state

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        state: SNNState | None = None,
        collect_stats: bool = False,
    ) -> Tuple[torch.Tensor, SNNState]:
        batch_size, seq_len, _ = obs_seq.shape
        if state is None:
            state = self.zero_state(batch_size, obs_seq.device)

        preds: List[torch.Tensor] = []
        for step_idx in range(seq_len):
            pred, state = self.step(obs_seq[:, step_idx], action_seq[:, step_idx], state)
            preds.append(pred)
            if collect_stats:
                self.coactivity.mul_(0.992).add_(
                    torch.matmul(state.trace.detach().T, state.trace.detach()) / float(max(batch_size, 1)),
                )
                self.usage.mul_(0.996).add_(
                    torch.matmul(state.spikes.detach().T, state.spikes.detach()) / float(max(batch_size, 1)),
                )
        return torch.stack(preds, dim=1), state

    def maybe_rewire(self, rng: random.Random) -> None:
        with torch.no_grad():
            for dst in range(self.hidden_dim):
                existing = torch.nonzero(self.rec_mask[dst] > 0.5, as_tuple=False).flatten().tolist()
                if len(existing) > MODEL_IN_DEGREE - REWIRE_PRUNE_PER_NODE:
                    existing_scores = [
                        (
                            abs(float(self.recurrent[dst, src])) * float(self.usage[dst, src] + 1e-6),
                            src,
                        )
                        for src in existing
                    ]
                    existing_scores.sort(key=lambda item: item[0])
                    for _, src in existing_scores[:REWIRE_PRUNE_PER_NODE]:
                        self.rec_mask[dst, src] = 0.0
                        self.recurrent[dst, src] = 0.0

                available = [
                    src
                    for src in range(self.hidden_dim)
                    if self.rec_mask[dst, src] < 0.5 and src != dst
                ]
                available.sort(key=lambda src: float(self.coactivity[dst, src]), reverse=True)
                if not available:
                    continue
                grew = 0
                for src in available:
                    self.rec_mask[dst, src] = 1.0
                    self.recurrent[dst, src] = torch.empty(1).normal_(0.0, 0.08).item()
                    grew += 1
                    if grew >= REWIRE_GROW_PER_NODE:
                        break


def weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.8, 1.0, 0.9, 0.9, 1.7, 1.5, 1.5], device=pred.device)
    return ((pred - target) ** 2 * weights).mean()


def batch_iter(x: torch.Tensor, a: torch.Tensor, y: torch.Tensor, batch_size: int, rng: random.Random) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    indices = list(range(x.shape[0]))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        idx = torch.tensor(chosen, dtype=torch.long)
        yield x[idx], a[idx], y[idx]


def evaluate_world_model(
    model: SparseCyclicSNN,
    x: torch.Tensor,
    a: torch.Tensor,
    y: torch.Tensor,
) -> WorldModelMetrics:
    model.eval()
    with torch.no_grad():
        preds, _ = model.forward_sequence(x, a, collect_stats=False)
        loss = float(weighted_loss(preds, y).item())
        height_mae = float(torch.abs(preds[:, :, 4] - y[:, :, 4]).mean().item())
    return WorldModelMetrics(val_loss=loss, val_height_mae=height_mae)


def train_world_model(
    seed: int,
    hidden_dim: int = MODEL_HIDDEN,
    in_degree: int = MODEL_IN_DEGREE,
    epochs: int = MODEL_EPOCHS,
) -> Tuple[SparseCyclicSNN, WorldModelMetrics]:
    train_eps = collect_dataset(
        seed=seed,
        episodes=TRAIN_EPISODES,
        horizon=TRAIN_HORIZON,
        disturbance_scale=0.38,
        disturbance_mode=DEFAULT_DISTURBANCE_MODE,
        policy_mix="mixed",
    )
    val_eps = collect_dataset(
        seed=seed + 991,
        episodes=VAL_EPISODES,
        horizon=TRAIN_HORIZON,
        disturbance_scale=0.50,
        disturbance_mode=DEFAULT_DISTURBANCE_MODE,
        policy_mix="mixed",
    )
    train_x, train_a, train_y = build_sequence_samples(train_eps, seq_len=SEQ_LEN)
    val_x, val_a, val_y = build_sequence_samples(val_eps, seq_len=SEQ_LEN)

    model = SparseCyclicSNN(OBS_DIM, hidden_dim=hidden_dim, action_dim=ACTION_DIM, in_degree=in_degree, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    rng = random.Random(seed)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_a, batch_y in batch_iter(train_x, train_a, train_y, BATCH_SIZE, rng):
            optimizer.zero_grad(set_to_none=True)
            preds, _ = model.forward_sequence(batch_x, batch_a, collect_stats=True)
            loss = weighted_loss(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % REWIRE_INTERVAL == 0:
            model.maybe_rewire(rng)

    return model, evaluate_world_model(model, val_x, val_a, val_y)


def planner_score_from_state(state: StickState, depth: int) -> float:
    if depth <= 0:
        return 0.0
    best = float("-inf")
    for action in ACTIONS:
        next_state = transition_dynamics(state, action, disturbance=0.0)
        score = (toward_value(height_from_theta(next_state.theta)) - away_value(height_from_theta(next_state.theta))) + (
            PLANNER_DISCOUNT * planner_score_from_state(next_state, depth - 1)
        )
        best = max(best, score)
    return best


def planner_action(state: StickState, depth: int = PLANNER_DEPTH) -> str:
    best_action = ACTIONS[0]
    best_score = float("-inf")
    for action in ACTIONS:
        next_state = transition_dynamics(state, action, disturbance=0.0)
        score = (toward_value(height_from_theta(next_state.theta)) - away_value(height_from_theta(next_state.theta))) + (
            PLANNER_DISCOUNT * planner_score_from_state(next_state, depth - 1)
        )
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def prediction_error_to_beta(error_ema: float) -> float:
    accuracy = math.exp(-error_ema / ACCURACY_SCALE)
    return BETA_MIN + (BETA_MAX - BETA_MIN) * accuracy


def action_entropy(probs: Sequence[float]) -> float:
    total = 0.0
    for prob in probs:
        if prob > 1e-12:
            total -= prob * math.log(prob + 1e-12)
    return total


def choose_imagined_base_action(
    model: SparseCyclicSNN,
    obs: torch.Tensor,
    state: SNNState,
) -> str:
    best_action = ACTIONS[0]
    best_score = float("-inf")
    for action in ACTIONS:
        pred, _ = model.step(obs.unsqueeze(0), torch.tensor([action_pulse(action)], dtype=torch.float32), state.clone())
        score = obs_value(pred[0].tolist())
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def imagined_rollout(
    model: SparseCyclicSNN,
    obs: torch.Tensor,
    state: SNNState,
    first_action: str | None,
    horizon: int,
) -> List[torch.Tensor]:
    current_obs = obs.clone()
    current_state = state.clone()
    predictions: List[torch.Tensor] = []
    for step_idx in range(horizon):
        action = first_action if step_idx == 0 else choose_imagined_base_action(model, current_obs, current_state)
        pulse_tensor = torch.tensor([action_pulse(action)], dtype=torch.float32)
        pred, current_state = model.step(current_obs.unsqueeze(0), pulse_tensor, current_state)
        current_obs = pred[0].detach()
        predictions.append(current_obs.clone())
    return predictions


def signed_action_score(
    model: SparseCyclicSNN,
    obs: torch.Tensor,
    state: SNNState,
    action: str,
    horizon: int,
) -> Tuple[float, float, float]:
    with_action = imagined_rollout(model, obs, state, first_action=action, horizon=horizon)
    without_action = imagined_rollout(model, obs, state, first_action=None, horizon=horizon)

    base_pred, _ = model.step(obs.unsqueeze(0), torch.tensor([action_pulse(action)], dtype=torch.float32), state.clone())
    immediate = obs_value(base_pred[0].tolist())

    pos = 0.0
    neg = 0.0
    branch_value = 0.0
    for step_idx, (full_obs, null_obs) in enumerate(zip(with_action, without_action)):
        discount = FA_DISCOUNT**step_idx
        full_height = obs_height(full_obs.tolist())
        null_height = obs_height(null_obs.tolist())
        branch_value += discount * (toward_value(full_height) - NEG_WEIGHT * away_value(full_height))
        pos += discount * (toward_value(full_height) - toward_value(null_height))
        neg += discount * (away_value(full_height) - away_value(null_height))
    total = (
        BASE_WEIGHT * immediate
        + PROSPECTIVE_WEIGHT * branch_value
        + DELTA_WEIGHT * (pos - NEG_WEIGHT * neg)
    )
    return total, pos, neg


def prediction_only_score(
    model: SparseCyclicSNN,
    obs: torch.Tensor,
    state: SNNState,
    action: str,
) -> float:
    pred, _ = model.step(obs.unsqueeze(0), torch.tensor([action_pulse(action)], dtype=torch.float32), state.clone())
    return obs_value(pred[0].tolist())


def choose_action_with_policy(
    policy_name: str,
    model: SparseCyclicSNN,
    obs: torch.Tensor,
    state: SNNState,
    beta: float,
    rng: random.Random,
) -> Tuple[str, Dict[str, float], List[float]]:
    logits: List[float] = []
    debug: Dict[str, float] = {}
    for action in ACTIONS:
        if policy_name == "prediction_only":
            score = prediction_only_score(model, obs, state, action)
            debug[f"{action}_score"] = score
        elif policy_name == "signed_prospective":
            score, pos, neg = signed_action_score(model, obs, state, action, horizon=FA_HORIZON)
            debug[f"{action}_score"] = score
            debug[f"{action}_pos"] = pos
            debug[f"{action}_neg"] = neg
        else:
            raise ValueError(f"unknown policy_name: {policy_name}")
        logits.append(beta * score)

    probs_tensor = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0)
    probs = probs_tensor.tolist()
    action = rng.choices(list(ACTIONS), weights=probs, k=1)[0]
    return action, debug, probs


def evaluate_policy(
    model: SparseCyclicSNN,
    seed: int,
    policy_name: str,
    disturbance_scale: float,
) -> EvalMetrics:
    rng = random.Random(seed)
    total_steps = 0
    total_band_hits = 0
    total_abs_error = 0.0
    planner_hits = 0
    beta_values: List[float] = []
    entropy_values: List[float] = []

    model.eval()
    with torch.no_grad():
        for episode_idx in range(EVAL_EPISODES):
            state = random_initial_state(rng)
            disturbances = sample_disturbance_sequence(
                random.Random(seed * 1000 + episode_idx),
                EVAL_HORIZON,
                disturbance_scale,
                mode=DEFAULT_DISTURBANCE_MODE,
            )
            snn_state = model.zero_state(batch_size=1, device=torch.device("cpu"))
            error_ema = 0.08

            for disturbance in disturbances:
                obs_vec = torch.tensor(state_to_obs(state), dtype=torch.float32)
                beta = prediction_error_to_beta(error_ema)
                planner_target = planner_action(state)
                action, _, probs = choose_action_with_policy(policy_name, model, obs_vec, snn_state, beta, rng)
                pred_next, updated_state = model.step(
                    obs_vec.unsqueeze(0),
                    torch.tensor([action_pulse(action)], dtype=torch.float32),
                    snn_state.clone(),
                )
                next_state = transition_dynamics(state, action, disturbance=disturbance)
                target_obs = torch.tensor(state_to_obs(next_state), dtype=torch.float32)

                pred_error = float(weighted_loss(pred_next[0], target_obs).item())
                error_ema = 0.94 * error_ema + 0.06 * pred_error

                snn_state = updated_state
                state = next_state

                height = height_from_theta(state.theta)
                total_band_hits += int(band_distance(height) <= 1e-8)
                total_abs_error += abs(height - TARGET_HEIGHT)
                planner_hits += int(action == planner_target)
                total_steps += 1
                beta_values.append(beta)
                entropy_values.append(action_entropy(probs))

    return EvalMetrics(
        in_band_rate=total_band_hits / float(max(1, total_steps)),
        mean_abs_height_error=total_abs_error / float(max(1, total_steps)),
        planner_agreement=planner_hits / float(max(1, total_steps)),
        mean_beta=statistics.mean(beta_values) if beta_values else 0.0,
        mean_action_entropy=statistics.mean(entropy_values) if entropy_values else 0.0,
    )


def format_markdown(summary: RunSummary) -> str:
    lines = [
        "# Sparse Cyclic SNN Stick-Height Control",
        "",
        "## World Model",
        f"- val_loss: {summary.world_model['val_loss']:.6f}",
        f"- val_height_mae: {summary.world_model['val_height_mae']:.6f}",
        "",
        "## Evaluation",
    ]
    for name, metrics in summary.evaluation.items():
        lines.append(f"### {name}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value:.6f}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_experiment(seed: int) -> RunSummary:
    torch.manual_seed(seed)
    random.seed(seed)
    model, world_metrics = train_world_model(seed=seed)

    nominal_prediction = evaluate_policy(model, seed=seed + 101, policy_name="prediction_only", disturbance_scale=0.45)
    nominal_signed = evaluate_policy(model, seed=seed + 202, policy_name="signed_prospective", disturbance_scale=0.45)
    stress_prediction = evaluate_policy(model, seed=seed + 303, policy_name="prediction_only", disturbance_scale=0.72)
    stress_signed = evaluate_policy(model, seed=seed + 404, policy_name="signed_prospective", disturbance_scale=0.72)

    return RunSummary(
        config={
            "seed": seed,
            "target_height": TARGET_HEIGHT,
            "target_band": TARGET_BAND,
            "hidden_dim": MODEL_HIDDEN,
            "fa_horizon": FA_HORIZON,
            "disturbance_mode": DEFAULT_DISTURBANCE_MODE,
        },
        world_model={
            "val_loss": world_metrics.val_loss,
            "val_height_mae": world_metrics.val_height_mae,
        },
        evaluation={
            "nominal_prediction_only": nominal_prediction.__dict__,
            "nominal_signed_prospective": nominal_signed.__dict__,
            "stress_prediction_only": stress_prediction.__dict__,
            "stress_signed_prospective": stress_signed.__dict__,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse cyclic SNN stick-height control.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--val-episodes", type=int, default=VAL_EPISODES)
    parser.add_argument("--epochs", type=int, default=MODEL_EPOCHS)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--eval-horizon", type=int, default=EVAL_HORIZON)
    parser.add_argument("--target-height", type=float, default=TARGET_HEIGHT)
    parser.add_argument("--target-band", type=float, default=TARGET_BAND)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("snn_cyclic_stick_height_control.json"),
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=Path("snn_cyclic_stick_height_control.md"),
    )
    args = parser.parse_args()

    globals()["TRAIN_EPISODES"] = args.train_episodes
    globals()["VAL_EPISODES"] = args.val_episodes
    globals()["MODEL_EPOCHS"] = args.epochs
    globals()["EVAL_EPISODES"] = args.eval_episodes
    globals()["EVAL_HORIZON"] = args.eval_horizon
    globals()["TARGET_HEIGHT"] = args.target_height
    globals()["TARGET_BAND"] = args.target_band

    summary = run_experiment(seed=args.seed)

    args.json.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
    args.md.write_text(format_markdown(summary), encoding="utf-8")

    print(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
