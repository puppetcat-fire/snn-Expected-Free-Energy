#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stick-height control with softmax semantic future attention.

This version folds together the later design discussions:

1. Candidate actions are scored in separate imagined branches.
2. Action selection uses a softmax policy.
3. Positive and negative target semantics are represented as explicit tokens.
4. Future attention measures how much those semantic tokens attend back to the
   current candidate action token.
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
SEMANTIC_SCALE = 0.22
EPISODE_HORIZON = 60

HISTORY_STEPS = 6
FA_HORIZON = 4
FA_DISCOUNT = 0.88
PLANNER_DEPTH = 5
PLANNER_DISCOUNT = 0.90

TOKEN_DIM = 10
STATE_DIM = 4


@dataclass(frozen=True)
class StickState:
    theta: float
    omega: float
    drive: float


@dataclass(frozen=True)
class PredictedState:
    theta: float
    omega: float
    drive: float
    height: float


@dataclass(frozen=True)
class EvalScenario:
    history_states: Tuple[StickState, ...]
    history_actions: Tuple[str, ...]
    disturbances: Tuple[float, ...]
    action_draws: Tuple[float, ...]


@dataclass(frozen=True)
class WorldModelMetrics:
    val_loss: float
    val_height_mae: float


@dataclass(frozen=True)
class EvalMetrics:
    in_band_rate: float
    survival_rate: float
    mean_abs_height_error: float
    planner_agreement: float
    mean_episode_length: float
    mean_action_entropy: float


@dataclass(frozen=True)
class RepresentativeCase:
    theta: float
    omega: float
    drive: float
    height: float
    planner_action: str
    prediction_only_action: str
    positive_fa_action: str
    signed_semantic_fa_action: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def height_from_theta(theta: float) -> float:
    return math.cos(theta)


def state_height(state: StickState) -> float:
    return height_from_theta(state.theta)


def failed_state(theta: float) -> bool:
    return abs(theta) >= FAIL_ANGLE


def failed(state: StickState) -> bool:
    return failed_state(state.theta)


def band_distance(height: float) -> float:
    return max(0.0, abs(height - TARGET_HEIGHT) - TARGET_BAND)


def toward_value(height: float) -> float:
    return clamp(1.0 - band_distance(height) / SEMANTIC_SCALE, 0.0, 1.0)


def away_value(height: float) -> float:
    return clamp(band_distance(height) / SEMANTIC_SCALE, 0.0, 1.0)


def viability_score(theta: float, height: float) -> float:
    if failed_state(theta):
        return -3.0
    return toward_value(height) - away_value(height)


def state_to_target(state: StickState) -> Tuple[float, float, float, float]:
    return (
        state.theta / math.pi,
        state.omega / MAX_OMEGA,
        state.drive / MAX_DRIVE,
        state_height(state),
    )


def vector_to_prediction(pred: Sequence[float]) -> PredictedState:
    theta = clamp(float(pred[0]), -1.0, 1.0) * math.pi
    omega = clamp(float(pred[1]), -1.0, 1.0) * MAX_OMEGA
    drive = clamp(float(pred[2]), -1.0, 1.0) * MAX_DRIVE
    height = clamp(float(pred[3]), -1.0, 1.0)
    return PredictedState(theta=theta, omega=omega, drive=drive, height=height)


def state_token_from_values(theta: float, omega: float, drive: float, height: float) -> List[float]:
    return [theta / math.pi, omega / MAX_OMEGA, drive / MAX_DRIVE, height, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def state_token(state: StickState) -> List[float]:
    return state_token_from_values(state.theta, state.omega, state.drive, state_height(state))


def predicted_state_token(pred: PredictedState) -> List[float]:
    return state_token_from_values(pred.theta, pred.omega, pred.drive, pred.height)


def action_token(action: str) -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, ACTION_VALUES[action], 0.0, 0.0, 1.0, 0.0, 0.0]


def toward_token(height: float) -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, toward_value(height), 0.0, 0.0, 1.0, 0.0]


def away_token(height: float) -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, away_value(height), 0.0, 0.0, 0.0, 1.0]


def transition_dynamics(state: StickState, action: str, disturbance: float) -> StickState:
    drive = clamp(DRIVE_DECAY * state.drive + DRIVE_PUSH * ACTION_VALUES[action], -MAX_DRIVE, MAX_DRIVE)
    omega = DAMPING * state.omega + TORQUE_SCALE * drive - GRAVITY_SCALE * math.sin(state.theta) + disturbance
    omega = clamp(omega, -MAX_OMEGA, MAX_OMEGA)
    theta = wrap_angle(state.theta + DT * omega)
    return StickState(theta=theta, omega=omega, drive=drive)


def deterministic_step(state: StickState, action: str) -> StickState:
    return transition_dynamics(state, action, disturbance=0.0)


def stochastic_step(state: StickState, action: str, rng: random.Random, noise_std: float) -> StickState:
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
    theta_bin, omega_bin, drive_bin = quantize_planner_state(state)
    key = (theta_bin, omega_bin, drive_bin, depth)
    if key in PLANNER_CACHE.values:
        return PLANNER_CACHE.values[key]
    decoded = dequantize_planner_state(theta_bin, omega_bin, drive_bin)
    if failed(decoded):
        return -3.0
    best = float("-inf")
    for action in ACTIONS:
        next_state = deterministic_step(decoded, action)
        total = viability_score(next_state.theta, state_height(next_state)) + PLANNER_DISCOUNT * planner_value(next_state, depth - 1)
        best = max(best, total)
    PLANNER_CACHE.values[key] = best
    return best


def planner_action_scores(state: StickState, depth: int = PLANNER_DEPTH) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for action in ACTIONS:
        next_state = deterministic_step(state, action)
        scores[action] = viability_score(next_state.theta, state_height(next_state)) + PLANNER_DISCOUNT * planner_value(next_state, depth - 1)
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


def build_step_tokens(states: Sequence[StickState], actions: Sequence[str]) -> List[List[float]]:
    assert len(states) == len(actions)
    tokens: List[List[float]] = []
    for state, action in zip(states, actions):
        height = state_height(state)
        tokens.append(state_token(state))
        tokens.append(toward_token(height))
        tokens.append(away_token(height))
        tokens.append(action_token(action))
    return tokens


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
            start_idx = step_idx - history_steps + 1
            states_window = states[start_idx : step_idx + 1]
            actions_window = actions[start_idx : step_idx + 1]
            inputs.append(build_step_tokens(states_window, actions_window))
            targets.append(list(state_to_target(states[step_idx + 1])))
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

    def predict_next(self, tokens: torch.Tensor) -> torch.Tensor:
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
            pred = model.predict_next(batch_x)
            losses.append(float(F.mse_loss(pred, batch_y).item()))
            height_errors.extend(torch.abs(pred[:, 3] - batch_y[:, 3]).cpu().tolist())
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
        max_seq_len=4 * HISTORY_STEPS + 4 * FA_HORIZON + 8,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    best_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in batch_iter(train_x, train_y, batch_size=batch_size, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            pred = model.predict_next(batch_x)
            loss = F.mse_loss(pred, batch_y)
            total_loss = loss + 0.35 * F.l1_loss(pred[:, 3], batch_y[:, 3])
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        metrics = evaluate_world_model(model, val_x, val_y, batch_size=batch_size)
        if metrics.val_loss < best_loss:
            best_loss = metrics.val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate_world_model(model, val_x, val_y, batch_size=batch_size)


def history_prefix_tokens(states: Sequence[StickState], actions: Sequence[str], candidate_action: str) -> List[List[float]]:
    assert len(states) == len(actions) + 1
    tokens = build_step_tokens(states[:-1], actions)
    current_state = states[-1]
    current_height = state_height(current_state)
    tokens.append(state_token(current_state))
    tokens.append(toward_token(current_height))
    tokens.append(away_token(current_height))
    tokens.append(action_token(candidate_action))
    return tokens


def predict_next_state(model: StickWorldModel, tokens: Sequence[Sequence[float]]) -> PredictedState:
    batch = torch.tensor([tokens], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = model.predict_next(batch)[0].detach().cpu().tolist()
    return vector_to_prediction(pred)


def softmax_probs(logits: Sequence[float], temperature: float) -> List[float]:
    scaled = [value / max(temperature, 1e-6) for value in logits]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def sample_from_probs(actions: Sequence[str], probs: Sequence[float], draw: float) -> str:
    cumulative = 0.0
    for action, prob in zip(actions, probs):
        cumulative += prob
        if draw <= cumulative:
            return action
    return actions[-1]


def entropy_from_probs(probs: Sequence[float]) -> float:
    total = 0.0
    for prob in probs:
        if prob > 1e-12:
            total -= prob * math.log(prob)
    return total


def imagined_semantic_scores(
    model: StickWorldModel,
    history_states: Sequence[StickState],
    history_actions: Sequence[str],
    candidate_action: str,
    horizon: int,
    discount: float,
    rollout_mode: str,
) -> Dict[str, float]:
    tokens = history_prefix_tokens(history_states, history_actions, candidate_action)
    candidate_action_index = len(tokens) - 1

    first_pred = predict_next_state(model, tokens)
    base_score = viability_score(first_pred.theta, first_pred.height)

    pos_fa = 0.0
    neg_fa = 0.0
    current_pred = first_pred

    for step_idx in range(horizon):
        tokens.append(predicted_state_token(current_pred))
        tokens.append(toward_token(current_pred.height))
        tokens.append(away_token(current_pred.height))

        batch = torch.tensor([tokens], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            _, attn = model.encode(batch)
        mean_attn = attn.mean(dim=1)[0]
        toward_idx = len(tokens) - 2
        away_idx = len(tokens) - 1
        pos_fa += (discount**step_idx) * float(mean_attn[toward_idx, candidate_action_index].item()) * toward_value(current_pred.height)
        neg_fa += (discount**step_idx) * float(mean_attn[away_idx, candidate_action_index].item()) * away_value(current_pred.height)

        if step_idx + 1 >= horizon:
            break
        if rollout_mode == "repeat":
            rollout_action = candidate_action
        elif rollout_mode == "greedy_prediction":
            branch_scores: Dict[str, float] = {}
            for action in ACTIONS:
                next_pred = predict_next_state(model, list(tokens) + [action_token(action)])
                branch_scores[action] = viability_score(next_pred.theta, next_pred.height)
            rollout_action = argmax_action(branch_scores)
        else:
            raise ValueError(f"unknown rollout mode: {rollout_mode}")

        tokens.append(action_token(rollout_action))
        current_pred = predict_next_state(model, tokens)

    return {"base": base_score, "pos_fa": pos_fa, "neg_fa": neg_fa}


def policy_logits(
    policy_name: str,
    model: StickWorldModel,
    history_states: Sequence[StickState],
    history_actions: Sequence[str],
    fa_weight: float,
    rollout_mode: str,
) -> Dict[str, float]:
    if policy_name == "planner":
        return planner_action_scores(history_states[-1])

    logits: Dict[str, float] = {}
    for action in ACTIONS:
        branch = imagined_semantic_scores(
            model=model,
            history_states=history_states,
            history_actions=history_actions,
            candidate_action=action,
            horizon=FA_HORIZON,
            discount=FA_DISCOUNT,
            rollout_mode=rollout_mode,
        )
        if policy_name == "prediction_only":
            logits[action] = branch["base"]
        elif policy_name == "positive_fa":
            logits[action] = branch["base"] + fa_weight * branch["pos_fa"]
        elif policy_name == "signed_semantic_fa":
            logits[action] = branch["base"] + fa_weight * (branch["pos_fa"] - branch["neg_fa"])
        else:
            raise ValueError(f"unknown policy: {policy_name}")
    return logits


def choose_action(
    policy_name: str,
    model: StickWorldModel,
    history_states: Sequence[StickState],
    history_actions: Sequence[str],
    fa_weight: float,
    rollout_mode: str,
    temperature: float,
    draw: float,
) -> Tuple[str, float]:
    if policy_name == "random":
        probs = [0.5, 0.5]
        return sample_from_probs(ACTIONS, probs, draw), entropy_from_probs(probs)

    logits = policy_logits(
        policy_name=policy_name,
        model=model,
        history_states=history_states,
        history_actions=history_actions,
        fa_weight=fa_weight,
        rollout_mode=rollout_mode,
    )
    if policy_name == "planner":
        return argmax_action(logits), 0.0

    probs = softmax_probs([logits[action] for action in ACTIONS], temperature=temperature)
    return sample_from_probs(ACTIONS, probs, draw), entropy_from_probs(probs)


def build_eval_scenarios(
    seed: int,
    episodes: int,
    history_steps: int,
    horizon: int,
    noise_std: float,
) -> List[EvalScenario]:
    rng = random.Random(seed)
    scenarios: List[EvalScenario] = []
    for _ in range(episodes):
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
        action_draws = tuple(rng.random() for _ in range(horizon))
        scenarios.append(
            EvalScenario(
                history_states=tuple(history_states),
                history_actions=tuple(history_actions),
                disturbances=disturbances,
                action_draws=action_draws,
            )
        )
    return scenarios


def evaluate_policy(
    policy_name: str,
    model: StickWorldModel,
    scenarios: Sequence[EvalScenario],
    horizon: int,
    fa_weight: float,
    rollout_mode: str,
    temperature: float,
) -> EvalMetrics:
    total_band_hits = 0
    total_steps = 0
    survived = 0
    total_abs_height_error = 0.0
    planner_matches = 0
    planner_total = 0
    episode_lengths: List[int] = []
    entropies: List[float] = []

    for scenario in scenarios:
        history_states = list(scenario.history_states)
        history_actions = list(scenario.history_actions)
        state = history_states[-1]
        failed_episode = False
        steps_taken = 0

        for disturbance, draw in zip(scenario.disturbances[:horizon], scenario.action_draws[:horizon]):
            if failed(state):
                failed_episode = True
                break

            action, entropy = choose_action(
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
        mean_action_entropy=statistics.mean(entropies) if entropies else 0.0,
    )


def representative_cases(
    model: StickWorldModel,
    fa_weight: float,
    rollout_mode: str,
    temperature: float,
) -> List[RepresentativeCase]:
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
                prediction_only_action=choose_action("prediction_only", model, history_states, history_actions, fa_weight, rollout_mode, temperature, 0.5)[0],
                positive_fa_action=choose_action("positive_fa", model, history_states, history_actions, fa_weight, rollout_mode, temperature, 0.5)[0],
                signed_semantic_fa_action=choose_action("signed_semantic_fa", model, history_states, history_actions, fa_weight, rollout_mode, temperature, 0.5)[0],
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
        "mean_action_entropy": pack_metric([item.mean_action_entropy for item in metrics]),
    }


def format_metric(entry: Dict[str, float], pct: bool = False) -> str:
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    return f"{entry['mean'] * scale:.2f}±{entry['std'] * scale:.2f}{suffix}"


def write_markdown(path: str, summary: Dict[str, object], cases: Sequence[RepresentativeCase]) -> None:
    world_model = summary["world_model"]
    nominal_policies = summary["nominal_policies"]
    stress_policies = summary["stress_policies"]
    lines = [
        "# Stick Height Softmax Semantic-FA Experiment",
        "",
        "Counterfactual action branches are scored separately, then converted into a softmax policy.",
        "",
        f"- `target_height = {TARGET_HEIGHT:.2f}`",
        f"- `target_band = ±{TARGET_BAND:.2f}`",
        f"- `episode_horizon = {EPISODE_HORIZON}`",
        f"- `history_steps = {HISTORY_STEPS}`",
        f"- `fa_horizon = {FA_HORIZON}`",
        f"- `offline_noise_mode = {summary['config']['offline_noise_mode']}`",
        f"- `rollout_mode = {summary['config']['rollout_mode']}`",
        f"- `policy_temperature = {summary['config']['policy_temperature']:.2f}`",
        "",
        "## World Model",
        "",
        f"- `val_loss = {world_model['val_loss']['mean']:.4f} ± {world_model['val_loss']['std']:.4f}`",
        f"- `val_height_mae = {world_model['val_height_mae']['mean']:.4f} ± {world_model['val_height_mae']['std']:.4f}`",
        "",
        "## Nominal Policy Results",
        "",
        "| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_name, label in (
        ("random", "Random"),
        ("prediction_only", "Prediction only"),
        ("positive_fa", "Positive-only FA"),
        ("signed_semantic_fa", "Signed semantic FA"),
        ("planner", "Planner"),
    ):
        block = nominal_policies[policy_name]
        lines.append(
            f"| {label} | {format_metric(block['in_band_rate'], pct=True)} | "
            f"{format_metric(block['survival_rate'], pct=True)} | {format_metric(block['mean_abs_height_error'])} | "
            f"{format_metric(block['planner_agreement'], pct=True)} | {format_metric(block['mean_episode_length'])} | "
            f"{format_metric(block['mean_action_entropy'])} |"
        )

    lines.extend(
        [
            "",
            "## Stress Policy Results",
            "",
            "| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_name, label in (
        ("random", "Random"),
        ("prediction_only", "Prediction only"),
        ("positive_fa", "Positive-only FA"),
        ("signed_semantic_fa", "Signed semantic FA"),
        ("planner", "Planner"),
    ):
        block = stress_policies[policy_name]
        lines.append(
            f"| {label} | {format_metric(block['in_band_rate'], pct=True)} | "
            f"{format_metric(block['survival_rate'], pct=True)} | {format_metric(block['mean_abs_height_error'])} | "
            f"{format_metric(block['planner_agreement'], pct=True)} | {format_metric(block['mean_episode_length'])} | "
            f"{format_metric(block['mean_action_entropy'])} |"
        )

    lines.extend(
        [
            "",
            "## Representative States",
            "",
            "| Theta | Omega | Drive | Height | Planner | Prediction | Positive FA | Signed semantic FA |",
            "|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for case in cases:
        lines.append(
            f"| {case.theta:.2f} | {case.omega:.2f} | {case.drive:.2f} | {case.height:.2f} | "
            f"{case.planner_action} | {case.prediction_only_action} | {case.positive_fa_action} | {case.signed_semantic_fa_action} |"
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stick-height control with softmax semantic future attention.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--offline-episodes", type=int, default=40)
    parser.add_argument("--offline-horizon", type=int, default=EPISODE_HORIZON)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--fa-weight", type=float, default=0.7)
    parser.add_argument("--policy-temperature", type=float, default=0.05)
    parser.add_argument("--offline-noise-mode", type=str, default="mixed", choices=("nominal", "high", "mixed"))
    parser.add_argument("--offline-high-noise-scale", type=float, default=1.6)
    parser.add_argument("--stress-noise-scale", type=float, default=1.8)
    parser.add_argument("--rollout-mode", type=str, default="repeat", choices=("repeat", "greedy_prediction"))
    parser.add_argument("--json", type=str, default="stick_height_softmax_semantic_fa_results.json")
    parser.add_argument("--md", type=str, default="stick_height_softmax_semantic_fa_results.md")
    args = parser.parse_args()

    world_model_metrics: List[WorldModelMetrics] = []
    nominal_policy_metrics: Dict[str, List[EvalMetrics]] = {
        name: [] for name in ("random", "prediction_only", "positive_fa", "signed_semantic_fa", "planner")
    }
    stress_policy_metrics: Dict[str, List[EvalMetrics]] = {
        name: [] for name in ("random", "prediction_only", "positive_fa", "signed_semantic_fa", "planner")
    }
    last_model: StickWorldModel | None = None

    for run_idx in range(args.runs):
        base_seed = 400 + 41 * run_idx
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
                    scenarios=nominal_scenarios,
                    horizon=EPISODE_HORIZON,
                    fa_weight=args.fa_weight,
                    rollout_mode=args.rollout_mode,
                    temperature=args.policy_temperature,
                )
            )
            stress_policy_metrics[policy_name].append(
                evaluate_policy(
                    policy_name=policy_name,
                    model=model,
                    scenarios=stress_scenarios,
                    horizon=EPISODE_HORIZON,
                    fa_weight=args.fa_weight,
                    rollout_mode=args.rollout_mode,
                    temperature=args.policy_temperature,
                )
            )

    assert last_model is not None
    cases = representative_cases(last_model, args.fa_weight, args.rollout_mode, args.policy_temperature)

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
            "policy_temperature": args.policy_temperature,
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

    print("=" * 120)
    print("Stick height softmax semantic-FA experiment")
    print("Interpretation: future toward/away semantic tokens vote for or against current candidate actions.")
    print("=" * 120)
    print(
        f"World model | val_loss={format_metric(summary['world_model']['val_loss'])} | "
        f"val_height_mae={format_metric(summary['world_model']['val_height_mae'])}"
    )
    print("-" * 120)
    print("Nominal noise")
    print(f"{'Policy':22s} | {'InBand':>12s} | {'Survival':>11s} | {'Height MAE':>12s} | {'PlanAgree':>11s} | {'EpLen':>8s} | {'Entropy':>8s}")
    print("-" * 120)
    for policy_name, label in (
        ("random", "Random"),
        ("prediction_only", "Prediction only"),
        ("positive_fa", "Positive-only FA"),
        ("signed_semantic_fa", "Signed semantic FA"),
        ("planner", "Planner"),
    ):
        block = summary["nominal_policies"][policy_name]
        print(
            f"{label:22s} | {format_metric(block['in_band_rate'], pct=True):>12s} | "
            f"{format_metric(block['survival_rate'], pct=True):>11s} | {format_metric(block['mean_abs_height_error']):>12s} | "
            f"{format_metric(block['planner_agreement'], pct=True):>11s} | {format_metric(block['mean_episode_length']):>8s} | "
            f"{format_metric(block['mean_action_entropy']):>8s}"
        )

    print("-" * 120)
    print("Stress noise")
    print(f"{'Policy':22s} | {'InBand':>12s} | {'Survival':>11s} | {'Height MAE':>12s} | {'PlanAgree':>11s} | {'EpLen':>8s} | {'Entropy':>8s}")
    print("-" * 120)
    for policy_name, label in (
        ("random", "Random"),
        ("prediction_only", "Prediction only"),
        ("positive_fa", "Positive-only FA"),
        ("signed_semantic_fa", "Signed semantic FA"),
        ("planner", "Planner"),
    ):
        block = summary["stress_policies"][policy_name]
        print(
            f"{label:22s} | {format_metric(block['in_band_rate'], pct=True):>12s} | "
            f"{format_metric(block['survival_rate'], pct=True):>11s} | {format_metric(block['mean_abs_height_error']):>12s} | "
            f"{format_metric(block['planner_agreement'], pct=True):>11s} | {format_metric(block['mean_episode_length']):>8s} | "
            f"{format_metric(block['mean_action_entropy']):>8s}"
        )

    print("\nRepresentative states")
    for case in cases:
        print(
            f"theta={case.theta:+.2f}, omega={case.omega:+.2f}, drive={case.drive:+.2f}, height={case.height:.2f} | "
            f"planner={case.planner_action:5s} | prediction={case.prediction_only_action:5s} | "
            f"positive={case.positive_fa_action:5s} | signed={case.signed_semantic_fa_action:5s}"
        )

    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown(args.md, summary, cases)
    print(f"\nRaw results written to {args.json}")
    print(f"Markdown summary written to {args.md}")


if __name__ == "__main__":
    main()
