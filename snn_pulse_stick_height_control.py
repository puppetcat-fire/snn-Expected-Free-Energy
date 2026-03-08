"""
Pulse-node sparse cyclic SNN stick-height control.

This version keeps the same stick environment as the current cyclic SNN
controller, but changes the information flow into a pulse-node form:

- continuous state -> discrete multi-hot pulse nodes
- sparse cyclic SNN predicts next-step pulse-node activations
- action choice uses signed prospective contribution on future pulse needs

The goal is to move closer to the original "known input node activation"
framing without yet introducing node birth/death or explicit two-hop latent
credit.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import snn_cyclic_stick_height_control as base


P_THETA_LEFT = 0
P_THETA_RIGHT = 1
P_ANGLE_LOW = 2
P_ANGLE_TARGET = 3
P_ANGLE_HIGH = 4
P_OMEGA_FAST_LEFT = 5
P_OMEGA_SLOW_LEFT = 6
P_OMEGA_ZERO = 7
P_OMEGA_SLOW_RIGHT = 8
P_OMEGA_FAST_RIGHT = 9
P_LEFT_ACTIVE = 10
P_LEFT_STRONG = 11
P_RIGHT_ACTIVE = 12
P_RIGHT_STRONG = 13
P_HEIGHT_FAR_LOW = 14
P_HEIGHT_LOW_NEAR = 15
P_HEIGHT_IN_BAND = 16
P_HEIGHT_HIGH_NEAR = 17
P_HEIGHT_FAR_HIGH = 18
P_STABLE = 19

PULSE_DIM = 20

LOSS_WEIGHTS = torch.tensor(
    [
        0.35,
        0.35,
        0.30,
        0.55,
        0.30,
        0.40,
        0.32,
        0.32,
        0.32,
        0.40,
        0.22,
        0.30,
        0.22,
        0.30,
        0.85,
        0.70,
        1.85,
        0.70,
        0.85,
        1.00,
    ],
    dtype=torch.float32,
)

TRAIN_EPISODES = 120
VAL_EPISODES = 28
SEQ_LEN = 18
MODEL_HIDDEN = 72
INIT_ACTIVE_HIDDEN = 40
MODEL_IN_DEGREE = 6
MODEL_EPOCHS = 14
MODEL_LR = 2.0e-3
BATCH_SIZE = 32

FA_HORIZON = 8
FA_DISCOUNT = 0.86
NEG_WEIGHT = 1.0
BASE_WEIGHT = 0.95
PROSPECTIVE_WEIGHT = 1.20
DELTA_WEIGHT = 0.85
LATENT_WEIGHT = 0.25
LATENT_TOPK = 2

EVAL_EPISODES = 12
EVAL_HORIZON = 96

BETA_MIN = 1.10
BETA_MAX = 5.50
ACCURACY_SCALE = 0.12

MIN_ACTIVE_HIDDEN = 28
GROWTH_LOSS_THRESHOLD = 0.065
NODE_ACTIVITY_DECAY = 0.985
NODE_BIRTHS_PER_EVENT = 2
NODE_DEATHS_PER_EVENT = 1
IMPORTANCE_DECAY = 0.92
NEW_NODE_IMPORTANCE_THRESHOLD = 0.015
NEW_NODE_PROBATION_EVENTS = 2


@dataclass(frozen=True)
class WorldModelMetrics:
    val_loss: float
    val_in_band_bce: float
    active_nodes: float
    births: float
    deaths: float
    mean_new_node_importance: float


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


@dataclass(frozen=True)
class GrowthSnapshot:
    epoch: int
    val_loss: float
    val_in_band_bce: float
    active_nodes: int
    births: int
    deaths: int
    active_mask: List[int]
    node_importance: List[float]
    node_age: List[int]


def pulse_from_state(state: base.StickState) -> List[float]:
    pulses = [0.0] * PULSE_DIM
    target_angle = math_acos_target()
    angle_margin = 0.16
    theta_abs = abs(state.theta)
    height = base.height_from_theta(state.theta)
    left_alpha = base.left_alpha(state)
    right_alpha = base.right_alpha(state)

    if state.theta < 0.0:
        pulses[P_THETA_LEFT] = 1.0
    else:
        pulses[P_THETA_RIGHT] = 1.0

    if theta_abs < target_angle - angle_margin:
        pulses[P_ANGLE_LOW] = 1.0
    elif theta_abs <= target_angle + angle_margin:
        pulses[P_ANGLE_TARGET] = 1.0
    else:
        pulses[P_ANGLE_HIGH] = 1.0

    omega = state.omega
    if omega <= -0.75:
        pulses[P_OMEGA_FAST_LEFT] = 1.0
    elif omega <= -0.18:
        pulses[P_OMEGA_SLOW_LEFT] = 1.0
    elif omega < 0.18:
        pulses[P_OMEGA_ZERO] = 1.0
    elif omega < 0.75:
        pulses[P_OMEGA_SLOW_RIGHT] = 1.0
    else:
        pulses[P_OMEGA_FAST_RIGHT] = 1.0

    if left_alpha > 0.12:
        pulses[P_LEFT_ACTIVE] = 1.0
    if left_alpha > 0.55:
        pulses[P_LEFT_STRONG] = 1.0
    if right_alpha > 0.12:
        pulses[P_RIGHT_ACTIVE] = 1.0
    if right_alpha > 0.55:
        pulses[P_RIGHT_STRONG] = 1.0

    far_margin = base.TARGET_BAND + 0.16
    if height < base.TARGET_HEIGHT - far_margin:
        pulses[P_HEIGHT_FAR_LOW] = 1.0
    elif height < base.TARGET_HEIGHT - base.TARGET_BAND:
        pulses[P_HEIGHT_LOW_NEAR] = 1.0
    elif height <= base.TARGET_HEIGHT + base.TARGET_BAND:
        pulses[P_HEIGHT_IN_BAND] = 1.0
    elif height <= base.TARGET_HEIGHT + far_margin:
        pulses[P_HEIGHT_HIGH_NEAR] = 1.0
    else:
        pulses[P_HEIGHT_FAR_HIGH] = 1.0

    if pulses[P_HEIGHT_IN_BAND] > 0.5 and pulses[P_OMEGA_ZERO] > 0.5:
        pulses[P_STABLE] = 1.0
    return pulses


def math_acos_target() -> float:
    return torch.arccos(torch.tensor(base.clamp(base.TARGET_HEIGHT, -1.0, 1.0))).item()


def positive_mass(pulses: Sequence[float]) -> float:
    return (
        1.35 * pulses[P_HEIGHT_IN_BAND]
        + 0.70 * pulses[P_STABLE]
        + 0.24 * pulses[P_ANGLE_TARGET]
        + 0.12 * pulses[P_OMEGA_ZERO]
    )


def negative_mass(pulses: Sequence[float]) -> float:
    return (
        1.00 * pulses[P_HEIGHT_FAR_LOW]
        + 0.55 * pulses[P_HEIGHT_LOW_NEAR]
        + 0.55 * pulses[P_HEIGHT_HIGH_NEAR]
        + 1.00 * pulses[P_HEIGHT_FAR_HIGH]
        + 0.20 * pulses[P_OMEGA_FAST_LEFT]
        + 0.20 * pulses[P_OMEGA_FAST_RIGHT]
        + 0.10 * pulses[P_ANGLE_HIGH]
    )


def pulse_score(pulses: Sequence[float]) -> float:
    return positive_mass(pulses) - negative_mass(pulses)


def action_entropy(probs: Sequence[float]) -> float:
    total = 0.0
    for prob in probs:
        if prob > 1e-12:
            total -= prob * torch.log(torch.tensor(prob + 1e-12)).item()
    return total


class PulseSNNState:
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

    def clone(self) -> "PulseSNNState":
        return PulseSNNState(
            membrane=self.membrane.clone(),
            spikes=self.spikes.clone(),
            trace=self.trace.clone(),
            adapt=self.adapt.clone(),
            action_rise=self.action_rise.clone(),
            action_decay=self.action_decay.clone(),
        )


class PulseCyclicSNN(nn.Module):
    def __init__(self, hidden_dim: int, in_degree: int, seed: int) -> None:
        super().__init__()
        rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.obs_proj = nn.Linear(PULSE_DIM, hidden_dim)
        self.action_proj = nn.Linear(base.ACTION_DIM * 2, hidden_dim)
        self.decoder = nn.Linear(hidden_dim + base.ACTION_DIM, PULSE_DIM)
        self.recurrent = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.11)
        self.register_buffer("rec_mask", base.make_sparse_mask(hidden_dim, in_degree, rng))
        self.register_buffer("coactivity", torch.zeros(hidden_dim, hidden_dim))
        self.register_buffer("usage", torch.zeros(hidden_dim, hidden_dim))
        active_mask = torch.zeros(hidden_dim)
        active_mask[: min(hidden_dim, INIT_ACTIVE_HIDDEN)] = 1.0
        self.register_buffer("active_mask", active_mask)
        self.register_buffer("node_activity", torch.zeros(hidden_dim))
        self.register_buffer("node_importance", torch.zeros(hidden_dim))
        node_age = torch.zeros(hidden_dim, dtype=torch.long)
        node_age[: min(hidden_dim, INIT_ACTIVE_HIDDEN)] = 10_000
        self.register_buffer("node_age", node_age)
        self.births = 0
        self.deaths = 0

        self.membrane_decay = 0.86
        self.trace_decay = 0.91
        self.adapt_decay = 0.93
        self.adapt_strength = 0.22
        self.threshold = 0.18
        self.spike_temp = 0.28

        self.register_buffer("action_rise_decay", torch.full((base.ACTION_DIM,), base.ACTION_RISE_FACTOR))
        self.register_buffer("action_decay_decay", torch.full((base.ACTION_DIM,), base.ACTION_DECAY_FACTOR))

    def zero_state(self, batch_size: int, device: torch.device) -> PulseSNNState:
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        action_zeros = torch.zeros(batch_size, base.ACTION_DIM, device=device)
        return PulseSNNState(
            membrane=zeros.clone(),
            spikes=zeros.clone(),
            trace=zeros.clone(),
            adapt=zeros.clone(),
            action_rise=action_zeros.clone(),
            action_decay=action_zeros.clone(),
        )

    def masked_recurrent(self) -> torch.Tensor:
        node_mask = self.active_mask.unsqueeze(0) * self.active_mask.unsqueeze(1)
        return self.recurrent * self.rec_mask * node_mask

    def active_count(self) -> int:
        return int(torch.count_nonzero(self.active_mask > 0.5).item())

    def inactive_indices(self) -> List[int]:
        return torch.nonzero(self.active_mask < 0.5, as_tuple=False).flatten().tolist()

    def active_indices(self) -> List[int]:
        return torch.nonzero(self.active_mask > 0.5, as_tuple=False).flatten().tolist()

    def mean_new_node_importance(self) -> float:
        indices = [
            idx
            for idx in self.active_indices()
            if self.node_age[idx].item() < 10_000
        ]
        if not indices:
            return 0.0
        return float(self.node_importance[indices].mean().item())

    def step(
        self,
        pulses: torch.Tensor,
        action_pulses: torch.Tensor,
        state: PulseSNNState,
        ablate_hidden: Sequence[int] | None = None,
    ) -> Tuple[torch.Tensor, PulseSNNState]:
        action_rise = state.action_rise * self.action_rise_decay + action_pulses
        action_decay = state.action_decay * self.action_decay_decay + action_pulses
        action_alpha = torch.relu(action_decay - action_rise) * base.ACTION_GAIN
        action_alpha_norm = torch.clamp(action_alpha / base.MAX_ACTION_ALPHA, 0.0, 1.0)

        rec_drive = F.linear(state.trace, self.masked_recurrent())
        total_drive = self.obs_proj(pulses) + self.action_proj(torch.cat([action_pulses, action_alpha_norm], dim=-1)) + rec_drive
        total_drive = total_drive * self.active_mask.unsqueeze(0)

        membrane = self.membrane_decay * state.membrane + total_drive - self.adapt_strength * state.adapt
        spikes = torch.sigmoid((membrane - self.threshold) / self.spike_temp)
        adapt = self.adapt_decay * state.adapt + spikes
        trace = self.trace_decay * state.trace + spikes
        membrane = membrane * self.active_mask.unsqueeze(0)
        spikes = spikes * self.active_mask.unsqueeze(0)
        adapt = adapt * self.active_mask.unsqueeze(0)
        trace = trace * self.active_mask.unsqueeze(0)

        if ablate_hidden:
            index = torch.tensor(list(ablate_hidden), dtype=torch.long, device=membrane.device)
            membrane = membrane.index_fill(1, index, 0.0)
            spikes = spikes.index_fill(1, index, 0.0)
            adapt = adapt.index_fill(1, index, 0.0)
            trace = trace.index_fill(1, index, 0.0)

        pred = torch.sigmoid(self.decoder(torch.cat([trace, action_alpha_norm], dim=-1)))
        next_state = PulseSNNState(
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
        pulse_seq: torch.Tensor,
        action_seq: torch.Tensor,
        state: PulseSNNState | None = None,
        collect_stats: bool = False,
        return_trace_seq: bool = False,
    ) -> Tuple[torch.Tensor, PulseSNNState, torch.Tensor | None]:
        batch_size, seq_len, _ = pulse_seq.shape
        if state is None:
            state = self.zero_state(batch_size, pulse_seq.device)

        preds: List[torch.Tensor] = []
        trace_seq: List[torch.Tensor] = []
        for step_idx in range(seq_len):
            pred, state = self.step(pulse_seq[:, step_idx], action_seq[:, step_idx], state)
            preds.append(pred)
            if return_trace_seq:
                trace_seq.append(state.trace)
            if collect_stats:
                self.coactivity.mul_(0.992).add_(
                    torch.matmul(state.trace.detach().T, state.trace.detach()) / float(max(batch_size, 1))
                )
                self.usage.mul_(0.996).add_(
                    torch.matmul(state.spikes.detach().T, state.spikes.detach()) / float(max(batch_size, 1))
                )
                self.node_activity.mul_(NODE_ACTIVITY_DECAY).add_(
                    (1.0 - NODE_ACTIVITY_DECAY) * state.spikes.detach().mean(dim=0)
                )
        stacked_trace = torch.stack(trace_seq, dim=1) if return_trace_seq else None
        return torch.stack(preds, dim=1), state, stacked_trace

    def maybe_rewire(self, rng: random.Random) -> None:
        with torch.no_grad():
            for dst in self.active_indices():
                existing = torch.nonzero(self.rec_mask[dst] > 0.5, as_tuple=False).flatten().tolist()
                if len(existing) > MODEL_IN_DEGREE - 1:
                    ranked_existing = [
                        (abs(float(self.recurrent[dst, src])) * float(self.usage[dst, src] + 1e-6), src)
                        for src in existing
                    ]
                    ranked_existing.sort(key=lambda item: item[0])
                    _, src = ranked_existing[0]
                    self.rec_mask[dst, src] = 0.0
                    self.recurrent[dst, src] = 0.0

                available = [src for src in self.active_indices() if self.rec_mask[dst, src] < 0.5 and src != dst]
                available.sort(key=lambda src: float(self.coactivity[dst, src]), reverse=True)
                if available:
                    src = available[0]
                    self.rec_mask[dst, src] = 1.0
                    self.recurrent[dst, src] = torch.empty(1).normal_(0.0, 0.08).item()

    def maybe_grow_or_prune(self, last_val_loss: float, rng: random.Random) -> None:
        with torch.no_grad():
            active_before = self.active_indices()
            for idx in active_before:
                if self.node_age[idx].item() < 10_000:
                    self.node_age[idx] += 1

            if last_val_loss > GROWTH_LOSS_THRESHOLD and self.inactive_indices():
                dormant = self.inactive_indices()
                active = self.active_indices()
                ranked_active = sorted(active, key=lambda idx: float(self.node_activity[idx]), reverse=True)
                for new_idx in dormant[:NODE_BIRTHS_PER_EVENT]:
                    self.active_mask[new_idx] = 1.0
                    self.recurrent[new_idx, :] = 0.0
                    self.recurrent[:, new_idx] = 0.0
                    self.rec_mask[new_idx, :] = 0.0
                    self.rec_mask[:, new_idx] = 0.0
                    self.coactivity[new_idx, :] = 0.0
                    self.coactivity[:, new_idx] = 0.0
                    self.usage[new_idx, :] = 0.0
                    self.usage[:, new_idx] = 0.0
                    self.node_activity[new_idx] = 0.0
                    self.node_importance[new_idx] = 0.0
                    self.node_age[new_idx] = 0
                    for src in ranked_active[: MODEL_IN_DEGREE]:
                        if src == new_idx:
                            continue
                        self.rec_mask[new_idx, src] = 1.0
                        self.recurrent[new_idx, src] = torch.empty(1).normal_(0.0, 0.08).item()
                    for dst in ranked_active[: max(1, MODEL_IN_DEGREE // 2)]:
                        if dst == new_idx:
                            continue
                        self.rec_mask[dst, new_idx] = 1.0
                        self.recurrent[dst, new_idx] = torch.empty(1).normal_(0.0, 0.08).item()
                    self.births += 1

            if self.active_count() > MIN_ACTIVE_HIDDEN:
                active = self.active_indices()
                ranked_weak = sorted(
                    active,
                    key=lambda idx: (
                        0 if (
                            self.node_age[idx].item() >= NEW_NODE_PROBATION_EVENTS
                            and self.node_importance[idx].item() < NEW_NODE_IMPORTANCE_THRESHOLD
                        ) else 1,
                        float(self.node_importance[idx]),
                        float(self.node_activity[idx]),
                    ),
                )
                removed = 0
                for idx in ranked_weak:
                    if idx < INIT_ACTIVE_HIDDEN:
                        continue
                    if self.node_age[idx].item() < NEW_NODE_PROBATION_EVENTS:
                        continue
                    if self.node_importance[idx].item() >= NEW_NODE_IMPORTANCE_THRESHOLD and removed == 0:
                        continue
                    self.active_mask[idx] = 0.0
                    self.recurrent[idx, :] = 0.0
                    self.recurrent[:, idx] = 0.0
                    self.rec_mask[idx, :] = 0.0
                    self.rec_mask[:, idx] = 0.0
                    self.coactivity[idx, :] = 0.0
                    self.coactivity[:, idx] = 0.0
                    self.usage[idx, :] = 0.0
                    self.usage[:, idx] = 0.0
                    self.node_activity[idx] = 0.0
                    self.node_importance[idx] = 0.0
                    self.node_age[idx] = 0
                    self.deaths += 1
                    removed += 1
                    if removed >= NODE_DEATHS_PER_EVENT or self.active_count() <= MIN_ACTIVE_HIDDEN:
                        break


def build_sequence_samples(
    episodes: Sequence[Tuple[List[base.StickState], List[str]]],
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pulse_sequences: List[List[List[float]]] = []
    action_sequences: List[List[List[float]]] = []
    target_sequences: List[List[List[float]]] = []
    for states, actions in episodes:
        pulses = [pulse_from_state(state) for state in states]
        if len(actions) < seq_len:
            continue
        for start in range(0, len(actions) - seq_len + 1):
            pulse_sequences.append(pulses[start : start + seq_len])
            action_sequences.append([base.action_pulse(action) for action in actions[start : start + seq_len]])
            target_sequences.append(pulses[start + 1 : start + seq_len + 1])
    return (
        torch.tensor(pulse_sequences, dtype=torch.float32),
        torch.tensor(action_sequences, dtype=torch.float32),
        torch.tensor(target_sequences, dtype=torch.float32),
    )


def weighted_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = LOSS_WEIGHTS.to(pred.device)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    return (bce * weights).mean()


def batch_iter(x: torch.Tensor, a: torch.Tensor, y: torch.Tensor, batch_size: int, rng: random.Random) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    indices = list(range(x.shape[0]))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        idx = torch.tensor(chosen, dtype=torch.long)
        yield x[idx], a[idx], y[idx]


def evaluate_world_model(model: PulseCyclicSNN, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> WorldModelMetrics:
    model.eval()
    with torch.no_grad():
        preds, _, _ = model.forward_sequence(x, a, collect_stats=False)
        loss = float(weighted_bce(preds, y).item())
        in_band_bce = float(F.binary_cross_entropy(preds[:, :, P_HEIGHT_IN_BAND], y[:, :, P_HEIGHT_IN_BAND]).item())
    return WorldModelMetrics(
        val_loss=loss,
        val_in_band_bce=in_band_bce,
        active_nodes=float(model.active_count()),
        births=float(model.births),
        deaths=float(model.deaths),
        mean_new_node_importance=model.mean_new_node_importance(),
    )


def _train_world_model_internal(seed: int, collect_history: bool) -> Tuple[PulseCyclicSNN, WorldModelMetrics, List[GrowthSnapshot]]:
    train_eps = base.collect_dataset(
        seed=seed,
        episodes=TRAIN_EPISODES,
        horizon=base.TRAIN_HORIZON,
        disturbance_scale=0.40,
        disturbance_mode=base.DEFAULT_DISTURBANCE_MODE,
        policy_mix="mixed",
    )
    val_eps = base.collect_dataset(
        seed=seed + 991,
        episodes=VAL_EPISODES,
        horizon=base.TRAIN_HORIZON,
        disturbance_scale=0.54,
        disturbance_mode=base.DEFAULT_DISTURBANCE_MODE,
        policy_mix="mixed",
    )
    train_x, train_a, train_y = build_sequence_samples(train_eps, seq_len=SEQ_LEN)
    val_x, val_a, val_y = build_sequence_samples(val_eps, seq_len=SEQ_LEN)

    model = PulseCyclicSNN(hidden_dim=MODEL_HIDDEN, in_degree=MODEL_IN_DEGREE, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    rng = random.Random(seed)
    history: List[GrowthSnapshot] = []

    for epoch in range(MODEL_EPOCHS):
        model.train()
        for batch_x, batch_a, batch_y in batch_iter(train_x, train_a, train_y, BATCH_SIZE, rng):
            optimizer.zero_grad(set_to_none=True)
            preds, _, trace_seq = model.forward_sequence(
                batch_x,
                batch_a,
                collect_stats=True,
                return_trace_seq=True,
            )
            if trace_seq is not None:
                decoder_salience = torch.matmul(
                    LOSS_WEIGHTS.to(batch_x.device),
                    torch.abs(model.decoder.weight[:, : model.hidden_dim]),
                )
                trace_activity = trace_seq.detach().abs().mean(dim=(0, 1))
                importance = trace_activity * decoder_salience.detach()
                model.node_importance.mul_(IMPORTANCE_DECAY).add_((1.0 - IMPORTANCE_DECAY) * importance)
            loss = weighted_bce(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 3 == 0:
            model.maybe_rewire(rng)
            metrics = evaluate_world_model(model, val_x, val_a, val_y)
            model.maybe_grow_or_prune(metrics.val_loss, rng)
            if collect_history:
                final_metrics = evaluate_world_model(model, val_x, val_a, val_y)
                history.append(
                    GrowthSnapshot(
                        epoch=epoch + 1,
                        val_loss=final_metrics.val_loss,
                        val_in_band_bce=final_metrics.val_in_band_bce,
                        active_nodes=int(final_metrics.active_nodes),
                        births=int(final_metrics.births),
                        deaths=int(final_metrics.deaths),
                        active_mask=[int(value) for value in model.active_mask.tolist()],
                        node_importance=[float(value) for value in model.node_importance.tolist()],
                        node_age=[int(value) for value in model.node_age.tolist()],
                    )
                )

    final_metrics = evaluate_world_model(model, val_x, val_a, val_y)
    if collect_history and (not history or history[-1].epoch != MODEL_EPOCHS):
        history.append(
            GrowthSnapshot(
                epoch=MODEL_EPOCHS,
                val_loss=final_metrics.val_loss,
                val_in_band_bce=final_metrics.val_in_band_bce,
                active_nodes=int(final_metrics.active_nodes),
                births=int(final_metrics.births),
                deaths=int(final_metrics.deaths),
                active_mask=[int(value) for value in model.active_mask.tolist()],
                node_importance=[float(value) for value in model.node_importance.tolist()],
                node_age=[int(value) for value in model.node_age.tolist()],
            )
        )
    return model, final_metrics, history


def train_world_model(seed: int) -> Tuple[PulseCyclicSNN, WorldModelMetrics]:
    model, metrics, _ = _train_world_model_internal(seed, collect_history=False)
    return model, metrics


def train_world_model_with_history(seed: int) -> Tuple[PulseCyclicSNN, WorldModelMetrics, List[GrowthSnapshot]]:
    return _train_world_model_internal(seed, collect_history=True)


def prediction_error_to_beta(error_ema: float) -> float:
    accuracy = torch.exp(torch.tensor(-error_ema / ACCURACY_SCALE)).item()
    return BETA_MIN + (BETA_MAX - BETA_MIN) * accuracy


def choose_imagined_base_action(model: PulseCyclicSNN, pulses: torch.Tensor, state: PulseSNNState) -> str:
    best_action = base.ACTIONS[0]
    best_score = float("-inf")
    for action in base.ACTIONS:
        pred, _ = model.step(pulses.unsqueeze(0), torch.tensor([base.action_pulse(action)], dtype=torch.float32), state.clone())
        score = pulse_score(pred[0].tolist())
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def imagined_rollout(
    model: PulseCyclicSNN,
    pulses: torch.Tensor,
    state: PulseSNNState,
    first_action: str | None,
    horizon: int,
    ablate_hidden: Sequence[int] | None = None,
) -> List[torch.Tensor]:
    current_pulses = pulses.clone()
    current_state = state.clone()
    predictions: List[torch.Tensor] = []
    for step_idx in range(horizon):
        action = first_action if step_idx == 0 else choose_imagined_base_action(model, current_pulses, current_state)
        pred, current_state = model.step(
            current_pulses.unsqueeze(0),
            torch.tensor([base.action_pulse(action)], dtype=torch.float32),
            current_state,
            ablate_hidden=ablate_hidden,
        )
        current_pulses = pred[0].detach()
        predictions.append(current_pulses.clone())
    return predictions


def top_latent_indices(state: PulseSNNState, topk: int) -> List[int]:
    if topk <= 0:
        return []
    scores = (state.trace[0] + 0.5 * state.spikes[0]).detach()
    if int(torch.count_nonzero(scores > 1e-5).item()) == 0:
        return []
    k = min(topk, scores.shape[0])
    values, indices = torch.topk(scores, k=k)
    chosen: List[int] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        if value <= 1e-5:
            continue
        chosen.append(int(index))
    return chosen


def prediction_only_score(model: PulseCyclicSNN, pulses: torch.Tensor, state: PulseSNNState, action: str) -> float:
    pred, _ = model.step(pulses.unsqueeze(0), torch.tensor([base.action_pulse(action)], dtype=torch.float32), state.clone())
    return pulse_score(pred[0].tolist())


def signed_action_score(
    model: PulseCyclicSNN,
    pulses: torch.Tensor,
    state: PulseSNNState,
    action: str,
    include_latent_twohop: bool,
) -> Tuple[float, float, float]:
    with_action = imagined_rollout(model, pulses, state, first_action=action, horizon=FA_HORIZON)
    without_action = imagined_rollout(model, pulses, state, first_action=None, horizon=FA_HORIZON)
    base_pred, first_state = model.step(
        pulses.unsqueeze(0),
        torch.tensor([base.action_pulse(action)], dtype=torch.float32),
        state.clone(),
    )
    latent_masked = None
    if include_latent_twohop:
        latent_nodes = top_latent_indices(first_state, LATENT_TOPK)
        latent_masked = imagined_rollout(
            model,
            pulses,
            state,
            first_action=action,
            horizon=FA_HORIZON,
            ablate_hidden=latent_nodes,
        )
    immediate = pulse_score(base_pred[0].tolist())

    branch_pos = 0.0
    branch_neg = 0.0
    delta_pos = 0.0
    delta_neg = 0.0
    latent_pos = 0.0
    latent_neg = 0.0
    for step_idx, (full_pulses, null_pulses) in enumerate(zip(with_action, without_action)):
        discount = FA_DISCOUNT**step_idx
        full_list = full_pulses.tolist()
        null_list = null_pulses.tolist()
        branch_pos += discount * positive_mass(full_list)
        branch_neg += discount * negative_mass(full_list)
        delta_pos += discount * (positive_mass(full_list) - positive_mass(null_list))
        delta_neg += discount * (negative_mass(full_list) - negative_mass(null_list))
        if latent_masked is not None:
            masked_list = latent_masked[step_idx].tolist()
            latent_pos += discount * (positive_mass(full_list) - positive_mass(masked_list))
            latent_neg += discount * (negative_mass(full_list) - negative_mass(masked_list))

    total = (
        BASE_WEIGHT * immediate
        + PROSPECTIVE_WEIGHT * (branch_pos - NEG_WEIGHT * branch_neg)
        + DELTA_WEIGHT * (delta_pos - NEG_WEIGHT * delta_neg)
    )
    if include_latent_twohop:
        total += LATENT_WEIGHT * (latent_pos - NEG_WEIGHT * latent_neg)
    return total, delta_pos + latent_pos, delta_neg + latent_neg


def choose_action_with_policy(
    policy_name: str,
    model: PulseCyclicSNN,
    pulses: torch.Tensor,
    state: PulseSNNState,
    beta: float,
    rng: random.Random,
) -> Tuple[str, List[float]]:
    scores: List[float] = []
    for action in base.ACTIONS:
        if policy_name == "prediction_only":
            score = prediction_only_score(model, pulses, state, action)
        elif policy_name == "signed_pulse":
            score, _, _ = signed_action_score(model, pulses, state, action, include_latent_twohop=False)
        elif policy_name == "signed_twohop_pulse":
            score, _, _ = signed_action_score(model, pulses, state, action, include_latent_twohop=True)
        else:
            raise ValueError(f"unknown policy_name: {policy_name}")
        scores.append(score)
    logits = torch.tensor([beta * value for value in scores], dtype=torch.float32)
    probs = torch.softmax(logits, dim=0).tolist()
    action = rng.choices(list(base.ACTIONS), weights=probs, k=1)[0]
    return action, probs


def evaluate_policy(model: PulseCyclicSNN, seed: int, policy_name: str, disturbance_scale: float) -> EvalMetrics:
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
            state = base.random_initial_state(rng)
            disturbances = base.sample_disturbance_sequence(
                random.Random(seed * 1000 + episode_idx),
                EVAL_HORIZON,
                disturbance_scale,
                mode=base.DEFAULT_DISTURBANCE_MODE,
            )
            snn_state = model.zero_state(batch_size=1, device=torch.device("cpu"))
            error_ema = 0.10
            for disturbance in disturbances:
                pulse_vec = torch.tensor(pulse_from_state(state), dtype=torch.float32)
                beta = prediction_error_to_beta(error_ema)
                planner_target = base.planner_action(state)
                action, probs = choose_action_with_policy(policy_name, model, pulse_vec, snn_state, beta, rng)
                pred_next, updated_state = model.step(
                    pulse_vec.unsqueeze(0),
                    torch.tensor([base.action_pulse(action)], dtype=torch.float32),
                    snn_state.clone(),
                )
                next_state = base.transition_dynamics(state, action, disturbance=disturbance)
                target_pulses = torch.tensor(pulse_from_state(next_state), dtype=torch.float32)
                pred_error = float(weighted_bce(pred_next[0], target_pulses).item())
                error_ema = 0.94 * error_ema + 0.06 * pred_error

                snn_state = updated_state
                state = next_state

                height = base.height_from_theta(state.theta)
                total_band_hits += int(base.band_distance(height) <= 1e-8)
                total_abs_error += abs(height - base.TARGET_HEIGHT)
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
        "# Pulse-Node Sparse Cyclic SNN Stick-Height Control",
        "",
        "## World Model",
        f"- val_loss: {summary.world_model['val_loss']:.6f}",
        f"- val_in_band_bce: {summary.world_model['val_in_band_bce']:.6f}",
        f"- active_nodes: {summary.world_model['active_nodes']:.0f}",
        f"- births: {summary.world_model['births']:.0f}",
        f"- deaths: {summary.world_model['deaths']:.0f}",
        f"- mean_new_node_importance: {summary.world_model['mean_new_node_importance']:.6f}",
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
    model, world_metrics = train_world_model(seed)

    nominal_prediction = evaluate_policy(model, seed + 101, "prediction_only", disturbance_scale=0.45)
    nominal_signed = evaluate_policy(model, seed + 202, "signed_pulse", disturbance_scale=0.45)
    nominal_twohop = evaluate_policy(model, seed + 252, "signed_twohop_pulse", disturbance_scale=0.45)
    stress_prediction = evaluate_policy(model, seed + 303, "prediction_only", disturbance_scale=0.72)
    stress_signed = evaluate_policy(model, seed + 404, "signed_pulse", disturbance_scale=0.72)
    stress_twohop = evaluate_policy(model, seed + 454, "signed_twohop_pulse", disturbance_scale=0.72)

    return RunSummary(
        config={
            "seed": seed,
            "target_height": base.TARGET_HEIGHT,
            "target_band": base.TARGET_BAND,
            "pulse_dim": PULSE_DIM,
            "hidden_dim": MODEL_HIDDEN,
            "init_active_hidden": INIT_ACTIVE_HIDDEN,
            "fa_horizon": FA_HORIZON,
            "latent_weight": LATENT_WEIGHT,
            "latent_topk": LATENT_TOPK,
            "disturbance_mode": base.DEFAULT_DISTURBANCE_MODE,
        },
        world_model={
            "val_loss": world_metrics.val_loss,
            "val_in_band_bce": world_metrics.val_in_band_bce,
            "active_nodes": world_metrics.active_nodes,
            "births": world_metrics.births,
            "deaths": world_metrics.deaths,
            "mean_new_node_importance": world_metrics.mean_new_node_importance,
        },
        evaluation={
            "nominal_prediction_only": nominal_prediction.__dict__,
            "nominal_signed_pulse": nominal_signed.__dict__,
            "nominal_signed_twohop_pulse": nominal_twohop.__dict__,
            "stress_prediction_only": stress_prediction.__dict__,
            "stress_signed_pulse": stress_signed.__dict__,
            "stress_signed_twohop_pulse": stress_twohop.__dict__,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pulse-node sparse cyclic SNN stick-height control.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, default=Path("snn_pulse_stick_height_control.json"))
    parser.add_argument("--md", type=Path, default=Path("snn_pulse_stick_height_control.md"))
    args = parser.parse_args()

    summary = run_experiment(args.seed)
    args.json.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
    args.md.write_text(format_markdown(summary), encoding="utf-8")
    print(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
