#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-attention FA experiment.

This version replaces the synthetic attention operator with a fixed causal
Transformer world model trained on POMDP trajectories. Oracle FA labels are
computed from the world model's real self-attention weights during imagined
future rollouts.

Pipeline
--------
1. Train or load a fixed transformer world model on offline trajectories.
2. Extract oracle FA labels from real attention in imagined rollouts.
3. Train a lightweight FA predictor on those labels.
4. Run a closed-loop experiment where only FA predictor / policy updates.
5. Compare against Exact EFE and a myopic baseline.

No third-party dependencies are required beyond torch and numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: Tuple[str, ...] = ("eat", "inspect", "move", "wait")
ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTIONS)}
BELIEF_BUCKETS: Tuple[float, ...] = tuple(i / 10.0 for i in range(11))
MAX_ENERGY = 4
EPISODE_HORIZON = 12
DEFAULT_PLAN_DEPTH = 3
DEFAULT_INFO_WEIGHT = 0.85
DEFAULT_DISCOUNT = 0.85
FA_TARGET_HORIZON = 4
FA_TARGET_DISCOUNT = 0.82
DEFAULT_FA_INFO_BONUS = 0.75
DEFAULT_FA_POLICY_BASE_WEIGHT = 1.25

PAD_ID = 0
BOS_ID = 1
OBS_OFFSET = 2
ACT_OFFSET = OBS_OFFSET + 10
EOS_ID = ACT_OFFSET + len(ACTIONS)
VOCAB_SIZE = EOS_ID + 1
MAX_SEQ_LEN = 1 + 2 * EPISODE_HORIZON + 2 * FA_TARGET_HORIZON + 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class VisibilityConfig:
    name: str
    passive_accuracy: float
    inspect_accuracy: float


VISIBILITY_CONFIGS: Tuple[VisibilityConfig, ...] = (
    VisibilityConfig(name="full", passive_accuracy=1.00, inspect_accuracy=1.00),
    VisibilityConfig(name="partial", passive_accuracy=0.65, inspect_accuracy=0.95),
    VisibilityConfig(name="hard_partial", passive_accuracy=0.52, inspect_accuracy=0.88),
)
CFG_BY_NAME = {cfg.name: cfg for cfg in VISIBILITY_CONFIGS}
INITIAL_RICH_PROB = 0.55
PREFERENCE_PROBS = {0: 0.005, 1: 0.040, 2: 0.160, 3: 0.340, 4: 0.455}
PREFERENCE_UTILITY = {0: -4.0, 1: -1.2, 2: 0.8, 3: 1.4, 4: 1.8}


def clamp_energy(value: int) -> int:
    return max(0, min(MAX_ENERGY, value))


def nearest_bucket(value: float) -> float:
    return min(BELIEF_BUCKETS, key=lambda bucket: abs(bucket - value))


def round_belief(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 3)


def bernoulli_entropy(prob: float) -> float:
    prob = min(1.0 - 1e-12, max(1e-12, prob))
    return -(prob * math.log(prob) + (1.0 - prob) * math.log(1.0 - prob))


def normalized_entropy(prob: float) -> float:
    return bernoulli_entropy(prob) / math.log(2.0)


def preference_risk(energy: int) -> float:
    return -math.log(PREFERENCE_PROBS[energy])


def preference_utility(energy: int) -> float:
    return PREFERENCE_UTILITY[energy]


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def zscore_dict(scores: Dict[str, float]) -> Dict[str, float]:
    values = list(scores.values())
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = math.sqrt(max(variance, 1e-12))
    return {key: (value - mean_value) / std_value for key, value in scores.items()}


def softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
    scaled = [value / max(temperature, 1e-8) for value in values]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def sample_from_probs(actions: Sequence[str], probs: Sequence[float], rng: random.Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    for action, prob in zip(actions, probs):
        cumulative += prob
        if draw <= cumulative:
            return action
    return actions[-1]


def cue_accuracy(cfg: VisibilityConfig, action: str) -> float:
    return cfg.inspect_accuracy if action == "inspect" else cfg.passive_accuracy


def next_energy(energy: int, site_rich: int, action: str) -> int:
    if action == "eat":
        return clamp_energy(energy + 2) if site_rich else clamp_energy(energy - 1)
    return clamp_energy(energy - 1)


def site_transition(site_rich: int, action: str) -> Iterable[Tuple[int, float]]:
    if action == "eat":
        if site_rich:
            return ((1, 0.35), (0, 0.65))
        return ((0, 0.85), (1, 0.15))
    if action == "move":
        return ((1, 0.65), (0, 0.35))
    if site_rich:
        return ((1, 0.85), (0, 0.15))
    return ((1, 0.25), (0, 0.75))


def sample_site_transition(site_rich: int, action: str, rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for next_site, prob in site_transition(site_rich, action):
        cumulative += prob
        if draw <= cumulative:
            return next_site
    return 0


def sample_cue(site_rich: int, cfg: VisibilityConfig, action: str, rng: random.Random) -> int:
    acc = cue_accuracy(cfg, action)
    return site_rich if rng.random() <= acc else 1 - site_rich


def initial_posterior_from_cue(prior_rich: float, cue: int, cfg: VisibilityConfig) -> float:
    rich_like = cfg.passive_accuracy if cue == 1 else 1.0 - cfg.passive_accuracy
    barren_like = 1.0 - rich_like
    denom = prior_rich * rich_like + (1.0 - prior_rich) * barren_like
    if denom <= 0.0:
        return prior_rich
    return (prior_rich * rich_like) / denom


def initial_belief(cfg: VisibilityConfig, site_rich: int, rng: random.Random) -> float:
    cue = sample_cue(site_rich, cfg, "wait", rng)
    return initial_posterior_from_cue(INITIAL_RICH_PROB, cue, cfg)


def observation_model(
    energy: int,
    belief_rich: float,
    action: str,
    cfg: VisibilityConfig,
) -> Tuple[float, Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    branch_probs: Dict[Tuple[int, int], float] = {}
    for current_site, current_prob in ((1, belief_rich), (0, 1.0 - belief_rich)):
        if current_prob <= 0.0:
            continue
        energy_next = next_energy(energy, current_site, action)
        for next_site, trans_prob in site_transition(current_site, action):
            branch_probs[(energy_next, next_site)] = branch_probs.get((energy_next, next_site), 0.0) + current_prob * trans_prob
    prior_rich = sum(prob for (_, next_site), prob in branch_probs.items() if next_site == 1)
    obs_probs: Dict[Tuple[int, int], float] = {}
    rich_mass: Dict[Tuple[int, int], float] = {}
    acc = cue_accuracy(cfg, action)
    for (energy_next, next_site), branch_prob in branch_probs.items():
        for cue in (0, 1):
            likelihood = acc if cue == next_site else 1.0 - acc
            obs = (energy_next, cue)
            obs_probs[obs] = obs_probs.get(obs, 0.0) + branch_prob * likelihood
            if next_site == 1:
                rich_mass[obs] = rich_mass.get(obs, 0.0) + branch_prob * likelihood
    posteriors: Dict[Tuple[int, int], float] = {}
    for obs, prob in obs_probs.items():
        posteriors[obs] = prior_rich if prob <= 0.0 else rich_mass.get(obs, 0.0) / prob
    return prior_rich, obs_probs, posteriors


@lru_cache(maxsize=None)
def exact_efe_scores_cached(
    energy: int,
    rounded_belief: float,
    cfg_name: str,
    depth: int,
    info_weight: float,
    discount: float,
) -> Tuple[float, ...]:
    if energy <= 0 or depth <= 0:
        return tuple(0.0 for _ in ACTIONS)
    cfg = CFG_BY_NAME[cfg_name]
    belief_rich = float(rounded_belief)
    scores: List[float] = []
    for action in ACTIONS:
        prior_rich, obs_probs, posteriors = observation_model(energy, belief_rich, action, cfg)
        expected_risk = sum(prob * preference_risk(obs_energy) for (obs_energy, _), prob in obs_probs.items())
        expected_entropy = sum(prob * bernoulli_entropy(posteriors[obs]) for obs, prob in obs_probs.items())
        info_gain = bernoulli_entropy(prior_rich) - expected_entropy
        future_cost = 0.0
        if depth > 1:
            for obs, prob in obs_probs.items():
                obs_energy, _cue = obs
                if obs_energy <= 0:
                    continue
                next_scores = exact_efe_scores_cached(obs_energy, round_belief(posteriors[obs]), cfg_name, depth - 1, info_weight, discount)
                future_cost += prob * min(next_scores)
        scores.append(expected_risk - info_weight * info_gain + discount * future_cost)
    return tuple(scores)


def exact_efe_action_scores(
    energy: int,
    belief_rich: float,
    cfg: VisibilityConfig,
    depth: int = DEFAULT_PLAN_DEPTH,
    info_weight: float = DEFAULT_INFO_WEIGHT,
    discount: float = DEFAULT_DISCOUNT,
) -> Dict[str, float]:
    scores = exact_efe_scores_cached(energy, round_belief(belief_rich), cfg.name, depth, info_weight, discount)
    return {action: score for action, score in zip(ACTIONS, scores)}


def argmin_action(action_scores: Dict[str, float]) -> str:
    return min(ACTIONS, key=lambda action: (action_scores[action], action))


def argmax_action(action_scores: Dict[str, float]) -> str:
    return max(ACTIONS, key=lambda action: (action_scores[action], action))


def immediate_expected_utility_scores(energy: int, belief_rich: float) -> Dict[str, float]:
    scores = {}
    for action in ACTIONS:
        expected = 0.0
        for current_site, prob in ((1, belief_rich), (0, 1.0 - belief_rich)):
            if prob <= 0.0:
                continue
            expected += prob * preference_utility(next_energy(energy, current_site, action))
        scores[action] = expected
    return scores


def expected_info_gain_scores(energy: int, belief_rich: float, cfg: VisibilityConfig) -> Dict[str, float]:
    scores = {}
    for action in ACTIONS:
        prior_rich, obs_probs, posteriors = observation_model(energy, belief_rich, action, cfg)
        expected_entropy = sum(prob * bernoulli_entropy(posteriors[obs]) for obs, prob in obs_probs.items())
        scores[action] = bernoulli_entropy(prior_rich) - expected_entropy
    return scores


def obs_token_id(energy: int, cue: int) -> int:
    return OBS_OFFSET + energy * 2 + cue


def decode_obs_token(token_id: int) -> Tuple[int, int]:
    raw = token_id - OBS_OFFSET
    return raw // 2, raw % 2


def action_token_id(action: str) -> int:
    return ACT_OFFSET + ACTION_TO_IDX[action]


def decode_action_token(token_id: int) -> str:
    return ACTIONS[token_id - ACT_OFFSET]

@dataclass
class OfflineEpisode:
    tokens: List[int]
    observations: List[Tuple[int, int]]


@dataclass
class ContextRecord:
    history_tokens: List[int]
    energy: int
    belief: float


@dataclass
class WorldModelMetrics:
    train_loss: float
    val_loss: float
    val_next_token_acc: float
    val_obs_token_acc: float


@dataclass
class EvalMetrics:
    avg_return: float
    survival_rate: float
    safe_step_rate: float
    avg_realized_fa: float
    action_entropy: float
    eat_rate: float
    inspect_rate: float
    move_rate: float
    wait_rate: float


@dataclass
class AlignmentMetrics:
    fa_fit_mae: float
    predictor_vs_oracle_agreement: float
    oracle_vs_exact_agreement: float
    predictor_vs_exact_agreement: float
    predictor_exact_regret: float
    oracle_exact_regret: float


@dataclass
class BenchmarkRow:
    depth: int
    exact_ms_per_decision: float
    fa_ms_per_decision: float
    speedup: float


@dataclass
class RoundResult:
    round_idx: int
    behavior: EvalMetrics
    alignment: AlignmentMetrics
    policy_kl: float


@dataclass
class SeedResult:
    visibility: str
    seed: int
    world_model: WorldModelMetrics
    baseline_myopic: EvalMetrics
    baseline_exact: EvalMetrics
    rounds: List[RoundResult]
    benchmark: List[BenchmarkRow]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        need_weights: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights


class CausalTransformerWorldModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 96, nhead: int = 4, num_layers: int = 3, dim_ff: int = 192, dropout: float = 0.1, max_len: int = MAX_SEQ_LEN) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor] | None]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = input_ids.eq(PAD_ID)
        all_attn: List[torch.Tensor] = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=return_attn)
            if return_attn:
                all_attn.append(attn)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits, x, all_attn if return_attn else None


class FAPredictor(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, hidden: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if action_emb.dim() == 1:
            action_emb = action_emb.unsqueeze(0)
        return self.net(torch.cat([hidden, action_emb], dim=-1)).squeeze(-1)


def mixed_policy_action(energy: int, belief: float, cfg: VisibilityConfig, rng: random.Random) -> str:
    draw = rng.random()
    if cfg.passive_accuracy < 1.0 and energy >= 2 and normalized_entropy(belief) >= 0.82 and draw < 0.25:
        return "inspect"
    if draw < 0.33:
        return rng.choice(ACTIONS)
    if draw < 0.66:
        return argmax_action(immediate_expected_utility_scores(energy, belief))
    return argmin_action(exact_efe_action_scores(energy, belief, cfg))


def offline_policy_action(policy_name: str, energy: int, belief: float, cfg: VisibilityConfig, rng: random.Random) -> str:
    if policy_name == "mixed":
        return mixed_policy_action(energy, belief, cfg, rng)
    if policy_name == "random":
        return rng.choice(ACTIONS)
    if policy_name == "myopic":
        return argmax_action(immediate_expected_utility_scores(energy, belief))
    if policy_name == "exact":
        return argmin_action(exact_efe_action_scores(energy, belief, cfg))
    if policy_name == "entropy":
        return argmax_action(expected_info_gain_scores(energy, belief, cfg))
    raise ValueError(f"Unknown offline policy: {policy_name}")


def simulate_real_step(
    energy: int,
    site_rich: int,
    belief: float,
    action: str,
    cfg: VisibilityConfig,
    rng: random.Random,
) -> Tuple[int, int, int, float, float, float]:
    entropy_before = bernoulli_entropy(belief)
    next_site = sample_site_transition(site_rich, action, rng)
    energy_after = next_energy(energy, site_rich, action)
    cue = sample_cue(next_site, cfg, action, rng)
    _prior, _obs_probs, posteriors = observation_model(energy, belief, action, cfg)
    belief_after = posteriors.get((energy_after, cue), belief)
    entropy_after = bernoulli_entropy(belief_after)
    return energy_after, next_site, cue, belief_after, entropy_before, entropy_after


def generate_offline_dataset(
    cfg: VisibilityConfig,
    episodes: int,
    seed: int,
    offline_policy_name: str = "mixed",
) -> List[OfflineEpisode]:
    rng = random.Random(seed)
    data: List[OfflineEpisode] = []
    for _ in range(episodes):
        energy = 3 if rng.random() < 0.6 else 2
        site_rich = 1 if rng.random() < INITIAL_RICH_PROB else 0
        cue0 = sample_cue(site_rich, cfg, "wait", rng)
        belief = initial_posterior_from_cue(INITIAL_RICH_PROB, cue0, cfg)
        tokens = [BOS_ID, obs_token_id(energy, cue0)]
        observations = [(energy, cue0)]
        for _step in range(EPISODE_HORIZON):
            if energy <= 0:
                break
            action = offline_policy_action(offline_policy_name, energy, belief, cfg, rng)
            tokens.append(action_token_id(action))
            energy, site_rich, cue, belief, _e0, _e1 = simulate_real_step(energy, site_rich, belief, action, cfg, rng)
            tokens.append(obs_token_id(energy, cue))
            observations.append((energy, cue))
        tokens.append(EOS_ID)
        data.append(OfflineEpisode(tokens=tokens, observations=observations))
    return data


def pad_batch(seqs: Sequence[List[int]]) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    batch = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    for idx, seq in enumerate(seqs):
        batch[idx, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch

def train_world_model(
    cfg: VisibilityConfig,
    checkpoint_path: str,
    offline_episodes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    retrain: bool,
    d_model: int = 96,
    nhead: int = 4,
    num_layers: int = 3,
    dim_ff: int = 192,
    dropout: float = 0.1,
    offline_policy_name: str = "mixed",
) -> Tuple[CausalTransformerWorldModel, WorldModelMetrics]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = CausalTransformerWorldModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
    ).to(DEVICE)
    metrics = WorldModelMetrics(train_loss=0.0, val_loss=0.0, val_next_token_acc=0.0, val_obs_token_acc=0.0)
    if os.path.exists(checkpoint_path) and not retrain:
        payload = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(payload['model'])
        metrics = WorldModelMetrics(**payload['metrics'])
        model.eval()
        return model, metrics

    data = generate_offline_dataset(cfg, episodes=offline_episodes, seed=seed, offline_policy_name=offline_policy_name)
    rng = random.Random(seed)
    rng.shuffle(data)
    split = int(0.85 * len(data))
    train_data, val_data = data[:split], data[split:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def iterate_batches(dataset: Sequence[OfflineEpisode]) -> Iterable[torch.Tensor]:
        order = list(range(len(dataset)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_eps = [dataset[idx].tokens for idx in order[start:start + batch_size]]
            yield pad_batch(batch_eps).to(DEVICE)

    best_val = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in iterate_batches(train_data):
            optimizer.zero_grad(set_to_none=True)
            logits, _hidden, _attn = model(batch, return_attn=False)
            targets = batch[:, 1:]
            supervised_mask = ((targets.ge(OBS_OFFSET) & targets.lt(ACT_OFFSET)) | targets.eq(EOS_ID)) & targets.ne(PAD_ID)
            token_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction='none').view_as(targets)
            loss = token_loss.masked_select(supervised_mask).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        next_correct = 0
        next_total = 0
        obs_correct = 0
        obs_total = 0
        with torch.no_grad():
            for start in range(0, len(val_data), batch_size):
                batch = pad_batch([ep.tokens for ep in val_data[start:start + batch_size]]).to(DEVICE)
                logits, _hidden, _attn = model(batch, return_attn=False)
                targets = batch[:, 1:]
                supervised_mask = ((targets.ge(OBS_OFFSET) & targets.lt(ACT_OFFSET)) | targets.eq(EOS_ID)) & targets.ne(PAD_ID)
                token_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction='none').view_as(targets)
                loss = token_loss.masked_select(supervised_mask).mean()
                val_losses.append(loss.item())
                preds = logits[:, :-1, :].argmax(dim=-1)
                next_correct += preds.eq(targets).masked_select(supervised_mask).sum().item()
                next_total += supervised_mask.sum().item()
                obs_mask = supervised_mask & targets.ge(OBS_OFFSET) & targets.lt(ACT_OFFSET)
                if obs_mask.any():
                    obs_correct += preds.eq(targets).masked_select(obs_mask).sum().item()
                    obs_total += obs_mask.sum().item()
        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = sum(val_losses) / max(1, len(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            metrics = WorldModelMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                val_next_token_acc=next_correct / float(max(1, next_total)),
                val_obs_token_acc=obs_correct / float(max(1, obs_total)),
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    torch.save({'model': model.state_dict(), 'metrics': metrics.__dict__}, checkpoint_path)
    return model, metrics


def world_model_hidden(world_model: CausalTransformerWorldModel, history_tokens: List[int]) -> torch.Tensor:
    with torch.no_grad():
        tokens = torch.tensor([history_tokens], dtype=torch.long, device=DEVICE)
        _logits, hidden, _attn = world_model(tokens, return_attn=False)
    return hidden[0, -1].detach()


def predictor_action_scores(world_model: CausalTransformerWorldModel, predictor: FAPredictor, history_tokens: List[int]) -> Dict[str, float]:
    with torch.no_grad():
        hidden = world_model_hidden(world_model, history_tokens)
        action_embs = world_model.token_emb.weight[ACT_OFFSET:ACT_OFFSET + len(ACTIONS)]
        hidden_batch = hidden.unsqueeze(0).expand(len(ACTIONS), -1)
        scores = predictor(hidden_batch, action_embs).detach().cpu().tolist()
    return {action: score for action, score in zip(ACTIONS, scores)}


class RealAttentionFAPolicy:
    def __init__(self, world_model: CausalTransformerWorldModel, predictor: FAPredictor, lambda_fa: float, base_weight: float = DEFAULT_FA_POLICY_BASE_WEIGHT, temperature: float = 0.85, epsilon: float = 0.08) -> None:
        self.world_model = world_model
        self.predictor = predictor
        self.lambda_fa = lambda_fa
        self.base_weight = base_weight
        self.temperature = temperature
        self.epsilon = epsilon

    def score_dict(self, history_tokens: List[int], energy: int, belief: float) -> Dict[str, float]:
        base_scores = immediate_expected_utility_scores(energy, belief)
        fa_scores = predictor_action_scores(self.world_model, self.predictor, history_tokens)
        fa_z = zscore_dict(fa_scores)
        return {action: self.base_weight * base_scores[action] + self.lambda_fa * fa_z[action] for action in ACTIONS}

    def action_probs(self, history_tokens: List[int], energy: int, belief: float, explore: bool = False) -> Dict[str, float]:
        scores = self.score_dict(history_tokens, energy, belief)
        probs = softmax([scores[action] for action in ACTIONS], temperature=self.temperature)
        if explore and self.epsilon > 0.0:
            uniform = 1.0 / len(ACTIONS)
            probs = [(1.0 - self.epsilon) * prob + self.epsilon * uniform for prob in probs]
        return {action: prob for action, prob in zip(ACTIONS, probs)}

    def sample_action(self, history_tokens: List[int], energy: int, belief: float, rng: random.Random, explore: bool = False) -> str:
        prob_dict = self.action_probs(history_tokens, energy, belief, explore=explore)
        return sample_from_probs(ACTIONS, [prob_dict[action] for action in ACTIONS], rng)


def collect_contexts(policy, cfg: VisibilityConfig, episodes: int, max_contexts: int, seed: int) -> List[ContextRecord]:
    rng = random.Random(seed)
    contexts: List[ContextRecord] = []
    for _ in range(episodes):
        energy = 3 if rng.random() < 0.6 else 2
        site_rich = 1 if rng.random() < INITIAL_RICH_PROB else 0
        cue0 = sample_cue(site_rich, cfg, "wait", rng)
        belief = initial_posterior_from_cue(INITIAL_RICH_PROB, cue0, cfg)
        history_tokens = [BOS_ID, obs_token_id(energy, cue0)]
        for _step in range(EPISODE_HORIZON):
            if energy <= 0:
                break
            contexts.append(ContextRecord(history_tokens=list(history_tokens), energy=energy, belief=belief))
            action = policy.sample_action(history_tokens, energy, belief, rng, explore=True)
            history_tokens.append(action_token_id(action))
            energy, site_rich, cue, belief, _ent0, _ent1 = simulate_real_step(energy, site_rich, belief, action, cfg, rng)
            history_tokens.append(obs_token_id(energy, cue))
            if len(contexts) >= max_contexts:
                return contexts
    return contexts[:max_contexts]


def train_fa_predictor(
    world_model: CausalTransformerWorldModel,
    predictor: FAPredictor,
    contexts: Sequence[ContextRecord],
    acting_policy,
    cfg: VisibilityConfig,
    rollouts_per_action: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    info_bonus_weight: float,
    oracle_gate_mode: str,
) -> None:
    rng = random.Random(seed)
    features_h: List[torch.Tensor] = []
    features_a: List[torch.Tensor] = []
    labels: List[float] = []
    with torch.no_grad():
        action_embs = world_model.token_emb.weight[ACT_OFFSET:ACT_OFFSET + len(ACTIONS)].detach().clone()
    for ctx in contexts:
        hidden = world_model_hidden(world_model, ctx.history_tokens).cpu()
        for action_idx, action in enumerate(ACTIONS):
            label = oracle_fa_from_world_model(
                world_model,
                ctx.history_tokens,
                ctx.belief,
                action,
                acting_policy,
                cfg,
                rollouts_per_action,
                rng,
                info_bonus_weight=info_bonus_weight,
                gate_mode=oracle_gate_mode,
            )
            features_h.append(hidden)
            features_a.append(action_embs[action_idx].cpu())
            labels.append(label)

    if not labels:
        return

    h_tensor = torch.stack(features_h).to(DEVICE)
    a_tensor = torch.stack(features_a).to(DEVICE)
    y_tensor = torch.tensor(labels, dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr)
    order = list(range(len(labels)))
    rng.shuffle(order)
    for _epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = predictor(h_tensor[idx], a_tensor[idx])
            loss = F.mse_loss(pred, y_tensor[idx])
            loss.backward()
            optimizer.step()


def policy_kl(world_model: CausalTransformerWorldModel, policy_prev, policy_curr, contexts: Sequence[ContextRecord]) -> float:
    if policy_prev is None:
        return 0.0
    vals = []
    for ctx in contexts:
        prev_probs = policy_prev.action_probs(ctx.history_tokens, ctx.energy, ctx.belief, explore=False)
        curr_probs = policy_curr.action_probs(ctx.history_tokens, ctx.energy, ctx.belief, explore=False)
        kl = 0.0
        for action in ACTIONS:
            p = max(prev_probs[action], 1e-12)
            q = max(curr_probs[action], 1e-12)
            kl += p * math.log(p / q)
        vals.append(kl)
    return sum(vals) / float(max(1, len(vals)))

class MyopicPolicyWrapper:
    def sample_action(self, history_tokens: List[int], energy: int, belief: float, rng: random.Random, explore: bool = False) -> str:
        del history_tokens, rng, explore
        return argmax_action(immediate_expected_utility_scores(energy, belief))

    def action_probs(self, history_tokens: List[int], energy: int, belief: float, explore: bool = False) -> Dict[str, float]:
        del history_tokens, explore
        best = argmax_action(immediate_expected_utility_scores(energy, belief))
        return {action: (1.0 if action == best else 0.0) for action in ACTIONS}


class ExactEFEPolicyWrapper:
    def __init__(self, cfg: VisibilityConfig) -> None:
        self.cfg = cfg

    def sample_action(self, history_tokens: List[int], energy: int, belief: float, rng: random.Random, explore: bool = False) -> str:
        del history_tokens, rng, explore
        return argmin_action(exact_efe_action_scores(energy, belief, self.cfg))

    def action_probs(self, history_tokens: List[int], energy: int, belief: float, explore: bool = False) -> Dict[str, float]:
        del history_tokens, explore
        best = argmin_action(exact_efe_action_scores(energy, belief, self.cfg))
        return {action: (1.0 if action == best else 0.0) for action in ACTIONS}


class EntropyAwareRolloutPolicy:
    def __init__(self, cfg: VisibilityConfig, utility_weight: float = 1.0, info_weight: float = 2.0) -> None:
        self.cfg = cfg
        self.utility_weight = utility_weight
        self.info_weight = info_weight

    def score_dict(self, energy: int, belief: float) -> Dict[str, float]:
        utility_scores = zscore_dict(immediate_expected_utility_scores(energy, belief))
        info_scores = zscore_dict(expected_info_gain_scores(energy, belief, self.cfg))
        return {
            action: self.utility_weight * utility_scores[action] + self.info_weight * info_scores[action]
            for action in ACTIONS
        }

    def sample_action(self, history_tokens: List[int], energy: int, belief: float, rng: random.Random, explore: bool = False) -> str:
        del history_tokens, rng, explore
        return argmax_action(self.score_dict(energy, belief))

    def action_probs(self, history_tokens: List[int], energy: int, belief: float, explore: bool = False) -> Dict[str, float]:
        del history_tokens, explore
        best = argmax_action(self.score_dict(energy, belief))
        return {action: (1.0 if action == best else 0.0) for action in ACTIONS}


def oracle_fa_from_world_model(
    world_model: CausalTransformerWorldModel,
    history_tokens: List[int],
    belief: float,
    candidate_action: str,
    acting_policy,
    cfg: VisibilityConfig,
    rollouts: int,
    rng: random.Random,
    info_bonus_weight: float = DEFAULT_FA_INFO_BONUS,
    horizon: int = FA_TARGET_HORIZON,
    gate_mode: str = "full",
) -> float:
    current_energy, _current_cue = decode_obs_token(history_tokens[-1])
    total = 0.0
    for _ in range(rollouts):
        seq = list(history_tokens)
        belief_now = belief
        energy_now = current_energy
        action_now = candidate_action
        seq.append(action_token_id(action_now))
        action_pos = len(seq) - 1
        value = 0.0
        for step_idx in range(horizon):
            with torch.no_grad():
                inp = torch.tensor([seq], dtype=torch.long, device=DEVICE)
                logits, _hidden, _attn = world_model(inp, return_attn=False)
                next_logits = logits[0, -1, OBS_OFFSET:ACT_OFFSET]
                obs_probs = torch.softmax(next_logits, dim=-1)
                obs_token = torch.multinomial(obs_probs, 1).item() + OBS_OFFSET
            seq.append(obs_token)
            with torch.no_grad():
                inp2 = torch.tensor([seq], dtype=torch.long, device=DEVICE)
                _logits2, _hidden2, attn_list = world_model(inp2, return_attn=True)
                attn = attn_list[-1].mean(dim=1)[0]
                attn_to_action = attn[-1, action_pos].item()
            next_energy_obs, next_cue_obs = decode_obs_token(obs_token)
            prev_entropy = bernoulli_entropy(belief_now)
            _prior, _obs_probs, posteriors = observation_model(energy_now, belief_now, action_now, cfg)
            belief_next = posteriors.get((next_energy_obs, next_cue_obs), belief_now)
            info_gain = max(0.0, prev_entropy - bernoulli_entropy(belief_next))
            preference_gate = max(0.0, preference_utility(next_energy_obs))
            info_gate = info_bonus_weight * info_gain
            if gate_mode == "attention_only":
                gate = 1.0
            elif gate_mode == "preference_only":
                gate = preference_gate
            elif gate_mode == "info_only":
                gate = info_gate
            elif gate_mode == "full":
                gate = preference_gate + info_gate
            else:
                raise ValueError(f"Unknown oracle gate mode: {gate_mode}")
            value += (FA_TARGET_DISCOUNT ** step_idx) * gate * attn_to_action
            energy_now = next_energy_obs
            belief_now = belief_next
            if energy_now <= 0 or step_idx == horizon - 1:
                break
            action_now = acting_policy.sample_action(seq, energy_now, belief_now, rng, explore=False)
            seq.append(action_token_id(action_now))
        total += value
    return total / float(max(1, rollouts))


def evaluate_behavior(policy, cfg: VisibilityConfig, episodes: int, seed: int) -> EvalMetrics:
    rng = random.Random(seed)
    total_return = 0.0
    survived = 0
    safe_steps = 0
    total_steps = 0
    action_counter: Counter[str] = Counter()
    entropies: List[float] = []
    realized_fa_values: List[float] = []
    dummy_world_model = None

    for _ in range(episodes):
        energy = 3 if rng.random() < 0.6 else 2
        site_rich = 1 if rng.random() < INITIAL_RICH_PROB else 0
        cue0 = sample_cue(site_rich, cfg, "wait", rng)
        belief = initial_posterior_from_cue(INITIAL_RICH_PROB, cue0, cfg)
        history_tokens = [BOS_ID, obs_token_id(energy, cue0)]
        step_infos: List[Tuple[float, float, float]] = []
        for _step in range(EPISODE_HORIZON):
            if energy <= 0:
                break
            if hasattr(policy, 'action_probs'):
                probs = policy.action_probs(history_tokens, energy, belief, explore=False)
                entropies.append(-sum(prob * math.log(max(prob, 1e-12)) for prob in probs.values() if prob > 0.0))
            action = policy.sample_action(history_tokens, energy, belief, rng, explore=False)
            action_counter[action] += 1
            prev_entropy = bernoulli_entropy(belief)
            history_tokens.append(action_token_id(action))
            energy, site_rich, cue, belief, _e0, _e1 = simulate_real_step(energy, site_rich, belief, action, cfg, rng)
            info_gain = max(0.0, prev_entropy - bernoulli_entropy(belief))
            gate = max(0.0, preference_utility(energy)) + 0.75 * info_gain
            step_infos.append((gate, 1.0 if action == 'inspect' else 0.0, 1.0 if action == 'eat' else 0.0))
            history_tokens.append(obs_token_id(energy, cue))
            total_steps += 1
            total_return += preference_utility(energy)
            if energy >= 2:
                safe_steps += 1
        if energy > 0:
            survived += 1
        for idx, (gate, inspect_flag, eat_flag) in enumerate(step_infos):
            pseudo_attn = 0.55 * eat_flag + 0.35 * inspect_flag + 0.10
            realized_fa_values.append((FA_TARGET_DISCOUNT ** idx) * gate * pseudo_attn)

    return EvalMetrics(
        avg_return=total_return / float(episodes),
        survival_rate=survived / float(episodes),
        safe_step_rate=safe_steps / float(max(1, total_steps)),
        avg_realized_fa=sum(realized_fa_values) / float(len(realized_fa_values)) if realized_fa_values else 0.0,
        action_entropy=sum(entropies) / float(len(entropies)) if entropies else 0.0,
        eat_rate=action_counter['eat'] / float(max(1, total_steps)),
        inspect_rate=action_counter['inspect'] / float(max(1, total_steps)),
        move_rate=action_counter['move'] / float(max(1, total_steps)),
        wait_rate=action_counter['wait'] / float(max(1, total_steps)),
    )


def evaluate_alignment(
    world_model: CausalTransformerWorldModel,
    predictor: FAPredictor,
    acting_policy,
    cfg: VisibilityConfig,
    contexts: Sequence[ContextRecord],
    oracle_rollouts: int,
    seed: int,
    info_bonus_weight: float,
    oracle_gate_mode: str,
) -> AlignmentMetrics:
    rng = random.Random(seed)
    abs_errors: List[float] = []
    predictor_oracle = 0
    oracle_exact = 0
    predictor_exact = 0
    predictor_regrets: List[float] = []
    oracle_regrets: List[float] = []
    for ctx in contexts:
        oracle_scores = {
            action: oracle_fa_from_world_model(
                world_model,
                ctx.history_tokens,
                ctx.belief,
                action,
                acting_policy,
                cfg,
                oracle_rollouts,
                rng,
                info_bonus_weight=info_bonus_weight,
                gate_mode=oracle_gate_mode,
            )
            for action in ACTIONS
        }
        pred_scores = predictor_action_scores(world_model, predictor, ctx.history_tokens)
        exact_scores = exact_efe_action_scores(ctx.energy, ctx.belief, cfg)
        for action in ACTIONS:
            abs_errors.append(abs(pred_scores[action] - oracle_scores[action]))
        pred_action = argmax_action(pred_scores)
        oracle_action = argmax_action(oracle_scores)
        exact_action = argmin_action(exact_scores)
        predictor_oracle += int(pred_action == oracle_action)
        oracle_exact += int(oracle_action == exact_action)
        predictor_exact += int(pred_action == exact_action)
        predictor_regrets.append(exact_scores[pred_action] - exact_scores[exact_action])
        oracle_regrets.append(exact_scores[oracle_action] - exact_scores[exact_action])
    denom = float(max(1, len(contexts)))
    return AlignmentMetrics(
        fa_fit_mae=sum(abs_errors) / float(max(1, len(abs_errors))),
        predictor_vs_oracle_agreement=predictor_oracle / denom,
        oracle_vs_exact_agreement=oracle_exact / denom,
        predictor_vs_exact_agreement=predictor_exact / denom,
        predictor_exact_regret=sum(predictor_regrets) / denom,
        oracle_exact_regret=sum(oracle_regrets) / denom,
    )

def benchmark_runtime(
    world_model: CausalTransformerWorldModel,
    predictor: FAPredictor,
    cfg: VisibilityConfig,
    contexts: Sequence[ContextRecord],
    max_depth: int,
    repeats: int,
) -> List[BenchmarkRow]:
    contexts = list(contexts)[: min(40, len(contexts))]
    rows: List[BenchmarkRow] = []
    for depth in range(2, max_depth + 1):
        exact_efe_scores_cached.cache_clear()
        start = time.perf_counter()
        exact_calls = 0
        for _ in range(repeats):
            for ctx in contexts:
                exact_efe_action_scores(ctx.energy, ctx.belief, cfg, depth=depth)
                exact_calls += 1
        exact_elapsed = time.perf_counter() - start

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        fa_calls = 0
        for _ in range(repeats):
            for ctx in contexts:
                _scores = predictor_action_scores(world_model, predictor, ctx.history_tokens)
                fa_calls += 1
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        fa_elapsed = time.perf_counter() - start

        exact_ms = (exact_elapsed / float(max(1, exact_calls))) * 1000.0
        fa_ms = (fa_elapsed / float(max(1, fa_calls))) * 1000.0
        rows.append(BenchmarkRow(depth=depth, exact_ms_per_decision=exact_ms, fa_ms_per_decision=fa_ms, speedup=exact_ms / max(fa_ms, 1e-12)))
    return rows


def run_seed(
    cfg: VisibilityConfig,
    seed: int,
    world_model_ckpt: str,
    offline_episodes: int,
    world_model_epochs: int,
    world_model_batch: int,
    retrain_world_model: bool,
    world_model_d_model: int,
    world_model_nhead: int,
    world_model_layers: int,
    world_model_dim_ff: int,
    world_model_dropout: float,
    rounds: int,
    collect_episodes: int,
    max_train_contexts: int,
    max_eval_contexts: int,
    eval_episodes: int,
    target_rollouts: int,
    oracle_rollouts: int,
    lambda_fa: float,
    fa_info_bonus: float,
    oracle_rollout_policy_name: str,
    oracle_gate_mode: str,
    policy_base_weight: float,
    offline_policy_name: str,
    predictor_epochs: int,
    predictor_batch: int,
    benchmark_depth: int,
    benchmark_repeats: int,
    skip_benchmark: bool,
) -> SeedResult:
    world_model, wm_metrics = train_world_model(
        cfg=cfg,
        checkpoint_path=world_model_ckpt,
        offline_episodes=offline_episodes,
        epochs=world_model_epochs,
        batch_size=world_model_batch,
        lr=3e-4,
        seed=seed,
        retrain=retrain_world_model,
        d_model=world_model_d_model,
        nhead=world_model_nhead,
        num_layers=world_model_layers,
        dim_ff=world_model_dim_ff,
        dropout=world_model_dropout,
        offline_policy_name=offline_policy_name,
    )
    predictor = FAPredictor(world_model.d_model).to(DEVICE)
    myopic_policy = MyopicPolicyWrapper()
    exact_policy = ExactEFEPolicyWrapper(cfg)
    entropy_policy = EntropyAwareRolloutPolicy(cfg)
    baseline_myopic = evaluate_behavior(myopic_policy, cfg, episodes=eval_episodes, seed=seed + 11)
    baseline_exact = evaluate_behavior(exact_policy, cfg, episodes=eval_episodes, seed=seed + 19)

    prev_policy = None
    rounds_out: List[RoundResult] = []
    kl_contexts = collect_contexts(myopic_policy, cfg, episodes=20, max_contexts=80, seed=seed + 23)
    for round_idx in range(rounds + 1):
        current_lambda = 0.0 if round_idx == 0 else lambda_fa
        current_policy = RealAttentionFAPolicy(
            world_model,
            predictor,
            lambda_fa=current_lambda,
            base_weight=policy_base_weight,
            temperature=0.85,
            epsilon=max(0.04, 0.10 - 0.01 * round_idx),
        )
        oracle_policy = current_policy
        if oracle_rollout_policy_name == 'entropy':
            oracle_policy = entropy_policy
        elif oracle_rollout_policy_name == 'myopic':
            oracle_policy = myopic_policy
        elif oracle_rollout_policy_name == 'exact':
            oracle_policy = exact_policy
        behavior = evaluate_behavior(current_policy, cfg, episodes=eval_episodes, seed=seed + 100 * round_idx + 3)
        eval_contexts = collect_contexts(current_policy, cfg, episodes=25, max_contexts=max_eval_contexts, seed=seed + 100 * round_idx + 5)
        alignment = evaluate_alignment(
            world_model,
            predictor,
            oracle_policy,
            cfg,
            eval_contexts,
            oracle_rollouts,
            seed + 100 * round_idx + 7,
            info_bonus_weight=fa_info_bonus,
            oracle_gate_mode=oracle_gate_mode,
        )
        rounds_out.append(RoundResult(round_idx=round_idx, behavior=behavior, alignment=alignment, policy_kl=policy_kl(world_model, prev_policy, current_policy, kl_contexts)))

        if round_idx == rounds:
            prev_policy = current_policy
            break
        train_contexts = collect_contexts(current_policy, cfg, episodes=collect_episodes, max_contexts=max_train_contexts, seed=seed + 100 * round_idx + 13)
        train_fa_predictor(
            world_model,
            predictor,
            train_contexts,
            oracle_policy,
            cfg,
            target_rollouts,
            predictor_epochs,
            predictor_batch,
            1e-3,
            seed + 100 * round_idx + 17,
            info_bonus_weight=fa_info_bonus,
            oracle_gate_mode=oracle_gate_mode,
        )
        prev_policy = current_policy

    benchmark: List[BenchmarkRow] = []
    if not skip_benchmark:
        benchmark_contexts = collect_contexts(exact_policy, cfg, episodes=20, max_contexts=80, seed=seed + 900)
        benchmark = benchmark_runtime(world_model, predictor, cfg, benchmark_contexts, max_depth=benchmark_depth, repeats=benchmark_repeats)
    return SeedResult(
        visibility=cfg.name,
        seed=seed,
        world_model=wm_metrics,
        baseline_myopic=baseline_myopic,
        baseline_exact=baseline_exact,
        rounds=rounds_out,
        benchmark=benchmark,
    )


def aggregate_results(seed_results: Sequence[SeedResult]) -> Dict[str, Dict[str, object]]:
    by_visibility: Dict[str, List[SeedResult]] = defaultdict(list)
    for item in seed_results:
        by_visibility[item.visibility].append(item)
    summary: Dict[str, Dict[str, object]] = {}
    for visibility, results in by_visibility.items():
        final_rounds = [item.rounds[-1] for item in results]
        def pack(values: Sequence[float]) -> Dict[str, float]:
            mean_value, std_value = mean_std(values)
            return {'mean': mean_value, 'std': std_value}
        bench_map = defaultdict(list)
        for item in results:
            for row in item.benchmark:
                bench_map[row.depth].append(row)
        summary[visibility] = {
            'world_model': {
                'val_loss': pack([item.world_model.val_loss for item in results]),
                'val_next_token_acc': pack([item.world_model.val_next_token_acc for item in results]),
                'val_obs_token_acc': pack([item.world_model.val_obs_token_acc for item in results]),
            },
            'baseline_myopic': {
                'avg_return': pack([item.baseline_myopic.avg_return for item in results]),
                'survival_rate': pack([item.baseline_myopic.survival_rate for item in results]),
                'safe_step_rate': pack([item.baseline_myopic.safe_step_rate for item in results]),
            },
            'baseline_exact': {
                'avg_return': pack([item.baseline_exact.avg_return for item in results]),
                'survival_rate': pack([item.baseline_exact.survival_rate for item in results]),
                'safe_step_rate': pack([item.baseline_exact.safe_step_rate for item in results]),
            },
            'final_round': {
                'avg_return': pack([item.behavior.avg_return for item in final_rounds]),
                'survival_rate': pack([item.behavior.survival_rate for item in final_rounds]),
                'safe_step_rate': pack([item.behavior.safe_step_rate for item in final_rounds]),
                'fa_fit_mae': pack([item.alignment.fa_fit_mae for item in final_rounds]),
                'predictor_vs_oracle_agreement': pack([item.alignment.predictor_vs_oracle_agreement for item in final_rounds]),
                'oracle_vs_exact_agreement': pack([item.alignment.oracle_vs_exact_agreement for item in final_rounds]),
                'predictor_vs_exact_agreement': pack([item.alignment.predictor_vs_exact_agreement for item in final_rounds]),
                'predictor_exact_regret': pack([item.alignment.predictor_exact_regret for item in final_rounds]),
                'oracle_exact_regret': pack([item.alignment.oracle_exact_regret for item in final_rounds]),
                'policy_kl': pack([item.policy_kl for item in final_rounds]),
            },
            'benchmark': {
                str(depth): {
                    'exact_ms_per_decision': pack([row.exact_ms_per_decision for row in rows]),
                    'fa_ms_per_decision': pack([row.fa_ms_per_decision for row in rows]),
                    'speedup': pack([row.speedup for row in rows]),
                }
                for depth, rows in bench_map.items()
            },
        }
    return summary

def print_summary(seed_results: Sequence[SeedResult], summary: Dict[str, Dict[str, object]]) -> None:
    print('=' * 112)
    print(f'Real-attention FA vs Exact EFE with a fixed transformer world model | device={DEVICE}')
    print('Interpretation: this is a stronger empirical package because oracle FA comes from real model attention.')
    print('=' * 112)
    for visibility in (cfg.name for cfg in VISIBILITY_CONFIGS):
        if visibility not in summary:
            continue
        block = summary[visibility]
        print(f'\n[Visibility: {visibility}]')
        wm = block['world_model']
        print(f"World model obs acc: {wm['val_obs_token_acc']['mean'] * 100:.2f}±{wm['val_obs_token_acc']['std'] * 100:.2f}% | next-token acc: {wm['val_next_token_acc']['mean'] * 100:.2f}±{wm['val_next_token_acc']['std'] * 100:.2f}%")
        print(f"{'Policy':14s} | {'Return':>10s} | {'Survival':>10s} | {'SafeRate':>10s} | {'Pred/Oracle':>11s} | {'Oracle/Exact':>12s} | {'Pred/Exact':>10s} | {'Regret':>8s}")
        print('-' * 112)
        def fmt(entry: Dict[str, float], pct: bool = False) -> str:
            return f"{entry['mean'] * 100:6.2f}±{entry['std'] * 100:4.2f}%" if pct else f"{entry['mean']:6.3f}±{entry['std']:4.3f}"
        print(f"{'Myopic':14s} | {fmt(block['baseline_myopic']['avg_return']):>10s} | {fmt(block['baseline_myopic']['survival_rate'], pct=True):>10s} | {fmt(block['baseline_myopic']['safe_step_rate'], pct=True):>10s} | {'-':>11s} | {'-':>12s} | {'-':>10s} | {'-':>8s}")
        print(f"{'Exact EFE':14s} | {fmt(block['baseline_exact']['avg_return']):>10s} | {fmt(block['baseline_exact']['survival_rate'], pct=True):>10s} | {fmt(block['baseline_exact']['safe_step_rate'], pct=True):>10s} | {'-':>11s} | {'-':>12s} | {'-':>10s} | {'-':>8s}")
        print(f"{'Final FA loop':14s} | {fmt(block['final_round']['avg_return']):>10s} | {fmt(block['final_round']['survival_rate'], pct=True):>10s} | {fmt(block['final_round']['safe_step_rate'], pct=True):>10s} | {fmt(block['final_round']['predictor_vs_oracle_agreement'], pct=True):>11s} | {fmt(block['final_round']['oracle_vs_exact_agreement'], pct=True):>12s} | {fmt(block['final_round']['predictor_vs_exact_agreement'], pct=True):>10s} | {fmt(block['final_round']['predictor_exact_regret']):>8s}")
        if block['benchmark']:
            depth = max(int(item) for item in block['benchmark'].keys())
            bench = block['benchmark'][str(depth)]
            print(f"Depth {depth} benchmark: Exact {bench['exact_ms_per_decision']['mean']:.3f} ms/decision, FA {bench['fa_ms_per_decision']['mean']:.3f} ms/decision, speedup {bench['speedup']['mean']:.2f}x")


def write_markdown_summary(path: str, summary: Dict[str, Dict[str, object]]) -> None:
    lines = [
        '# Real-Attention FA Experiment Summary',
        '',
        '- Oracle FA labels come from a fixed transformer world model\'s real self-attention.',
        '- Exact EFE is an external planning baseline.',
        '- Only FA predictor / policy are updated in the closed loop; the world model stays fixed.',
        '',
    ]
    for visibility in (cfg.name for cfg in VISIBILITY_CONFIGS):
        if visibility not in summary:
            continue
        block = summary[visibility]
        lines.append(f'## {visibility}')
        lines.append('')
        lines.append(f"World model observation-token accuracy: {block['world_model']['val_obs_token_acc']['mean'] * 100:.2f}±{block['world_model']['val_obs_token_acc']['std'] * 100:.2f}%")
        lines.append('')
        lines.append('| Metric | Myopic | Exact EFE | Final FA loop |')
        lines.append('|---|---:|---:|---:|')
        def render(entry: Dict[str, float], pct: bool = False) -> str:
            return f"{entry['mean'] * 100:.2f}±{entry['std'] * 100:.2f}%" if pct else f"{entry['mean']:.3f}±{entry['std']:.3f}"
        lines.append(f"| Avg return | {render(block['baseline_myopic']['avg_return'])} | {render(block['baseline_exact']['avg_return'])} | {render(block['final_round']['avg_return'])} |")
        lines.append(f"| Survival rate | {render(block['baseline_myopic']['survival_rate'], pct=True)} | {render(block['baseline_exact']['survival_rate'], pct=True)} | {render(block['final_round']['survival_rate'], pct=True)} |")
        lines.append(f"| Safe-step rate | {render(block['baseline_myopic']['safe_step_rate'], pct=True)} | {render(block['baseline_exact']['safe_step_rate'], pct=True)} | {render(block['final_round']['safe_step_rate'], pct=True)} |")
        lines.append(f"| FA fit MAE | - | - | {render(block['final_round']['fa_fit_mae'])} |")
        lines.append(f"| Predictor vs Oracle agreement | - | - | {render(block['final_round']['predictor_vs_oracle_agreement'], pct=True)} |")
        lines.append(f"| Oracle FA vs Exact EFE agreement | - | - | {render(block['final_round']['oracle_vs_exact_agreement'], pct=True)} |")
        lines.append(f"| Predictor FA vs Exact EFE agreement | - | - | {render(block['final_round']['predictor_vs_exact_agreement'], pct=True)} |")
        lines.append(f"| Predictor exact regret | - | - | {render(block['final_round']['predictor_exact_regret'])} |")
        lines.append('')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def world_model_checkpoint_name(cfg: VisibilityConfig, d_model: int, layers: int, nhead: int, offline_policy_name: str) -> str:
    ckpt = f'real_attention_world_model_{cfg.name}_d{d_model}_l{layers}_h{nhead}.pt'
    if offline_policy_name != 'mixed':
        return ckpt[:-3] + f'_data-{offline_policy_name}.pt'
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description='Real-attention FA experiment with a fixed transformer world model.')
    parser.add_argument('--visibility', type=str, default='all', choices=['all', 'full', 'partial', 'hard_partial'])
    parser.add_argument('--train-world-model-only', action='store_true')
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--offline-episodes', type=int, default=1200)
    parser.add_argument('--world-model-epochs', type=int, default=10)
    parser.add_argument('--world-model-batch', type=int, default=64)
    parser.add_argument('--world-model-d-model', type=int, default=96)
    parser.add_argument('--world-model-nhead', type=int, default=4)
    parser.add_argument('--world-model-layers', type=int, default=3)
    parser.add_argument('--world-model-dim-ff', type=int, default=192)
    parser.add_argument('--world-model-dropout', type=float, default=0.1)
    parser.add_argument('--retrain-world-model', action='store_true')
    parser.add_argument('--collect-episodes', type=int, default=45)
    parser.add_argument('--max-train-contexts', type=int, default=140)
    parser.add_argument('--max-eval-contexts', type=int, default=80)
    parser.add_argument('--eval-episodes', type=int, default=90)
    parser.add_argument('--target-rollouts', type=int, default=6)
    parser.add_argument('--oracle-rollouts', type=int, default=10)
    parser.add_argument('--predictor-epochs', type=int, default=10)
    parser.add_argument('--predictor-batch', type=int, default=64)
    parser.add_argument('--lambda-fa', type=float, default=1.0)
    parser.add_argument('--policy-base-weight', type=float, default=DEFAULT_FA_POLICY_BASE_WEIGHT)
    parser.add_argument('--offline-policy', type=str, default='mixed', choices=['mixed', 'random', 'myopic', 'exact', 'entropy'])
    parser.add_argument('--oracle-gate-mode', type=str, default='full', choices=['full', 'attention_only', 'preference_only', 'info_only'])
    parser.add_argument('--fa-info-bonus', type=float, default=DEFAULT_FA_INFO_BONUS)
    parser.add_argument('--oracle-rollout-policy', type=str, default='current', choices=['current', 'entropy', 'myopic', 'exact'])
    parser.add_argument('--benchmark-depth', type=int, default=5)
    parser.add_argument('--benchmark-repeats', type=int, default=1)
    parser.add_argument('--skip-benchmark', action='store_true')
    parser.add_argument('--json', type=str, default='real_attention_fa_results.json')
    parser.add_argument('--md', type=str, default='real_attention_fa_summary.md')
    args = parser.parse_args()

    seed_results: List[SeedResult] = []
    selected_cfgs = VISIBILITY_CONFIGS if args.visibility == 'all' else (CFG_BY_NAME[args.visibility],)
    if args.train_world_model_only:
        trained: List[Dict[str, object]] = []
        for cfg_idx, cfg in enumerate(selected_cfgs):
            ckpt = world_model_checkpoint_name(cfg, args.world_model_d_model, args.world_model_layers, args.world_model_nhead, args.offline_policy)
            seed = 202 + cfg_idx * 100
            _world_model, wm_metrics = train_world_model(
                cfg=cfg,
                checkpoint_path=ckpt,
                offline_episodes=args.offline_episodes,
                epochs=args.world_model_epochs,
                batch_size=args.world_model_batch,
                lr=3e-4,
                seed=seed,
                retrain=True,
                d_model=args.world_model_d_model,
                nhead=args.world_model_nhead,
                num_layers=args.world_model_layers,
                dim_ff=args.world_model_dim_ff,
                dropout=args.world_model_dropout,
                offline_policy_name=args.offline_policy,
            )
            trained.append({'visibility': cfg.name, 'checkpoint': ckpt, 'world_model': wm_metrics.__dict__})
        payload = {'device': str(DEVICE), 'mode': 'train_world_model_only', 'config': vars(args), 'trained': trained}
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for cfg_idx, cfg in enumerate(selected_cfgs):
        ckpt = world_model_checkpoint_name(cfg, args.world_model_d_model, args.world_model_layers, args.world_model_nhead, args.offline_policy)
        for seed_idx in range(args.seeds):
            seed = 202 + cfg_idx * 100 + seed_idx * 17
            seed_results.append(run_seed(
                cfg=cfg,
                seed=seed,
                world_model_ckpt=ckpt,
                offline_episodes=args.offline_episodes,
                world_model_epochs=args.world_model_epochs,
                world_model_batch=args.world_model_batch,
                retrain_world_model=args.retrain_world_model or seed_idx > 0,
                world_model_d_model=args.world_model_d_model,
                world_model_nhead=args.world_model_nhead,
                world_model_layers=args.world_model_layers,
                world_model_dim_ff=args.world_model_dim_ff,
                world_model_dropout=args.world_model_dropout,
                rounds=args.rounds,
                collect_episodes=args.collect_episodes,
                max_train_contexts=args.max_train_contexts,
                max_eval_contexts=args.max_eval_contexts,
                eval_episodes=args.eval_episodes,
                target_rollouts=args.target_rollouts,
                oracle_rollouts=args.oracle_rollouts,
                lambda_fa=args.lambda_fa,
                fa_info_bonus=args.fa_info_bonus,
                oracle_rollout_policy_name=args.oracle_rollout_policy,
                oracle_gate_mode=args.oracle_gate_mode,
                policy_base_weight=args.policy_base_weight,
                offline_policy_name=args.offline_policy,
                predictor_epochs=args.predictor_epochs,
                predictor_batch=args.predictor_batch,
                benchmark_depth=args.benchmark_depth,
                benchmark_repeats=args.benchmark_repeats,
                skip_benchmark=args.skip_benchmark,
            ))

    summary = aggregate_results(seed_results)
    print_summary(seed_results, summary)
    payload = {
        'device': str(DEVICE),
        'config': vars(args),
        'summary': summary,
        'seed_results': [
            {
                'visibility': item.visibility,
                'seed': item.seed,
                'world_model': item.world_model.__dict__,
                'baseline_myopic': item.baseline_myopic.__dict__,
                'baseline_exact': item.baseline_exact.__dict__,
                'rounds': [{'round_idx': rr.round_idx, 'behavior': rr.behavior.__dict__, 'alignment': rr.alignment.__dict__, 'policy_kl': rr.policy_kl} for rr in item.rounds],
                'benchmark': [row.__dict__ for row in item.benchmark],
            }
            for item in seed_results
        ],
    }
    with open(args.json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_markdown_summary(args.md, summary)
    print(f'\nRaw results written to {args.json}')
    print(f'Markdown summary written to {args.md}')


if __name__ == '__main__':
    main()


