# Stick Height Softmax Semantic-FA Experiment

Counterfactual action branches are scored separately, then converted into a softmax policy.

- `target_height = 0.72`
- `target_band = ±0.08`
- `episode_horizon = 60`
- `history_steps = 6`
- `fa_horizon = 4`
- `offline_noise_mode = mixed`
- `rollout_mode = repeat`
- `policy_temperature = 0.05`

## World Model

- `val_loss = 0.0140 ± 0.0000`
- `val_height_mae = 0.0738 ± 0.0000`

## Nominal Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 16.99±0.00% | 50.00±0.00% | 0.21±0.00 | 52.61±0.00% | 38.25±0.00 | 0.69±0.00 |
| Prediction only | 26.02±0.00% | 37.50±0.00% | 0.19±0.00 | 63.57±0.00% | 33.62±0.00 | 0.42±0.00 |
| Positive-only FA | 29.08±0.00% | 50.00±0.00% | 0.17±0.00 | 63.86±0.00% | 46.00±0.00 | 0.38±0.00 |
| Signed semantic FA | 37.36±0.00% | 50.00±0.00% | 0.14±0.00 | 70.88±0.00% | 45.50±0.00 | 0.37±0.00 |
| Planner | 82.71±0.00% | 100.00±0.00% | 0.05±0.00 | 100.00±0.00% | 60.00±0.00 | 0.00±0.00 |

## Stress Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 12.46±0.00% | 37.50±0.00% | 0.25±0.00 | 50.17±0.00% | 36.12±0.00 | 0.69±0.00 |
| Prediction only | 21.40±0.00% | 0.00±0.00% | 0.22±0.00 | 52.56±0.00% | 26.88±0.00 | 0.45±0.00 |
| Positive-only FA | 26.70±0.00% | 25.00±0.00% | 0.19±0.00 | 62.62±0.00% | 25.75±0.00 | 0.37±0.00 |
| Signed semantic FA | 27.50±0.00% | 12.50±0.00% | 0.20±0.00 | 61.50±0.00% | 25.00±0.00 | 0.39±0.00 |
| Planner | 78.96±0.00% | 100.00±0.00% | 0.06±0.00 | 100.00±0.00% | 60.00±0.00 | 0.00±0.00 |

## Representative States

| Theta | Omega | Drive | Height | Planner | Prediction | Positive FA | Signed semantic FA |
|---:|---:|---:|---:|---|---|---|---|
| -0.95 | 0.10 | -0.30 | 0.58 | left | left | right | right |
| -0.65 | 0.60 | 0.45 | 0.80 | right | left | left | left |
| 0.30 | 0.00 | 0.10 | 0.96 | right | right | right | right |
| 0.95 | -0.10 | 0.30 | 0.58 | right | right | right | right |