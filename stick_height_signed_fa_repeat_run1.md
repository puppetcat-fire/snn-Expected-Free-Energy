# Stick Height Signed-FA Experiment

Target-band control using a transformer world model and real self-attention.

- `target_height = 0.72`
- `target_band = ±0.08`
- `episode_horizon = 60`
- `history_steps = 6`
- `fa_horizon = 4`
- `offline_noise_mode = mixed`
- `rollout_mode = repeat`

## World Model

- `val_loss = 0.0107 ± 0.0000`
- `val_height_mae = 0.0815 ± 0.0000`

## Nominal Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length |
|---|---:|---:|---:|---:|---:|
| Random | 13.62±0.00% | 37.50±0.00% | 0.24±0.00 | 53.05±0.00% | 34.88±0.00 |
| Prediction only | 55.32±0.00% | 87.50±0.00% | 0.10±0.00 | 75.18±0.00% | 52.88±0.00 |
| Positive-only FA | 55.32±0.00% | 87.50±0.00% | 0.10±0.00 | 75.89±0.00% | 52.88±0.00 |
| Signed FA | 78.48±0.00% | 62.50±0.00% | 0.07±0.00 | 82.73±0.00% | 41.25±0.00 |
| Planner | 76.25±0.00% | 100.00±0.00% | 0.09±0.00 | 100.00±0.00% | 60.00±0.00 |

## Stress Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length |
|---|---:|---:|---:|---:|---:|
| Random | 20.44±0.00% | 25.00±0.00% | 0.21±0.00 | 50.31±0.00% | 39.75±0.00 |
| Prediction only | 39.58±0.00% | 100.00±0.00% | 0.13±0.00 | 74.58±0.00% | 60.00±0.00 |
| Positive-only FA | 38.96±0.00% | 100.00±0.00% | 0.12±0.00 | 71.25±0.00% | 60.00±0.00 |
| Signed FA | 67.45±0.00% | 75.00±0.00% | 0.08±0.00 | 81.36±0.00% | 47.62±0.00 |
| Planner | 76.67±0.00% | 100.00±0.00% | 0.07±0.00 | 100.00±0.00% | 60.00±0.00 |

## Representative States

| Theta | Omega | Drive | Height | Planner | Prediction | Positive FA | Signed FA |
|---:|---:|---:|---:|---|---|---|---|
| -0.95 | 0.10 | -0.30 | 0.58 | left | left | left | left |
| -0.65 | 0.60 | 0.45 | 0.80 | right | right | right | left |
| 0.30 | 0.00 | 0.10 | 0.96 | right | right | right | left |
| 0.95 | -0.10 | 0.30 | 0.58 | right | right | right | right |