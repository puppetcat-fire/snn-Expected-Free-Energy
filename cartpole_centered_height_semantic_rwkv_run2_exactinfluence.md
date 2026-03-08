# Camera-Centered CartPole Height Softmax Semantic-RWKV Experiment

Camera-centered continuing CartPole with random horizontal disturbances.

- `target_height = 0.90`
- `target_band = +/-0.05`
- `episode_horizon = 60`
- `history_steps = 6`
- `fa_horizon = 4`
- `offline_noise_mode = mixed`
- `disturbance_mode = impulse`
- `rollout_mode = repeat`
- `policy_temperature = 0.08`

## World Model

- `val_loss = 0.0531 +/- 0.0000`
- `val_height_mae = 0.1167 +/- 0.0000`

## Nominal Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 6.67+/-0.00% | 100.00+/-0.00% | 0.78+/-0.00 | 52.08+/-0.00% | 60.00+/-0.00 | 0.69+/-0.00 |
| Prediction only | 5.00+/-0.00% | 100.00+/-0.00% | 0.82+/-0.00 | 54.17+/-0.00% | 60.00+/-0.00 | 0.57+/-0.00 |
| Positive-only FA | 5.00+/-0.00% | 100.00+/-0.00% | 0.82+/-0.00 | 54.17+/-0.00% | 60.00+/-0.00 | 0.57+/-0.00 |
| Signed semantic FA | 5.00+/-0.00% | 100.00+/-0.00% | 0.82+/-0.00 | 54.17+/-0.00% | 60.00+/-0.00 | 0.57+/-0.00 |
| Planner | 17.50+/-0.00% | 100.00+/-0.00% | 0.75+/-0.00 | 100.00+/-0.00% | 60.00+/-0.00 | 0.00+/-0.00 |

## Stress Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 12.08+/-0.00% | 100.00+/-0.00% | 0.73+/-0.00 | 52.92+/-0.00% | 60.00+/-0.00 | 0.69+/-0.00 |
| Prediction only | 12.08+/-0.00% | 100.00+/-0.00% | 0.72+/-0.00 | 54.17+/-0.00% | 60.00+/-0.00 | 0.52+/-0.00 |
| Positive-only FA | 12.08+/-0.00% | 100.00+/-0.00% | 0.72+/-0.00 | 54.17+/-0.00% | 60.00+/-0.00 | 0.52+/-0.00 |
| Signed semantic FA | 12.08+/-0.00% | 100.00+/-0.00% | 0.72+/-0.00 | 54.17+/-0.00% | 60.00+/-0.00 | 0.52+/-0.00 |
| Planner | 46.67+/-0.00% | 100.00+/-0.00% | 0.52+/-0.00 | 100.00+/-0.00% | 60.00+/-0.00 | 0.00+/-0.00 |

## Representative States

| X | Xdot | Theta | ThetaDot | Height | Planner | Prediction | Positive FA | Signed semantic FA |
|---:|---:|---:|---:|---:|---|---|---|---|
| -0.20 | 0.15 | -0.60 | 0.40 | 0.82 | left | right | right | right |
| 0.00 | 0.00 | 0.45 | 0.00 | 0.90 | right | right | right | right |
| 0.35 | -0.25 | 0.57 | -0.55 | 0.84 | right | right | right | right |
| -0.55 | 0.40 | -0.27 | 0.35 | 0.96 | right | right | right | right |