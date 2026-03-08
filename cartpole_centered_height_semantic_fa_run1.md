# Camera-Centered CartPole Height Softmax Semantic-FA Experiment

Camera-centered continuing CartPole with random horizontal disturbances.

- `target_height = 0.90`
- `target_band = +/-0.05`
- `episode_horizon = 150`
- `history_steps = 6`
- `fa_horizon = 4`
- `offline_noise_mode = mixed`
- `rollout_mode = repeat`
- `policy_temperature = 0.08`

## World Model

- `val_loss = 0.0090 +/- 0.0000`
- `val_height_mae = 0.0278 +/- 0.0000`

## Nominal Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 10.61+/-0.00% | 100.00+/-0.00% | 0.75+/-0.00 | 48.44+/-0.00% | 150.00+/-0.00 | 0.69+/-0.00 |
| Prediction only | 10.11+/-0.00% | 100.00+/-0.00% | 0.75+/-0.00 | 48.39+/-0.00% | 150.00+/-0.00 | 0.68+/-0.00 |
| Positive-only FA | 11.56+/-0.00% | 100.00+/-0.00% | 0.74+/-0.00 | 48.22+/-0.00% | 150.00+/-0.00 | 0.68+/-0.00 |
| Signed semantic FA | 10.56+/-0.00% | 100.00+/-0.00% | 0.73+/-0.00 | 49.67+/-0.00% | 150.00+/-0.00 | 0.68+/-0.00 |
| Planner | 43.28+/-0.00% | 100.00+/-0.00% | 0.51+/-0.00 | 100.00+/-0.00% | 150.00+/-0.00 | 0.00+/-0.00 |

## Stress Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 8.67+/-0.00% | 100.00+/-0.00% | 0.76+/-0.00 | 48.89+/-0.00% | 150.00+/-0.00 | 0.69+/-0.00 |
| Prediction only | 8.33+/-0.00% | 100.00+/-0.00% | 0.78+/-0.00 | 48.39+/-0.00% | 150.00+/-0.00 | 0.68+/-0.00 |
| Positive-only FA | 8.89+/-0.00% | 100.00+/-0.00% | 0.77+/-0.00 | 48.89+/-0.00% | 150.00+/-0.00 | 0.68+/-0.00 |
| Signed semantic FA | 10.00+/-0.00% | 100.00+/-0.00% | 0.74+/-0.00 | 49.44+/-0.00% | 150.00+/-0.00 | 0.67+/-0.00 |
| Planner | 69.94+/-0.00% | 100.00+/-0.00% | 0.23+/-0.00 | 100.00+/-0.00% | 150.00+/-0.00 | 0.00+/-0.00 |

## Representative States

| X | Xdot | Theta | ThetaDot | Height | Planner | Prediction | Positive FA | Signed semantic FA |
|---:|---:|---:|---:|---:|---|---|---|---|
| -0.20 | 0.15 | -0.60 | 0.40 | 0.82 | left | left | left | left |
| 0.00 | 0.00 | 0.45 | 0.00 | 0.90 | right | left | left | left |
| 0.35 | -0.25 | 0.57 | -0.55 | 0.84 | right | left | left | left |
| -0.55 | 0.40 | -0.27 | 0.35 | 0.96 | right | left | left | left |