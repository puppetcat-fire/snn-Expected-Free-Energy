# Camera-Centered CartPole Height Softmax Semantic-FA Experiment

Camera-centered continuing CartPole with random horizontal disturbances.

- `target_height = 0.90`
- `target_band = +/-0.05`
- `episode_horizon = 150`
- `history_steps = 6`
- `fa_horizon = 12`
- `offline_noise_mode = mixed`
- `disturbance_mode = impulse`
- `rollout_mode = repeat`
- `policy_temperature = 0.08`

## World Model

- `val_loss = 0.0135 +/- 0.0000`
- `val_height_mae = 0.0461 +/- 0.0000`

## Nominal Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 10.92+/-0.00% | 100.00+/-0.00% | 0.70+/-0.00 | 48.33+/-0.00% | 150.00+/-0.00 | 0.69+/-0.00 |
| Prediction only | 9.50+/-0.00% | 100.00+/-0.00% | 0.69+/-0.00 | 50.67+/-0.00% | 150.00+/-0.00 | 0.56+/-0.00 |
| Positive-only FA | 10.50+/-0.00% | 100.00+/-0.00% | 0.69+/-0.00 | 49.25+/-0.00% | 150.00+/-0.00 | 0.56+/-0.00 |
| Signed semantic FA | 9.00+/-0.00% | 100.00+/-0.00% | 0.70+/-0.00 | 47.00+/-0.00% | 150.00+/-0.00 | 0.56+/-0.00 |
| Planner | 49.08+/-0.00% | 100.00+/-0.00% | 0.47+/-0.00 | 100.00+/-0.00% | 150.00+/-0.00 | 0.00+/-0.00 |

## Stress Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 11.17+/-0.00% | 100.00+/-0.00% | 0.76+/-0.00 | 51.50+/-0.00% | 150.00+/-0.00 | 0.69+/-0.00 |
| Prediction only | 12.58+/-0.00% | 100.00+/-0.00% | 0.71+/-0.00 | 48.67+/-0.00% | 150.00+/-0.00 | 0.58+/-0.00 |
| Positive-only FA | 13.25+/-0.00% | 100.00+/-0.00% | 0.72+/-0.00 | 49.00+/-0.00% | 150.00+/-0.00 | 0.58+/-0.00 |
| Signed semantic FA | 12.67+/-0.00% | 100.00+/-0.00% | 0.73+/-0.00 | 49.58+/-0.00% | 150.00+/-0.00 | 0.58+/-0.00 |
| Planner | 70.33+/-0.00% | 100.00+/-0.00% | 0.25+/-0.00 | 100.00+/-0.00% | 150.00+/-0.00 | 0.00+/-0.00 |

## Representative States

| X | Xdot | Theta | ThetaDot | Height | Planner | Prediction | Positive FA | Signed semantic FA |
|---:|---:|---:|---:|---:|---|---|---|---|
| -0.20 | 0.15 | -0.60 | 0.40 | 0.82 | left | left | left | left |
| 0.00 | 0.00 | 0.45 | 0.00 | 0.90 | right | left | left | left |
| 0.35 | -0.25 | 0.57 | -0.55 | 0.84 | right | left | left | left |
| -0.55 | 0.40 | -0.27 | 0.35 | 0.96 | right | right | right | right |