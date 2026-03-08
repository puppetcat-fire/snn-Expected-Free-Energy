# CartPole Height Softmax Semantic-FA Experiment

Standard CartPole physics with a continuing target-height variant.

- `target_height = 0.90`
- `target_band = +/-0.05`
- `episode_horizon = 150`
- `history_steps = 6`
- `fa_horizon = 4`
- `offline_noise_mode = mixed`
- `rollout_mode = repeat`
- `policy_temperature = 0.08`

## World Model

- `val_loss = 0.0128 +/- 0.0000`
- `val_height_mae = 0.0375 +/- 0.0000`

## Nominal Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 11.60+/-0.00% | 75.00+/-0.00% | 0.76+/-0.00 | 49.23+/-0.00% | 145.50+/-0.00 | 0.69+/-0.00 |
| Prediction only | 12.03+/-0.00% | 37.50+/-0.00% | 0.69+/-0.00 | 53.73+/-0.00% | 134.00+/-0.00 | 0.59+/-0.00 |
| Positive-only FA | 12.81+/-0.00% | 37.50+/-0.00% | 0.68+/-0.00 | 53.82+/-0.00% | 135.62+/-0.00 | 0.59+/-0.00 |
| Signed semantic FA | 10.69+/-0.00% | 50.00+/-0.00% | 0.69+/-0.00 | 55.43+/-0.00% | 138.00+/-0.00 | 0.58+/-0.00 |
| Planner | 22.38+/-0.00% | 0.00+/-0.00% | 0.62+/-0.00 | 100.00+/-0.00% | 60.88+/-0.00 | 0.00+/-0.00 |

## Stress Policy Results

| Policy | In-band rate | Survival | Height MAE | Planner agreement | Episode length | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 10.31+/-0.00% | 62.50+/-0.00% | 0.72+/-0.00 | 49.46+/-0.00% | 138.25+/-0.00 | 0.69+/-0.00 |
| Prediction only | 11.38+/-0.00% | 50.00+/-0.00% | 0.73+/-0.00 | 49.17+/-0.00% | 127.38+/-0.00 | 0.59+/-0.00 |
| Positive-only FA | 11.12+/-0.00% | 50.00+/-0.00% | 0.74+/-0.00 | 49.06+/-0.00% | 125.88+/-0.00 | 0.59+/-0.00 |
| Signed semantic FA | 9.53+/-0.00% | 62.50+/-0.00% | 0.73+/-0.00 | 49.43+/-0.00% | 132.50+/-0.00 | 0.57+/-0.00 |
| Planner | 41.67+/-0.00% | 0.00+/-0.00% | 0.28+/-0.00 | 100.00+/-0.00% | 60.00+/-0.00 | 0.00+/-0.00 |

## Representative States

| X | Xdot | Theta | ThetaDot | Height | Planner | Prediction | Positive FA | Signed semantic FA |
|---:|---:|---:|---:|---:|---|---|---|---|
| -0.20 | 0.15 | -0.60 | 0.40 | 0.82 | left | right | right | right |
| 0.00 | 0.00 | 0.45 | 0.00 | 0.90 | right | left | left | left |
| 0.35 | -0.25 | 0.57 | -0.55 | 0.84 | right | right | right | right |
| -0.55 | 0.40 | -0.27 | 0.35 | 0.96 | right | left | left | left |