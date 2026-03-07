# Real-Attention FA Experiment Summary

- Oracle FA labels come from a fixed transformer world model's real self-attention.
- Exact EFE is an external planning baseline.
- Only FA predictor / policy are updated in the closed loop; the world model stays fixed.

## partial

World model observation-token accuracy: 46.15±0.13%

| Metric | Myopic | Exact EFE | Final FA loop |
|---|---:|---:|---:|
| Avg return | 0.134±0.336 | 6.131±1.289 | 2.450±0.987 |
| Survival rate | 7.92±0.42% | 42.92±7.08% | 25.83±4.17% |
| Safe-step rate | 64.55±1.29% | 81.89±2.91% | 72.22±2.75% |
| FA fit MAE | - | - | 0.042±0.005 |
| Predictor vs Oracle agreement | - | - | 66.67±6.25% |
| Oracle FA vs Exact EFE agreement | - | - | 54.17±7.29% |
| Predictor FA vs Exact EFE agreement | - | - | 53.12±11.46% |
| Predictor exact regret | - | - | 0.339±0.082 |
