# Real-Attention FA Experiment Summary

- Oracle FA labels come from a fixed transformer world model's real self-attention.
- Exact EFE is an external planning baseline.
- Only FA predictor / policy are updated in the closed loop; the world model stays fixed.

## partial

World model observation-token accuracy: 46.15±0.13%

| Metric | Myopic | Exact EFE | Final FA loop |
|---|---:|---:|---:|
| Avg return | 0.134±0.336 | 6.131±1.289 | 0.633±0.761 |
| Survival rate | 7.92±0.42% | 42.92±7.08% | 12.08±5.42% |
| Safe-step rate | 64.55±1.29% | 81.89±2.91% | 67.08±2.64% |
| FA fit MAE | - | - | 0.031±0.001 |
| Predictor vs Oracle agreement | - | - | 78.65±1.56% |
| Oracle FA vs Exact EFE agreement | - | - | 55.73±7.81% |
| Predictor FA vs Exact EFE agreement | - | - | 57.29±8.33% |
| Predictor exact regret | - | - | 0.275±0.066 |
