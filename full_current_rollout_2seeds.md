# Real-Attention FA Experiment Summary

- Oracle FA labels come from a fixed transformer world model's real self-attention.
- Exact EFE is an external planning baseline.
- Only FA predictor / policy are updated in the closed loop; the world model stays fixed.

## full

World model observation-token accuracy: 66.98±0.58%

| Metric | Myopic | Exact EFE | Final FA loop |
|---|---:|---:|---:|
| Avg return | 1.346±0.099 | 13.242±0.662 | 6.044±0.999 |
| Survival rate | 15.00±0.00% | 78.75±2.08% | 42.92±7.08% |
| Safe-step rate | 69.07±0.47% | 92.28±1.17% | 80.71±1.46% |
| FA fit MAE | - | - | 0.041±0.005 |
| Predictor vs Oracle agreement | - | - | 64.06±6.77% |
| Oracle FA vs Exact EFE agreement | - | - | 70.31±8.85% |
| Predictor FA vs Exact EFE agreement | - | - | 59.90±5.73% |
| Predictor exact regret | - | - | 0.361±0.018 |
