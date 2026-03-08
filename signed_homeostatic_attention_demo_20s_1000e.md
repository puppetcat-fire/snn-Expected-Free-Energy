# Signed Homeostatic Attention Demo

- `rise_only_attention`: reward only future glucose increases.
- `signed_homeostatic_attention`: reward future movement toward the safe range, penalize movement away from it.
- `exact_homeostatic_planner`: expected-value dynamic-programming baseline.

| Policy | Avg return | Survival rate | Safe-step rate | Agreement with exact |
|---|---:|---:|---:|---:|
| Rise-only attention | -11.92±0.03 | 0.00±0.00% | 17.85±0.42% | 33.33±0.00% |
| Signed homeostatic attention | 24.51±0.03 | 100.00±0.00% | 86.19±0.15% | 100.00±0.00% |
| Exact homeostatic planner | 24.51±0.03 | 100.00±0.00% | 86.19±0.15% | 100.00±0.00% |

## Representative States

| Glucose | Carry | Rise-only action | Signed action | Exact action |
|---:|---:|---|---|---|
| 3 | 0 | eat | eat | eat |
| 5 | 0 | eat | wait | wait |
| 7 | 0 | eat | insulin | insulin |