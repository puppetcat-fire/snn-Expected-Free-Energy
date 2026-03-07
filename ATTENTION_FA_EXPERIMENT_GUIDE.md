# Attention-defined FA Experiment Guide

## What this package tests

This package is designed to support a stronger version of the FA hypothesis:

- FA is defined from future self-attention labels, not from EFE supervision.
- Exact EFE is used only as an external planning baseline.
- A closed loop is included: intervention changes the data distribution, which changes future attention targets, which updates the FA predictor, which changes the policy again.

## Main script

- `attention_fa_closed_loop_experiment.py`

## Default run

```bash
python attention_fa_closed_loop_experiment.py
```

Outputs:

- `attention_fa_closed_loop_results.json`
- `attention_fa_closed_loop_summary.md`

## Recommended stronger run

```bash
python attention_fa_closed_loop_experiment.py --seeds 6 --rounds 3 --collect-episodes 120 --eval-episodes 180 --target-rollouts 16 --oracle-rollouts 32
```

## Metrics to inspect

- `fa_fit_mae`: whether the predictor actually learns future-attention labels.
- `predictor_vs_oracle_agreement`: whether learned FA matches oracle FA.
- `oracle_vs_exact_agreement`: whether attention-defined FA itself aligns with Exact EFE.
- `predictor_vs_exact_agreement`: whether the learned FA policy approximates Exact EFE action ranking.
- `predictor_exact_regret`: exact-planning regret of the learned FA policy.
- `avg_return`, `survival_rate`, `safe_step_rate`: whether the closed loop forms useful behavior.
- runtime benchmark: whether FA keeps an engineering advantage over Exact EFE planning.

## Interpretation boundary

This experiment package can provide strong empirical support. It does not by itself prove that FA is identical to EFE.
