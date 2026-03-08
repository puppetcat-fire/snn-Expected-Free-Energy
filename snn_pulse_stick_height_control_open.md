# Pulse-Node Sparse Cyclic SNN Stick-Height Control

## World Model
- val_loss: 0.074617
- val_in_band_bce: 0.254433
- active_nodes: 44
- births: 8
- deaths: 4

## Evaluation
### nominal_prediction_only
- in_band_rate: 0.171875
- mean_abs_height_error: 0.781615
- planner_agreement: 0.481771
- mean_beta: 2.185637
- mean_action_entropy: 0.691750

### nominal_signed_pulse
- in_band_rate: 0.238715
- mean_abs_height_error: 0.549084
- planner_agreement: 0.578993
- mean_beta: 1.905473
- mean_action_entropy: 0.431842

### nominal_signed_twohop_pulse
- in_band_rate: 0.263021
- mean_abs_height_error: 0.494998
- planner_agreement: 0.586806
- mean_beta: 1.858127
- mean_action_entropy: 0.420045

### stress_prediction_only
- in_band_rate: 0.195312
- mean_abs_height_error: 0.720462
- planner_agreement: 0.525174
- mean_beta: 2.304090
- mean_action_entropy: 0.691899

### stress_signed_pulse
- in_band_rate: 0.278646
- mean_abs_height_error: 0.447305
- planner_agreement: 0.602431
- mean_beta: 1.879131
- mean_action_entropy: 0.403137

### stress_signed_twohop_pulse
- in_band_rate: 0.245660
- mean_abs_height_error: 0.520655
- planner_agreement: 0.564236
- mean_beta: 1.887389
- mean_action_entropy: 0.420498
