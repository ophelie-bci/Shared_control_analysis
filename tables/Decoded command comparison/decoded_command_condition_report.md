# Decoded-command comparison between conditions

For both BCI-only and shared-control trials, metrics were computed from `modelVelocity`, the raw decoded BCI command before RT-V2 modifies the executed velocity.

## Session-level paired comparisons

- Appearing Obstacle, decoded_speed_mean: BCI-only 2.8, shared-control 2.86, shared-minus-BCI 0.0591; Wilcoxon p=0.03607.
- Appearing Obstacle follow-up, decoded_speed_mean: BCI-only 3, shared-control 3.01, shared-minus-BCI 0.0125; Wilcoxon p=0.5703.
- Fixed Obstacle, decoded_speed_mean: BCI-only 2.45, shared-control 2.93, shared-minus-BCI 0.484; Wilcoxon p=0.0008495.
- Appearing Obstacle, angular_error_to_target_deg_mean: BCI-only 35.6, shared-control 35.1, shared-minus-BCI -0.538; Wilcoxon p=0.6226.
- Appearing Obstacle follow-up, angular_error_to_target_deg_mean: BCI-only 35.3, shared-control 32, shared-minus-BCI -3.26; Wilcoxon p=0.03906.
- Fixed Obstacle, angular_error_to_target_deg_mean: BCI-only 46.7, shared-control 34.2, shared-minus-BCI -12.4; Wilcoxon p=0.0005517.
- Appearing Obstacle, lateral_command_abs_mean: BCI-only 1.24, shared-control 1.32, shared-minus-BCI 0.081; Wilcoxon p=0.02299.
- Appearing Obstacle follow-up, lateral_command_abs_mean: BCI-only 1.27, shared-control 1.25, shared-minus-BCI -0.0218; Wilcoxon p=0.5703.
- Fixed Obstacle, lateral_command_abs_mean: BCI-only 1.17, shared-control 1.32, shared-minus-BCI 0.148; Wilcoxon p=4.935e-05.
- Appearing Obstacle, lateral_command_rms: BCI-only 1.45, shared-control 1.53, shared-minus-BCI 0.0833; Wilcoxon p=0.03234.
- Appearing Obstacle follow-up, lateral_command_rms: BCI-only 1.47, shared-control 1.44, shared-minus-BCI -0.0266; Wilcoxon p=0.4961.
- Fixed Obstacle, lateral_command_rms: BCI-only 1.36, shared-control 1.52, shared-minus-BCI 0.162; Wilcoxon p=0.0001278.
- Appearing Obstacle, command_heading_variability_deg: BCI-only 22.5, shared-control 21.2, shared-minus-BCI -1.3; Wilcoxon p=0.04456.
- Appearing Obstacle follow-up, command_heading_variability_deg: BCI-only 21.3, shared-control 17.8, shared-minus-BCI -3.54; Wilcoxon p=0.003906.
- Fixed Obstacle, command_heading_variability_deg: BCI-only 26.9, shared-control 20.5, shared-minus-BCI -6.4; Wilcoxon p=2.384e-06.
- Appearing Obstacle, command_speed_sd: BCI-only 0.83, shared-control 0.794, shared-minus-BCI -0.0359; Wilcoxon p=0.001694.
- Appearing Obstacle follow-up, command_speed_sd: BCI-only 0.824, shared-control 0.784, shared-minus-BCI -0.0399; Wilcoxon p=0.01172.
- Fixed Obstacle, command_speed_sd: BCI-only 0.745, shared-control 0.815, shared-minus-BCI 0.0703; Wilcoxon p=0.02767.
- Appearing Obstacle, command_acceleration_mean: BCI-only 5.14, shared-control 5.11, shared-minus-BCI -0.027; Wilcoxon p=0.6226.
- Appearing Obstacle follow-up, command_acceleration_mean: BCI-only 5.37, shared-control 5.34, shared-minus-BCI -0.0295; Wilcoxon p=0.3594.
- Fixed Obstacle, command_acceleration_mean: BCI-only 5.18, shared-control 5.32, shared-minus-BCI 0.143; Wilcoxon p=0.03267.

## Suggested reviewer wording

We directly compared the raw decoded BCI commands before RT-V2 modification across interleaved BCI-only and shared-control trials. This diagnostic did not support a simple claim that decoded commands were identical across conditions: Fixed Obstacle trials showed larger condition-related command differences, while Appearing Obstacle trials showed more modest differences, with angular error largely similar in the original appearing-obstacle sessions and speed/lateral-command differences much smaller than the behavioral success gain. We therefore treat this as a diagnostic control rather than the primary rebuttal; the stronger evidence comes from analyses showing that RT-V2 continued to use instantaneous BCI input outside safety overrides and that follow-up reactive obstacle metrics improved when no hidden-obstacle prior was available.
