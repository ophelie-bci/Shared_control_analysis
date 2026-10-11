# Obstacle Session Metrics

Trial trajectories were summarized for obstacle tasks by AI condition and then paired at the session level.
Path efficiency was computed on successful trials as the straight-line distance from movement start to the edge of the 2.3 m target window divided by actual trajectory path length; the inverse path-length ratio is retained as an audit field.
Target-window entry used the configured 2.3 m target-window radius around the cued target.
Time to first target entry is movement onset to first entry into the target window.
Total acquisition time is movement onset to successful completion, using `targetReached` when present and otherwise `trial.stop`; it is reported only for successful trials.
Primary movement speed is mean executed speed on successful trials only. Mean speed over all included trials is retained as `mean_executed_speed_mps` for the optional overall task-level speed analysis.
Primary trajectory smoothness is mean squared jerk on successful trials only. All-trial MSJ is retained as `smoothness_mean_squared_jerk` for the optional overall task-level smoothness analysis.
`inferred_time_to_dwell_start_s` is also retained as an audit field based on 10 consecutive target-window samples, but it is not the primary acquisition-time metric.
Obstacle clearance is the minimum avatar-obstacle center distance minus the avatar and obstacle radii.
Autonomous override was quantified from explicit aiControlOn/aiControlOff intervals as override samples divided by velocity-log samples.
The reviewer-facing override event count treats one 150-ms safety override as three 50-ms velocity-log bins, i.e. `n_override_samples / 3`; explicit `aiControlOn` episode counts are retained separately as a raw-log audit field.

The trial-level table contains 2,053 trials across 22 sessions.

## Override Summary

- Total 150-ms override-event equivalents: 3,783.3.
- Total explicit `aiControlOn` episodes retained as raw-log audit field: 775.
- Shared-control trials with any explicit override: 72.4%.
- Median 150-ms override-event equivalents per shared-control trial: 3.7 (IQR 0.0-5.7).
- Among shared-control trials with any explicit override, the median was 4.7 150-ms event equivalents (IQR 3.3-6.3).
- Mean 150-ms override-event equivalents per shared-control trial: 3.54.
- Median override time per shared-control trial: 0.530 s.
- Mean override time per shared-control trial: 0.526 s.
- Fraction of total shared-control time in explicit override: 9.98%.

## Session-Level Comparisons

- Appearing Obstacle, successful_target_window_path_ratio: mean AI=1.42, No AI=1.47, delta=-0.05007, n=10, Wilcoxon p=0.084.
- Appearing Obstacle, successful_target_window_path_efficiency: mean AI=0.7072, No AI=0.6893, delta=0.01787, n=10, Wilcoxon p=0.232.
- Appearing Obstacle, time_to_first_target_entry_s: mean AI=3.796, No AI=3.372, delta=0.4234, n=10, Wilcoxon p=0.00586.
- Appearing Obstacle, time_to_successful_dwell_s: mean AI=4.976, No AI=4.424, delta=0.5512, n=10, Wilcoxon p=0.00391.
- Appearing Obstacle, successful_mean_executed_speed_mps: mean AI=1.876, No AI=2.219, delta=-0.3425, n=10, Wilcoxon p=0.00195.
- Appearing Obstacle, mean_executed_speed_mps: mean AI=1.92, No AI=1.868, delta=0.05225, n=10, Wilcoxon p=0.432.
- Appearing Obstacle, obstacle_clearance_m: mean AI=0.7576, No AI=0.2621, delta=0.4956, n=10, Wilcoxon p=0.00195.
- Appearing Obstacle, successful_smoothness_mean_squared_jerk: mean AI=3.747e+05, No AI=5.203e+05, delta=-1.457e+05, n=10, Wilcoxon p=0.0195.
- Appearing Obstacle, smoothness_mean_squared_jerk: mean AI=3.407e+05, No AI=4.097e+05, delta=-6.9e+04, n=10, Wilcoxon p=0.084.
- Appearing Obstacle, n_150ms_override_events: mean AI=5.289, No AI=0, delta=5.289, n=10, Wilcoxon p=0.00195.
- Appearing Obstacle, override_fraction: mean AI=0.1829, No AI=0, delta=0.1829, n=10, Wilcoxon p=0.00195.
- Fixed Obstacle, successful_target_window_path_ratio: mean AI=1.393, No AI=1.43, delta=-0.03712, n=12, Wilcoxon p=0.0122.
- Fixed Obstacle, successful_target_window_path_efficiency: mean AI=0.7215, No AI=0.7091, delta=0.01244, n=12, Wilcoxon p=0.0771.
- Fixed Obstacle, time_to_first_target_entry_s: mean AI=3.349, No AI=2.907, delta=0.4412, n=12, Wilcoxon p=0.000488.
- Fixed Obstacle, time_to_successful_dwell_s: mean AI=3.97, No AI=3.434, delta=0.5355, n=12, Wilcoxon p=0.000488.
- Fixed Obstacle, successful_mean_executed_speed_mps: mean AI=2.021, No AI=2.323, delta=-0.3025, n=12, Wilcoxon p=0.000488.
- Fixed Obstacle, mean_executed_speed_mps: mean AI=1.973, No AI=2.116, delta=-0.1434, n=12, Wilcoxon p=0.0161.
- Fixed Obstacle, obstacle_clearance_m: mean AI=1.045, No AI=0.5594, delta=0.4852, n=12, Wilcoxon p=0.000488.
- Fixed Obstacle, successful_smoothness_mean_squared_jerk: mean AI=3.926e+05, No AI=5.242e+05, delta=-1.316e+05, n=12, Wilcoxon p=0.0161.
- Fixed Obstacle, smoothness_mean_squared_jerk: mean AI=3.808e+05, No AI=4.521e+05, delta=-7.134e+04, n=12, Wilcoxon p=0.0161.
- Fixed Obstacle, n_150ms_override_events: mean AI=2.407, No AI=0, delta=2.407, n=12, Wilcoxon p=0.000488.
- Fixed Obstacle, override_fraction: mean AI=0.08741, No AI=0, delta=0.08741, n=12, Wilcoxon p=0.000488.
