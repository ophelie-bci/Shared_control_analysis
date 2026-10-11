# Figure 3C Original Session Audit

The original `plot_summary_scatter()` allows only `AI Obstacle`, `AI Appearing Obstacle`, and `AI Appearing Obstacle 2`; `AI Respawn` is not plotted in Figure 3C.
The original Figure 3C path creates one plotted point per monkey x experiment x target, not one point per target x session.

## Default run_success_rate paths found under base_dir

| audit                          | monkey   | source_monkey_folder   | experiment            | source_experiment_folder   |   n_plotted_target_points |   n_available_session_files |   min_sessions_with_both_conditions |   median_sessions_with_both_conditions |   max_sessions_with_both_conditions |
|:-------------------------------|:---------|:-----------------------|:----------------------|:---------------------------|--------------------------:|----------------------------:|------------------------------------:|---------------------------------------:|------------------------------------:|
| default_run_success_rate_paths | Monkey 1 | Monkey 1               | AI Appearing Obstacle | AI Appearing Obstacle      |                         5 |                           9 |                                   9 |                                      9 |                                   9 |
| default_run_success_rate_paths | Monkey 1 | Monkey 1               | AI Obstacle           | AI Obstacle                |                         5 |                          11 |                                  11 |                                     11 |                                  11 |

## Mapped manuscript-source paths found under base_dir

| audit                          | monkey   | source_monkey_folder   | experiment              | source_experiment_folder   |   n_plotted_target_points |   n_available_session_files |   min_sessions_with_both_conditions |   median_sessions_with_both_conditions |   max_sessions_with_both_conditions |
|:-------------------------------|:---------|:-----------------------|:------------------------|:---------------------------|--------------------------:|----------------------------:|------------------------------------:|---------------------------------------:|------------------------------------:|
| mapped_manuscript_source_paths | Monkey 1 | Monkey 3               | AI Appearing Obstacle   | AI Appearing Obstacle      |                         5 |                           9 |                                   9 |                                      9 |                                   9 |
| mapped_manuscript_source_paths | Monkey 1 | Monkey 3               | AI Appearing Obstacle 2 | AI Appearing Obstacle 2    |                         5 |                           7 |                                   7 |                                      7 |                                   7 |
| mapped_manuscript_source_paths | Monkey 1 | Monkey 3               | AI Obstacle             | AI Obstacle                |                         5 |                          12 |                                  12 |                                     12 |                                  12 |
| mapped_manuscript_source_paths | Monkey 2 | Monkey 1               | AI Appearing Obstacle   | AI Appearing Obstacle      |                         5 |                           9 |                                   9 |                                      9 |                                   9 |
| mapped_manuscript_source_paths | Monkey 2 | Monkey 1               | AI Obstacle             | AI Obstacle                |                         5 |                          11 |                                  11 |                                     11 |                                  11 |

## Important Interpretation

A plotted target point is based on pooled No-AI target success for the x-axis and the mean of session-level target gains for the y-axis.
The audit reports `n_sessions_with_both_conditions` because a target-session can only provide an empirical session gain when both AI and No-AI trials are present for that target.
