# Follow-up Reactive Obstacle Analysis

Dataset: Monkey 1 Appearing Obstacle six-month follow-up, hidden-obstacle prior absent.

Geometry-based trajectory metrics were computed for the 7 follow-up sessions with available `_trials.pkl` trajectory files.

Trials were aligned to obstacle appearance at `t=0`. The primary reactive outcomes were post-appearance collision probability and minimum post-appearance clearance. As a supporting mechanistic outcome, heading/velocity reorientation was quantified as the change in radial velocity away from the obstacle from the 1 s pre-appearance window to the 1 s post-appearance window.

## Session-Level Paired Comparisons

- Post-appearance collision probability: BCI-only 18.7%, shared-control 0.0%, difference -18.7 pp, Wilcoxon p=0.01562; collision reduction in 7/7 sessions.
- Minimum post-appearance clearance: BCI-only 0.25 m, shared-control 0.78 m, difference 0.53 m, Wilcoxon p=0.01562; clearance increase in 7/7 sessions.
- Supporting mechanistic outcome, change in radial velocity away from obstacle: BCI-only -0.13 m/s, shared-control 0.31 m/s, difference 0.45 m/s, Wilcoxon p=0.01562; increase in 7/7 sessions.

## Target-Adjusted Trial-Level Models

| status   | outcome                        | model_type   | term           |      coef |   se_cluster_session |           p |    ci_low |   ci_high |   n_trials |
|:---------|:-------------------------------|:-------------|:---------------|----------:|---------------------:|------------:|----------:|----------:|-----------:|
| ok       | post_collision                 | ols          | shared_control | -0.184494 |            0.030899  | 2.35992e-09 | -0.245055 | -0.123933 |        739 |
| ok       | min_post_clearance_m           | ols          | shared_control |  0.53015  |            0.0378888 | 1.73811e-44 |  0.455889 |  0.60441  |        739 |
| ok       | delta_radial_velocity_away_mps | ols          | shared_control |  0.41897  |            0.0633219 | 3.67777e-11 |  0.294862 |  0.543079 |        739 |

## Suggested Manuscript Wording

In the follow-up trajectory sessions where no hidden-obstacle prior was available, shared control reduced post-appearance collision probability and increased obstacle clearance after the obstacle became visible, with both effects observed consistently across all usable sessions. In the same sessions, the change in radial velocity away from the obstacle also increased under shared control. These results provide direct evidence for online reactive obstacle avoidance by RT-V2 after sensory evidence became available, rather than a benefit that depended on advance spatial knowledge from a hidden obstacle prior.
