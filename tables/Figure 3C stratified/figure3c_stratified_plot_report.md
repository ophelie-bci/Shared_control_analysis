# Figure 3C Stratified Plots

These plots use the same target-summary points as the original Figure 3C provenance audit: one point per monkey x task x target from `mapped_manuscript_source_paths`.

## Point Counts

| monkey   | task_label          |   n_points |   min_baseline |   median_baseline |   max_baseline |
|:---------|:--------------------|-----------:|---------------:|------------------:|---------------:|
| Monkey 1 | Appearing obstacle  |          5 |          11.11 |             46.03 |          58.18 |
| Monkey 1 | Fixed obstacle      |          5 |          26.36 |             52.90 |          59.70 |
| Monkey 1 | Follow-up appearing |          5 |          15.24 |             47.06 |          56.86 |
| Monkey 2 | Appearing obstacle  |          5 |           1.92 |             28.26 |          39.58 |
| Monkey 2 | Fixed obstacle      |          5 |          14.53 |             37.72 |          48.31 |

## Quadratic Fits

| stratification   | group               |   n |   quadratic_beta |   linear_beta |   intercept |      r2 |   model_p |   quadratic_p |
|:-----------------|:--------------------|----:|-----------------:|--------------:|------------:|--------:|----------:|--------------:|
| task             | Fixed obstacle      |  10 |         -0.0279  |        2.27   |     -9.282  | 0.5397  |  0.06616  |      0.04629  |
| task             | Appearing obstacle  |  10 |         -0.03889 |        2.655  |      0.7738 | 0.8354  |  0.001808 |      0.001929 |
| task             | Follow-up appearing |   5 |          0.04114 |       -3.458  |     86.41   | 0.5181  |  0.4819   |      0.5298   |
| monkey           | Monkey 1            |  15 |          0.00029 |       -0.1802 |     36.08   | 0.07881 |  0.6111   |      0.9842   |
| monkey           | Monkey 2            |  10 |         -0.04366 |        2.885  |     -4.55   | 0.6873  |  0.0171   |      0.05347  |

## Outputs

figures\Extended Data Figure 3\Figure 3C stratified\figure3c_stratified_by_task.svg
figures\Extended Data Figure 3\Figure 3C stratified\figure3c_stratified_by_monkey.svg
figures\Extended Data Figure 3\Figure 3C stratified\figure3c_task_fixed_obstacle.svg
figures\Extended Data Figure 3\Figure 3C stratified\figure3c_task_appearing_obstacle.svg
figures\Extended Data Figure 3\Figure 3C stratified\figure3c_task_follow_up_appearing.svg
figures\Extended Data Figure 3\Figure 3C stratified\figure3c_monkey_monkey_1.svg
figures\Extended Data Figure 3\Figure 3C stratified\figure3c_monkey_monkey_2.svg
