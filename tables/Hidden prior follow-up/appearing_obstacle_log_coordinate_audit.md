# Appearing-obstacle log coordinate audit

Date: 2026-08-31

## Result

The earlier interpretation was too strong. The original logs show that the controller-side `Obstacle0` entry matches the nominal `ShowTarget.obstaclePosition`, but the actually rendered obstacle is logged separately as `recv [obstacleAppeared]` and is not the same coordinate.

The nominal `ShowTarget.obstaclePosition` is exactly halfway in x/z from the origin/start position to the target position in all non-NaN cases. In original sessions, the controller-side `Obstacle0` matched that same nominal midpoint exactly. The actually appeared obstacle was not exactly at the midpoint.

## Summary

| dataset        |   n_showtarget_trials |   n_show_obstacle_metadata |   n_controller_obstacle0 |   n_obstacle_appeared_events |   median_controller_to_show_xz_m |   median_appeared_to_show_xz_m |   median_appeared_to_controller_xz_m |   max_show_to_target_midpoint_xz_m |   max_controller_to_target_midpoint_xz_m |   median_appeared_to_target_midpoint_xz_m |
|:---------------|----------------------:|---------------------------:|-------------------------:|-----------------------------:|---------------------------------:|-------------------------------:|-------------------------------------:|-----------------------------------:|-----------------------------------------:|------------------------------------------:|
| Original Loki  |                  1099 |                        831 |                      831 |                          811 |                            0.000 |                          0.905 |                                0.905 |                              0.000 |                                    0.000 |                                     0.905 |
| Original Maui  |                  1122 |                        776 |                      776 |                          769 |                            0.000 |                          0.867 |                                0.867 |                              0.000 |                                    0.000 |                                     0.867 |
| Follow-up Maui |                  2071 |                       1451 |                        0 |                         1427 |                          nan     |                          0.816 |                              nan     |                              0.000 |                                  nan     |                                     0.816 |

## Example

Original Maui, first appearing-obstacle trial:

```text
ShowTarget.obstaclePosition = [-3.5, 0, 3]
Obstacle0 = [-3.5, 0.0, 3.0]
obstacleAppeared.obstaclePosition = [-2.6586, 0.3000, 2.6782]
targetPosition = [-7.0, 0.75, 6.0]
target midpoint from origin = [-3.5, 0.75, 3.0]
```

Thus the hidden controller obstacle was at the nominal midpoint location, not at the exact rendered location that appeared later.

## Corrected wording

Use:

> In the original Appearing Obstacle sessions, when an appearing-obstacle trial was configured, the shared-control controller received a non-rendered `Obstacle0` entry at the nominal `ShowTarget.obstaclePosition`, which was exactly halfway between the start position and target in x/z. The obstacle later rendered to the monkey was logged separately as `obstacleAppeared.obstaclePosition` and was not exactly at this midpoint, consistent with the rendered obstacle being computed from cursor position at the moment of appearance. Thus the hidden controller obstacle provided advance information about the upcoming obstacle region, but not the exact trial-specific appeared coordinate. In the six-month follow-up logs, the nominal future `obstaclePosition` remained in the trial metadata, but no `Obstacle0` controller-list entries were found.
