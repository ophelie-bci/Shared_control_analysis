# Time-Resolved BCI Influence on Executed Command

Shared-control obstacle trials were analyzed after excluding explicit safety-override bins and a 150-ms buffer around each override epoch.

The retained sample table contains 169,560 non-override velocity bins from 2,796 shared-control trials across 51 sessions.
Across trials with velocity logs, 38,973 bins were removed as explicit override plus pad windows.

## Fixed Obstacle
- early-0-30: mean unique current-BCI beta = 0.348; minimum bin beta = 0.092; 3/3 bins positive; maximum bin p = 3.56e-36; mean directional alignment = 0.867.
- middle-40-70: mean unique current-BCI beta = 0.148; minimum bin beta = 0.111; 3/3 bins positive; maximum bin p = 2.79e-17; mean directional alignment = 0.946.
- late-70-100: mean unique current-BCI beta = 0.080; minimum bin beta = 0.063; 3/3 bins positive; maximum bin p = 2.26e-15; mean directional alignment = 0.925.

## Appearing Obstacle
- early-0-30: mean unique current-BCI beta = 0.328; minimum bin beta = 0.141; 3/3 bins positive; maximum bin p = 1.06e-46; mean directional alignment = 0.906.
- middle-40-70: mean unique current-BCI beta = 0.098; minimum bin beta = 0.086; 3/3 bins positive; maximum bin p = 4.97e-14; mean directional alignment = 0.909.
- late-70-100: mean unique current-BCI beta = 0.114; minimum bin beta = 0.109; 3/3 bins positive; maximum bin p = 2.7e-17; mean directional alignment = 0.942.

Reviewer-facing summary:

Outside the explicit 150-ms safety-override windows, the component of current decoded BCI velocity orthogonal to the instantaneous target direction retained a positive unique contribution to the corresponding executed-velocity component through the middle and late portions of both obstacle tasks, after accounting for previous executed command and prior confidence.
