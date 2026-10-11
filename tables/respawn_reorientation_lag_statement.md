# Respawn Reorientation Lag

Shared-control Respawn trials were aligned to the target jump.
Evidence is defined as the unit velocity projection toward the new target minus the projection toward the old target.

Decoded BCI switch latency: successful 389.7 ms, unsuccessful 616.6 ms.
Executed switch latency: successful 424.7 ms, unsuccessful 619.3 ms.
Controller lag (executed minus BCI): successful 0.0 ms, unsuccessful 59.3 ms; difference 59.3 ms.
Pre-jump alpha: successful 0.3, unsuccessful 0.3.

Nearest-neighbor control matched unsuccessful trials to successful trials within monkey by pre-jump alpha.
Matched controller-lag difference: 41.4 ms over 69 pairs; median absolute alpha mismatch 0.0.

BCI-matched control matched unsuccessful trials to successful trials within monkey using t_BCI, E_BCI from 0.2-0.8 s, and pre-jump alpha.
BCI-matched controller-lag difference: 48.3 ms over 69 pairs; p = 0.0.

successful non-crossing fractions: BCI 22.7%, executed 23.4%.
unsuccessful non-crossing fractions: BCI 44.8%, executed 71.8%.

Trial-level logistic regression controlled for t_BCI, controller lag, pre-jump alpha, and early BCI evidence, with cluster-robust standard errors by session.
Controller-lag odds ratio per z-scored increase: 0.6; cluster-robust p = 0.1. Rows with undefined latency predictors were excluded from this supporting model and are counted in the non-crossing table.
