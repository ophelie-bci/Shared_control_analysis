# Extended Data Table 1 Candidate Audit

This candidate uses reproduced session-level summaries for session counts, means, sample SDs, paired mean gains, and paired two-sided Wilcoxon p-values.
Chance values are retained from the previous Extended Data Table 1 and are marked as legacy values.

## Candidate table

| Monkey | Task | Number of sessions | BCI-only (mean +/- SD) | Shared-control (mean +/- SD) | Legacy chance level | Shared-control Gain (pp) | p-value (paired Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Monkey 1 | Fixed Obstacle | 12 | 49 +/- 10 | 81 +/- 8 | 14 | 31.6 | 4.88 x 10^-4 |
| Monkey 1 | Appearing Obstacle | 10 | 39 +/- 15 | 67 +/- 16 | 18 | 27.4 | 1.95 x 10^-3 |
| Monkey 1 | Appearing Obstacle (6 months later) | 9 | 57 +/- 12 | 80 +/- 10 | 16 | 23.0 | 3.91 x 10^-3 |
| Monkey 1 | Respawn | 9 | 79 +/- 8 | 67 +/- 10 | 24 | -12.6 | 1.17 x 10^-2 |
| Monkey 2 | Fixed Obstacle | 11 | 34 +/- 8 | 61 +/- 9 | 18 | 27.1 | 9.77 x 10^-4 |
| Monkey 2 | Appearing Obstacle | 9 | 22 +/- 9 | 56 +/- 9 | 21 | 33.7 | 3.91 x 10^-3 |

## Previous CSV comparison

No cell differences from the previous CSV after renaming `Chance level` to `Legacy chance level`.

## Latest manuscript embedded table comparison

The latest manuscript embedded Extended Data Table 1 was not found inside this repository, so every manuscript-table cell comparison is unresolved.

## Figure 3 annotation comparison

- Monkey 1 | Appearing Obstacle (6 months later) / Legacy chance level: Figure 3 annotation `17` vs candidate `16`
- Monkey 1 | Respawn / BCI-only (mean +/- SD): Figure 3 annotation `80 +/- 8` vs candidate `79 +/- 8`
- Monkey 2 | Appearing Obstacle / BCI-only (mean +/- SD): Figure 3 annotation `22 +/- 10` vs candidate `22 +/- 9`

## Original chance procedure

Legacy chance values come from the original success-rate outputs. The original helper performs target permutations using successful recorded trajectories, with the obstacle list fixed in recorded-trial order.

Suggested Methods wording: Random-target chance was estimated by replaying successful recorded trajectories while randomly permuting the empirical target labels across trials; obstacle positions remained fixed to the recorded trials, and success was scored with the same target-entry and obstacle-collision criteria used by the original success-rate analysis.
