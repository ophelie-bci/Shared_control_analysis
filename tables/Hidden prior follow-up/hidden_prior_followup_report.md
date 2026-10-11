# Hidden-prior follow-up shared-control gain analysis

Restricted to Monkey 1 Appearing Obstacle sessions.

## Session-level gain comparison

- Hidden prior present: n=10 sessions, mean gain=27.4 pp, median gain=28.0 pp.
- Hidden prior absent: n=9 sessions, mean gain=23.0 pp, median gain=24.0 pp.
- Difference in mean gains, hidden minus no-hidden: 4.4 pp (bootstrap 95% CI -5.8 to 15.0 pp).
- Two-sided Mann-Whitney rank-sum test: U=52.0, p=0.5956.

## Trial-level interaction model

Model: `success ~ shared_control * followup + C(target)`, cluster-robust SEs by session.
- Interaction coefficient `shared_control:followup`: beta=-0.275, SE=0.344, p=0.424, OR=0.76.
- Interpretation: this tests whether the shared-control effect changed when the hidden-obstacle prior was removed.

## Pooled trial success rates

| prior_condition   | condition      |   n_trials |   n_success |   success_rate |   success_pct |
|:------------------|:---------------|-----------:|------------:|---------------:|--------------:|
| Hidden prior      | BCI-only       |        275 |         110 |       0.4      |       40      |
| Hidden prior      | Shared-control |        378 |         262 |       0.693122 |       69.3122 |
| No hidden prior   | BCI-only       |        357 |         213 |       0.596639 |       59.6639 |
| No hidden prior   | Shared-control |        382 |         306 |       0.801047 |       80.1047 |

## Suggested manuscript wording

Shared control improved success substantially in both experiments, and we found no evidence that the shared-control gain differed after the hidden-obstacle prior was removed.
