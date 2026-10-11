# AI-only baseline and candidate-set-size analysis

## Definition

The strict AI-only baseline is defined as: environment, candidate targets, and
the autonomous controller, with no decoded BCI command and no BCI-derived command
history. It must not retain a history derived from previous BCI commands, because
that would leak neural information into the autonomous condition.

For each replayed trial, measure:

- success at the hidden cued target;
- which candidate target the autonomous controller reaches;
- collision and failure rate.

The critical strict-baseline result is AI-only at K = 5, where all five targets
are available, the cued target is hidden, and there is no neural command or
history. This is the number to compare against random-target chance, BCI-only,
and the actual shared-control system.

The existing permutation-derived baseline should be reported as
random-target chance, not simply chance. It tests random target selection, not
autonomous-controller performance.

## Candidate-set-size sweep

For every trial, evaluate candidate-set size K = 1, 2, 3, 4, 5. Always include
the true cued target and include K - 1 distractors from the remaining targets.
Because this task has five targets, all distractor combinations can be evaluated
exhaustively.

K = 1 is not ordinary AI-only; it is goal-revealed/oracle autonomy, because the
controller is given the true target. K = 5 corresponds to the actual experimental
condition.

The main curve to generate first is P_AI-only(K). A shared-control K-sweep,
P_shared(K), is useful only if replaying the actual BCI commands under modified
candidate sets is explicitly performed; it is a separate counterfactual from the
strict AI-only baseline.

If P_AI-only is close to random-target chance at K = 5, then geometry and
candidate-target knowledge are not sufficient to solve the task. If P_AI-only is
above random-target chance, that structure should be reported explicitly, and
the key actual-condition contrast becomes shared-control performance at K = 5
minus AI-only performance at K = 5.

## Interpretation table

| Condition | Target information | Neural input | Main purpose |
| --- | --- | --- | --- |
| Random-target chance | none | none | statistical/random-selection reference |
| AI-only, K = 5 | 5 candidates, cued target hidden | none | autonomous baseline |
| BCI-only | cued target encoded only through neural command | yes | neural-control baseline |
| Shared control | 5 candidates, cued target hidden | yes | actual proposed system |
| AI-only, K = 1 | true target effectively revealed | none | autonomous upper bound |

This analysis tests sensitivity to increasing goal ambiguity within the present
five-target task; it should not be described as demonstrating scaling to
arbitrarily complex real-world environments.
