# BCI-only Learning Across Original and Follow-up Appearing Obstacle Sessions

This analysis uses Monkey 1 BCI-only trials from the original Appearing Obstacle experiment and the six-month follow-up without the hidden-obstacle prior.

## Original-session BCI-only trend

Across the original sessions, BCI-only success increased with session number (Spearman rho=0.515, p=0.1276; linear slope=2.88 pp/session).
The primary inference is session-level here; the trial-level clustered logistic model was skipped because this run used the raw session summary table rather than reloading the multi-GB trial pickles.

## End-of-original to follow-up transition

The final 3 original sessions averaged 52.8% BCI-only success; the first 3 follow-up sessions averaged 52.4% (follow-up minus final-original=-0.4 pp).

Individual BCI-only session values are saved in `bci_only_learning_session_values.csv`.

## Suggested reviewer wording

The apparent increase from the original experiment mean to the six-month follow-up mean partly reflects learning already present within the original Appearing Obstacle sessions. BCI-only success showed a positive, non-significant trend across the original sessions (Spearman rho=0.52, p=0.128), and the first follow-up sessions began near the level reached at the end of the original experiment (final three original sessions 52.8% vs first three follow-up sessions 52.4%). Thus, the 39-40% original mean should not be interpreted as the BCI-only level immediately before the six-month gap.
