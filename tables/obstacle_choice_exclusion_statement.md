# Obstacle Choice-State Exclusion Audit

The decoded-choice analysis excluded time steps when the avatar center was within 1.5 m of the obstacle center in the x-z plane, after applying the model-velocity speed threshold of 0.1.

Across included obstacle trials, 85,905 of 418,026 speed-valid model-velocity time steps were excluded (20.55%).
Median trial-level excluded fraction: 12.05%.
Median session-level excluded fraction: 16.65%.
Maximum session-level excluded fraction: 55.16%.

See `obstacle_choice_exclusion_by_trial.csv` and `obstacle_choice_exclusion_by_session.csv` for the per-trial and per-session values.
