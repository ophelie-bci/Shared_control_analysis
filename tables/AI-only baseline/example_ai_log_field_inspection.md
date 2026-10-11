# Example raw AI log field inspection

Date: 2026-08-28

Inspected file:

`<base_dir>\Monkey 3\AI Obstacle\aiLog\navdecodingsphereaiobstacle_Maui_20241101_1044_A.log`

## File status

- File exists.
- Size: approximately 9.5 MB.

## Fields directly observed in the raw log scan

Representative log lines include:

- `recv [StartTrial]: {'aiVelocityFactor': 1, 'index': 1}`
- `recv [ShowTarget]: {'targetPosition': [7.0, 0.75, 6.0], 'obstaclePosition': [3.5, 0, 3], 'targetJumpPosition': [nan, nan, nan]}`
- `Obstacle List`
- `Obstacle0:[3.5, 0.0, 3.0]`
- `Velocity input:[...]`
- `Entropy ub:...`
- `Entropy lb:...`

The scan found repeated nonzero `Velocity input:[...]` records. This is the neural/decoded command stream sent into the AI controller.

## Parser-confirmed fields for this log family

The local Unity-log parser `tnsbmi/tnsbmi/navigationdata_ai.py` extracts these fields from AI task logs:

- `Experiment`
- `StartTrial`
- `StopTrial`
- `TargetVisible`
- `TargetJump`
- `UnityTargetPosition`
- `AvatarPosition`
- `AvatarVelocity`
- `AiVelocity`
- `Velocity`
- `SendingVelocityToAi`
- `StuckBehindTree`
- `Obstacle appeared`
- `message`
- `ObstaclePosition`
- `targetReached`
- `targetJumpPosition`

For AI-on trials, the parser uses:

- `AiVelocity` as the recorded AI/output velocity stream.
- `SendingVelocityToAi` as the recorded model/BCI input velocity stream.

## Controller-state fields not observed

The direct scan did not reveal lines corresponding to:

- full prior distribution or GMM parameters
- posterior distribution
- candidate trajectories or trajectory samples
- trajectory costs
- history buffer
- model/controller object
- checkpoint reference
- scaler/config/model path
- planner/Trajectron/node object
- all five candidate target coordinates in a single replay-ready structure

## Interpretation

This raw log appears richer than the derived `_aitrials.pkl` because it preserves human-readable event lines, but the observed fields still look like recorded online streams and task geometry rather than enough controller state to rerun RT-V2 with neural input removed.

The most useful extra information in the raw log is the distinction between:

- `Velocity input` / `SendingVelocityToAi`: decoded neural command entering the AI controller.
- `AiVelocity` / output velocity: recorded controller output during the actual online trial.

That distinction helps audit the online shared-control behavior, but it still does not by itself create the strict AI-only baseline, because the recorded output was generated after receiving nonzero input history.

## Limitation

A full-file custom parser pass over the mounted `X:` file was blocked by the environment approval/usage limit during this inspection. The conclusions above are based on the successful targeted raw-log scan plus the local parser source for this log family.
