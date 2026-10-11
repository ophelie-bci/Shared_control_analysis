# Example `_aitrials.pkl` field inspection

Date: 2026-08-28

Inspected file:

`<base_dir>\Monkey 1\AI Obstacle\aiLog\navdecodingsphereaiobstacle_Loki_20241223_1106_E_aitrials.pkl`

## Root object

- Type: `tuple`
- Length: `2`
- Nonempty trial list: tuple element `1`
- Trial count: `101`

## Trial-level fields found

Each of the 101 trial dictionaries contained:

- `trial`
- `start`
- `stop`
- `answer`
- `aiControlOn`
- `aiControlOff`
- `aiVelocities`
- `multipleAiControlBlocks`
- `targetPosition`
- `obstaclePosition`

## `aiVelocities` record fields found

Across 4,315 inspected `aiVelocities` records:

- `Input`
- `InputTimestamp`
- `Output`
- `OutputTimestamp`
- `EntropyUb`
- `EntropyLb`
- `Latency`
- `AvatarPosition`

## Searched-for controller/replay fields

The nested key scan did not find fields corresponding to:

- prior distribution or GMM parameters
- posterior distribution
- candidate trajectories or trajectory samples
- trajectory costs
- history buffer
- model/controller object
- checkpoint reference
- scaler/config/model path
- planner/Trajectron/node object
- all five candidate target coordinates

Only `targetPosition`, `obstaclePosition`, `Input`, and `Output`-related keys matched the requested field categories.

## Interpretation

This example `_aitrials.pkl` appears to contain recorded online trial metadata and recorded per-sample AI stream values, not enough RT-V2 controller state to rerun a strict AI-only counterfactual.

The presence of nonzero `Input` values confirms that the recorded stream includes neural/decoded command history. Therefore, reusing recorded `Output` values as an AI-only baseline would not answer the strict reviewer question.

Example nonzero record:

```text
trial: 1
aiVelocities record index: 4
Input: [0.020157070830464363, -0.0002525904565118253, -0.038033999502658844]
Output: (0, 0, 0)
InputTimestamp: array([10017.])
OutputTimestamp: array([10017.])
EntropyUb: nan
EntropyLb: nan
AvatarPosition: [0.0, 0.0, 0.0]
```
