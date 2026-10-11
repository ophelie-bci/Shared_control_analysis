# Respawn Failure Mechanism Decomposition

Shared-control Respawn failures were split into two mechanisms.

## 1. Stable decoded-BCI reorientation

- Incorrect shared-control trials without sustained BCI reorientation within 2 s: 24.6% (64/260).
- Correct shared-control trials without sustained BCI reorientation within 2 s: 0.2% (1/535).

These non-reorientation failures cannot be attributed solely to RT-V2 temporal inertia because the decoded BCI command did not provide a stable new-target switch within the analysis window.

## 2. Latency after BCI reorientation

- Among trials where BCI did reorient, incorrect trials had later BCI reorientation than correct trials: 683.0 ms vs 384.3 ms, delta 298.7 ms, p=1.95e-21.
- The BCI-only incorrect-correct latency penalty was 93.2 ms, p=0.00975.
- The median latency penalty was 205.5 ms larger in shared control than BCI-only; robust OLS interaction coefficient 175.1 ms, p=0.00105.
- Among shared-control trials where both BCI input and executed output reoriented, the BCI-to-executed reorientation lag was 59.3 ms for incorrect trials vs 0.0 ms for correct trials, delta 59.3 ms, p=7.07e-05.
