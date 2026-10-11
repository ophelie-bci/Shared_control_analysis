# Respawn BCI Reorientation Sensitivity

Decoded BCI reorientation was summarized as cumulative probability over 0-3 s and at 1.0, 1.5, 2.0, and 3.0 s windows.
A reorientation counted for a given window only when its full 200 ms sustained-positive evidence segment fit inside that window.

## Primary 2 s window

- Shared control correct: 99.8% reoriented (534/535); median latency 384.3 ms.
- Shared control incorrect: 75.4% reoriented (196/260); median latency 683.0 ms.
- BCI-only correct: 100.0% reoriented (635/635); median latency 449.8 ms.
- BCI-only incorrect: 73.3% reoriented (121/165); median latency 543.0 ms.

## Tests

- Shared control incorrect vs correct: latency p=1.95e-21; crossing-rate p=1.16e-32; log-rank p=3.88e-56.
- BCI-only incorrect vs correct: latency p=0.00975; crossing-rate p=4.1e-33; log-rank p=1.21e-29.
- Shared-control penalty vs BCI-only penalty: median penalty contrast 205.5 ms; robust OLS interaction p=0.00105.
