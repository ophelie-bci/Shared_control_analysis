# Exact-Zero Baseline Directional Quality Audit

BCI-only baseline success was separated into exact 0%, >0-10%, >10-20%, and >20% target x session rows.

## BCI-only Trial Counts

```csv
baseline_group,n_target_session_rows,mean_n_bci_only_trials,median_n_bci_only_trials,min_n_bci_only_trials,max_n_bci_only_trials,median_success_step_pct
exact_0,36,5.777777777777778,5.0,3,11,20.0
gt0_to_10,11,10.545454545454545,10.0,10,12,10.0
gt10_to_20,31,7.967741935483871,8.0,5,12,12.5
gt20,177,8.915254237288135,10.0,4,15,10.0
```

## Directional Quality

```csv
baseline_group,n_target_session_rows,n_bci_only_trials_with_velocity,mean_trial_cosine,ci95_trial_cosine,mean_trial_angle_deg,ci95_trial_angle_deg,median_trial_cosine
exact_0,36,208,0.4936288506619527,0.043948707301655524,57.286303940355545,3.234984638948965,0.4643631472047294
gt0_to_10,11,116,0.4675180924858863,0.07024732489812326,58.77827761851399,4.950187885111067,0.5003726172895178
```

## Interpretation

Exact-zero baseline rows contributed 208 BCI-only trials with usable decoded velocity. Mean trial-level cosine alignment was 0.494 +/- 0.044 (95% CI), corresponding to a mean angular error of 57.3 deg.
For >0-10% rows, mean trial-level cosine alignment was 0.468 +/- 0.070 across 116 trials.
