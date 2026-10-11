# Shared-control Adaptation Analysis

Within each session, decoded-command metrics were modeled as condition + trial progress + condition x trial progress. A separate BCI-only carryover analysis tested whether BCI-only trial metrics changed with the fraction of preceding trials that were shared-control trials.

## Condition-specific within-session drift

- Appearing Obstacle, angular_error_to_target_deg_mean: median interaction slope 3.874; 12/19 positive; two-sided Wilcoxon p=0.5412, BH q=0.7644.
- Appearing Obstacle, command_heading_variability_deg: median interaction slope -0.04126; 9/19 positive; two-sided Wilcoxon p=0.5153, BH q=0.7644.
- Appearing Obstacle, command_speed_sd: median interaction slope -1.399e-05; 9/19 positive; two-sided Wilcoxon p=0.6794, BH q=0.7644.
- Appearing Obstacle, decoded_speed_mean: median interaction slope -0.04408; 6/19 positive; two-sided Wilcoxon p=0.5153, BH q=0.7644.
- Appearing Obstacle, lateral_command_abs_mean: median interaction slope 0.06973; 13/19 positive; two-sided Wilcoxon p=0.2253, BH q=0.676.
- Appearing Obstacle, neural_target_evidence: median interaction slope 0.07523; 13/19 positive; two-sided Wilcoxon p=0.1688, BH q=0.676.
- Appearing Obstacle follow-up, angular_error_to_target_deg_mean: median interaction slope 5.257; 9/9 positive; two-sided Wilcoxon p=0.003906, BH q=0.07031.
- Appearing Obstacle follow-up, command_heading_variability_deg: median interaction slope 2.181; 5/9 positive; two-sided Wilcoxon p=0.3008, BH q=0.7644.
- Appearing Obstacle follow-up, command_speed_sd: median interaction slope -0.02397; 3/9 positive; two-sided Wilcoxon p=0.6523, BH q=0.7644.
- Appearing Obstacle follow-up, decoded_speed_mean: median interaction slope -0.08224; 3/9 positive; two-sided Wilcoxon p=0.1289, BH q=0.676.
- Appearing Obstacle follow-up, lateral_command_abs_mean: median interaction slope 0.1127; 8/9 positive; two-sided Wilcoxon p=0.01172, BH q=0.1055.
- Appearing Obstacle follow-up, neural_target_evidence: median interaction slope -0.04437; 2/9 positive; two-sided Wilcoxon p=0.4258, BH q=0.7644.
- Fixed Obstacle, angular_error_to_target_deg_mean: median interaction slope 5.951; 14/23 positive; two-sided Wilcoxon p=0.3604, BH q=0.7644.
- Fixed Obstacle, command_heading_variability_deg: median interaction slope 1.689; 14/23 positive; two-sided Wilcoxon p=0.56, BH q=0.7644.
- Fixed Obstacle, command_speed_sd: median interaction slope -0.003607; 10/23 positive; two-sided Wilcoxon p=0.8229, BH q=0.8229.
- Fixed Obstacle, decoded_speed_mean: median interaction slope 0.01647; 12/23 positive; two-sided Wilcoxon p=0.754, BH q=0.7983.
- Fixed Obstacle, lateral_command_abs_mean: median interaction slope 0.06464; 15/23 positive; two-sided Wilcoxon p=0.665, BH q=0.7644.
- Fixed Obstacle, neural_target_evidence: median interaction slope 0.03587; 15/23 positive; two-sided Wilcoxon p=0.2002, BH q=0.676.

## BCI-only carryover across exposure timescales

- Appearing Obstacle, angular_error_to_target_deg_mean, previous_trial_shared: median slope 1.63; 10/19 positive; two-sided Wilcoxon p=0.1564, BH q=0.5474.
- Appearing Obstacle, command_heading_variability_deg, previous_trial_shared: median slope 1.006; 12/19 positive; two-sided Wilcoxon p=0.1336, BH q=0.5474.
- Appearing Obstacle, command_speed_sd, previous_trial_shared: median slope -0.0179; 6/19 positive; two-sided Wilcoxon p=0.07283, BH q=0.5098.
- Appearing Obstacle, decoded_speed_mean, previous_trial_shared: median slope 0.0144; 12/19 positive; two-sided Wilcoxon p=0.2253, BH q=0.5915.
- Appearing Obstacle, lateral_command_abs_mean, previous_trial_shared: median slope 0.03883; 15/19 positive; two-sided Wilcoxon p=0.02041, BH q=0.4287.
- Appearing Obstacle, neural_target_evidence, previous_trial_shared: median slope 0.02329; 11/19 positive; two-sided Wilcoxon p=0.9843, BH q=1.
- Appearing Obstacle, success, previous_trial_shared: median slope -0.008767; 9/19 positive; two-sided Wilcoxon p=0.7381, BH q=1.
- Appearing Obstacle follow-up, angular_error_to_target_deg_mean, previous_trial_shared: median slope 1.259; 6/9 positive; two-sided Wilcoxon p=0.4961, BH q=0.8682.
- Appearing Obstacle follow-up, command_heading_variability_deg, previous_trial_shared: median slope -1.571; 4/9 positive; two-sided Wilcoxon p=0.9102, BH q=1.
- Appearing Obstacle follow-up, command_speed_sd, previous_trial_shared: median slope -0.00964; 4/9 positive; two-sided Wilcoxon p=0.8203, BH q=1.
- Appearing Obstacle follow-up, decoded_speed_mean, previous_trial_shared: median slope -0.004141; 3/9 positive; two-sided Wilcoxon p=0.5703, BH q=0.9213.
- Appearing Obstacle follow-up, lateral_command_abs_mean, previous_trial_shared: median slope 0.05603; 6/9 positive; two-sided Wilcoxon p=0.6523, BH q=0.9785.
- Appearing Obstacle follow-up, neural_target_evidence, previous_trial_shared: median slope 0.01278; 6/9 positive; two-sided Wilcoxon p=0.3008, BH q=0.6316.
- Appearing Obstacle follow-up, success, previous_trial_shared: median slope 0.005968; 5/9 positive; two-sided Wilcoxon p=1, BH q=1.
- Fixed Obstacle, angular_error_to_target_deg_mean, previous_trial_shared: median slope -3.93; 8/23 positive; two-sided Wilcoxon p=0.2002, BH q=0.5915.
- Fixed Obstacle, command_heading_variability_deg, previous_trial_shared: median slope -1.638; 6/23 positive; two-sided Wilcoxon p=0.05625, BH q=0.5098.
- Fixed Obstacle, command_speed_sd, previous_trial_shared: median slope 0.01141; 14/23 positive; two-sided Wilcoxon p=0.9881, BH q=1.
- Fixed Obstacle, decoded_speed_mean, previous_trial_shared: median slope 0.022; 14/23 positive; two-sided Wilcoxon p=0.3001, BH q=0.6316.
- Fixed Obstacle, lateral_command_abs_mean, previous_trial_shared: median slope 0.007357; 12/23 positive; two-sided Wilcoxon p=0.482, BH q=0.8682.
- Fixed Obstacle, neural_target_evidence, previous_trial_shared: median slope -0.01996; 10/23 positive; two-sided Wilcoxon p=0.9406, BH q=1.
- Fixed Obstacle, success, previous_trial_shared: median slope 0.03251; 16/23 positive; two-sided Wilcoxon p=0.1341, BH q=0.5474.
- Appearing Obstacle, angular_error_to_target_deg_mean, preceding_3_shared_fraction: median slope -0.9315; 6/19 positive; two-sided Wilcoxon p=0.5412, BH q=0.8823.
- Appearing Obstacle, command_heading_variability_deg, preceding_3_shared_fraction: median slope -0.7923; 9/19 positive; two-sided Wilcoxon p=0.7983, BH q=0.8823.
- Appearing Obstacle, command_speed_sd, preceding_3_shared_fraction: median slope -0.03272; 7/19 positive; two-sided Wilcoxon p=0.1564, BH q=0.6569.
- Appearing Obstacle, decoded_speed_mean, preceding_3_shared_fraction: median slope 0.01658; 11/19 positive; two-sided Wilcoxon p=0.5678, BH q=0.8823.
- Appearing Obstacle, lateral_command_abs_mean, preceding_3_shared_fraction: median slope -0.002439; 8/19 positive; two-sided Wilcoxon p=0.4653, BH q=0.8823.
- Appearing Obstacle, neural_target_evidence, preceding_3_shared_fraction: median slope -0.018; 9/19 positive; two-sided Wilcoxon p=0.5949, BH q=0.8823.
- Appearing Obstacle, success, preceding_3_shared_fraction: median slope 0.009147; 10/19 positive; two-sided Wilcoxon p=0.768, BH q=0.8823.
- Appearing Obstacle follow-up, angular_error_to_target_deg_mean, preceding_3_shared_fraction: median slope 2.886; 7/9 positive; two-sided Wilcoxon p=0.2031, BH q=0.7109.
- Appearing Obstacle follow-up, command_heading_variability_deg, preceding_3_shared_fraction: median slope 0.36; 5/9 positive; two-sided Wilcoxon p=0.4961, BH q=0.8823.
- Appearing Obstacle follow-up, command_speed_sd, preceding_3_shared_fraction: median slope -0.02548; 2/9 positive; two-sided Wilcoxon p=0.1289, BH q=0.6569.
- Appearing Obstacle follow-up, decoded_speed_mean, preceding_3_shared_fraction: median slope -0.04908; 4/9 positive; two-sided Wilcoxon p=0.6523, BH q=0.8823.
- Appearing Obstacle follow-up, lateral_command_abs_mean, preceding_3_shared_fraction: median slope 0.1289; 6/9 positive; two-sided Wilcoxon p=0.25, BH q=0.75.
- Appearing Obstacle follow-up, neural_target_evidence, preceding_3_shared_fraction: median slope -0.03509; 4/9 positive; two-sided Wilcoxon p=1, BH q=1.
- Appearing Obstacle follow-up, success, preceding_3_shared_fraction: median slope -0.04576; 2/9 positive; two-sided Wilcoxon p=0.02734, BH q=0.2687.
- Fixed Obstacle, angular_error_to_target_deg_mean, preceding_3_shared_fraction: median slope -0.6172; 11/23 positive; two-sided Wilcoxon p=0.601, BH q=0.8823.
- Fixed Obstacle, command_heading_variability_deg, preceding_3_shared_fraction: median slope 1.034; 14/23 positive; two-sided Wilcoxon p=0.7314, BH q=0.8823.
- Fixed Obstacle, command_speed_sd, preceding_3_shared_fraction: median slope 0.01366; 13/23 positive; two-sided Wilcoxon p=0.9168, BH q=0.9627.
- Fixed Obstacle, decoded_speed_mean, preceding_3_shared_fraction: median slope 0.07163; 16/23 positive; two-sided Wilcoxon p=0.03543, BH q=0.2687.
- Fixed Obstacle, lateral_command_abs_mean, preceding_3_shared_fraction: median slope 0.03715; 15/23 positive; two-sided Wilcoxon p=0.03838, BH q=0.2687.
- Fixed Obstacle, neural_target_evidence, preceding_3_shared_fraction: median slope -0.02125; 11/23 positive; two-sided Wilcoxon p=0.7314, BH q=0.8823.
- Fixed Obstacle, success, preceding_3_shared_fraction: median slope -0.005544; 11/23 positive; two-sided Wilcoxon p=0.665, BH q=0.8823.
- Appearing Obstacle, angular_error_to_target_deg_mean, preceding_5_shared_fraction: median slope 2.269; 13/19 positive; two-sided Wilcoxon p=0.2253, BH q=0.5833.
- Appearing Obstacle, command_heading_variability_deg, preceding_5_shared_fraction: median slope 0.524; 11/19 positive; two-sided Wilcoxon p=0.6794, BH q=0.8917.
- Appearing Obstacle, command_speed_sd, preceding_5_shared_fraction: median slope -0.04709; 6/19 positive; two-sided Wilcoxon p=0.02041, BH q=0.4287.
- Appearing Obstacle, decoded_speed_mean, preceding_5_shared_fraction: median slope 0.07613; 12/19 positive; two-sided Wilcoxon p=0.2935, BH q=0.6005.
- Appearing Obstacle, lateral_command_abs_mean, preceding_5_shared_fraction: median slope 0.07675; 13/19 positive; two-sided Wilcoxon p=0.2413, BH q=0.5833.
- Appearing Obstacle, neural_target_evidence, preceding_5_shared_fraction: median slope -0.06452; 8/19 positive; two-sided Wilcoxon p=0.3955, BH q=0.6921.
- Appearing Obstacle, success, preceding_5_shared_fraction: median slope -0.07253; 8/19 positive; two-sided Wilcoxon p=0.9843, BH q=1.
- Appearing Obstacle follow-up, angular_error_to_target_deg_mean, preceding_5_shared_fraction: median slope 4.396; 5/9 positive; two-sided Wilcoxon p=0.25, BH q=0.5833.
- Appearing Obstacle follow-up, command_heading_variability_deg, preceding_5_shared_fraction: median slope -0.4722; 4/9 positive; two-sided Wilcoxon p=1, BH q=1.
- Appearing Obstacle follow-up, command_speed_sd, preceding_5_shared_fraction: median slope -0.03465; 3/9 positive; two-sided Wilcoxon p=0.1289, BH q=0.5833.
- Appearing Obstacle follow-up, decoded_speed_mean, preceding_5_shared_fraction: median slope -0.02892; 3/9 positive; two-sided Wilcoxon p=0.5703, BH q=0.7984.
- Appearing Obstacle follow-up, lateral_command_abs_mean, preceding_5_shared_fraction: median slope 0.08389; 5/9 positive; two-sided Wilcoxon p=0.4961, BH q=0.7803.
- Appearing Obstacle follow-up, neural_target_evidence, preceding_5_shared_fraction: median slope -0.01754; 4/9 positive; two-sided Wilcoxon p=0.8203, BH q=0.9873.
- Appearing Obstacle follow-up, success, preceding_5_shared_fraction: median slope -0.08267; 2/9 positive; two-sided Wilcoxon p=0.2031, BH q=0.5833.
- Fixed Obstacle, angular_error_to_target_deg_mean, preceding_5_shared_fraction: median slope -0.819; 11/23 positive; two-sided Wilcoxon p=0.9406, BH q=1.
- Fixed Obstacle, command_heading_variability_deg, preceding_5_shared_fraction: median slope -2.181; 9/23 positive; two-sided Wilcoxon p=0.3146, BH q=0.6005.
- Fixed Obstacle, command_speed_sd, preceding_5_shared_fraction: median slope 0.01167; 14/23 positive; two-sided Wilcoxon p=0.5202, BH q=0.7803.
- Fixed Obstacle, decoded_speed_mean, preceding_5_shared_fraction: median slope 0.01561; 12/23 positive; two-sided Wilcoxon p=0.2002, BH q=0.5833.
- Fixed Obstacle, lateral_command_abs_mean, preceding_5_shared_fraction: median slope 0.01564; 13/23 positive; two-sided Wilcoxon p=0.2226, BH q=0.5833.
- Fixed Obstacle, neural_target_evidence, preceding_5_shared_fraction: median slope -0.02248; 11/23 positive; two-sided Wilcoxon p=0.8462, BH q=0.9873.
- Fixed Obstacle, success, preceding_5_shared_fraction: median slope 0.1273; 14/23 positive; two-sided Wilcoxon p=0.2226, BH q=0.5833.
- Appearing Obstacle, angular_error_to_target_deg_mean, preceding_shared_fraction: median slope 9.123; 11/19 positive; two-sided Wilcoxon p=0.4413, BH q=0.6487.
- Appearing Obstacle, command_heading_variability_deg, preceding_shared_fraction: median slope 0.6215; 11/19 positive; two-sided Wilcoxon p=0.1688, BH q=0.5064.
- Appearing Obstacle, command_speed_sd, preceding_shared_fraction: median slope -0.146; 3/19 positive; two-sided Wilcoxon p=0.0006447, BH q=0.01354.
- Appearing Obstacle, decoded_speed_mean, preceding_shared_fraction: median slope 0.4999; 14/19 positive; two-sided Wilcoxon p=0.01808, BH q=0.1899.
- Appearing Obstacle, lateral_command_abs_mean, preceding_shared_fraction: median slope 0.4691; 14/19 positive; two-sided Wilcoxon p=0.06021, BH q=0.3117.
- Appearing Obstacle, neural_target_evidence, preceding_shared_fraction: median slope 0.02526; 11/19 positive; two-sided Wilcoxon p=0.5949, BH q=0.7349.
- Appearing Obstacle, success, preceding_shared_fraction: median slope 0.2672; 11/19 positive; two-sided Wilcoxon p=0.3321, BH q=0.5811.
- Appearing Obstacle follow-up, angular_error_to_target_deg_mean, preceding_shared_fraction: median slope 14.42; 5/9 positive; two-sided Wilcoxon p=0.4961, BH q=0.6511.
- Appearing Obstacle follow-up, command_heading_variability_deg, preceding_shared_fraction: median slope 9.386; 5/9 positive; two-sided Wilcoxon p=0.3008, BH q=0.5811.
- Appearing Obstacle follow-up, command_speed_sd, preceding_shared_fraction: median slope -0.04695; 4/9 positive; two-sided Wilcoxon p=0.4258, BH q=0.6487.
- Appearing Obstacle follow-up, decoded_speed_mean, preceding_shared_fraction: median slope -0.1684; 2/9 positive; two-sided Wilcoxon p=0.05469, BH q=0.3117.
- Appearing Obstacle follow-up, lateral_command_abs_mean, preceding_shared_fraction: median slope -0.109; 4/9 positive; two-sided Wilcoxon p=0.6523, BH q=0.7592.
- Appearing Obstacle follow-up, neural_target_evidence, preceding_shared_fraction: median slope -0.08964; 4/9 positive; two-sided Wilcoxon p=0.8203, BH q=0.8613.
- Appearing Obstacle follow-up, success, preceding_shared_fraction: median slope -0.2985; 2/9 positive; two-sided Wilcoxon p=0.07422, BH q=0.3117.
- Fixed Obstacle, angular_error_to_target_deg_mean, preceding_shared_fraction: median slope 13.11; 13/23 positive; two-sided Wilcoxon p=0.6869, BH q=0.7592.
- Fixed Obstacle, command_heading_variability_deg, preceding_shared_fraction: median slope 5.537; 12/23 positive; two-sided Wilcoxon p=0.4634, BH q=0.6487.
- Fixed Obstacle, command_speed_sd, preceding_shared_fraction: median slope 0.1406; 12/23 positive; two-sided Wilcoxon p=0.1511, BH q=0.5064.
- Fixed Obstacle, decoded_speed_mean, preceding_shared_fraction: median slope 0.1578; 12/23 positive; two-sided Wilcoxon p=0.2861, BH q=0.5811.
- Fixed Obstacle, lateral_command_abs_mean, preceding_shared_fraction: median slope 0.3014; 13/23 positive; two-sided Wilcoxon p=0.2112, BH q=0.5543.
- Fixed Obstacle, neural_target_evidence, preceding_shared_fraction: median slope -0.2332; 10/23 positive; two-sided Wilcoxon p=0.3146, BH q=0.5811.
- Fixed Obstacle, success, preceding_shared_fraction: median slope -0.04066; 11/23 positive; two-sided Wilcoxon p=0.9881, BH q=0.9881.

## BCI-only carryover collinearity diagnostic

- Appearing Obstacle, preceding_3_shared_fraction: median |r|=0.068, p90 |r|=0.130, max |r|=0.192; median VIF=1.00, p90 VIF=1.02, max VIF=1.04; median condition number=7.38, max=8.85. Sessions with VIF>5: 0/19; condition number>30: 0/19.
- Appearing Obstacle, preceding_5_shared_fraction: median |r|=0.089, p90 |r|=0.188, max |r|=0.204; median VIF=1.01, p90 VIF=1.04, max VIF=1.04; median condition number=8.34, max=10.51. Sessions with VIF>5: 0/19; condition number>30: 0/19.
- Appearing Obstacle, preceding_shared_fraction: median |r|=0.301, p90 |r|=0.458, max |r|=0.525; median VIF=1.10, p90 VIF=1.27, max VIF=1.38; median condition number=20.05, max=26.69. Sessions with VIF>5: 0/19; condition number>30: 0/19.
- Appearing Obstacle, previous_trial_shared: median |r|=0.079, p90 |r|=0.178, max |r|=0.187; median VIF=1.01, p90 VIF=1.03, max VIF=1.04; median condition number=7.29, max=8.13. Sessions with VIF>5: 0/19; condition number>30: 0/19.
- Appearing Obstacle follow-up, preceding_3_shared_fraction: median |r|=0.069, p90 |r|=0.126, max |r|=0.173; median VIF=1.00, p90 VIF=1.02, max VIF=1.03; median condition number=7.12, max=7.43. Sessions with VIF>5: 0/9; condition number>30: 0/9.
- Appearing Obstacle follow-up, preceding_5_shared_fraction: median |r|=0.099, p90 |r|=0.151, max |r|=0.211; median VIF=1.01, p90 VIF=1.02, max VIF=1.05; median condition number=7.41, max=7.80. Sessions with VIF>5: 0/9; condition number>30: 0/9.
- Appearing Obstacle follow-up, preceding_shared_fraction: median |r|=0.164, p90 |r|=0.386, max |r|=0.422; median VIF=1.03, p90 VIF=1.18, max VIF=1.22; median condition number=22.66, max=39.37. Sessions with VIF>5: 0/9; condition number>30: 2/9.
- Appearing Obstacle follow-up, previous_trial_shared: median |r|=0.035, p90 |r|=0.111, max |r|=0.162; median VIF=1.00, p90 VIF=1.01, max VIF=1.03; median condition number=7.08, max=7.46. Sessions with VIF>5: 0/9; condition number>30: 0/9.
- Fixed Obstacle, preceding_3_shared_fraction: median |r|=0.078, p90 |r|=0.192, max |r|=0.200; median VIF=1.01, p90 VIF=1.04, max VIF=1.04; median condition number=7.42, max=9.77. Sessions with VIF>5: 0/23; condition number>30: 0/23.
- Fixed Obstacle, preceding_5_shared_fraction: median |r|=0.087, p90 |r|=0.187, max |r|=0.235; median VIF=1.01, p90 VIF=1.04, max VIF=1.06; median condition number=8.77, max=12.69. Sessions with VIF>5: 0/23; condition number>30: 0/23.
- Fixed Obstacle, preceding_shared_fraction: median |r|=0.348, p90 |r|=0.494, max |r|=0.632; median VIF=1.14, p90 VIF=1.32, max VIF=1.67; median condition number=21.89, max=26.22. Sessions with VIF>5: 0/23; condition number>30: 0/23.
- Fixed Obstacle, previous_trial_shared: median |r|=0.078, p90 |r|=0.174, max |r|=0.214; median VIF=1.01, p90 VIF=1.03, max VIF=1.05; median condition number=7.38, max=9.23. Sessions with VIF>5: 0/23; condition number>30: 0/23.

## Formal task interaction checks

- command_speed_sd, preceding_shared_fraction, Appearing Obstacle vs Fixed Obstacle: Appearing-minus-comparison median slope difference -0.2866; Mann-Whitney p=0.009949, BH q=0.3748.
- decoded_speed_mean, preceding_shared_fraction, Appearing Obstacle vs Appearing Obstacle follow-up: Appearing-minus-comparison median slope difference 0.6683; Mann-Whitney p=0.01822, BH q=0.3748.
- command_heading_variability_deg, previous_trial_shared, Appearing Obstacle vs Fixed Obstacle: Appearing-minus-comparison median slope difference 2.644; Mann-Whitney p=0.02008, BH q=0.3748.
- success, preceding_shared_fraction, Appearing Obstacle vs Appearing Obstacle follow-up: Appearing-minus-comparison median slope difference 0.5657; Mann-Whitney p=0.03043, BH q=0.4027.
- angular_error_to_target_deg_mean, previous_trial_shared, Appearing Obstacle vs Fixed Obstacle: Appearing-minus-comparison median slope difference 5.56; Mann-Whitney p=0.03595, BH q=0.4027.
- command_speed_sd, preceding_5_shared_fraction, Appearing Obstacle vs Fixed Obstacle: Appearing-minus-comparison median slope difference -0.05875; Mann-Whitney p=0.04589, BH q=0.4283.
- lateral_command_abs_mean, preceding_3_shared_fraction, Appearing Obstacle vs Appearing Obstacle follow-up: Appearing-minus-comparison median slope difference -0.1313; Mann-Whitney p=0.1045, BH q=0.729.
- lateral_command_abs_mean, preceding_3_shared_fraction, Appearing Obstacle vs Fixed Obstacle: Appearing-minus-comparison median slope difference -0.03959; Mann-Whitney p=0.1058, BH q=0.729.
- success, previous_trial_shared, Appearing Obstacle vs Fixed Obstacle: Appearing-minus-comparison median slope difference -0.04128; Mann-Whitney p=0.1172, BH q=0.729.
- lateral_command_abs_mean, previous_trial_shared, Appearing Obstacle vs Fixed Obstacle: Appearing-minus-comparison median slope difference 0.03147; Mann-Whitney p=0.1724, BH q=0.8278.

## Do exposure-associated commands predict BCI-only behavior?

- Appearing Obstacle, angular_error_to_target_deg_mean vs command_speed_sd: median slope -12.92; 5/19 positive; p=0.01597, q=0.0213.
- Appearing Obstacle, angular_error_to_target_deg_mean vs decoded_speed_mean: median slope -4.342; 5/19 positive; p=0.009453, q=0.01418.
- Appearing Obstacle, success vs command_speed_sd: median slope 0.928; 17/19 positive; p=1.907e-05, q=5.722e-05.
- Appearing Obstacle, success vs decoded_speed_mean: median slope 0.3871; 18/19 positive; p=1.144e-05, q=4.578e-05.
- Appearing Obstacle follow-up, angular_error_to_target_deg_mean vs command_speed_sd: median slope -31.67; 0/9 positive; p=0.003906, q=0.006696.
- Appearing Obstacle follow-up, angular_error_to_target_deg_mean vs decoded_speed_mean: median slope -13.31; 2/9 positive; p=0.01953, q=0.02344.
- Appearing Obstacle follow-up, success vs command_speed_sd: median slope 1.089; 9/9 positive; p=0.003906, q=0.006696.
- Appearing Obstacle follow-up, success vs decoded_speed_mean: median slope 0.4237; 9/9 positive; p=0.003906, q=0.006696.
- Fixed Obstacle, angular_error_to_target_deg_mean vs command_speed_sd: median slope 39.06; 13/23 positive; p=0.02767, q=0.03019.
- Fixed Obstacle, angular_error_to_target_deg_mean vs decoded_speed_mean: median slope -30.3; 1/23 positive; p=7.153e-07, q=8.583e-06.
- Fixed Obstacle, success vs decoded_speed_mean: median slope 0.3975; 20/23 positive; p=7.868e-06, q=4.578e-05.

## Suggested reviewer wording

We tested for progressive strategic adaptation by asking whether raw decoded-command metrics changed differently for shared-control versus BCI-only trials over the course of a session, whether BCI-only commands changed after recent shared-control exposure, whether the exposure slopes differed formally across tasks, and whether exposure-associated command metrics predicted BCI-only behavior. Recent one-trial and 3-5-trial exposure effects were weak after multiple-comparison correction, whereas the cumulative original Appearing Obstacle effect on BCI-only command-speed variability remained significant. Thus the data support a cautious carryover/adaptation interpretation on a slower within-session timescale rather than a strong immediate trial-to-trial effect. Formal task-interaction tests did not survive correction, so apparent task specificity should be described as descriptive.
