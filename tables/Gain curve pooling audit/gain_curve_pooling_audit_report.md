# Gain Curve Pooling Audit

Rows are target x session observations matching the reviewer-requested variables: BCI-only success and shared-control gain. The table contains 255 observations from 51 sessions.
The rows retain both the manuscript-facing monkey label and the source data folder so the animal/session mapping is auditable.

## Main Model Check

The shared-shape quadratic model included common linear/quadratic baseline terms plus monkey and task intercepts. Its clustered quadratic coefficient was -0.977 (p=1.45e-06).
Allowing monkey-specific linear/quadratic terms gave LR p=0.018; allowing task-specific terms gave LR p=0.0372.
AIC was lowest for monkey_specific_quadratic; BIC was lowest for common_quadratic_monkey_task_intercepts. Cross-validated MSEs were very similar across the candidate models.
The binomial/log-odds robustness check found a significant shared-control x baseline interaction but did not retain a significant quadratic shared-control x baseline^2 term, consistent with ceiling/mathematical-coupling concerns on the raw gain scale.

## Coverage By Baseline Range

```csv
baseline_bin,Monkey 1_Appearing Obstacle,Monkey 1_Appearing Obstacle follow-up,Monkey 1_Fixed Obstacle,Monkey 2_Appearing Obstacle,Monkey 2_Fixed Obstacle
0-20%,17,6,10,26,19
20-40%,10,5,10,10,18
40-60%,13,14,20,7,12
60-80%,6,14,15,2,3
80-100%,4,6,5,0,3
```

## Baseline Distribution

```csv
monkey,task,n_observations,min_baseline,median_baseline,max_baseline,mean_gain
Monkey 1,Appearing Obstacle,50,0.0,0.3333333333333333,1.0,0.2963607503607503
Monkey 1,Appearing Obstacle follow-up,45,0.0,0.5714285714285714,1.0,0.24516372516372517
Monkey 1,Fixed Obstacle,60,0.0,0.5,0.9090909090909091,0.3184163059163059
Monkey 2,Appearing Obstacle,45,0.0,0.16666666666666666,0.6666666666666666,0.33609587942921276
Monkey 2,Fixed Obstacle,55,0.0,0.3333333333333333,0.9,0.2712075298438935
```

## Separate Quadratic Fits

```csv
grouping,n,quadratic_beta,quadratic_p_clustered,linear_beta,r2,monkey,task
monkey,155,-0.5830735832205328,0.043936025237353224,-0.43098761902936245,0.3051703240192295,Monkey 1,
monkey,100,-1.2632891835618167,0.00016267515675221392,-0.1516754134798703,0.0927693442169577,Monkey 2,
monkey x task,50,-0.4901466506419867,0.3025879364694517,-0.32530065139227116,0.1573486849981397,Monkey 1,Appearing Obstacle
monkey x task,45,0.03694180606314244,0.9277211163284841,-0.7905270292722478,0.5589358152926589,Monkey 1,Appearing Obstacle follow-up
monkey x task,60,-0.7016209362957821,0.007591450433967722,-0.39928013519162786,0.33688450456515096,Monkey 1,Fixed Obstacle
monkey x task,45,-2.1690094563584763,0.059776457066069995,-0.33436148949303446,0.10931083836636324,Monkey 2,Appearing Obstacle
monkey x task,55,-1.2339829189123706,0.00010659908023752782,-0.07900831297403166,0.12900347819135338,Monkey 2,Fixed Obstacle
```

## Nested Model Comparison

```csv
model,n,df_model,aic,bic,r2,quadratic_beta,quadratic_p_clustered,lr_vs_common_stat,lr_vs_common_df,lr_vs_common_p
common_quadratic_monkey_task_intercepts,255,5.0,-16.2548889426634,4.992692328287156,0.1951721157184102,-0.9773388211753626,1.4489977286732445e-06,,,
monkey_specific_quadratic,255,7.0,-20.291775562880673,8.038332798386733,0.220142474190965,-0.6010857909227072,0.03878394476023474,8.036886620217274,2.0,0.017980933902397026
task_specific_quadratic,255,9.0,-18.452657450362665,16.959978001221593,0.2267231015689205,-1.0391900128905074,0.002774487158470698,10.197768507699266,4.0,0.03722486196276369
full_monkey_task_shape,255,11.0,-18.76483919763865,23.73032334426246,0.2396896687663992,-0.7058421046777428,0.09475684145654044,14.509950254975251,6.0,0.024430519319767252
```

## Cross-Validated Error

```csv
model,n_folds,mean_cv_mse,median_cv_mse
common_quadratic,5,0.056063231076336775,0.05364860730874514
monkey_specific,5,0.0558203440588243,0.05049474729119402
task_specific,5,0.055437030423834334,0.05587068912737106
```

## Binomial/Log-Odds Robustness

```csv
model,term,beta,se_clustered,p_clustered
binomial_logit_success_rate,shared_control:x_centered,-1.8433461057620324,0.3224000139391519,1.0805529125191236e-08
binomial_logit_success_rate,shared_control:x_centered2,-1.5086749511065305,1.175348966111926,0.19928283788241485
```

## Figure

figures\Extended Data Figure 3\Gain curve pooling audit\gain_curve_by_monkey_task.svg
