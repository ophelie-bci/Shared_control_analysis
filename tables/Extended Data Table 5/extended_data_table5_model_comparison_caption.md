# Extended Data Table 5 | Comparison of shared and group-specific models relating baseline BCI performance to shared-control gain

Models were fitted to 255 target × session observations from 51 sessions. The common-shape model included shared linear and quadratic effects of baseline BCI performance with monkey and task intercepts. Monkey-specific and task-specific models additionally allowed the linear and quadratic terms to vary by monkey or task, respectively. Models were compared with likelihood-ratio tests relative to the common-shape model, AIC, BIC and grouped cross-validation by session.

```csv
Model,Shape allowed to vary by,df,AIC,BIC,R²,LR test vs common,Mean CV MSE
Common-shape,—,5,-16.25,4.99,0.195,—,0.05606
Monkey-specific,Monkey,7,-20.29,8.04,0.220,p=0.018,0.05582
Task-specific,Task,9,-18.45,16.96,0.227,p=0.037,0.05544
Monkey × task-specific,Monkey × task,11,-18.76,23.73,0.240,p=0.024,0.05642
```
