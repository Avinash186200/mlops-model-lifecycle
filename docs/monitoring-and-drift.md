# Monitoring and drift

## What is watched

Each month, after the production model scores the month:

| signal | how | needs labels? | alert when |
|---|---|---|---|
| feature drift | PSI per feature (10 quantile bins from the reference; category shares for categoricals) and two-sample KS for numbers | no | any feature PSI > 0.2 |
| prediction drift | PSI of this month's scores vs the scores on the model's holdout | no | PSI > 0.2 |
| performance | AUC and ECE on the previous month, whose labels just arrived | yes (one month late) | AUC more than 0.03 below the model's holdout AUC, or ECE > 0.05 |

The reference for drift is the model's own training window. A new model brings a new reference.

PSI thresholds follow the usual rule of thumb (under 0.1 stable, 0.1-0.2 watch, over 0.2 act). I report KS as well because it gives a p-value, but with 3,000 rows a month KS flags tiny, harmless differences as significant, so I do not alert on it.

## What it saw (v1, never retrained)

| month | max feature PSI | feature | prediction PSI | AUC on previous month | alerts |
|---|---|---|---|---|---|
| 6-8 | 0.004-0.007 | - | 0.004-0.006 | 0.791-0.805 | none |
| 9 | 1.017 | usage_hours | 0.161 | 0.792 | feature drift |
| 10-14 | 0.904-1.018 | usage_hours | 0.144-0.191 | 0.799-0.819 | feature drift |
| 15-18 | 0.949-1.012 | usage_hours | 0.148-0.179 | 0.777-0.811 | feature drift |
| 19 | 1.015 | usage_hours | 0.156 | 0.772 | feature drift, AUC |
| 20, 23 | 0.905-0.936 | usage_hours | 0.154-0.175 | 0.788-0.797 | feature drift |
| 21-22 | 0.948-1.113 | usage_hours | 0.155-0.186 | 0.767-0.774 | feature drift, AUC |

Full table in [results/RESULTS.md](../results/RESULTS.md).

## What I learned from it

1. **Feature drift found the feature shift at once, and it did not matter.** `usage_hours` PSI went from 0.007 to 1.017 at month 9 (KS 0.016 to 0.410). AUC in months 9-13 was 0.799-0.819, no worse than before. The churn rate rose, but the model already knew low usage means churn.
2. **Prediction drift stayed under the 0.2 threshold** (peak 0.191 at month 14) even though the churn rate went from 17% to 25%. With a lower threshold it would have fired, and also been a false alarm.
3. **The concept shift was invisible to every unlabelled signal.** No feature distribution changed at month 15. Only the performance alert caught it, and only at month 19: labels arrive a month late, and the AUC loss grew gradually (0.784, 0.777, 0.792, 0.772 on months 15-18) until month 18 crossed the 0.03 line.
4. **Calibration** stayed inside 0.05 for v1 in every month (highest 0.037), so the ECE alert never fired in the never-retrain run.

So: unlabelled drift signals are good at catching broken feeds and changes in who your users are. They do not tell you the model is wrong. Only labels do, and labels are late.

## Alert truth

To score alerts I use the simulation's hindsight: an alert at the end of month m is true if the same model loses more than 0.02 AUC on month m+1 versus its holdout AUC. For never-retrain: 6 true, 8 false, 1 unevaluated (month 23). Most "true" ones from month 14 on were fired by usage drift while the real cause was the concept shift, so I would not call them good detections.

## Code

`monitoring.py`: `psi_numeric`, `psi_categorical`, `feature_drift`, `alerts`, `AlertConfig`. The API's SQLite prediction log (`serving.py`) stores features and scores per request, which is what a scheduled job would read to compute the same report on live traffic.
