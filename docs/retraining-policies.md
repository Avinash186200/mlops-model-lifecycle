# Retraining policies

## Setup

- v1 (gradient boosting) is trained on months 0-4, checked on month 5, and serves from month 6. 18 months are served (6-23).
- At the end of month m: monitoring runs (see [monitoring-and-drift.md](monitoring-and-drift.md)); a policy may train a candidate on months m-6 to m-2 and gate it on month m-1 against production. A promoted candidate serves from month m+1.
- Each policy has its own registry under `artifacts/sim/<policy>/`, so every promotion and rejection goes through the real `Registry.promote` code and is in an audit log (also saved in `results/retraining.json`).

| policy | rule |
|---|---|
| never | keep v1. Monitoring still runs and alerts are counted. |
| monthly | train a candidate every month; the gate decides. |
| drift | train a candidate only in months with an alert. If it is rejected, move the drift reference to the recent window (the shift was looked at and did not need a new model). The AUC reference is not moved, so a real loss still alerts. |

## Result

| policy | mean AUC | months 6-14 | months 15-23 | min AUC | mean ECE | retrains | promoted | alert months | true | false | missed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| never | 0.792 | 0.804 | 0.780 | 0.766 | 0.022 | 0 | 0 | 15 | 6 | 8 | 0 |
| monthly | 0.815 | 0.804 | 0.826 | 0.781 | 0.019 | 17 | 6 | 6 | 1 | 5 | 2 |
| drift | 0.808 | 0.804 | 0.813 | 0.772 | 0.020 | 6 | 1 | 6 | 1 | 5 | 3 |

(`results/retraining.json` counts the initial v1 too, so it shows trainings 1 / 18 / 7 and promotions 1 / 7 / 2.)

AUC by month for the months after the concept shift:

| month | never | monthly | drift |
|---|---|---|---|
| 15 | 0.784 | 0.784 | 0.784 |
| 16 | 0.777 | 0.781 | 0.777 |
| 17 | 0.792 | 0.796 | 0.792 |
| 18 | 0.772 | 0.804 | 0.772 |
| 19 | 0.797 | 0.845 | 0.797 |
| 20 | 0.774 | 0.858 | 0.858 |
| 21 | 0.767 | 0.857 | 0.847 |
| 22 | 0.788 | 0.859 | 0.850 |
| 23 | 0.766 | 0.854 | 0.839 |

## Reading it

- **Months 6-14 are identical for all three.** Nothing promoted before month 14 because nothing beat v1. Monthly promoted at the end of month 14 (gain 0.0065), which affects months 15 onward.
- **Monthly adapted first.** Its training window started to include post-shift months at month 17, and from then it promoted every month through 21 (gains 0.036, 0.023, 0.020, 0.006, 0.005). 11 of its 17 candidates were rejected as no better.
- **Drift-triggered wasted 5 retrains on the feature shift** (months 9-13, all rejected) and then reacted to the concept shift only at month 19 through the performance alert, when it promoted a model with +0.0745 AUC on the holdout. From month 20 it is close to monthly.
- **Never retraining cost 0.06-0.09 AUC** in months 20-23 compared with either of the other two.

## What I would choose, and when

- **Retraining is cheap and automated (this case):** monthly retrain with the gate. The gate is the safety, the schedule removes the detection lag. Drift monitoring still runs, to catch broken feeds and to explain changes.
- **Retraining is expensive or needs human sign-off:** drift-triggered, but with the performance alert as the main trigger and feature drift as a prompt to look, not to retrain. Accept a lag of about label delay + the time for the loss to cross the threshold (4 months here).
- **Either way:** the gate is the part that matters most. In this run it blocked 16 candidates that were no better, plus the three bad ones in the registry demo.

## Caveats

- One seed, one synthetic dataset, planted step changes. Monthly AUC noise is about ±0.01.
- The 5-month training window is a choice. A shorter window would adapt faster to the concept shift and be noisier before it.
- I did not try sample weighting toward recent months, or a policy that retrains on a performance alert only. Both are cheap to add in `simulate.py`.
