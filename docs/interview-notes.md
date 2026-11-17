# Interview notes

Short answers in plain words, based on the numbers in `results/`. The data is synthetic with planted shifts, and I say so when asked.

## Drift

**How do you detect drift?**
Three signals. PSI per feature against the training data, PSI on the score distribution, and real performance (AUC, calibration) once labels arrive. The first two need no labels, so they are fast. The third is the only one that tells you the model is actually worse.

**Did feature drift monitoring work?**
It found the planted usage shift immediately: PSI went from 0.007 to 1.017 in the month it happened. But AUC did not drop at all, so in terms of model quality it was a false alarm. And it completely missed the concept shift, because the feature distributions did not change, only what they mean.

**So how did you catch the concept shift?**
With the performance alert on delayed labels. It fired four months after the shift, because labels come a month late and the AUC loss built up slowly until it crossed my 0.03 threshold.

**Why PSI and not KS?**
I compute both. KS gives a p-value, but with 3,000 rows a month it calls tiny differences significant. PSI with a 0.2 threshold is easier to tune and explain. I alert on PSI and keep KS for the report.

**What threshold do you use?**
PSI above 0.2 per feature, PSI above 0.2 on scores, AUC more than 0.03 below the model's own holdout AUC, ECE above 0.05. Prediction PSI peaked at 0.191 even when the churn rate went from 17% to 25%, so the threshold matters a lot.

## Retraining

**When do you retrain?**
In my simulation, monthly retraining with a gate gave the best AUC: 0.815 mean over 18 months, against 0.808 for drift-triggered and 0.792 for never retraining. After the concept shift it was 0.826 vs 0.813 vs 0.780.

**Then why not always retrain monthly?**
Because it cost 17 training runs against 6, and 11 of the 17 were rejected by the gate as no better. Here retraining takes seconds, so monthly is the right call. If each retrain needs GPU hours or a sign-off, drift-triggered is the better trade, and I accept that it reacts late.

**What does drift-triggered get wrong?**
It wasted 5 retrains on the feature shift (all rejected, the model was fine) and it reacted to the concept shift four months late. After it reacted, it was close to monthly: 0.858 vs 0.858 in month 20.

**How do you stop drift alerts from retraining every month?**
When a drift-triggered candidate is rejected, I move the drift reference to the recent window, because the shift was checked and did not need a new model. I do not move the AUC reference, so a real loss still alerts.

## Gates and rollback

**What stops a bad model from going to production?**
A gate with three checks on a holdout month neither model trained on: beat production AUC by at least 0.005, ECE at most 0.05, and the feed passes the schema checks. All three must pass.

**Give an example where the gate mattered.**
A gradient boosting model with balanced class weights had AUC 0.803, within 0.003 of production. Its ECE was 0.214, because class weights inflate the probabilities. It also missed the AUC bar (-0.0025 against +0.005), but only the calibration check shows what actually broke.

**And the schema gate?**
I fed a holdout with 10% unknown contract values. The bad rows get dropped before scoring, so the AUC still looked fine. The schema check caught that 10% of the feed was broken, above my 5% limit.

**How do you roll back?**
Every version stays on disk with its hash. Rollback reads the promotion history from the audit log, puts the previous production version back, archives the current one, and logs the reason. The API picks it up on `/reload`. It is a metadata change, nothing is rebuilt. In the demo I rolled back from v2 to v1 and then promoted v2 again through the gate.

## Packaging

**What goes in a model card?**
The data window (train months and holdout month), holdout metrics (AUC, PR-AUC, Brier, ECE, base rate), the feature list, intended use, what it should not be used for, and known limits, for example "labels arrive one month late, so the newest month is never in training".

**What else is in the package?**
The pipeline artifact with preprocessing inside it, a signature with feature types, ranges and allowed values, and a SHA-256 hash. The hash is checked on every load; a tampered file is refused. The API checks the signature at start-up.

**Why is preprocessing inside the artifact?**
So training and serving cannot encode features differently. That is one of the most common silent bugs in production ML.

## Models

**Why gradient boosting over logistic regression?**
It was about 0.01 AUC better (0.805 vs 0.794 on validation) because the data has threshold effects. Both were well calibrated. I kept the logistic regression as the baseline that a new model has to beat.

**Why not LightGBM?**
sklearn's HistGradientBoosting is the same idea and keeps the install small. For this data size there is no difference worth a dependency.

## Honesty

**What would you not claim from this?**
That monthly retraining is always best. The shifts are planted step changes on synthetic data with one seed. Monthly AUC noise is about ±0.01, so I trust the gap between monthly and never retraining after month 17 (0.03-0.09), not the gap between monthly and drift-triggered retraining from month 20 (0.00-0.02) or the differences up to month 17. The true/false alert counts depend on my own definition of a real loss (0.02 AUC on the next month).
