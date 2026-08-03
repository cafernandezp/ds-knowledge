# Feature Importance in Tree Ensembles — Gain/MDI vs Permutation Importance, and Which Split to Use

> **Problem.** You have a fitted XGBoost or Random Forest classifier (credit default prediction) and want to
> rank input features by importance. Two candidate methods: (1) the model's native **gain**-based (XGBoost) /
> **MDI**-based (Random Forest) importance, and (2) **permutation importance**. Open question: on **which
> dataset** (train / validation / test / OOT) should importance be computed.
> **Assumptions:** binary classification target, `xgboost.XGBClassifier` or `sklearn.RandomForestClassifier`,
> TRAIN/OOT split already in place (per your existing pipeline), scikit-learn ≥1.5, xgboost ≥2.x.

## TL;DR

- **Gain (XGBoost) / MDI (Random Forest)** measure how much a feature reduced *training-time* impurity or
  loss while the trees were being built. They are **free** (already computed during `fit`) but **structurally
  tied to the training set** and **biased toward high-cardinality / continuous features**.
- **Permutation importance** measures the **drop in a chosen metric** when a feature's values are shuffled,
  evaluated on **whatever dataset you pass in**. It is **model-agnostic**, **not free** (re-scores the model
  `n_repeats` times per feature), and its meaning changes with the split you choose.
- **Which split:** compute permutation importance on a **held-out set (validation or OOT)**, never on train
  only, and never on the same test/OOT sample you'll later use to report the final generalization metric if
  importance results will drive feature-selection decisions. Gain/MDI has no "which split" choice — it is
  always training-set derived by construction; you cannot get a held-out version of it from the native API.
- If gain (train) ranks a feature high but permutation on OOT ranks it near zero → treat it as a signal of
  overfitting to that feature, not a contradiction to resolve by trusting gain.
- For regulatory/documentation purposes (credit risk model), permutation importance on OOT + SHAP
  (`TreeExplainer`) is the defensible combination; gain/MDI alone is not sufficient evidence of a feature's
  real contribution.

## Comparison

| | Gain (XGBoost) / MDI (Random Forest) | Permutation importance |
|---|---|---|
| **What it measures** | Impurity/loss reduction attributed to a feature while building trees | Drop in a scoring metric when a feature is shuffled |
| **Dataset used** | Always training-set statistics (no split choice) | Any dataset you pass — train, validation, test, OOT |
| **Model dependency** | Tree-specific (native attribute) | Model-agnostic (any fitted estimator + scorer) |
| **Cost** | Free (byproduct of `fit`) | `O(n_features × n_repeats)` re-scoring passes |
| **Bias** | Toward high-cardinality / continuous features | Diluted by correlated/multicollinear features |
| **Reflects generalization?** | No — training-time only | Yes, if run on held-out data |
| **Typical use** | Fast diagnostic during training, overfitting check | Reportable, defensible importance for interpretation/selection |

## Gain-based (XGBoost) / MDI-based (Random Forest) importance

**What.** XGBoost's default `feature_importances_` uses `importance_type="gain"`: the average gain across all
splits where a feature is used; `total_gain` sums instead of averages, `weight` counts split occurrences, and
`cover` averages the Hessian-weighted number of observations affected<cite index="1-1,7-1">the average gain across splits where the feature was used, and weight counts how many times a feature is used to split data, with gain being the default type for the scikit-learn-style API</cite>. Random Forest's `feature_importances_` is
Mean Decrease in Impurity (MDI): impurity decrease per split, weighted by the fraction of samples reaching
that node, accumulated across all trees<cite index="24-1">feature importances are computed as the mean and standard deviation of accumulated impurity decrease within each tree</cite>.

**Pros.**
- Zero marginal cost — already computed during `fit`.
- Useful as a fast, in-training diagnostic (e.g., spotting a feature the model leans on heavily during fitting).

**Risks.**
- <cite index="21-1">Biased towards high cardinality features, and computed on training set statistics — so importance can be high even for features that are not predictive of the target, as long as the model has the capacity to use them to overfit</cite>.
- <cite index="23-1">Impurity-based feature importance for trees is strongly biased and favors high-cardinality features (typically numerical) over low-cardinality features such as binary or small-category categoricals</cite>.
- `weight`/frequency specifically favors numerical/high-cardinality features over low-cardinality ones like a
  binary flag, even when the binary flag is highly predictive<cite index="6-1">a binary feature can be used at most once per tree while a higher-cardinality feature can appear at multiple levels, so the binary feature gets low importance by frequency/weight but high importance by gain and coverage</cite>.
- No native way to compute this "on a held-out set" — it isn't a scored quantity, it's a property of how the
  fitted trees were built. Don't interpret it as a generalization measure.
- Rankings across `gain`/`weight`/`cover` frequently disagree on the same model<cite index="8-1">running the importance function with gain, weight and cover metrics can yield three different rankings of which features are most important</cite>.

**How / verify.**
```python
import numpy as np
from xgboost import XGBClassifier

# fit on train only
model = XGBClassifier(n_estimators=300, max_depth=4, random_state=42, eval_metric="logloss")
model.fit(X_train, y_train)

# native importance types — all training-set derived, no split argument exists
for imp_type in ["gain", "weight", "cover", "total_gain"]:
    booster = model.get_booster()
    booster.feature_names = list(X_train.columns)
    scores = booster.get_score(importance_type=imp_type)
    top5 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(imp_type, top5)

# Random Forest MDI equivalent
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
mdi = np.argsort(rf.feature_importances_)[::-1][:5]
print(X_train.columns[mdi].tolist())
```

## Permutation importance

**What.** Fit the model once; on a chosen dataset, evaluate a baseline metric, then shuffle one feature's
column (breaking its relationship with the target and other features) and re-evaluate; importance is the
metric drop, averaged over `n_repeats` shuffles<cite index="17-1">the estimator is required to be fitted; a baseline metric is evaluated on a dataset, then a feature column is permuted and the metric evaluated again — the permutation importance is the difference between the baseline metric and the metric after permutation</cite>.

**Pros.**
- Model-agnostic — same recipe for XGBoost, Random Forest, or any other estimator with a `scoring` function.
- <cite index="23-1">Does not exhibit the high-cardinality bias that impurity-based importance has, and can be computed with any performance metric, on any model class</cite>.
- Directly answers "how much does this feature matter for the metric I actually care about" (AUC, logloss,
  KS), not an internal split-quality proxy.

**Risks.**
- <cite index="18-1">Permutation importances can be computed either on the training set or on a held-out testing/validation set; using a held-out set highlights which features contribute most to the generalization power of the model, and features important on the training set but not on the held-out set may indicate the model is overfitting</cite>. Computing it only on train and stopping there under-uses the method.
- Correlated / multicollinear features dilute each other's scores: <cite index="11-1">when a dataset contains multicollinear features, permutation importance can show that none of the features are important, in contradiction with high test accuracy, because permuting one feature has little effect when a correlated feature carries the same information</cite>. This is a distinct, non-additive failure mode — it doesn't mean the feature group is unimportant, it means credit for it is shared.
- Cost scales with `n_features × n_repeats` re-predictions; expensive on large boosted ensembles over large
  OOT samples — subsample or reduce `n_repeats` if this becomes a bottleneck.
- If you use the same test/OOT set both to compute permutation importance for feature-selection decisions
  *and* to report the model's final generalization metric, that final metric is no longer an unbiased estimate
  (classic test-set reuse leakage — not specific to this method, but easy to trigger with it).

**How / verify.**
```python
from sklearn.inspection import permutation_importance

# use a held-out split the model never saw during fit or tuning (validation / OOT), not train
result = permutation_importance(
    model, X_valid, y_valid,
    scoring="roc_auc", n_repeats=20, random_state=42, n_jobs=-1
)
order = result.importances_mean.argsort()[::-1]
for i in order[:10]:
    print(f"{X_valid.columns[i]:25s} {result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")

# optional: same call on X_train, y_train to compare — a feature high on train but ~0 on valid/OOT
# signals overfitting to that feature rather than a real generalization contribution
```

## Which dataset to compute importance on

- **Gain / MDI:** no choice to make — always training-set derived by construction. There is no "test-set
  gain" in the native API; don't try to interpret it as one.
- **Permutation importance — recommended: validation or OOT, not train, and not the final untouched test set
  if the ranking will drive decisions.**
  - **Train:** shows what the model *leaned on to fit* the training data — useful only as a train-vs-holdout
    contrast to flag overfitting, not as a standalone importance ranking.
  - **Validation / OOT (your existing TRAIN/OOT split):** <cite index="18-1">using a held-out set makes it possible to highlight which features contribute the most to the generalization power of the inspected model</cite> — this is what you want for interpretation, documentation, or feature-selection
    calls.
  - **Final test/OOT reserved for reporting:** keep it untouched by any importance-driven decision; if you
    must reuse it, treat any subsequent "final" metric as no longer strictly unbiased.

## Problem-specific considerations (credit default model)

- Given your existing TRAIN/OOT cohort structure: run permutation importance on the **OOT** sample with
  `scoring` matched to your evaluation metric (AUC or KS, not accuracy), not on TRAIN.
- Gain/MDI on TRAIN is still worth keeping as a **cheap diagnostic**: a feature with high train-gain and
  near-zero OOT-permutation importance is a concrete, quantified overfitting/leakage flag — check that
  feature for target leakage or population-instability (PSI) issues before dropping it outright.
- Relative to `mean(|SHAP|)` (already your preferred robustness benchmark over native XGBoost importance):
  the three methods answer different questions and aren't redundant —
  - gain/MDI → training-time split-structure signal (cheap, biased, train-only by construction),
  - permutation importance (on OOT) → generalization-level sensitivity of the chosen metric,
  - SHAP (`TreeExplainer`) → additive, consistent per-prediction attribution, useful for individual credit
    decision explanations, not just global ranking.
  For model-risk documentation (SR 26-2 / EBA-style), permutation importance on OOT + SHAP is the defensible
  pairing; gain/MDI alone should not be presented as evidence of a feature's real contribution.
- High-cardinality categorical features (e.g., product type, branch code) after one-hot encoding will inflate
  `weight` specifically — if you must use a native importance type for such features, prefer `gain`/`total_gain`
  over `weight`, and consider aggregating importance across the dummy columns of the same original variable
  before comparing to continuous features.

## Diagnostics & pitfalls

- **Leakage via test-set reuse:** decide upfront which held-out sample is "for tuning/importance/feature
  selection" (validation) vs. "for final reporting" (OOT/test), and don't let importance-driven decisions
  touch the final reporting sample.
- **Correlated features:** before trusting a low permutation-importance score, check pairwise (Spearman)
  correlation; for correlated groups, either cluster and keep one representative, or permute the group jointly
  and attribute the combined drop to the group rather than each member individually.
- **Sample size / stability:** report `importances_mean ± importances_std` over `n_repeats`, and ideally
  repeat across CV folds — a ranking based on a single pass on a small OOT sample is not stable.
- **Don't mix importance types across features without normalizing intent:** `weight` vs `gain` vs `cover`
  can rank the same model differently; pick one type and state which, rather than eyeballing whichever looks
  most convincing.
- **Impurity-based importance is never a generalization statement:** even after retraining or refitting, MDI
  and gain remain properties of the training fit — a high value never implies the feature will hold up on
  unseen data.

## Decision rule

1. Need a free, in-training diagnostic while iterating on features → use **gain** (XGBoost) or **MDI** (RF)
   on train; treat as provisional, not reportable.
2. Need a defensible ranking of what actually drives out-of-sample performance → use **permutation
   importance on validation/OOT**, scorer = your real evaluation metric.
3. Gain (train) high but permutation (OOT) ~0 for the same feature → investigate overfitting / leakage /
   population instability (PSI) for that feature; don't average the two rankings together.
4. Features are correlated/multicollinear → cluster first, or permute as a group; don't trust per-feature
   permutation scores in isolation.
5. Need per-prediction, additive explanations (individual credit decisions, adverse-action reasons) → neither
   of these; use SHAP `TreeExplainer`.

## References

1. XGBoost Python API Reference — `feature_importances_` / `importance_type`. https://xgboost.readthedocs.io/en/release_1.0.0/python/python_api.html
2. MLJAR — "Xgboost Feature Importance Computed in 3 Ways with Python". https://mljar.com/blog/feature-importance-xgboost/
3. "The Multiple faces of 'Feature importance' in XGBoost". https://medium.com/data-science/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
4. Forecastegy — "How to Get Feature Importance in XGBoost in Python". https://forecastegy.com/posts/xgboost-feature-importance-python/
5. Emily K Marsh — "Calculating XGBoost Feature Importance". https://medium.com/@emilykmarsh/xgboost-feature-importance-233ee27c33a4
6. scikit-learn — `permutation_importance` API reference. https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
7. scikit-learn User Guide — "5.2. Permutation feature importance". https://scikit-learn.org/stable/modules/permutation_importance.html
8. scikit-learn Example — "Permutation Importance vs Random Forest Feature Importance (MDI)". https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
9. scikit-learn Example — "Permutation Importance with Multicollinear or Correlated Features". https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
10. scikit-learn Example — "Feature importances with a forest of trees". https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
