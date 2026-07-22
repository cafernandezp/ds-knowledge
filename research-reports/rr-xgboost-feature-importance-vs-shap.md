# XGBoost Feature Importance — Native Metrics vs Mean |SHAP| Values

> **Problem.** Rank/quantify variable importance for a trained XGBoost model.
> **Assumptions.** `booster="gbtree"` (native importance is undefined for `gblinear`); tabular data, possibly with correlated/high-cardinality features; goal is *global* importance (per-variable ranking), not just local explanation.
> **Constraint compared.** `model.feature_importances_` / `Booster.get_score(importance_type=...)` (weight, gain, cover, total_gain, total_cover) vs `mean(|SHAP value|)` per feature (`shap.TreeExplainer`).

## TL;DR

- **Use mean |SHAP| as the default for reporting/production feature importance.** It is the only method with an axiomatic consistency guarantee and it aggregates *local* explanations, so it doubles as per-prediction interpretability with no extra model cost.
- **Native `gain`** is an acceptable, near-zero-cost proxy for quick iteration (feature selection during EDA, CI checks on many models), but it is **not consistency-guaranteed**: adding a redundant/correlated feature can lower another feature's importance even if the model relies on it just as much.
- **Native `weight`** is biased toward high-cardinality / continuous features (more possible split points → more splits) and should not be used as a primary importance metric.
- **`cover`** is rarely the right axis alone; use it as a secondary diagnostic, not a ranking metric.
- SHAP values get **diluted** by correlated/duplicate features under `colsample_bytree < 1.0`, same failure mode as gain-based importance — this is a data problem (multicollinearity), not something either method fixes on its own. Address it with clustering/redundancy checks before trusting either ranking.
- Cost trade-off: SHAP is $O(TLD^2)$ via Tree SHAP (T trees, L leaves, D depth) vs near-free gain/weight/cover lookups — for very large models/datasets, sample rows for SHAP rather than skipping it.

## Comparison table

| Axis | `weight` | `gain` | `cover` | `total_gain` / `total_cover` | mean(&#124;SHAP&#124;) |
|---|---|---|---|---|---|
| Definition | # times feature used to split <cite index="15-1">Weight: This is the simplest type of feature importance. It's just the number of times a feature is used to split the data across all trees.</cite> | Avg loss improvement per split on that feature <cite index="15-1">Gain: This is a bit more complex. Gain measures the average improvement in (training set) loss brought by a feature.</cite> | Avg Hessian (≈ sample weight) covered by splits on that feature <cite index="15-1">Cover: Cover measures the number of data points a given feature affects. This is done via taking the average of the hessian values of all splits the feature is used in.</cite> | Sum instead of mean of gain/cover <cite index="15-1">Total_gain: This is similar to gain, but instead of the average, it gives you the total improvement brought by a feature.</cite> | Mean absolute Shapley value per feature across a dataset (Tree SHAP exact algorithm) |
| Theoretical grounding | Heuristic (split count) | Heuristic (impurity/loss reduction) | Heuristic (sample coverage) | Heuristic | Game-theoretic: unique solution satisfying efficiency, symmetry, dummy, additivity <cite index="4-1">mean(&#124;Tree SHAP&#124;). A global attribution method based on the average magnitude of the individualized Tree SHAP attributions.</cite> |
| Consistency (a feature's importance never decreases when the model relies on it more) | ✗ Not guaranteed | ✗ Not guaranteed <cite index="2-1">Using gain to calculate feature importance leads to a bias towards splits lower in the tree.</cite> | ✗ Not guaranteed | ✗ Not guaranteed | ✓ Proven consistent (Tree SHAP) |
| High-cardinality bias | Strong — <cite index="15-1">In the case of high cardinality categorical features, the 'weight' type of feature importance will be biased towards those features, as they naturally have more splits.</cite> | Present, weaker | Moderate | Same as gain/cover | Reduced (attribution tied to actual output change, not split count) |
| Behavior under `colsample_bytree<1.0` + duplicated/correlated feature | Diluted across duplicates | Diluted, but empirically **does not dilute the same way SHAP does** in some regimes — <cite index="1-1">No dilution occurs because information gain is not shared across trees. SHAP works differently because it considers all trees in the ensemble and distributes feature importance across them.</cite> | Diluted | Diluted | Diluted (shared across duplicate/correlated copies) |
| Local (per-prediction) explanations | No | No | No | No | Yes — same values aggregate to global importance |
| Compute cost | ~free (already computed during training) | ~free | ~free | ~free | $O(TLD^2)$ via Tree SHAP; needs a scoring pass over data |
| Comparable across different models | Poor | Poor | Poor | Poor | Better (same axioms), but still model- and data-dependent |
| Scikit-learn parity | `RandomForestClassifier.feature_importances_` only implements a gain/impurity analogue; **weight/frequency has no scikit-learn equivalent** <cite index="14-1">Looking into the documentation of scikit-lean ensembles, the weight/frequency feature importance is not implemented. This might indicate that this type of feature importance is less indicative of the predictive contribution of a feature for the whole model.</cite> | — | — | — | Model-agnostic (works identically for RF, XGBoost, LightGBM via TreeExplainer) |

## Native XGBoost importance (`weight` / `gain` / `cover` / `total_gain` / `total_cover`)

**What.** Five importance types exposed via `Booster.get_score(importance_type=...)` or `XGBClassifier(importance_type=...).feature_importances_`: <cite index="14-1">Available importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']</cite>. Default in the scikit-learn API is `"gain"` <cite index="11-1">importance_type (string, default "gain") – The feature importance type for the feature_importances_ property: either "gain", "weight", "cover", "total_gain" or "total_cover".</cite>, and only defined for `booster=gbtree` <cite index="11-1">Feature importance is only defined when the decision tree model is chosen as base learner (booster=gbtree). It is not defined for other base learner types, such as linear learners (booster=gblinear).</cite>.

**Pros.**
- Free — already computed as a training byproduct, no extra pass over data.
- `gain`/`total_gain` correlate reasonably well with actual predictive contribution in the common case (no severe multicollinearity, no aggressive column subsampling).
- Good for **fast triage** across many candidate models (e.g., inside a CV loop or AutoML search) where SHAP's cost isn't worth paying yet.

**Risks.**
- **Not axiom-consistent**: <cite index="2-1">Lundberg says, the tree SHAP method is mathematically equivalent to averaging differences in predictions over all possible orderings of the features, rather than just the ordering specified by their position in the tree — [gain does not have this property].</cite> A model can be modified to rely more on feature A, yet A's gain-based importance can go down.
- **`weight` is biased by cardinality/split opportunity**, not by actual predictive value — high-cardinality numeric features rack up splits regardless of usefulness.
- **Instability under correlated/duplicate features and `colsample_bytree<1.0`**: importance mass gets arbitrarily assigned to whichever duplicate the sampler happens to expose first <cite index="1-1">Without feature sampling (colsample_bytree = 1.0), XGBoost consistently picks the first occurrence of a feature for splits, assigning all SHAP values to it. With feature sampling (colsample_bytree < 1.0), some trees will exclude the first occurrence, forcing XGBoost to rely on its duplicate instead.</cite> — this dilution risk applies to gain too, and empirically doesn't always mirror SHAP's dilution pattern.
- The three metrics (`weight`/`gain`/`cover`) can and do **disagree with each other** on ranking, which alone is a signal none of them should be treated as ground truth <cite index="2-1">As can be seen, all three graphs provide three different answers to which features are the most important. This inconsistency across the three metrics is similar to what Scott Lundberg found with his exploration as well.</cite>.
- No per-prediction breakdown — can't explain an individual customer's score.

**How / verify.**
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X, y = make_classification(n_samples=2000, n_features=15, n_informative=6, random_state=42)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss",
)
model.fit(X_train, y_train)

booster = model.get_booster()
for imp_type in ["weight", "gain", "cover", "total_gain", "total_cover"]:
    scores = booster.get_score(importance_type=imp_type)
    top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(imp_type, top3)
```
Verify: rank features under all five types on **train only** (fit on `X_train`); if the top-k set changes materially across types, do not trust a single-metric ranking — escalate to SHAP.

## Mean |SHAP| values

**What.** Per-instance Shapley-value attributions computed exactly and efficiently for tree ensembles via **Tree SHAP** <cite index="4-1">Tree SHAP. A new individualized method we are proposing.</cite>; global importance = mean absolute value per feature over a dataset <cite index="4-1">mean(|Tree SHAP|). A global attribution method based on the average magnitude of the individualized Tree SHAP attributions.</cite>, distinct from drop-column importance <cite index="6-1">it is worth mentioning that the Shapley values are not the difference in the prediction when we remove that feature from the whole feature sets. Thus, its results are different from the drop-column feature importance study conducted in the previous section; the drop-column feature importance is based on the decrease in the model performance while SHAP is based on the magnitude of feature attributions.</cite>.

**Pros.**
- Only method proven both **consistent and locally accurate** among the ones compared <cite index="2-1">Running an experiment he found that only the SHAP value method was both consistent and accurate as it is the only method that does a fair allocation of profits using game theory.</cite>.
- Same computation gives **global ranking + local, per-instance explanation** for free (summary plots, dependence plots, waterfall per row) — useful for stakeholder-facing reports and model audits.
- Model-agnostic implementation (`shap.TreeExplainer` works the same way across XGBoost/LightGBM/CatBoost/RandomForest), so it's the right choice when comparing importance across different model families for the same problem.

**Risks.**
- **Compute cost**: exact Tree SHAP scales with tree count/depth/leaves; on very large datasets or ensembles, computing it on the full dataset can be expensive — <cite index="10-1">The computing feature importances with SHAP can be computationally expensive.</cite>. Mitigate by sampling rows (e.g., 5–20k) rather than the full test set.
- **Correlated/duplicate features still dilute SHAP** — attribution mass splits across the duplicates rather than concentrating on "the" true driver <cite index="1-1">SHAP works differently because it considers all trees in the ensemble and distributes feature importance across them.</cite>. Always screen for near-duplicate / highly correlated columns before trusting a SHAP ranking.
- Interpreting SHAP requires care with **interacting features** and non-additive effects; summary-plot magnitude is not the same as causal effect size.
- Library/version drift: always check the SHAP explainer type matches the booster (TreeExplainer for gbtree; different handling for `gblinear`/DART).

**How / verify.**
```python
import shap

# leakage-safe: fit already done on X_train above; explain on held-out data
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)          # shape: (n_samples, n_features)

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=X_test.columns
).sort_values(ascending=False)

print(mean_abs_shap.head(10))
# shap.summary_plot(shap_values, X_test, plot_type="bar")  # visual check
```
Verify: cross-check `mean_abs_shap` top-k against the native `gain` top-k from the previous section; large disagreement → inspect for correlated features (`X_train.corr()`), and re-derive importance on a de-duplicated feature set.

## Problem-specific considerations

- **Correlated / high-cardinality tabular features (common case here):** both families degrade under multicollinearity; run a correlation/clustering pass (e.g., hierarchical clustering on `1 - |corr|`) and pick one representative per cluster before final importance reporting, regardless of which metric you use.
- **Model comparison across families (XGBoost vs RandomForest vs LightGBM):** prefer SHAP — native `gain`/`weight` scales differ across libraries and aren't directly comparable; SHAP values are.
- **Classification with imbalanced target:** SHAP values respect the model's actual output space (log-odds or probability depending on `link`); confirm which space you're in before comparing magnitudes across models with different objectives.
- **Feature selection at scale (hundreds of models, e.g. hyperparameter search):** use `gain`/`total_gain` for cheap iteration; run SHAP only on the final selected model(s).

## Diagnostics & pitfalls

- **Data leakage:** compute both native importance and SHAP on the trained booster using held-out (`X_test`) data for the *interpretation* step; never fit encoders/scalers on the full dataset before the train/test split — any target-derived encoding must be fit inside CV folds.
- **Train/val/test discipline:** importance computed on training data reflects what the model *fit to*, including noise; when reporting "what drives predictions in the real world," prefer explaining on validation/test data.
- **Metric disagreement is a signal, not noise:** if `weight`, `gain`, and mean |SHAP| disagree materially on the top features, treat that as evidence of instability (correlated features, insufficient regularization, or `colsample_bytree` too aggressive) rather than picking whichever ranking is convenient.
- **Sampling for SHAP cost control:** subsample rows for `shap_values` computation on very large test sets; take a stratified sample if the target is imbalanced.
- **Don't conflate importance with causality:** both methods describe the *model's* behavior, not necessarily the true causal driver in the underlying process.

## Decision rule

1. Need per-prediction explanations (compliance, customer-facing "why this score") → **mean(|SHAP|)**, no alternative.
2. Need a fast, cheap ranking during iterative model/feature search (many models trained) → **`gain`/`total_gain`** as a first pass.
3. Final reported/production importance ranking, or comparing importance across model families → **mean(|SHAP|)**.
4. Any correlated/duplicate/high-cardinality features present → run a correlation/redundancy check **before** trusting either method; de-duplicate features first.
5. `weight` alone as the primary metric → avoid; use only as a secondary "how often is this feature touched" diagnostic.

## References

1. Feature Importance Dilution in XGBoost: Just Like SHAP or Different? — https://medium.com/@xwang222/feature-importance-dilution-in-xgboost-just-like-shap-or-different-274714987aea
2. Calculating XGBoost Feature Importance — https://medium.com/@emilykmarsh/xgboost-feature-importance-233ee27c33a4
3. Feature Importance and Feature Selection With XGBoost in Python — MachineLearningMastery.com — https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
4. Interpretable Machine Learning with XGBoost (Scott Lundberg) — https://medium.com/data-science/interpretable-machine-learning-with-xgboost-9ec80d148d27
5. Xgboost Feature Importance Computed in 3 Ways with Python — https://mljar.com/blog/feature-importance-xgboost/
6. MechProNet: Machine Learning Prediction of Mechanical Properties in Metal Additive Manufacturing — https://arxiv.org/pdf/2209.12605
7. The Multiple faces of 'Feature importance' in XGBoost — https://medium.com/data-science/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
8. How to Get Feature Importance in XGBoost in Python — Forecastegy — https://forecastegy.com/posts/xgboost-feature-importance-python/
9. Python API Reference — XGBoost documentation (importance_type) — https://xgboost.readthedocs.io/en/release_1.0.0/python/python_api.html
