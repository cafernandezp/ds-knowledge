# Tree Ensembles vs Linear Models — Structural Diagnostics (a priori) and Test-Set Significance (a posteriori)

> **Problem.** Binary PD model. Target = default within 18 months of credit opening. Train: 50,000 accounts, 24% default rate (≈12,000 events). Test: 15,000 accounts, 24% default (≈3,600 events / 11,400 non-events). Candidate families: logistic regression (raw, WoE-binned, or spline-expanded), Random Forest, XGBoost. Primary metric: AUC / Gini.
>
> **Two questions, deliberately separated.**
> - **Group 1 — structural (a priori):** does the data-generating process contain non-additive structure that a tree ensemble can exploit and an additive model cannot? Answered on **train + cross-validation**. Never on test.
> - **Group 2 — inferential (a posteriori):** given two trained models and one held-out sample, is the observed metric gap distinguishable from sampling noise? Answered on **test**, once.
>
> **Assumptions.** Tens of features (bureau + application), not thousands. Missing values encoded as `-1` in the source data and requiring explicit treatment. No severe train/test temporal drift (if there is, see §7.5 — that is a third, separate question). No cost matrix specified, so ranking metrics are primary and calibration is secondary but reported.

---

## TL;DR / Recommendation

1. **The two questions consume different data and cannot substitute for each other.** Group 1 bounds *what is achievable*; Group 2 bounds *what is measurable*. Running Group 2 without Group 1 wastes the test set on a hypothesis that may already be dead.
2. **The highest-value Group-1 procedure is the nested complexity ladder** (L0 linear → L1 additive-nonlinear → L2 pairwise → L3 unrestricted) fitted under one shared CV. `ΔAUC(L1 → L3)` is a direct **upper bound** on what any tree ensemble can add over a well-specified additive model. If it is below your relevance margin, the decision is made without touching test.
3. **Prefer exact constraints over post-hoc interaction metrics.** `interaction_cst="no_interactions"` (scikit-learn `HistGradientBoosting*`) and `interaction_constraints` (XGBoost) enforce additivity *by construction* [1][2][4], so the gap they produce is a clean bound. Friedman's H-statistic and SHAP interaction values *describe* a fitted model; they do not bound achievable performance.
4. **Every Group-2 test must be paired.** Both models score the same 15,000 accounts; their errors are correlated (ρ ≈ 0.85–0.95). Comparing two independent confidence intervals discards the covariance term and roughly halves effective power.
5. **Know your resolution before you test.** At AUC ≈ 0.75 with m=3,600 / n=11,400, `SE(AUC) ≈ 0.0051` (Hanley–McNeil) and `SE(ΔAUC) ≈ 0.0023–0.0029` paired. **Minimum detectable difference at 80% power ≈ 0.006–0.008 AUC ≈ 1.3–1.6 Gini points.** Smaller true gaps are invisible in this test set regardless of method.
6. **AUC is the least sensitive metric available to you.** Paired tests on **per-observation log-loss or Brier** have materially more power because the loss decomposes per observation while AUC does not.
7. **To claim equivalence, run TOST, not a significance test.** `p > 0.05` is not evidence of no difference; a two-one-sided-test against a pre-specified margin δ is.

---

## Comparison table — all procedures

| # | Procedure | Group | Data used | Answers | Cost | Bounds achievable gain? |
|---|---|---|---|---|---|---|
| A1 | Nested complexity ladder L0–L3 | 1 | Train + CV | Where does the gain live: curvature, pairs, or higher order? | Med | **Yes** |
| A2 | Exact interaction constraints | 1 | Train + CV | Contribution of interactions, isolated by construction | Low | **Yes** |
| A3 | Depth / stump sensitivity | 1 | Train + CV | Cheap proxy for A2 | Very low | Approximately |
| A4 | EBM vs GA2M (`interactions=0` vs pairs) | 1 | Train + CV | Same, with an interpretable model | Med | Yes (pairwise only) |
| A5 | Friedman H-statistic | 1 | Train/CV | *Which* pairs interact, and how strongly | High | No |
| A6 | SHAP interaction values (TreeSHAP) | 1 | Train/CV | Share of attribution carried by off-diagonal terms | Med–High | No |
| A7 | Univariate shape diagnostics (WoE/IV, spline LRT) | 1 | Train + CV | Is non-linearity needed at all (L0 → L1)? | Low | Partially |
| A8 | Learning curve | 1 | Train + CV | Is the binding constraint sample size or signal? | Med | No (orthogonal axis) |
| A9 | Zero-fit structural priors | 1 | Schema only | Rough prior on tree advantage before any fitting | Nil | No |
| B1 | DeLong paired test | 2 | Test | Is ΔAUC ≠ 0? | Low | — |
| B2 | Paired bootstrap | 2 | Test | Any metric (KS, capture rate, segment Gini, profit) | Med | — |
| B3 | Paired t / Wilcoxon on per-obs losses | 2 | Test | Is Δ log-loss / Δ Brier ≠ 0? (highest power) | Very low | — |
| B4 | McNemar at a fixed cutoff | 2 | Test | Does the accept/reject decision change? | Very low | — |
| B5 | TOST equivalence | 2 | Test | Can we *assert* the models are equivalent within δ? | Low | — |
| B6 | Multi-seed retraining envelope | 2 | Train (repeat) | How much of Δ is training randomness, not signal? | High | — |
| B7 | 5×2cv paired t-test (Dietterich) | 2 | CV (not test) | Algorithm-level comparison — **wrong tool here** (§B7) | High | — |

---

# Part A — Group 1: structural diagnostics (a priori)

The organising idea: these model families are **nested in structure**, not arbitrary rivals.

```
L0   logit(p) = β₀ + Σ βⱼ xⱼ                                 linear, additive
L1   logit(p) = β₀ + Σ fⱼ(xⱼ)                                 non-linear, additive        ← WoE / splines / GAM
L2   logit(p) = β₀ + Σ fⱼ(xⱼ) + Σ fⱼₖ(xⱼ, xₖ)                 + pairwise interactions     ← GA2M / pairwise-constrained GB
L3   logit(p) = f(x)                                          unrestricted                ← XGBoost, Random Forest
```

- `ΔAUC(L0 → L1)` = **curvature**. Recoverable *inside* the logistic family (WoE binning, splines). Does not justify trees.
- `ΔAUC(L1 → L2)` = **pairwise interaction**. This is where trees start to earn their place.
- `ΔAUC(L2 → L3)` = **higher-order interaction**. Rare in application/bureau credit data.
- `ΔAUC(L1 → L3)` = **the total ceiling for tree ensembles** over a correctly specified additive model.

---

## A1. Nested complexity ladder under one shared CV  *(the backbone)*

**What.** Fit all four levels under a single `StratifiedKFold` object with an identical preprocessing pipeline, and read off where the incremental AUC lives.

**Pros.**
- Directly produces the quantity that matters (`ΔAUC(L1→L3)`) rather than a proxy.
- Decomposes the gain into interpretable components (curvature vs interaction vs higher order), so the remediation is obvious: curvature → fix the logistic spec; interaction → move to trees.
- Uses no test data, so the result is free of the selection bias that contaminates test-set model shopping.
- Sits naturally inside the assignment requirement to "try 2 different modelling techniques" — the ladder *justifies* which 2.

**Risks.**
- **Leakage is the dominant failure mode.** A WoE table or target encoder fitted outside the CV loop inflates L1 and produces a false verdict of additivity. Every `y`-derived transform must live inside a `Pipeline` refit per fold.
- Unequal tuning effort across levels invalidates the comparison. L3 tuned with 200 trials against an untuned L1 measures your search budget, not structure.
- L1 realised as splines vs WoE can differ by a few tenths of a Gini point; report the stronger L1, otherwise you overstate the tree gain.
- CV fold-to-fold `std` is not a confidence interval — folds share training data and the scores are correlated.

**How / verify.**

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ladder = {
    # L0: linear, additive
    "L0_linear": make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000)),
    # L1a: non-linear, additive (spline-expanded logistic == GAM)
    "L1_spline": make_pipeline(SplineTransformer(n_knots=5, degree=3),
                               LogisticRegression(max_iter=5000, C=1.0)),
    # L1b: non-linear, additive (boosting with interactions switched off)
    "L1_additive_gb": HistGradientBoostingClassifier(
        interaction_cst="no_interactions", random_state=42),
    # L2: + pairwise interactions only
    "L2_pairwise_gb": HistGradientBoostingClassifier(
        interaction_cst="pairwise", random_state=42),
    # L3: unrestricted
    "L3_full_gb": HistGradientBoostingClassifier(random_state=42),
}

scores = {}
for name, model in ladder.items():
    s = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    scores[name] = s.mean()
    print(f"{name:16s} AUC {s.mean():.4f} ± {s.std():.4f}")

best_L1 = max(scores["L1_spline"], scores["L1_additive_gb"])
print(f"curvature   ΔAUC(L0→L1) = {best_L1 - scores['L0_linear']:+.4f}")
print(f"pairwise    ΔAUC(L1→L2) = {scores['L2_pairwise_gb'] - best_L1:+.4f}")
print(f"higher-ord. ΔAUC(L2→L3) = {scores['L3_full_gb'] - scores['L2_pairwise_gb']:+.4f}")
print(f"TREE CEILING ΔAUC(L1→L3) = {scores['L3_full_gb'] - best_L1:+.4f}")
```

Reference run on a synthetic set with strong planted interactions (6,000 rows, 12 features, 24% positives) — the ladder separates cleanly when interactions genuinely exist:

```
L0_linear        0.9130 ± 0.0070
L1_spline        0.9330 ± 0.0082
L1_additive_gb   0.9088 ± 0.0078
L2_pairwise_gb   0.9606 ± 0.0088     ← +0.028 over best L1: interactions are real
L3_full_gb       0.9791 ± 0.0039     ← +0.019 more: higher-order too
```

A credit portfolio typically produces the opposite pattern: a large `L0 → L1` step and a `L1 → L3` step under 0.01.

---

## A2. Exact interaction constraints  *(the clean bound)*

**What.** Restrict the learner so that no tree branch may split on more than one feature group. With singleton groups the boosted ensemble **is** an additive model by construction. Available as `interaction_cst={"no_interactions","pairwise"}` in scikit-learn's `HistGradientBoosting*` (added in 1.2) [1][2] and as `interaction_constraints` (nested list of allowed groups) in XGBoost, which requires `tree_method` set to `exact` or `hist` [4].

**Pros.**
- Holds the learner, loss, regularisation, binning and missing-value handling **fixed**, varying only the interaction budget. The resulting Δ is attributable to interactions alone — not to a change of algorithm.
- Removes the "maybe the logistic was just badly specified" objection, because the additive comparator is itself a boosted ensemble.
- Handles `-1` sentinels and missingness natively on both sides of the comparison, so the missingness advantage of trees does not contaminate the interaction estimate.

**Risks.**
- Constrained and unconstrained models need separate hyperparameter budgets; the constrained one usually wants more iterations to reach its own ceiling.
- XGBoost and other implementations differ in how they evaluate **overlapping** constraint groups along a branch — XGBoost permits some paths that a stricter reading would forbid [5]. Use disjoint groups (or the singleton/`no_interactions` case) to avoid ambiguity.
- `pairwise` at high dimensionality is expensive and can overfit; regularise it like any other level.

**How / verify.**

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p = X_train.shape[1]

common = dict(tree_method="hist", n_estimators=800, learning_rate=0.03,
              max_depth=4, subsample=0.8, colsample_bytree=0.8,
              eval_metric="logloss", random_state=42)

# additive by construction: every feature is its own interaction group
additive = XGBClassifier(interaction_constraints=[[j] for j in range(p)], **common)
full     = XGBClassifier(**common)

a = cross_val_score(additive, X_train, y_train, cv=cv, scoring="roc_auc").mean()
f = cross_val_score(full,     X_train, y_train, cv=cv, scoring="roc_auc").mean()
print(f"additive {a:.4f} | unrestricted {f:.4f} | interaction gain {f-a:+.4f}")
```

**Decision:** if `f - a` is below your pre-declared relevance margin (see §7.2), trees have nothing structural to offer and the choice reverts to interpretability and stability.

---

## A3. Depth / stump sensitivity  *(cheapest proxy)*

**What.** Sweep `max_depth ∈ {1, 2, 3, 4, 6, 8}` under one CV. Boosting with depth-1 stumps is exactly additive; depth-*d* admits interactions up to order *d*.

**Pros.**
- One loop, no extra dependencies, no explainability library.
- The depth at which the CV curve plateaus is a direct read on the effective interaction order in the data.
- Doubles as hyperparameter tuning, so the cost is already sunk.

**Risks.**
- Depth confounds interaction order with **capacity**: a deeper tree is also a more flexible univariate learner, so `Δ(depth 1 → 6)` slightly overstates the interaction contribution. A2 does not have this defect — prefer A2 when both are available.
- Depth is not comparable across implementations that grow leaf-wise (`max_leaf_nodes`) rather than depth-wise.
- Needs `n_estimators` and `learning_rate` re-tuned per depth, or shallow models are unfairly handicapped.

**How / verify.**

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for d in [1, 2, 3, 4, 6, 8]:
    m = XGBClassifier(max_depth=d, n_estimators=1200, learning_rate=0.03,
                      tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", random_state=42)
    s = cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"depth={d}  AUC {s.mean():.4f} ± {s.std():.4f}")
```

Reading: `depth=1 ≈ depth=6` ⇒ additive data. A plateau at `depth=2–3` ⇒ pairwise structure only, which a GA2M can capture while staying interpretable.

---

## A4. EBM vs GA2M  *(interpretable ladder)*

**What.** InterpretML's Explainable Boosting Machine fits `g(E[y]) = β₀ + Σ fᵢ(xᵢ) + Σ fᵢⱼ(xᵢ,xⱼ)` — a GA2M, cyclic-boosted one feature at a time with a low learning rate [3]. Setting `interactions=0` yields a pure GAM; the default (`"3x"` for the classifier) auto-selects pairwise terms [6].

**Pros.**
- The additive/pairwise split is a single parameter, so the A/B is trivial.
- Every term is plottable, which is exactly what step 6 of the assignment asks for ("describe the relationship between this variable and the target").
- Round-robin boosting over one feature at a time mitigates co-linearity effects in the shape functions [3] — useful given how correlated bureau features are.
- If EBM(pairwise) ≈ XGBoost, you can ship an interpretable model at no measured cost in discrimination. That is a strong, defensible result in a regulated setting.

**Risks.**
- Extra dependency (`interpret`), slower to fit than `HistGradientBoosting`.
- Captures pairwise terms only by default; it cannot rule out higher-order structure — pair it with A2/A3 for the L2 → L3 step.
- Auto-selected interaction pairs are chosen on the training fold; keep the whole fit inside the CV loop.

**How / verify.**

```python
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gam  = ExplainableBoostingClassifier(interactions=0, random_state=42)   # pure additive
ga2m = ExplainableBoostingClassifier(interactions="3x", random_state=42)  # + pairs

g  = cross_val_score(gam,  X_train, y_train, cv=cv, scoring="roc_auc").mean()
g2 = cross_val_score(ga2m, X_train, y_train, cv=cv, scoring="roc_auc").mean()
print(f"GAM {g:.4f} | GA2M {g2:.4f} | pairwise gain {g2-g:+.4f}")
```

---

## A5. Friedman H-statistic

**What.** For a fitted flexible model, the share of the joint partial-dependence variation not explained by the two marginal partial dependences:

```
H²ⱼₖ = Σᵢ [ PDⱼₖ(xᵢⱼ, xᵢₖ) − PDⱼ(xᵢⱼ) − PDₖ(xᵢₖ) ]²  /  Σᵢ PDⱼₖ(xᵢⱼ, xᵢₖ)²
```

**Pros.**
- Localises *which* feature pairs interact, which A1–A3 cannot. This is what turns "trees help" into an actionable feature-engineering brief (e.g. build an explicit `utilisation × tenure` term and keep the logistic model).
- Dimensionless and comparable across pairs.

**Risks.**
- **O(p²) partial-dependence evaluations** — expensive at 40 features, prohibitive beyond.
- Partial dependence assumes feature independence when marginalising; with correlated bureau variables it extrapolates into regions with no data and H² becomes unstable/inflated.
- Unstable denominator when the joint PD variation is near zero, producing spuriously large H² for irrelevant pairs.
- **Descriptive only.** A large H² on a fitted XGBoost does not prove the interaction generalises — it may be memorisation. Always confirm with the held-out Δ from A2.

**How / verify.** Compute from `sklearn.inspection.partial_dependence` on a subsample, then confirm the top-ranked pairs by re-running A2 with those pairs allowed and everything else additive. If allowing exactly those pairs recovers most of `ΔAUC(L1→L3)`, the finding is real and you can encode it as an explicit feature.

---

## A6. SHAP interaction values (TreeSHAP)

**What.** For tree models, an exact per-sample matrix of shape `(n_samples, n_features, n_features)` whose diagonal holds main effects and whose symmetric off-diagonal entries hold pairwise interaction effects; each sample's matrix sums to the model output minus the expected value [7][8].

**Pros.**
- Exact for tree ensembles and far cheaper than model-agnostic Shapley interaction estimation.
- Attribution lives in the additive **log-odds (margin) space** for binary XGBoost [7], which is exactly the scale where "is the model additive?" is the right question.
- Gives a single scalar summary — off-diagonal mass / total mass — that is easy to report to a risk committee.
- When no interactions exist, the interaction matrix collapses to a diagonal, so the diagnostic degenerates gracefully [8].

**Risks.**
- Same fundamental limitation as A5: it describes a *fitted* model, not the data-generating process. An overfit XGBoost will show interaction mass that does not replicate out-of-fold.
- Correlated features force a choice between `tree_path_dependent` and `interventional` handling of dependence; the two give different attributions and the interventional variant needs a background sample [7].
- Cost is quadratic in features; subsample rows.

**How / verify.**

```python
import numpy as np, shap
from xgboost import XGBClassifier

model = XGBClassifier(tree_method="hist", n_estimators=600, learning_rate=0.03,
                      max_depth=4, eval_metric="logloss", random_state=42)
model.fit(X_tr, y_tr)                       # inner training fold only

sub = X_val.sample(2000, random_state=42)   # held-out fold
iv = shap.TreeExplainer(model).shap_interaction_values(sub)   # (n, p, p)

absiv = np.abs(iv)
diag = np.trace(absiv, axis1=1, axis2=2).sum()
share_interaction = 1.0 - diag / absiv.sum()
print(f"share of |attribution| carried by interactions: {share_interaction:.1%}")
```

Rule of thumb: **< 5–10% ⇒ effectively additive**, and the ladder should confirm a negligible `ΔAUC(L1→L3)`. Compute it on a **held-out fold**, never on the training rows.

---

## A7. Univariate shape diagnostics — WoE/IV, spline LRT, Box–Tidwell

**What.** Tests of whether `logit(p)` is linear in each `xⱼ`, i.e. whether the `L0 → L1` step is needed. WoE binning per feature; a likelihood-ratio test of a spline-expanded logistic against the linear one, `2(ℓ_spline − ℓ_linear) ~ χ²_df`; Box–Tidwell for log-linearity of continuous predictors.

**Pros.**
- Cheap, per-feature, and directly actionable: the fix stays inside the logistic family.
- **Often the entire story in credit risk.** Most of the apparent "tree advantage" in naive comparisons is curvature that a WoE-transformed logistic absorbs completely.
- IV doubles as a feature screen, and WoE handles the `-1` sentinel cleanly as its own bin — which is itself informative, since missing bureau data is rarely missing at random.

**Risks.**
- **WoE is a target-derived transform: high leakage risk.** It must be fitted inside each CV fold. A globally-fitted WoE table makes L1 look artificially strong and biases the whole ladder toward "no trees needed".
- Over-binning memorises noise; enforce a minimum bin population (e.g. ≥5% of rows or ≥50 events) and monotonicity where the risk logic demands it.
- The LRT is anti-conservative if features were pre-selected on the same data.
- Silent on interactions — this diagnostic cannot answer the tree question, only the curvature question.

**How / verify.** Fit WoE inside a `Pipeline`; compare CV AUC of `logistic(raw)` vs `logistic(WoE)` vs `logistic(splines)`. Plot event rate by decile per feature and check that the WoE curve reproduces the shape. Any feature whose fitted shape is non-monotonic without a business rationale is a candidate for a monotonic constraint rather than extra flexibility.

---

## A8. Learning curve — a different axis

**What.** CV AUC as a function of training subsample size, for the additive model and the unrestricted model on the same axes.

**Pros.**
- Separates two failure modes that look identical in a single-point comparison: *no signal to gain* vs *not enough data yet to estimate the gain*.
- Directly informative for the roadmap: if L3 is still rising at 50,000 while L1 has flattened, more acquisition data (or a wider origination window) will eventually create the gap that is absent today.

**Risks.**
- Does not diagnose structure. A flat curve is consistent with both "additive data" and "capacity-limited model" — always read it alongside A1–A3.
- Subsampling must preserve the 24% event rate (stratify) and any temporal structure.
- Expensive: full refits at every sample size.

**How / verify.** `sklearn.model_selection.learning_curve` with `scoring="roc_auc"`, `cv=StratifiedKFold(5, shuffle=True, random_state=42)`, sizes `[0.1, 0.25, 0.5, 0.75, 1.0]`, for both the additive-constrained and unrestricted learner.

---

## A9. Zero-fit structural priors

**What.** Schema-level features that predict, before any fitting, whether trees will separate from a linear model.

| Signal | Favours additive/linear | Favours tree ensembles |
|---|---|---|
| Feature count `p` | tens | hundreds–thousands |
| Categorical cardinality | low (few levels) | high (ZIP, merchant, channel) |
| Missingness | rare, MCAR | frequent and **informative** (your `-1` sentinels) |
| Events per variable `EPV = 12000/p` | large (≥100) | matters less |
| Relationship shape | monotone in log-odds by domain logic | plateaus, thresholds, reversals |
| Feature provenance | curated bureau ratios | raw transactional/behavioural |

**Pros.** Free; sets expectations before any compute is spent; useful for scoping.

**Risks.** A prior, not evidence. It cannot substitute for A1–A2 and should never be the stated justification in a model document.

**How / verify.** For the stated setup — tens of curated features, 12,000 events, `-1`-encoded missingness — the prior is: **a large `L0 → L1` step from curvature and missingness handling, and a small `L1 → L3` step**. The one genuine tree advantage available here is native missing-value handling: gradient boosters learn a per-split direction for missing samples during training [1], while a logistic model requires an explicit imputation plus missing-indicator design. Verify by comparing `logistic(WoE with a dedicated missing bin)` against the tree — a well-built WoE bin usually closes most of that gap.

---

# Part B — Group 2: inference on the test set (a posteriori)

**The governing principle: pair everything.** Both models score the same 15,000 accounts, so

```
Var(Δ) = Var(A) + Var(B) − 2·Cov(A, B)
```

With ρ ≈ 0.9 the covariance term removes roughly 55% of the standard error of the difference. Comparing two separately-computed confidence intervals — the most common error in model comparison write-ups — discards that term and throws away about half the sample's worth of power.

---

## B1. DeLong paired test for ΔAUC

**What.** A non-parametric test for the difference between areas under **correlated** ROC curves, using generalised U-statistics to estimate the covariance matrix of the AUC estimators [9]. Closed-form, no resampling.

**Pros.**
- The standard for AUC comparison on a shared sample; recognised by clinical and statistical software as the recommended method for dependent curves [10].
- Exploits the pairing analytically, so it is both fast and near-optimal for this statistic.
- Extends to more than two models with the same covariance machinery.

**Risks.**
- Not in scikit-learn or scipy — you must implement it or take a dependency. Implementations with subtly wrong mid-rank handling silently return wrong `SE`; always assert the returned AUCs match `roc_auc_score` exactly.
- **Tests only ranking.** Two models with identical ranking but very different calibration are indistinguishable to DeLong.
- Asymptotic normality; fine at m=3,600 but not for small event counts or heavy ties (many tied scores from a coarse scorecard need mid-rank handling, included below).
- A non-significant result is *not* evidence of equivalence — see B5.

**How / verify.**

```python
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:                       # average ranks within tie groups
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n)
    out[order] = t + 1
    return out


def delong_paired(y, p_a, p_b):
    """Paired DeLong test. Returns (auc_a, auc_b), se_of_difference, z, p."""
    y = np.asarray(y).astype(int)
    P = np.vstack([np.asarray(p_a, float), np.asarray(p_b, float)])
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    X, Y = P[:, pos], P[:, neg]
    tx = np.vstack([_midrank(X[r]) for r in range(2)])
    ty = np.vstack([_midrank(Y[r]) for r in range(2)])
    tz = np.vstack([_midrank(np.concatenate([X[r], Y[r]])) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n           # placement values, positives
    v10 = 1.0 - (tz[:, m:] - ty) / m     # placement values, negatives
    S = np.cov(v01) / m + np.cov(v10) / n
    contrast = np.array([1.0, -1.0])
    se = float(np.sqrt(contrast @ S @ contrast))
    z = (aucs[0] - aucs[1]) / se
    return aucs, se, z, float(2 * stats.norm.sf(abs(z)))


aucs, se, z, p = delong_paired(y_test, p_xgb, p_logit)
assert np.allclose(aucs, [roc_auc_score(y_test, p_xgb),
                          roc_auc_score(y_test, p_logit)])   # sanity check
lo, hi = (aucs[0] - aucs[1]) - 1.96 * se, (aucs[0] - aucs[1]) + 1.96 * se
print(f"ΔAUC = {aucs[0]-aucs[1]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p = {p:.3f}")
print(f"ΔGini = {2*(aucs[0]-aucs[1]):+.4f}")
```

Validated on a 15,000-row simulation at 24% prevalence: the returned AUCs match `roc_auc_score` to machine precision, and with score correlation ρ=0.85 the paired `SE(Δ) = 0.0029`.

---

## B2. Paired bootstrap

**What.** Resample **accounts** (rows) with replacement, recompute *both* models' metrics on each identical resample, and build a percentile CI for the difference.

**Pros.**
- **Works for any metric**: KS, capture rate in the first decile, Gini within a segment, expected profit under a cost matrix, precision at the approval cutoff. DeLong covers AUC only.
- Preserves pairing automatically because both models see the same resampled index.
- Makes no distributional assumption and handles the 24%-prevalence asymmetry without correction.

**Risks.**
- **Resample rows, never the two models separately.** Independent resampling destroys the pairing and inflates the CI.
- With ~3,600 events, metrics defined on thin slices (top 1% capture) have wide CIs — report them, don't hide them.
- Use ≥2,000 replicates for stable 95% percentile bounds; stratified resampling on `y` reduces variance when the metric is prevalence-sensitive.
- Bootstrap on the test set does **not** account for training randomness (see B6).

**How / verify.**

```python
import numpy as np
from sklearn.metrics import roc_auc_score


def paired_bootstrap_delta(y, p_a, p_b, metric, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)                 # same index for BOTH models
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            deltas[b] = np.nan
            continue
        deltas[b] = metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])
    deltas = deltas[~np.isnan(deltas)]
    return deltas.mean(), np.percentile(deltas, [2.5, 97.5])


def capture_top_decile(y, p):
    k = max(1, int(0.10 * len(y)))
    top = np.argsort(-p)[:k]
    return y[top].sum() / y.sum()


for name, metric in [("AUC", roc_auc_score), ("top-decile capture", capture_top_decile)]:
    d, (lo, hi) = paired_bootstrap_delta(y_test, p_xgb, p_logit, metric)
    print(f"{name:20s} Δ = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
```

---

## B3. Paired test on per-observation losses  *(highest power)*

**What.** Log-loss and Brier score decompose per observation. Compute `dᵢ = ℓ(yᵢ, p_a,ᵢ) − ℓ(yᵢ, p_b,ᵢ)` and test `E[d] = 0` with a paired t-test (or Wilcoxon signed-rank, which is robust to the heavy tail log-loss produces near confident errors).

**Pros.**
- **The most sensitive of the Group-2 tests.** Because the loss is an average of 15,000 per-observation terms, the CLT works directly on `d̄` and the standard error is small.
- Detects the differences that AUC structurally cannot see: **calibration**. Random Forest compresses probabilities toward the mean; a heavily boosted XGBoost pushes them toward the extremes; a logistic model is calibrated by construction on its training distribution. All three can share an AUC.
- Proper scoring rules are the right objects for a PD model whose output feeds pricing, provisioning or expected-loss calculations — where the probability itself is the deliverable, not just the ranking.

**Risks.**
- Log-loss is unbounded and explodes on confident errors; clip predictions to `[ε, 1−ε]` or prefer Brier if a handful of observations dominate `d̄`.
- Sensitive to a monotone recalibration. A model that only loses on calibration can be fixed with isotonic or Platt scaling fitted on a **separate calibration split** — if you intend to recalibrate, compare post-calibration or the test answers a question you don't care about.
- Wilcoxon tests the median difference, not the mean; state which you report.

**How / verify.**

```python
import numpy as np
from scipy import stats

EPS = 1e-6

def per_obs_logloss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

d = per_obs_logloss(y_test, p_xgb) - per_obs_logloss(y_test, p_logit)
t, p_t = stats.ttest_rel(per_obs_logloss(y_test, p_xgb),
                         per_obs_logloss(y_test, p_logit))
w, p_w = stats.wilcoxon(d)
se = d.std(ddof=1) / np.sqrt(len(d))
print(f"Δ logloss = {d.mean():+.5f} ± {1.96*se:.5f} (95% CI)  "
      f"t p={p_t:.4f}  wilcoxon p={p_w:.4f}")   # negative Δ ⇒ xgb better
```

Also plot `sklearn.calibration.calibration_curve` for both models before concluding anything from AUC parity.

---

## B4. McNemar at a fixed decision cutoff

**What.** Once an approval cutoff is fixed, each model produces a binary accept/reject. McNemar tests marginal homogeneity on the 2×2 table of discordant pairs, using the exact binomial distribution by default (`exact=True`) or a chi-square approximation with continuity correction for large samples [11].

**Pros.**
- Tests the thing the business actually deploys: the **decision**, not the score.
- Uses only discordant pairs, which is exactly the right conditioning for a paired binary comparison.
- Trivially cheap, and its result is directly interpretable as "how many accounts would be classified differently, and is that asymmetric?".

**Risks.**
- **The cutoff must be pre-specified** (from a target approval rate or expected-loss policy). Choosing the cutoff that maximises the difference on the test set invalidates the p-value entirely.
- Ignores the magnitude of the score change — a hair's-breadth crossing counts the same as a large one.
- Two models can differ significantly by McNemar at one cutoff and not at another; report the cutoff and its business rationale.
- Discordant counts can be small if the models rank almost identically, which is itself the answer.

**How / verify.**

```python
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

cutoff = np.quantile(p_logit, 0.80)      # pre-specified: 20% rejection rate
a = (p_xgb   >= cutoff).astype(int)
b = (p_logit >= cutoff).astype(int)
correct_a, correct_b = (a == y_test), (b == y_test)

table = np.array([[np.sum(correct_a & correct_b),  np.sum(correct_a & ~correct_b)],
                  [np.sum(~correct_a & correct_b), np.sum(~correct_a & ~correct_b)]])
print(table)
print(mcnemar(table, exact=False, correction=True))
```

---

## B5. TOST — testing for equivalence

**What.** Two one-sided tests. Declare a relevance margin δ **in advance** (e.g. δ = 0.01 AUC = 2 Gini points) and reject the null of non-equivalence only if the whole CI for Δ lies inside `[−δ, +δ]`.

**Pros.**
- The only correct way to make the positive claim "these models perform equivalently", which is usually what you want to write in a model-selection memo justifying the simpler model.
- Forces an explicit, defensible relevance threshold instead of hiding behind `p > 0.05`.
- Reuses the DeLong `SE` — no additional machinery.

**Risks.**
- Requires δ to be set before seeing results; setting it afterwards is circular.
- **May be infeasible with your sample.** With `SE(Δ) ≈ 0.0025`, the 95% CI half-width is ≈0.005, so equivalence within δ=0.01 is demonstrable but δ=0.005 is not. Check feasibility before promising the analysis.
- Equivalence in AUC says nothing about equivalence in calibration or stability.

**How / verify.**

```python
import numpy as np
from scipy import stats

delta_margin = 0.01                      # pre-specified relevance margin, in AUC
diff = aucs[0] - aucs[1]                 # from delong_paired
p_lower = stats.norm.sf((diff + delta_margin) / se)      # H0: diff <= -delta
p_upper = stats.norm.cdf((diff - delta_margin) / se)     # H0: diff >= +delta
print(f"TOST p = {max(p_lower, p_upper):.4f} "
      f"({'equivalent' if max(p_lower, p_upper) < 0.05 else 'not shown equivalent'} "
      f"within ±{delta_margin})")
```

---

## B6. Multi-seed retraining envelope

**What.** Refit both candidates with several seeds (and, ideally, several train/validation splits), score each fit on the **same** test set, and look at the spread of Δ.

**Pros.**
- Captures a variance component that DeLong and the test-set bootstrap are structurally blind to: **training randomness** (bagging draws, column subsampling, split assignment, early-stopping fold).
- Frequently the same order of magnitude as the test sampling error, which means a Δ that looks significant under DeLong can vanish when the model is simply retrained with a different seed.
- Cheap insurance against shipping a result that does not reproduce.

**Risks.**
- Reusing the same test set across many retrains is fine for measuring the spread of Δ, but the *reported* headline number must still come from a single pre-declared fit — otherwise you are selecting on test.
- Expensive: `n_seeds ×` full training cost.
- Seed variance understates full pipeline variance (feature selection, binning cuts, hyperparameter search are also stochastic); include them in the loop if they are part of the production pipeline.

**How / verify.** Loop seeds `[0..9]`, refit both models on the fixed training set, record `ΔAUC` on test, and report `mean ± sd`. If `sd(Δ_seed)` is comparable to `SE(Δ_DeLong)`, state both in the model document and treat the total uncertainty as the combination.

---

## B7. 5×2cv paired t-test — critiqued, not recommended here

**What.** Dietterich's 5×2-fold cross-validated paired t-test: five replications of 2-fold CV, with a variance estimator designed to control the Type-I error inflation of naive k-fold paired t-tests (whose folds share training data and are therefore correlated).

**Why it is the wrong tool for this question.** It compares **learning algorithms** over resamplings of one dataset. Your Group-2 question is different: you have **two specific fitted models** and a held-out sample, and you want to know whether *these* models differ on *this* population. Answering that with 5×2cv discards the held-out test set, changes the estimand, and halves the training size in every fold — which biases both models downward and disproportionately penalises the higher-capacity one.

**Where it does belong:** inside **Group 1**, as a more rigorous alternative to comparing raw CV means across the ladder rungs — i.e. testing whether L3 beats L1 *as a procedure*. Used there, it is a reasonable upgrade over eyeballing `mean ± std` across folds.

**Also critiqued — three habits to drop:**
- **Comparing two independent confidence intervals.** Non-overlapping CIs imply a significant difference, but *overlapping CIs do not imply the reverse*. Use the paired CI on Δ.
- **Unpaired bootstrap** (resampling each model's predictions separately). Same defect, silently inflated CI.
- **Reporting `p > 0.05` as "the models are equivalent".** That is B5's job.

---

## 7. Problem-specific considerations

### 7.1 Your test set's resolution — compute this first

Hanley–McNeil variance for a single AUC, with `m` events and `n` non-events:

```
SE(AUC) = sqrt( [ A(1−A) + (m−1)(Q₁ − A²) + (n−1)(Q₂ − A²) ] / (m·n) )
    Q₁ = A / (2 − A),      Q₂ = 2A² / (1 + A)
```

At `A = 0.75`, `m = 3,600`, `n = 11,400`: **`SE(AUC) = 0.0051`**.

For the paired difference, `SE(Δ) ≈ SE(AUC) · sqrt(2(1 − ρ))`:

| ρ (estimator correlation) | SE(Δ) | MDE at 80% power | in Gini points |
|---|---|---|---|
| 0.80 | 0.0032 | 0.0090 | 1.8 |
| 0.85 | 0.0028 | 0.0078 | 1.6 |
| 0.90 | 0.0023 | 0.0064 | 1.3 |
| 0.95 | 0.0016 | 0.0045 | 0.9 |

**Consequence:** a true 1-Gini-point advantage for XGBoost is *not detectable* in this test set. Report this alongside the p-value; it converts "not significant" from a weak result into a quantified statement about measurement limits.

### 7.2 Set the relevance margin δ before anything else

Both groups need the same number. Group 1: is `ΔAUC(L1→L3)` above δ? Group 2: is the test CI for Δ wider than δ? Derive δ from the loss curve — how many Gini points translate into a material change in expected loss at your approval rate — not from convention.

### 7.3 The 18-month window compresses everything

The target is `default within 18 months`, so the label is contaminated by shocks unobservable at origination (unemployment, health, separation, macro cycle) and by definitional noise (cure, roll-back, charge-off timing). Label noise lowers the achievable AUC for **every** model family, which mechanically compresses the differences between them. It also means accounts with less than 18 months of observation must be excluded or handled with survival methods — including them with `y=0` creates a right-censoring bias that no amount of model tuning repairs.

### 7.4 The `-1` sentinel is a modelling decision, not a cleaning step

Missing bureau data is rarely missing at random in credit. Three defensible treatments, and they change the tree-vs-linear verdict:
- **Trees:** convert `-1` to `NaN` and let the learner assign a per-split direction during training [1]. This is a genuine, structural advantage of the tree family.
- **Logistic:** WoE with a dedicated missing bin. Usually recovers most of that advantage, and is more transparent.
- **Never:** mean-imputing `-1` values, which fabricates a spurious modal spike at the mean and destroys the informative signal.

Run the ladder with a **single, shared** missingness policy so the L1→L3 gap measures interactions rather than imputation quality.

### 7.5 Out-of-time is a different question again

If your test split is temporally posterior, a Δ on it confounds *discrimination difference* with *stability difference*. Neither DeLong nor a bootstrap separates them. Use a random split for Group-2 inference, and a separate out-of-time sample with PSI / vintage-level Gini for stability. The literature on credit-scoring benchmarks is explicit that ensemble advantages over logistic regression are sensitive to dataset characteristics and metric choice [12] — expect the out-of-time picture to differ from the in-time one.

### 7.6 Calibration is where the families actually differ

Even under AUC parity: logistic regression is calibrated by construction on its training distribution; Random Forest averages toward the base rate and compresses the tails; heavily boosted XGBoost pushes toward 0/1. For a PD model feeding provisioning or pricing, this is the difference that matters — and only B3 will see it.

---

## 8. Diagnostics & pitfalls

**Leakage and data separation**
- **Target-derived transforms inside CV.** WoE, IV screening, target encoding, and any binning cut chosen with reference to `y` must be refit within each fold. A global WoE table inflates the additive model and biases Group 1 toward "no trees needed".
- **Feature selection inside CV.** Screening by IV or univariate AUC on the full training set before CV leaks fold-validation information into every fold.
- **Test used more than once.** Every peek is a selection event. Do model selection in CV; touch test once, for the pre-declared comparison.
- **Multiplicity.** k pairwise comparisons on test require Holm correction, or pre-register a single primary comparison and label the rest exploratory.
- **Temporal leakage.** Features constructed with post-origination information (payment behaviour after the credit opening) will make every model look excellent and none of them deployable.

**Group-1-specific**
- Unequal tuning budgets across ladder rungs measure search effort, not structure.
- H-statistic and SHAP interaction values computed on training rows reflect memorisation; use a held-out fold.
- Partial dependence with correlated features extrapolates off-manifold; treat large H² on correlated pairs sceptically and confirm with A2.
- CV `std` across folds is not a confidence interval — folds share training data and are correlated.

**Group-2-specific**
- Unpaired analysis (the single most common error).
- Cutoff chosen post hoc for McNemar.
- `p > 0.05` reported as equivalence.
- Ignoring training-seed variance (B6).
- Metric mismatch: concluding "no difference" from AUC when the deliverable is a calibrated probability.
- Comparing models trained on different feature sets or different missingness policies and attributing the gap to the algorithm.

---

## 9. Decision rule

1. **Set δ** (relevance margin, in Gini points) from the business loss curve. Do this before any fitting.
2. **Fix one preprocessing policy** — missingness, outliers, feature set — shared across all model families.
3. **Run A7** (WoE/spline shapes). Large `L0 → L1`? Fix the logistic spec; that gain belongs to the linear family, not to trees.
4. **Run A1 + A2** under one CV. Read `ΔAUC(L1 → L3)`.
   - `< δ` → **stop. Ship the additive model.** No test-set comparison will change this, and it is now backed by a bound, not a hunch. Go to step 8.
   - `≥ δ` → continue.
5. **Run A3/A5/A6** to localise the interactions. If two or three pairs carry most of the gain, engineer them explicitly and re-run step 4 — you may recover the gain inside the logistic family and keep interpretability.
6. **Run A8.** If L3 is still climbing at 50,000 rows, note in the model document that the current verdict is sample-size-limited and revisit at the next data refresh.
7. **Compute the MDE** from §7.1. If `δ < MDE`, say so up front: the test set cannot adjudicate at your relevance threshold, and Group 1 is the deciding evidence.
8. **Fit the two pre-declared candidates on full train. Score test once.** Run B1 (ΔAUC + CI), B3 (Δ log-loss/Brier + calibration curves), B2 for the decile capture, B4 at the policy cutoff.
9. **If Δ is not significant and you want to claim parity**, run B5 (TOST). If the CI is contained in `[−δ, +δ]`, you have a positive equivalence result — the strongest possible justification for choosing the simpler model.
10. **Run B6** before publishing. If seed-to-seed spread rivals the test SE, report the combined uncertainty.
11. **Choose.** Where Δ is within δ, decide on interpretability, out-of-time stability (PSI, vintage Gini), calibration quality, monotonicity constraints, and governance cost — not on the third decimal of AUC.

---

## 10. References

1. scikit-learn — `HistGradientBoostingClassifier` (`interaction_cst`, native NaN handling, split-direction learning for missing values). https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
2. scikit-learn — `HistGradientBoostingRegressor` (`interaction_cst`, added in version 1.2). https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
3. InterpretML — Explainable Boosting Machine (GAM / GA2M formulation, round-robin cyclic boosting). https://interpret.ml/docs/ebm.html
4. XGBoost — Feature Interaction Constraints tutorial (`interaction_constraints`, `tree_method` requirement). https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
5. H2O — `interaction_constraints` (documents the difference in how XGBoost and GBM evaluate overlapping constraint groups along a branch). https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/interaction_constraints.html
6. InterpretML — `ExplainableBoostingClassifier` API (`interactions`, default `"3x"`, `exclude`, `max_interaction_bins`). https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html
7. SHAP — `shap.TreeExplainer` API (`shap_interaction_values` output shape and summation property; `tree_path_dependent` vs `interventional`). https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
8. SHAP — Basic SHAP Interaction Value Example in XGBoost (interaction matrix is diagonal when no interactions are present). https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
9. DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845. https://pubmed.ncbi.nlm.nih.gov/3203132/
10. MedCalc — Comparison of ROC curves (DeLong recommended for dependent curves; Hanley & McNeil variance formulas). https://www.medcalc.org/en/manual/comparison-of-roc-curves.php
11. statsmodels — `contingency_tables.mcnemar` (`exact` binomial vs chi-square approximation, continuity correction). https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html
12. Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: an update of research. *European Journal of Operational Research*, 247(1), 124–136. https://www.sciencedirect.com/science/article/abs/pii/S0377221715004208
13. Lessmann et al. (2015), accepted manuscript (open access). https://eprints.soton.ac.uk/377196/1/Lessmann_Benchmarking.pdf
