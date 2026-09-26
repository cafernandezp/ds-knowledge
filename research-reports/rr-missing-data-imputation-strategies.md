# Missing Data Imputation Strategies — Rigor Review & Decision Framework

> **Context.** Rigor evaluation of the 9 missing-data treatment methods from a shared infographic (listwise/column deletion, mean/median/mode imputation, KNN, MICE, forward-fill, interpolation), extended with methods the infographic omits. Target setting: tabular / structured pipelines (credit-risk, collections-style scoring), primary model = gradient-boosted trees (XGBoost) plus classical scikit-learn models, with panel/time-ordered features present. Assumptions: target is a probability or continuous score requiring calibration; production pipeline must respect train/OOT separation; imputers are fitted artifacts, versioned with the model.

## Which ML Models Need Missing-Value Treatment — and Which Don't

Two families of model exist with respect to missing data, and the split matters more than which imputation method is "best" in the abstract.

**The one-line mechanistic reason behind the whole split:** models in group A compute *arithmetic* on every feature dimension — a dot product, a distance, a mean, a gradient — and that arithmetic has no defined result when one of its inputs is `NaN` (`NaN` propagates through sums and products by IEEE-754 definition). Models in group B are all trees: a split only ever asks a *routing* question ("does this row go left or right?"), and "missing" can simply be treated as one more possible answer to that question, learned from data like any other. No arithmetic is ever performed on the missing value itself.

**A. Models that need a complete matrix — NaN causes a hard error, or a silently wrong fit.**
- **Linear/Logistic Regression, Ridge/Lasso/Elastic Net** — *why not:* fitting minimizes a loss built from `Xβ` (a sum of `feature × coefficient` across all columns) or solves `(XᵀX)⁻¹Xᵀy` in closed form; one `NaN` in a row makes that row's contribution to the sum/matrix `NaN`, which then poisons the whole gradient or the whole matrix inverse — there is no per-feature "skip" built into the linear-algebra formulation.
- **SVM (`SVC`/`SVR`)** — *why not:* the kernel function (linear dot product, RBF `exp(-γ‖x−x'‖²)`, etc.) needs every coordinate of both vectors to compute a distance or inner product; a missing coordinate leaves the distance undefined, not just imprecise.
- **k-Nearest Neighbors** *as a predictive model* (not the imputer from §6) — *why not:* same root cause as SVM — the standard distance metric sums squared differences across all dimensions, and a missing dimension breaks that sum (unless a NaN-aware metric like `nan_euclidean` is explicitly configured, which most classifiers don't do by default).
- **Naive Bayes** — *why not:* the prediction multiplies (or sums log-) per-feature likelihoods across all features for a given class; a feature with no value has no likelihood term to include, so the product/sum is either undefined or silently wrong depending on the implementation.
- **PCA / other linear dimensionality reduction, K-Means / clustering** — *why not:* PCA needs the full covariance matrix (or SVD) of the data, and K-Means needs a real-valued distance from every point to every centroid — both are matrix/distance operations with the same `NaN`-propagation problem as above.
- **Neural networks** (PyTorch, TensorFlow/Keras, plain MLPs) — *why not:* a forward pass is a chain of matrix multiplications and nonlinear activations; a `NaN` at the input layer multiplies into every downstream weight and propagates through the entire computation graph, corrupting the loss and every gradient in backpropagation, not just the neurons connected to that one input.
- **scikit-learn's `GradientBoostingClassifier`/`Regressor`** (the classic, non-histogram implementation) — *why not, unlike its sibling below:* it grows trees using exact/greedy split search directly on sorted, complete feature values; it was never adapted to reserve a "missing" bin or learn a default routing direction the way the newer histogram-based boosting was.

For all of these, one of the methods in §1–12/14 is mandatory before fitting — not a nice-to-have. Skipping it is either a `ValueError` at `fit()` time, or — worse, if NaNs were silently replaced by a sentinel like `0` upstream — a model that trains without error but has learned a wrong relationship.

**B. Models with native missing-value handling — imputation is optional, and sometimes counter-productive.**
- **XGBoost** — *why yes:* sparsity-aware split finding (§13). At each candidate split, the algorithm computes the gain from sending all missing-valued rows left, then the gain from sending them all right, and keeps whichever direction wins — a routing decision, not an arithmetic one, so no numeric value for the missing entries is ever needed. On by default, no flag required.
- **LightGBM** — *why yes:* the same routing idea, implemented on top of its histogram binning — missing values get their own bin and an optimal default direction is learned during training. `use_missing=True` by default (disable with `use_missing=False`).
- **CatBoost** — *why yes, slightly different mechanism:* controlled by `nan_mode` (default `"Min"`). Rather than *learning* a left/right direction per split like XGBoost, CatBoost treats a missing value as a fixed constant smaller (or, with `"Max"`, larger) than every observed value for that feature — this lets it slot into the same ordered/binned split search as any real value, and a split can still isolate it if that's where the gain is.
- **scikit-learn `HistGradientBoostingClassifier`/`Regressor`** — *why yes:* same histogram-binning mechanism as LightGBM — missing values occupy a dedicated bin, and the split search routes that bin independently of the observed-value bins.
- **scikit-learn `RandomForestClassifier`/`Regressor` and `DecisionTreeClassifier`/`Regressor`** — *why yes, in recent versions:* the tree grower learns, per split, whether missing-valued samples go left or right — the identical mechanism to XGBoost's sparsity-aware split finding (§13), just added later to these estimators' implementation.

For this family, imputing first and then feeding the model a completed matrix isn't *wrong*, but it discards exactly the property that makes tree ensembles attractive under MNAR (see the primer below): once a value is imputed, the model can only see your guess at what it might have been, not the fact that it was missing. If governance/explainability requires an explicit imputed value anyway — a real constraint at a regulated institution — pair the imputation with a missing indicator (§10), so the model still has both the point estimate and the missingness signal.

**Practical check.** "Handles missing values" is a property of a specific class *and* library version, not of "tree models" in general — plain `RandomForestClassifier`/`DecisionTreeClassifier` in scikit-learn didn't support NaN in older releases, and `GradientBoostingClassifier` still doesn't, while its histogram-based sibling does. `model.fit(X_with_nan, y)` raising `ValueError: Input X contains NaN` is the fastest way to confirm which side of the line a given estimator/version actually falls on — check the installed version's docs, don't assume from the model family alone.

## Missingness Mechanisms — Plain-Language Primer

Three acronyms recur throughout this report. They describe **why** a value is missing — not how much is missing — and that "why" is what determines whether a given method is statistically safe.

- **MCAR — Missing Completely At Random.** The chance a value is missing has nothing to do with any variable, observed or missing. *Example:* a sensor randomly drops a reading because of a flaky connection, unrelated to the reading itself or to anything else in the data. Rare in practice — it's the strong, "nothing to see here" case, and the only one where plain deletion or mean imputation are unbiased.
- **MAR — Missing At Random.** The chance a value is missing depends on *other variables you did observe*, but not on the missing value itself. *Example:* older customers are less likely to fill in an "email" field — missingness depends on `age` (which you have), not on the email address itself. Most real "random-looking" missingness in practice is actually this, not MCAR.
- **MNAR — Missing Not At Random.** The chance a value is missing depends on *the hidden value itself*, even after accounting for everything else you observed. *Example:* high-income customers are less likely to report their income — missingness depends on the very income figure that's hidden. In collections data: a payment stops being logged *because* the customer defaulted — the missingness is not noise, it's part of the event you're trying to predict.

**Why it matters.** MCAR and MAR can be "fixed" by the methods in this report: deletion is unbiased under MCAR; model-based imputation (KNN/MICE/MissForest) is unbiased under MAR, because the mechanism is fully explained by data you can see. MNAR **cannot** be fixed by imputation alone — no model can learn a dependency on data it never observes. It needs a different move: encode the missingness itself as a feature (§10 missing indicator), use a model that can route on missingness directly (§13 native tree handling), or explicitly model the mechanism.

**Practical rule of thumb** (not a proof, a starting heuristic): ask *"if I knew the true hidden value, would that change the probability it's missing?"* — if no, and no other variable explains it either, it's MCAR; if no, but some other column explains it, it's MAR; if yes, it's MNAR. When in doubt, §0 gives a testable check instead of guessing.

## TL;DR

- **Diagnose the missingness mechanism (MCAR/MAR/MNAR) before choosing a method.** It — not a fixed `%` threshold — determines which methods remain statistically valid.
- **For XGBoost/LightGBM**, prefer native sparsity-aware missing handling; it avoids a separate imputation model entirely and often outperforms manual imputation because missingness itself can be informative (MNAR).
- **For models needing a complete matrix and valid inference** (coefficients, SEs, p-values): MICE or MissForest run as *true* multiple imputation (m≥5, pooled via Rubin's rules) — not as a single pass, which is how most practitioners (and the infographic) actually use them.
- **Unconditional single imputation (mean/median/mode) is the weakest defensible option**, not the "simple, safe default" it's usually treated as — it deflates variance and biases every covariance involving the imputed feature, even under MCAR.
- **Always fit imputers on train only**, transform val/OOT/serving; wrap in a `Pipeline` so CV refits per fold. Two-sided interpolation and backward-fill are look-ahead leakage in any forecasting/scoring context.
- **Rigor ranking** (validity of inference, not raw point-accuracy): Multiple Imputation (MICE/MissForest + Rubin pooling) > single conditional model-based (MissForest/KNN/regression) > native tree handling > missing-indicator add-on > interpolation (context-dependent) > forward/backward fill > unconditional single imputation > deletion.

## Comparison table

| # | Method | Category | Validity requires | Preserves variance/covariance | Uncertainty quantified | Typical leakage risk | Best fit |
|---|--------|----------|--------------------|-------------------------------|--------------------------|------------------------|----------|
| 0 | Mechanism diagnosis (Little's/PKLM test) | Diagnostic, prerequisite | — | — | — | None | Always, first |
| 1 | Listwise deletion | Deletion | MCAR | N/A (drops rows) | No | Low | Very low missing rate, MCAR-plausible |
| 2 | Column deletion | Deletion | MCAR/MAR + low feature value | N/A | No | Low | High missing rate **and** low importance |
| 3 | Mean imputation | Unconditional single | MCAR | No | No | Medium | Quick baseline, symmetric numeric, non-tree fallback |
| 4 | Median imputation | Unconditional single | MCAR | No | No | Medium | Same as mean, skewed/outlier-heavy |
| 5 | Mode imputation | Unconditional single | MCAR | No | No | Medium | Low-cardinality categoricals |
| 6 | KNN imputation | Conditional single | MAR (local) | Partial | No | Medium-High | Multivariate structure, moderate dimensionality |
| 7 | MICE | Multiple (if m>1) | MAR | Yes, if m>1 + pooled | Yes, if m>1 | Medium | Formal inference under MAR |
| 8 | Forward/Backward Fill | Sequential | Slow-changing state; missingness ⟂ trend | No | No | High (NOCB = look-ahead) | Time-ordered, slowly-varying state fields |
| 9 | Interpolation | Sequential | Local smoothness, MAR along index | Partial | No | High (two-sided = look-ahead) | Continuous, smoothly-varying physical signals |
| 10 | Missing indicator + imputation | Add-on | None — valid even under MNAR | Improves base method | No | Low | Always, alongside 3–9/11/12 |
| 11 | Regression imputation (det./stoch.) | Conditional single | MAR + correct functional form | Stoch.: partial; Det.: no | No | Medium | Auditable single-feature models |
| 12 | MissForest | Conditional single (or MI) | MAR, no distributional assumption | Better than 6/11 | Only with MI extension | Medium | Mixed-type, nonlinear, no normality |
| 13 | Native tree handling (XGBoost/LightGBM) | Model-native | None — learned per split | N/A | No | Low | Tree ensembles already in stack, MNAR-tolerant |
| 14 | Deep generative (GAIN/DAE) | Conditional, learned | Large N to fit G+D | Yes, aims for full joint | Some variants | Medium-High | Large-N, complex joint structure |

## 0. Diagnose the Missingness Mechanism *(prerequisite, not optional)*

**What.** Classify missingness as MCAR / MAR / MNAR before selecting a treatment; the classification determines which downstream methods are unbiased (Rubin's taxonomy, formalized in Van Buuren, 2018).

**Pros.**
- Turns "which method" into a falsifiable, testable decision instead of a `%`-based heuristic.
- Little's chi-square test and the more robust PKLM test give an objective MCAR-vs-not signal.

**Risks.**
- Little's test assumes multivariate normality and doesn't reliably handle categorical or temporally-correlated data; failing to reject MCAR does **not** prove MCAR (absence of evidence).
- MAR vs. MNAR is fundamentally untestable from observed data alone — it always requires domain judgment (e.g. "income missing because unemployed" is MNAR by construction).

**How / verify.**
```python
# pip install qolmat
import numpy as np
from qolmat.analysis.holes_characterization import LittleTest, PKLMTest

rng = np.random.RandomState(0)
little_p = LittleTest(random_state=rng).test(df)   # H0: MCAR
pklm_p = PKLMTest(random_state=rng).test(df)        # more robust to non-normal/categorical data

# p < 0.05 on either -> reject MCAR -> treat as MAR/MNAR,
# avoid unconditional single imputation (methods 3-5) downstream.

# quick, cheap sanity check even without a formal test:
missing_flag = df["x"].isna()
target_gap = df.loc[missing_flag, "y"].mean() - df.loc[~missing_flag, "y"].mean()
# large |target_gap| => missingness likely carries signal => MNAR is plausible
```

**Critique when warranted.** This step is routinely skipped in practice — the infographic does too. Rules like "drop rows if <5% missing" or "drop columns if >50% missing" are not derived from a mechanism test and can silently bias a model even at low missing rates if the mechanism is MNAR.

## 1. Listwise Deletion (Drop Rows / Complete-Case Analysis)

**What.** Remove any row with ≥1 missing value.

**Pros.** No model to fit; preserves the true joint distribution of the retained rows; trivial to audit.

**Risks.** Unbiased only under MCAR; under MAR/MNAR the retained sample is a biased subsample, so both point estimates and their standard errors are wrong, not merely less precise (Little & Rubin's framework, formalized in Van Buuren, 2018). N-loss compounds multiplicatively across correlated missing columns.

**How / verify.**
```python
df_cc = df_train.dropna()  # fit/derive statistics on train split only; apply the same row filter logic downstream consistently
```

**Critique when warranted.** Popular as a "safe default," but only actually safe once §0 confirms MCAR-plausibility; otherwise it silently reweights the sample toward whichever subpopulation has fewer missing values — exactly the failure mode a `<5%` cutoff cannot detect.

## 2. Feature Deletion (Drop Columns)

**What.** Remove a feature entirely above a missing-rate threshold.

**Pros.** Removes unreliable columns from consideration; simplifies the pipeline.

**Risks.** Discards a feature that may carry real signal on its non-missing subset — for a tree ensemble with native handling (§13), a 60%-missing feature can still contribute. No missing-rate threshold in the literature is universal; the criterion should be *(mechanism) × (feature importance on the non-missing subset)*, not raw `%` alone.

**How / verify.**
```python
# check importance on the OBSERVED subset before dropping, not after
mask = df_train["x"].notna()
importance_on_observed = quick_shap_or_perm_importance(X_train[mask], y_train[mask], feature="x")
if importance_on_observed < threshold:
    df_train = df_train.drop(columns=["x"])
```

**Critique when warranted.** The infographic already gates this on "feature importance is low," which is the right instinct — the anti-pattern is treating the `%` cutoff as sufficient by itself instead of as a pre-filter for the importance check.

## 3. Mean Imputation

**What.** Replace missing values with the column mean, fitted on train.

**Pros.** Trivial, fast, unbiased for the mean statistic itself under MCAR.

**Risks.** Deflates variance and attenuates every covariance/correlation involving the imputed feature — even under MCAR — because a constant replaces a random quantity (Van Buuren, 2018). Downstream standard errors, hypothesis tests, and regression coefficients become invalid. Sensitive to outliers.

**How / verify.**
```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="mean").fit(X_train)   # fit on train fold only
X_train_i, X_val_i = imp.transform(X_train), imp.transform(X_val)
```

**Critique when warranted.** The most-criticized method in the missing-data literature precisely because it looks safe and isn't. It is statistically dominated by stochastic regression imputation (§11) or MICE (§7) at similar engineering cost, and by native tree handling (§13) when the model is XGBoost. Treat it as a sanity baseline, not a production choice.

## 4. Median Imputation

Same structural flaw as mean imputation — deflates variance, distorts covariance — but more robust to outliers/skew, which is a real (if narrow) advantage over §3.

```python
imp = SimpleImputer(strategy="median").fit(X_train)
```

## 5. Mode Imputation

**What.** Fill categoricals with the most frequent category.

**Pros.** Keeps a valid category label; simple for low-cardinality features.

**Risks.** Inflates the mode's frequency, can create a spurious majority class that biases a classifier's prior, ignores co-occurrence with other features.

**How / verify.**
```python
imp = SimpleImputer(strategy="most_frequent").fit(X_train)
```

## 6. KNN Imputation

**What.** Fill with a (possibly distance-weighted) average over the `k` nearest neighbors under a NaN-aware distance metric.

**Pros.** Conditions on multivariate structure, unlike §3–5; captures nonlinear local relationships without specifying a parametric form.

**Risks.** Requires prior scaling or large-scale features dominate the distance; curse of dimensionality degrades neighbor quality; `O(n²)`-ish at inference (not designed for Spark-scale data); still single-imputation — no uncertainty quantification; neighbors must be drawn only from the fitted train set, or test information leaks into imputed values.

**How / verify.**
```python
from sklearn.impute import KNNImputer
# sklearn defaults: n_neighbors=5, weights='uniform', metric='nan_euclidean'
imp = KNNImputer(n_neighbors=5, weights="distance").fit(X_train)
X_train_i = imp.transform(X_train)
X_val_i = imp.transform(X_val)   # neighbors always drawn from the fitted (train) set
```

**Critique when warranted.** A real improvement over unconditional imputation, but "considers feature similarity" does not by itself validate MAR — a poorly chosen distance metric on mixed-type data can encode spurious similarity.

## 7. MICE (Multiple Imputation by Chained Equations)

**What.** For each incomplete feature, regress it on all others (any conditional model per variable), cycle through features iteratively; repeated `m` times with independent random draws to produce `m` completed datasets, later pooled via Rubin's rules (van Buuren & Groothuis-Oudshoorn, 2011).

**Pros.** The only method among the infographic's nine that, used correctly (`m>1` + pooling), yields statistically valid point estimates *and* standard errors under MAR — that pooling step is MICE's actual source of rigor, not "chained equations" per se.

**Risks.** Most practitioners (and the infographic) run it as `m=1` — a single-imputation regression-based fill — which forfeits the uncertainty quantification that is MICE's entire statistical justification. Convergence is not guaranteed (inspect trace plots). Default conditional estimator (`BayesianRidge`) assumes roughly linear relationships unless swapped for a tree-based one.

**How / verify.**
```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401, required
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# m=1 (point-imputation) — fast, but NOT valid multiple imputation
imp = IterativeImputer(estimator=RandomForestRegressor(random_state=0), random_state=0).fit(X_train)
X_train_i = imp.transform(X_train)

# proper multiple imputation: m independent random_state draws, pooled per Rubin (1987)
# Q_bar = mean(Q_l);  T = W_bar + (1 + 1/m) * B   (within + between-imputation variance)
```

**Critique when warranted.** The infographic credits MICE for "preserving relationships better than simple imputation" without distinguishing it from KNN/regression on the dimension that actually matters — whether uncertainty is quantified. Run as `m=1`, MICE is just another single-imputation method.

## 8. Forward Fill (LOCF) *[+ Backward Fill / NOCB]*

**What.** Propagate the last (or next) observed value along an ordered index.

**Pros.** Trivial; plausible for slowly-changing state variables (e.g. a customer's static profile field).

**Risks.** Assumes missingness is unrelated to the trend between observations — false whenever the series is non-stationary or missingness correlates with a regime change (e.g. a payment stream stops being recorded right when a customer defaults — MNAR, and LOCF actively hides the event). Inflates autocorrelation, deflates variance of differences. Backward-fill (NOCB) is look-ahead leakage whenever used across a train/serving boundary.

**How / verify.**
```python
df_sorted = df.sort_values(["entity_id", "date"])
df_sorted["x"] = df_sorted.groupby("entity_id")["x"].ffill()   # .bfill() only for retrospective, non-production feature builds
```

**Critique when warranted.** The infographic's "use when data is time-ordered and changes slowly" is correct in spirit but omits the MNAR risk that is common precisely in collections/credit-risk data, where a stopped observation stream often *is* the signal.

## 9. Interpolation (linear / polynomial / spline)

**What.** Estimate the missing value from the local shape of an ordered series.

**Pros.** Better than LOCF for genuinely continuous, smoothly-varying signals; several functional forms available.

**Risks.** Standard interpolation uses points on both sides of the gap — leakage in any pipeline where future data wouldn't be available at scoring time. Not meaningful for categorical data (as the infographic notes) or jump-discontinuous series (transactions, defaults). High-order polynomial extrapolation is unstable (Runge's phenomenon).

**How / verify.**
```python
# only safe for retrospective feature construction, or when both endpoints
# are guaranteed available at scoring time:
df_sorted["x"] = df_sorted.groupby("entity_id")["x"].apply(
    lambda s: s.interpolate(method="linear", limit_direction="forward")
)
```

**Critique when warranted.** In an OOT-validated credit-risk pipeline this method needs an explicit look-ahead audit; the infographic doesn't flag it at all.

## 10. Missing Indicator / Flag Augmentation

**What.** Add a binary column marking where each feature was originally missing, alongside any imputed value.

**Pros.** Recovers part of the information destroyed by any single-value imputation; lets the downstream model learn "missing ⇒ different outcome" directly — the only method here that remains valid even under MNAR, since it doesn't guess the missing value, it encodes the missingness itself.

**Risks.** Doubles feature count on wide datasets; the indicator can leak target information if missingness was itself induced by a post-outcome process (e.g. a field only populated after a collection call happens) — check the causal timing before using it.

**How / verify.**
```python
imp = SimpleImputer(strategy="median", add_indicator=True).fit(X_train)
X_train_i = imp.transform(X_train)   # appends one 0/1 column per originally-missing feature
```

**Critique when warranted.** The cheapest rigor upgrade available — should be the default add-on to §3–9/11/12, not treated as a separate, optional technique.

## 11. Regression Imputation (deterministic & stochastic)

**What.** Fit `x_j = β^T z + ε` on complete cases; predict missing `x_j` from the other features `z`.

**Pros.** Conditions explicitly on the full feature set; interpretable, auditable coefficients.

**Risks.** Deterministic version (no noise term) deflates variance identically to mean imputation, only conditionally instead of globally. Stochastic version (`ε ~ N(0, σ̂²)` per draw) restores marginal variance but is still single-imputation unless repeated `m` times like MICE. Assumes the specified functional form is correct — misspecification biases imputed values.

**How / verify.**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

reg = LinearRegression().fit(Z_train_complete, x_train_complete)
resid_std = (x_train_complete - reg.predict(Z_train_complete)).std(ddof=1)
x_pred = reg.predict(Z_missing)
rng = np.random.default_rng(0)
x_stochastic = x_pred + rng.normal(0, resid_std, size=len(x_pred))   # restores variance
```

**Critique when warranted.** Rarely used standalone today — it's essentially a single-feature, single-pass version of MICE's conditional model. Use MICE directly unless a fully transparent, auditable single-equation imputation is required for governance (common in regulated credit scoring).

## 12. MissForest (Random-Forest Iterative Imputation)

**What.** MICE-style chained iteration where each conditional model is a random forest (Stekhoven & Bühlmann, 2012); handles continuous and categorical columns in one pass, no distributional assumption.

**Pros.** Captures nonlinearities/interactions that Bayesian-ridge-default MICE misses; reports an out-of-bag error per iteration as a built-in convergence diagnostic; consistently a top performer in mixed-type tabular benchmarks.

**Risks.** No closed-form uncertainty unless run as multiple imputation (`m` independent runs + pooling — an active extension, not the vanilla algorithm). Heavier than KNN/mean. Must be fit on train only, like MICE.

**How / verify.**
```python
# pip install miceforest
import miceforest as mf

kernel = mf.ImputationKernel(X_train, num_datasets=5, random_state=0)  # m=5 completed datasets
kernel.mice(iterations=5)
X_train_i = kernel.complete_data(dataset=0)                      # single completed set, or
X_train_pooled = [kernel.complete_data(d) for d in range(5)]     # for Rubin pooling
X_val_i = kernel.impute_new_data(X_val).complete_data(0)          # apply fitted models, no leakage
```

**Critique when warranted.** Given an XGBoost/scikit-learn stack, this — or MICE with a tree-based estimator — is the strongest general-purpose statistically-grounded upgrade over the infographic's KNN/mean/MICE trio. The infographic omits it despite it out-performing KNN and generic MICE in most published mixed-type benchmarks.

## 13. Native Missing-Value Handling in Tree Ensembles (XGBoost / LightGBM)

**What.** Sparsity-aware split finding: at every split, the algorithm evaluates sending missing-valued rows left or right and keeps whichever direction maximizes gain, learned per-split (not globally) (Chen & Guestrin, 2016).

**Pros.** No separate imputation model to fit, tune, or leak; the missingness pattern itself becomes informative — useful precisely when data is MNAR (e.g. a field going missing because a collection process stalled); zero preprocessing cost; both `xgboost` and LightGBM support it by default, as does `HistGradientBoostingClassifier` in scikit-learn.

**Risks.** XGBoost's `gblinear` booster treats missing as zero, **not** sparsity-aware — a documented gotcha when switching booster type. Not usable outside tree ensembles (logistic regression, SVM need a complete matrix). Handling is opaque relative to an explicit imputed value — harder to audit for a model-risk/validation function than a documented MICE/MissForest fill. Reliable only if missing rate/pattern is stable between train and serving — worth a drift check on missing rate itself, analogous to PSI.

**How / verify.**
```python
import numpy as np, xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)  # np.nan is the default
params = {"objective": "binary:logistic", "tree_method": "hist"}
booster = xgb.train(params, dtrain, num_boost_round=200)
# no imputer fit/transform step needed at all
```

**Critique when warranted.** For an XGBoost-centric stack this is arguably the single best default — it sidesteps every leakage risk above by construction. The trade-off is auditability: a regulated credit model may still require an explicit, documented imputation for governance/explainability, in which case MissForest + SHAP-on-the-imputed-set is the fallback.

## 14. Deep Generative Imputation (GAIN / Denoising Autoencoders)

**What.** A GAN-style generator fills masked values conditioned on observed ones; a discriminator is trained to detect which components were imputed, guided by a "hint" vector toward the true data distribution (Yoon, Jordon & van der Schaar, 2018).

**Pros.** Can model nonlinear, high-dimensional joint distributions beyond per-variable conditional models (MICE/MissForest); some variants (e.g. MIWAE) support genuinely multiple stochastic draws.

**Risks.** Data-hungry — needs large N to train generator+discriminator without overfitting, atypical for a single bank's tabular collections dataset relative to the image/genomics-scale corpora GAIN was benchmarked on. Adds a GAN to version, monitor, and retrain as a preprocessing step. Unstable training (mode collapse) needs its own diagnostics. Materially harder to explain to a model-risk function than MissForest/MICE.

**How / verify.**
```python
# pip install hyperimpute
from hyperimpute.plugins.imputers import Imputers

gain = Imputers().get("gain")             # wraps Yoon et al. (2018) GAIN
X_train_i = gain.fit_transform(X_train)   # fit strictly on train
X_val_i = gain.transform(X_val)
```

**Critique when warranted.** Overkill for most tabular credit-risk datasets at typical bank scale — included for completeness, not as a default. MissForest/MICE deliver most of the accuracy gain at a fraction of the engineering and governance cost.

## Problem-specific considerations

- **Target/calibration.** PD/collections outcomes need calibrated probabilities; single-imputation methods that shrink variance can distort the calibration curve. Re-run Platt/isotonic calibration after any change to the imputation strategy, not only after a model change.
- **Model family.** XGBoost primary → native handling (§13) is the lowest-risk default. Reserve MICE/MissForest for models needing a complete matrix (logistic baselines, SHAP-on-imputed-set decks) or where governance requires an explicit, documented imputed value.
- **Scale/infra.** At PySpark/Athena scale, `KNNImputer`/`IterativeImputer`/`miceforest` are single-node tools. For very large N, prefer native tree handling or a Spark-native `pyspark.ml.feature.Imputer` (mean/median) plus a missing indicator, not a full MICE/MissForest pass.
- **Financial data specifics.** Tied/round-number amounts (relevant to prior LOF-vs-DBSCAN work) affect KNN and interpolation the same way they affect distance-based anomaly detection — ties distort neighbor selection. Pair mode/median imputation of transaction-like amounts with a missing indicator rather than inventing a spurious "typical" value.
- **OOT validation.** A fitted imputer's statistics (means, neighbors, MICE/MissForest models) are part of the model artifact and must be versioned with it. Refitting an imputer on a newer OOT window without a corresponding model retrain is itself a drift source, distinct from feature-level PSI drift.

## Diagnostics & pitfalls

- **Mechanism test first (§0).** Little's/PKLM p-values, plus a cheap sanity check: compare `mean(y | x missing)` vs. `mean(y | x observed)` — a large gap signals MNAR, where unconditional single imputation is most dangerous.
- **Leakage.** Always `fit()` imputers on the train fold only, `transform()` val/OOT/serving. Wrap in a scikit-learn `Pipeline` so cross-validation refits the imputer per fold, never once globally.
- **Look-ahead in sequential methods.** Forward-fill is safe only strictly forward in time. Backward-fill and two-sided interpolation are leakage unless every endpoint used is guaranteed available at the actual production scoring timestamp — audit against that timestamp, not the batch file's row order.
- **Missing-rate drift.** Monitor `%` missing per feature between train and production the same way PSI is monitored for the feature distribution — a sudden jump in missing rate is model risk independent of how well the imputer performs on the training distribution.
- **Imputation-then-calibration order.** Recalibrate after any imputation-strategy change; imputation shifts the input distribution the calibration curve was fit on.
- **Dtype routing.** Mean/KNN/interpolation apply to numeric features only; route numeric vs. categorical through separate `ColumnTransformer` branches rather than one global imputer.

## Decision rule

1. Missing rate <5% **and** Little's/PKLM doesn't reject MCAR → listwise deletion (§1) is defensible. Otherwise → 2.
2. Model is XGBoost/LightGBM/HistGBM, no strict governance requirement for an explicit imputed value → native sparsity-aware handling (§13) + missing-rate drift monitor.
3. Model requires a complete matrix **and** formal inference validity matters (coefficients, SEs, CIs) → MICE or MissForest as true multiple imputation (m≥5, Rubin-pooled).
4. Model requires a complete matrix but only point-prediction accuracy matters → MissForest or MICE single-pass (m=1), always + missing indicator (§10).
5. Feature is time-ordered, slowly-changing state, and only past values are available at scoring time → forward fill only — never backward fill or two-sided interpolation across the scoring boundary.
6. Missing rate on a feature >50% → don't drop by `%` alone; check SHAP/importance on its non-missing subset first, drop only if importance is also low.
7. Whenever 3–6 are used → always add the missing indicator (§10); it stays valid even if the mechanism turns out to be MNAR.
8. Dataset is bank/Spark-scale and 1–4 are computationally intractable → Spark-native mean/median imputer + missing indicator as pragmatic fallback; prefer native tree handling wherever the downstream model allows it.

## References

1. Little, R.J.A. (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. *Journal of the American Statistical Association*, 83(404), 1198–1202. https://doi.org/10.1080/01621459.1988.10478722
2. Van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). Chapman & Hall/CRC. https://stefvanbuuren.name/fimd/
3. van Buuren, S., Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67. https://www.jstatsoft.org/article/view/v045i03
4. Stekhoven, D.J., Bühlmann, P. (2012). MissForest — non-parametric missing value imputation for mixed-type data. *Bioinformatics*, 28(1), 112–118. https://doi.org/10.1093/bioinformatics/btr597
5. Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. https://arxiv.org/pdf/1603.02754
6. XGBoost Developers. Frequently Asked Questions — missing value handling. https://xgboost.readthedocs.io/en/stable/faq.html
7. Yoon, J., Jordon, J., van der Schaar, M. (2018). GAIN: Missing Data Imputation using Generative Adversarial Nets. *Proceedings of the 35th ICML*, PMLR 80:5689–5698. https://proceedings.mlr.press/v80/yoon18a.html
8. scikit-learn. `sklearn.impute.SimpleImputer`. https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
9. scikit-learn. `sklearn.impute.KNNImputer`. https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
10. scikit-learn. `sklearn.impute.IterativeImputer` / `enable_iterative_imputer`. https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
11. scikit-learn. `sklearn.impute.MissingIndicator`. https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html
12. qolmat documentation. Tutorial for Testing the MCAR Case (Little's test, PKLM test). https://qolmat.readthedocs.io/en/main/examples/tutorials/plot_tuto_mcar.html
13. miceforest — Fast, Memory Efficient Imputation with LightGBM. https://pypi.org/project/miceforest/
14. HyperImpute documentation. `hyperimpute.plugins.imputers.plugin_gain` (GAIN implementation). https://hyperimpute.readthedocs.io/en/latest/generated/hyperimpute.plugins.imputers.plugin_gain.html
