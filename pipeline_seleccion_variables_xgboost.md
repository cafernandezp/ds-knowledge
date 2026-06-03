# Feature Selection Pipeline for Regression with XGBoost

**Problem context**

- Task: regression, final model XGBoost.
- Data: `df_train`, **≤ 10,000 rows**. All selection decisions are made **only on `df_train`**; the test set stays untouched until the very end.
- Phases run **in a chain**, one after another: the output (feature list) of one phase is the input of the next.
- Boruta is used in its **Boruta-SHAP** variant (SHAP importances instead of *gain*).

**Cross-cutting decision: a single estimator (XGBoost), not LightGBM**

With ≤ 10,000 rows speed is not a factor. XGBoost with `tree_method="hist"` already uses the same histogram algorithm that motivated LightGBM. Mixing two estimators introduces a *selection–model mismatch*: you would select variables according to LightGBM's inductive bias (its own binning, *leaf-wise* growth, different handling of interactions and NaNs) and then train XGBoost. You select **for** XGBoost, so you select **with** XGBoost.

---

## Phase map — what each phase controls

The "Selection-method family" column follows the taxonomy of Kuhn & Johnson (*Feature Engineering and Selection*, https://feat.engineering/): **Filter / Wrapper / Embedded (intrinsic)**, plus **Hybrid/iterative** and **Dimensionality reduction**. Filters analyze predictors once and pass the survivors on; wrappers iteratively search subsets guided by model performance; embedded/intrinsic methods select during model training (tree-based models qualify).

| Phase | Method | Selection-method family (Kuhn & Johnson) | Question it answers | Selection scope | Is train/val overfitting the key risk? |
|------|--------|------------------------------------------|---------------------|-----------------|-----------------------------------------|
| 1 | Spearman (threshold 0.85) | **Filter** — unsupervised, feature-vs-feature (high-correlation pruning) | Which features are redundant with each other? | Model-agnostic | **No** — fits no model |
| 2 | Boruta-SHAP | **Embedded / intrinsic** — tree-based importance + shadow test (all-relevant) | Which features beat noise? | All-relevant (inclusive) | **No** — controlled by *shadows* |
| 3 | Stability selection | **Hybrid / iterative** — resampling wrapped around an embedded base learner | Which features are selected consistently? | Robustness via resampling | **No** — resampling is the control |
| 4 | Backward selection (XGBoost) | **Wrapper** — greedy sequential backward elimination (SBS) | What is the minimal optimal subset? | Minimal-optimal, model-specific | **Yes — the dominant risk** |
| 5 *(added)* | Baseline check | *Validation / diagnostic* (not a selection family) | Does selection beat/match using everything? | Sanity check, via CV | Yes, via CV |

The progression is sound: **cheap → expensive**, **agnostic → model-specific**, **inclusive (all-relevant) → minimal (minimal-optimal)**. Note it also walks the Kuhn & Johnson families in increasing model-dependence: filter → embedded → hybrid → wrapper.

---

## Phase 1 — Spearman correlation (threshold 0.85)

### What the method does

Spearman measures **rank correlation**: instead of operating on the values, it operates on their sorted positions. It captures any **monotonic** relationship (not just linear), which makes it robust to monotonic transforms and to outliers.

Definition (no ties):

```
ρ = 1 − 6·Σ dᵢ² / (n·(n²−1))
```

where `dᵢ` is the rank difference of observation pair `i`. It is equivalent to Pearson correlation computed on the ranks.

### Family classification

**Filter, unsupervised (feature-vs-feature).** In the Kuhn & Johnson taxonomy this is a filter step — a one-time statistical analysis of the predictors, here unsupervised because it looks only at feature–feature redundancy, not at the target. It corresponds to "high pairwise correlation pruning".

### How it is applied here

- Compute the **feature–feature** `|ρ|` matrix (not feature–target).
- For each pair with `|ρ| ≥ 0.85`, keep **one** feature and drop the other: they are redundant for the model.
- "Without inferring null": no significance test on the coefficient; the magnitude threshold is used directly. Correct, because here we don't care whether the correlation is "statistically different from 0" but whether it is **practically high**.

### Overfitting analysis

**No overfitting is possible**: no model is fit, there is no train or validation. Spearman is a property of the joint distribution of the data. The train/val metric does not apply.

### Design critique and recommended adjustments

- **Trees tolerate collinearity.** XGBoost does not need decorrelation the way a linear model does: facing two correlated features it simply picks one at the split. So the real value of this phase is **not** "helping XGBoost converge" but **stabilizing phases 2 and 3**: when two features are highly correlated, they split the importance (SHAP or selection frequency), which dilutes the signal and can make both look weak in Boruta and stability.
- **Informed tie-break, not random.** When deciding which of the pair to keep, do not pick at random. Reasonable criteria: higher `|ρ_Spearman|` with the target, fewer NaNs, or lower cardinality/cost. Document the rule.
- **Real risk:** discarding a feature that was individually more predictive. That is why 0.85 is a prudent (high) threshold: it only removes near-total redundancy.

**Verdict:** keep it, understanding its justification is *stabilizing the downstream importance phases*, not a need of the final model.

---

## Phase 2 — Boruta-SHAP (keep Confirmed + Tentative)

### What the method does

Boruta is an **all-relevant** selection method: it looks for **all** features with real signal, not the minimal subset. Mechanics:

1. For each real feature, create a **shadow feature**: a copy with its values **randomly permuted**. By construction the shadow has no relationship with the target (it is noise with the same marginal distribution).
2. Train the model (here XGBoost) on real features + shadows.
3. Compute importances. In **Boruta-SHAP**, **SHAP values** are used instead of *gain*: SHAP attributes to each feature its marginal contribution averaged over coalitions, giving more consistent importances that are less biased toward high-cardinality features than *gain*.
4. A feature gets a **"hit"** in that iteration if its importance exceeds the **maximum** importance among **all** shadows.
5. Repeat over many iterations. Under the null hypothesis of irrelevance, the number of hits per feature follows a **Binomial(n_iter, 0.5)**. A test is applied:
   - **Confirmed:** hits significantly **above** chance.
   - **Rejected:** hits significantly **below**.
   - **Tentative (undecided):** neither → ambiguous signal.

### Family classification

**Embedded / intrinsic.** Boruta is built on the model's own (intrinsic) importance: tree-based models perform selection during training, and Boruta wraps that intrinsic importance with a shadow-feature significance test. In the Kuhn & Johnson scheme it sits with tree-based importance methods, extended into an all-relevant selector.

### Why keep Confirmed + Tentative

At this phase it pays to be **inclusive**: it is a wide sieve. Tentative features may carry weak or unstable signal that phases 3 and 4 will judge better. Dropping them here would prune too early.

### Overfitting analysis

**The train/val gap is not the relevant lens.** Boruta's logic is **self-referential**: it compares each feature against noise (shadows) trained in the **same** fit. If the model overfits and inflates importances, **it also inflates the shadows'** (which are noise with the same distribution), so the relative comparison stays valid. A moderately overfit model does not break Boruta.

Nuance: SHAP importances computed **on the train rows** reflect how the model *fit* train, and a heavily overfit model can over-credit noisy features it latched onto. The shadow control partially compensates, but not fully.

**Mitigations (preferable to an explicit train/val split inside Boruta):**
- Keep the estimator **regularized** (the `xgb_fast` with moderate `min_child_weight`, `subsample`/`colsample` < 1).
- Compute SHAP **out-of-fold** (on rows not used in that fit) for extra robustness.

**Verdict:** keep as is. I would not add a train/val split "to watch overfitting" because the shadow mechanism already serves that role; I would keep regularization and consider OOF SHAP.

---

## Phase 3 — Stability selection (threshold 0.7)

### What the method does

Stability selection (Meinshausen & Bühlmann) attacks a different problem from Boruta: **is the selection stable under small perturbations of the data?** A feature can beat noise in one fit yet be picked erratically depending on which rows enter.

Mechanics:

1. Generate `B` subsamples of `df_train` (typically 50% subsampling without replacement, or *bootstrap*).
2. On each subsample fit the model and determine a **selected set** (e.g. top-*k* by importance, or SHAP importance > 0). The **per-subsample selection rule must be defined explicitly**.
3. For each feature `k`, compute its **selection frequency**:

```
Π_k = (1/B) · Σ_b  1[ feature k selected in subsample b ]
```

4. Keep features with `Π_k ≥ π_thr`, here **0.7**.

### Family classification

**Hybrid / iterative.** Stability selection sits among the hybrid/iterative methods: it wraps an embedded base learner (here tree importance) inside a resampling loop and retains features chosen above a frequency threshold. It is neither a pure filter nor a single-pass wrapper — it combines resampling robustness with embedded importances.

### Theoretical guarantee

With `π_thr > 0.5`, the method bounds the expected number of false positives:

```
E[V] ≤ (1 / (2·π_thr − 1)) · q² / p
```

where `q` = mean number of features selected per subsample and `p` = total candidate features. This requires `π_thr > 0.5`; **0.7 is a solid choice** (false-positive control without being overly conservative).

### Overfitting analysis

**Resampling *is* the variance/overfitting control.** Only the **binary** output "was it selected in this subsample?" is used. Noise-driven selections appear intermittently and fail to clear the threshold; truly informative features reappear consistently. Watching the train/val gap inside each fit is **redundant** with what the phase already does by design: the cross-subsample frequency *is* the generalization proxy of the selection decision.

The parameters to watch are not train/val but: `B` (enough subsamples, e.g. 50–100), `q` (selection size per subsample) and `π_thr`.

### Design critique

- **Partial redundancy with Boruta.** Both are tree-importance-based filters. The key difference: Boruta answers "does it beat noise?" and stability answers "is it selected consistently?". A feature can pass Boruta and be unstable → stability **does** add new information, but the overlap is real.
- **Cost and overlap mitigation:** run stability **on the Boruta-reduced set** (as in this chain), which makes it cheap and turns it into a robustness filter rather than a from-scratch rediscovery. Correct.
- **Defining the per-subsample selection rule** is essential and often omitted. Recommended: top-*k* by SHAP, with *k* consistent with the `q` in the formula above.

**Verdict:** keep, run cheaply on the post-Boruta set, with a documented per-subsample selection rule.

---

## Phase 4 — Backward selection with XGBoost

### What the method does

**Minimal-optimal** and **model-specific** selection: start from the surviving features and **iteratively remove** the least useful one, measuring performance at each step, until you reach the smallest subset that does not degrade (or improves) the metric.

Mechanics (backward elimination, *greedy*):

1. Start with all candidate features. Compute the **cross-validated score**.
2. At each step, evaluate removing each remaining feature (or, more cheaply, remove the lowest-importance one) and keep the removal that **maximizes** the CV score.
3. Repeat, recording the CV score against the number of features.
4. Pick the final subset from the curve.

Here, unlike phases 2–3, the **real hyperparameters of the final model** are used (`min_child_weight=100`, `eta=0.01`, `n_estimators=200`, `max_depth=5`, etc.), because this phase must reflect the model that will be deployed.

### Family classification

**Wrapper.** This is the canonical wrapper: an iterative search procedure (sequential backward selection, SBS / stepwise) that repeatedly supplies feature subsets to the model and uses the resulting performance estimate to guide the next removal. It is the most model-dependent and most expensive family.

### Overfitting analysis — **the dominant risk of the whole chain**

This is the **only** phase where "watching train vs. validation overfitting" is exactly the right lens, for two distinct problems:

1. **Scoring on train is invalid.** Removing features almost never **worsens** the training error (the curve is monotonically decreasing or flat in flexibility), so a backward search guided by the train score is blind: it would always pick the full set or spurious decisions. **Cross-validated scoring is mandatory.**

2. **Selection-induced optimism (the subtler one).** Backward tries **many** subsets and keeps the best on validation. That "best of many" **overfits the validation estimate itself**: the CV score of the winning subset is optimistic relative to its true performance.

**Mandatory mitigations:**
- **CV with fixed folds** across all steps (same folds at every removal → fair comparisons).
- **Repeated CV** (several split seeds) to stabilize which feature to remove.
- **1-SE rule:** pick the **smallest** subset whose CV score is **within 1 standard error** of the best score, not the absolute-best score. This directly counters selection-induced optimism and favors parsimony.
- **Touch the test set only once** at the end, as an honest estimate. Never use it to guide removal.

### Design critique

- **Cost:** exhaustive backward is `O(p²)` fits. With the set already reduced by phases 1–3 it is affordable. If not, use **importance-based elimination** (always drop the lowest SHAP) instead of trying all: much cheaper, slightly less optimal.
- `min_child_weight=100` with ≤ 10,000 rows is **very aggressive**: each leaf requires ≥ 100 effective samples. With `max_depth=5` (up to 32 leaves) this prunes hard. It is strong regularization, defensible on small data to avoid overfitting, **but** during backward it can mask the usefulness of weak-signal features. Recommendation: either treat it as a hyperparameter to tune **after** fixing features, or relax it during this phase and restore it in the final model.

**Verdict:** keep, with repeated CV, fixed folds and the **1-SE rule** as non-negotiable safeguards.

---

## Phase 5 — *(Added)* Baseline check

### Why add it

After 4 phases of decisions taken on the **same** `df_train`, there is **accumulated optimism**: each phase picked what looked good on this data. The most basic evidence that the effort paid off is missing.

### What it does

- Compare, with the **same CV** as phase 4:
  - the score of the **final selected subset**, vs.
  - the score using **all features** (or the post-Spearman set).
- Acceptance criterion: the final subset must **match or beat** generalization with **fewer** variables. If it does not, some phase was too aggressive → review (typically backward with `min_child_weight=100`).
- *(Optional)* Re-run the whole pipeline with several seeds to measure the **stability of the selected set**. If the feature list changes a lot across seeds, the selection is not reliable.

---

## Recommended estimator configuration

```python
from xgboost import XGBRegressor

# Phases 2–3 (Boruta-SHAP, stability): cheap and regularized; here you only RANK
xgb_fast = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    tree_method="hist",
    learning_rate=0.05,      # higher than final: not tuning, just ranking
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,     # laxer than the final 100: don't prune weak signal yet
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=-1,
)

# Phase 4 (backward) and final model: REAL deployment hyperparameters
init_params = {
    "n_estimators": 200,
    "max_depth": 5,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "eta": 0.01,
    "seed": SEED,
    "gamma": 0,
    "min_child_weight": 100,  # review/tune after fixing features (see Phase 4)
}
```

---

## Cross-cutting diagnostics and risks

- **Data leakage:** all phases, **including Spearman**, must run only on `df_train`. If there is later an outer CV to evaluate the *full pipeline*, selection must go **inside** each fold of that CV, not before. Since here we work "only on `df_train`", the external test set remains an honest estimate and is touched only once.
- **Accumulated optimism:** 4 data-dependent decisions on the same train → the untouched test is the only unbiased estimate. Do not reuse it.
- **Spearman removes redundancy, not irrelevance:** do not confuse its role with Boruta's.
- **Boruta = all-relevant, Backward = minimal-optimal:** distinct, complementary goals; that is why the chain makes sense.
- **Seed stability:** with ≤ 10,000 rows, importances and selections can be sensitive to the split. Fix `random_state` everywhere and, if possible, verify stability across seeds.

## Critique summary

| Phase | Family | Decision | Reason |
|------|--------|----------|--------|
| 1 Spearman | Filter | **Keep** | Its real value is stabilizing 2–3, not decorrelating for the tree. Informed tie-break. |
| 2 Boruta-SHAP | Embedded | **Keep** | Shadow control makes watching train/val unnecessary. OOF SHAP optional. |
| 3 Stability | Hybrid/iterative | **Keep (cheap, post-Boruta)** | Resampling is the overfitting control. Partial overlap with Boruta, but adds robustness. Define the per-subsample selection rule. |
| 4 Backward | Wrapper | **Keep with safeguards** | **Only point where train/val overfitting is the central risk.** Repeated CV + fixed folds + 1-SE rule. Review `min_child_weight=100`. |
| 5 Baseline | Validation | **Add** | Evidence that selection matches/beats with fewer features. |

**LightGBM:** unnecessary across the whole flow. With ≤ 10,000 rows speed is not a factor and mixing estimators introduces a *selection–model mismatch*. Everything with XGBoost (`tree_method="hist"` in the cheap phases).

---

## Reference implementation (Python, functional)

Procedural/functional style, no OOP, functions only where they help. Real regression dataset (**California Housing**, built into scikit-learn, no external download), subsampled to ≤ 10,000 rows to match the context. `df` represents `df_train`: **everything runs only on it**.

> Requires `pip install xgboost shap scikit-learn scipy pandas numpy`. SHAP is mandatory because phase 2 is Boruta-**SHAP** and phase 3 uses SHAP importance for consistency.

### Setup, data and estimators

```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RepeatedKFold, cross_val_score
from xgboost import XGBRegressor
import shap

SEED = 42

# Real regression; subsample to <=10k. 'df' = df_train (sole source for selection).
df = fetch_california_housing(as_frame=True).frame.sample(8000, random_state=SEED).reset_index(drop=True)
TARGET = "MedHouseVal"
X, y = df.drop(columns=[TARGET]), df[TARGET]

def make_fast_model():
    # Phases 2-3: cheap and REGULARIZED. Here we only rank, not tune ->
    # high learning_rate, subsample/colsample <1, lax min_child_weight (don't prune weak signal yet).
    return XGBRegressor(
        n_estimators=300, max_depth=5, tree_method="hist", learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        objective="reg:squarederror", random_state=SEED, n_jobs=-1,
    )

def make_final_model():
    # Phase 4 and final model: REAL deployment hyperparameters.
    return XGBRegressor(
        n_estimators=200, max_depth=5, subsample=1.0, colsample_bytree=1.0,
        objective="reg:squarederror", learning_rate=0.01, gamma=0,
        min_child_weight=100, random_state=SEED, n_jobs=-1,
    )

def shap_importance(model, X):
    # Importance = mean |SHAP| per feature (more consistent than 'gain').
    sv = shap.TreeExplainer(model).shap_values(X)
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
```

### Phase 1 — Spearman (threshold 0.85)

`threshold=0.85` is high: it only removes near-total redundancy. **Informed** tie-break (keep the one more correlated with the target), not random.

```python
def spearman_filter(X, y, threshold=0.85):
    corr = X.corr(method="spearman").abs()
    tcorr = X.apply(lambda c: abs(spearmanr(c, y).statistic))  # |corr| with the target
    cols, drop = list(X.columns), set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in drop or b in drop or corr.loc[a, b] < threshold:
                continue
            drop.add(a if tcorr[a] < tcorr[b] else b)  # drop the less useful of the pair
    return [c for c in cols if c not in drop]
```

### Phase 2 — Boruta-SHAP

`n_iter=30` iterates shadows so the binomial test has power. Decision via binomial test vs. `p=0.5`. Keep **Confirmed + Tentative** (wide sieve).

```python
def boruta_shap(X, y, n_iter=30, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    hits = pd.Series(0, index=X.columns)
    for _ in range(n_iter):
        # Shadow = permuted copy (noise with the same marginal).
        shadow = X.apply(lambda c: rng.permutation(c.values)).add_prefix("shadow_")
        Xa = pd.concat([X.reset_index(drop=True), shadow], axis=1)
        imp = shap_importance(make_fast_model().fit(Xa, y), Xa)
        smax = imp[shadow.columns].max()                  # noise ceiling
        hits[X.columns] += (imp[X.columns] > smax).astype(int)
    def decide(h):
        if binomtest(h, n_iter, 0.5, alternative="greater").pvalue < alpha: return "Confirmed"
        if binomtest(h, n_iter, 0.5, alternative="less").pvalue   < alpha: return "Rejected"
        return "Tentative"
    status = hits.map(decide)
    return status[status != "Rejected"].index.tolist(), status
```

### Phase 3 — Stability selection (threshold 0.7)

`n_subsamples=50`, `frac=0.5` (subsampling without replacement). `top_k` defines the per-subsample selection rule (the `q` most important). `thr=0.7` satisfies the `>0.5` required by the false-positive bound.

```python
def stability_selection(X, y, n_subsamples=50, frac=0.5, top_k=None, thr=0.7, seed=SEED):
    rng = np.random.default_rng(seed)
    top_k = top_k or max(1, X.shape[1] // 2)   # q: features selected per subsample
    counts, n = pd.Series(0, index=X.columns), len(X)
    for _ in range(n_subsamples):
        idx = rng.choice(n, int(frac * n), replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
        imp = shap_importance(make_fast_model().fit(Xs, ys), Xs)
        counts[imp.nlargest(top_k).index] += 1
    freq = counts / n_subsamples
    return freq[freq >= thr].index.tolist(), freq
```

### Phase 4 — Backward selection (XGBoost) with repeated CV and 1-SE rule

**Greedy importance-based** elimination (`O(p)` fits, not `O(p²)`). **Repeated CV with fixed folds** and the **1-SE rule** against selection-induced optimism.

```python
def backward_selection(X, y, min_features=1, seed=SEED):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)  # fixed folds
    def cv_score(cols):
        s = cross_val_score(make_final_model(), X[cols], y, cv=cv,
                            scoring="neg_root_mean_squared_error")
        return s.mean(), s.std() / np.sqrt(len(s))   # mean and standard error
    feats, history = list(X.columns), []
    while True:
        history.append((list(feats), *cv_score(feats)))
        if len(feats) == min_features:
            break
        imp = shap_importance(make_final_model().fit(X[feats], y), X[feats])
        feats = [f for f in feats if f != imp.idxmin()]   # drop the least important
    best = max(history, key=lambda h: h[1])               # neg_rmse: higher is better
    thr = best[1] - best[2]                                # within 1 SE of the best
    chosen = min((h for h in history if h[1] >= thr), key=lambda h: len(h[0]))
    return chosen[0], history                              # most parsimonious subset
```

### Phase 5 — Baseline check

Evidence that selection **matches or beats** with fewer features (same CV).

```python
def baseline_check(X, y, selected, seed=SEED):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    def rmse(cols):
        s = cross_val_score(make_final_model(), X[cols], y, cv=cv,
                            scoring="neg_root_mean_squared_error")
        return -s.mean()
    return {"rmse_full": rmse(list(X.columns)), "n_full": X.shape[1],
            "rmse_selected": rmse(selected),    "n_selected": len(selected)}
```

### Orchestration (chain: each phase feeds the next)

```python
f1 = spearman_filter(X, y, threshold=0.85)
f2, status = boruta_shap(X[f1], y)
f3, freq    = stability_selection(X[f2], y, thr=0.7)
f4, history = backward_selection(X[f3], y)
report      = baseline_check(X, y, f4)

print("Phase 1 — Spearman      :", f1)
print("Phase 2 — Boruta-SHAP   :", f2, "\n  status:", status.to_dict())
print("Phase 3 — Stability     :", f3, "\n  freq  :", freq.round(2).to_dict())
print("Phase 4 — Backward final:", f4)
print("Phase 5 — Baseline      :", report)
# Accept the selection only if rmse_selected <= rmse_full (±SE) with fewer features.
```

> **Note:** California Housing has only 8 features, so the trimming will be modest — the code is **illustrative of the flow**. The pipeline's value grows with dimensionality. For reproducibility everything uses `random_state=SEED`; verify the selected set's stability by repeating with several seeds if the problem warrants it.
