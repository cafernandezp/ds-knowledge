# Feature-vs-Feature Association — Method Selection Summary

> **Created by:** Claude Sonnet 4.6
> **Date:** June 17 2026

---

> **Problem:** Pick the correct association measure for any pair of features, given their data types,
> to detect redundancy in a predictive-modeling pipeline.
>
> **Scope:** Consolidates four detailed reports (binary–binary, binary–continuous,
> categorical–continuous, categorical–binary) plus continuous–continuous and ordinal cases into one
> reference matrix.
>
> **Constraint:** standard Python stack (`pandas`, `scipy`, `sklearn`).

---

## Feature type definitions (read this first)

| Type | Definition (concise) | Order? | Distinct values | Simple example |
|---|---|---|---|---|
| **Binary** | Two categories, coded 0/1 | No (or trivial) | 2 | `has_purchased` ∈ {0, 1} |
| **Continuous** | Real-valued, measured on a numeric scale | Yes (full) | Many / infinite | `income` = 42 350.75 € |
| **Categorical (nominal)** | ≥3 categories with **no inherent order** | No | Few–many | `province` ∈ {Madrid, Barcelona, Sevilla} |
| **Ordinal** | Categories with a **meaningful order** but **unknown / unequal spacing** | Yes (rank only) | Few | `satisfaction` ∈ {low < medium < high} |

**Key distinctions that drive method choice:**
- **Nominal vs ordinal:** both are discrete categories; only ordinal has an order. `province` is nominal (Madrid is not "more" than Sevilla); `satisfaction` is ordinal (high > low). This determines whether rank-based correlation (Spearman) is meaningful.
- **Ordinal vs continuous:** both are ordered, but ordinal spacing is unknown (the gap low→medium need not equal medium→high), so you may only use the *ranks*, not the raw codes. → Spearman/Kendall, not Pearson.
- **Binary** is just a 2-level categorical; whether it is "nominal" (`province_madrid`) or "natural/ordinal" (`is_adult`) only matters when you consider rank-based methods.

---

## TL;DR / Recommendation

- **Continuous ↔ Continuous → Spearman** (robust default); Pearson only when linear + symmetric + outlier-free and you need r².
- **Binary ↔ Binary → φ** (= `df.corr(method="pearson")` on 0/1).
- **Binary ↔ Continuous → Spearman** (default) or point-biserial (= Pearson) if symmetric.
- **Categorical ↔ Continuous → η² (ANOVA)**; ε² (Kruskal–Wallis) if skewed.
- **Categorical ↔ {Categorical / Binary} → Cramér's V** (bias-corrected if high cardinality).
- **Anything involving Ordinal → Spearman / Kendall τ** (respects order, ignores spacing).
- **`df.corr()` is INVALID for any nominal categorical** — label codes have no real order.
- **Mutual information works for every combination** — see the dedicated section below for the caveats.

---

## The summary matrix

Symmetric — only the diagonal + upper triangle carry unique information.

| A \ B | Binary 0/1 | Continuous | Categorical (nominal) | Ordinal |
|---|---|---|---|---|
| **Binary 0/1** | **φ** = `corr(pearson)` | **Spearman** / point-biserial if symmetric | **Cramér's V** (= \|φ\| for 2×2) | **Spearman** (rank-biserial) |
| **Continuous** | Spearman / point-biserial | **Spearman** (Pearson if linear+symmetric) | **η² ANOVA** / ε² if skewed | **Spearman** |
| **Categorical (nominal)** | Cramér's V | η² ANOVA / ε² | **Cramér's V** (Bergsma corrected) | Cramér's V (ignores order) or η² |
| **Ordinal** | Spearman | Spearman | Cramér's V or η² | **Spearman / Kendall τ** |

**Color/family legend:**
- Correlation family (φ, Pearson, point-biserial, Spearman, Kendall) — for ordered or binary data.
- χ²-based (Cramér's V) — for categorical × categorical (binary = 2-level categorical).
- ANOVA-based (η², ε²) — for categorical × continuous.

**Transversal rule:** for any suspected **non-monotonic** dependence, no correlation in this matrix will detect it → use **mutual information** (see below).

---

## Continuous ↔ Continuous (the new cell)

**What.** Two real-valued features. The most common pairing and the home turf of Pearson and Spearman.

| Measure | Detects | Robust? | Python | Use when |
|---|---|---|---|---|
| **Pearson r** | Linear association | ❌ | `df.corr(method="pearson")` | Linear, symmetric, no outliers; need r² |
| **Spearman ρ** | Monotonic association | ✅ | `df.corr(method="spearman")` | **Default for EDA**; skew/outliers/unknown shape |
| **Kendall τ** | Concordance (monotonic) | ✅✅ | `df.corr(method="kendall")` | Small n (< 30) or many ties |
| **Distance corr.** | Any (incl. non-linear) | ✅ | `dcor.distance_correlation` | Suspect non-monotonic; need a [0,1] score |

**Pros / Risks.**
- Pearson is efficient and interpretable (r² = shared variance) but a single outlier can swing it by 0.5+; it returns ≈0 for a perfect non-linear relation like `y = x²`.
- Spearman captures any monotonic relation (`y = exp(x)` → ρ = 1, Pearson ≈ 0.7) and is robust to outliers, but misses non-monotonic shapes (`y = sin(x)` over [0, 2π] → ρ ≈ 0).
- Always plot — Anscombe's quartet: four datasets with identical r = 0.816 but radically different shapes.

**How / verify.**

```python
import numpy as np, pandas as pd
from scipy import stats

rng = np.random.default_rng(42)
x = rng.normal(size=500)
y = np.exp(x) + rng.normal(scale=0.1, size=500)   # monotonic but non-linear
df = pd.DataFrame({"x": x, "y": y})

print("Pearson :", round(df.corr(method="pearson").loc["x", "y"], 3))   # ~0.7
print("Spearman:", round(df.corr(method="spearman").loc["x", "y"], 3))  # ~1.0
print("Kendall :", round(df.corr(method="kendall").loc["x", "y"], 3))
```

**Redundancy threshold:** |r| or |ρ| > 0.8–0.9.

---

## Can mutual information be used for ALL combinations?

**Short answer: yes — MI is the one measure that applies to every pairing.** That is its main appeal as a universal screen. But "applies everywhere" does not mean "best everywhere."

**Why it is universal.** MI measures the reduction in uncertainty about one variable from knowing the other:

$$I(X; Y) = \sum_{x,y} p(x,y)\,\log\frac{p(x,y)}{p(x)\,p(y)}, \qquad I(X;Y) = 0 \iff X \perp Y$$

It makes **no assumption** about order, linearity, or monotonicity, so it is defined for binary, continuous, nominal, ordinal — any mix. It is the only entry in the matrix that detects **non-monotonic** dependence (`y = sin(x)`, U-shapes, variance-only differences).

**How to call it correctly per type — the catch:** sklearn's estimators need you to flag which features are discrete, because continuous and discrete MI use different estimators (k-NN for continuous, counting for discrete).

```python
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Continuous target  -> mutual_info_regression
# Discrete target    -> mutual_info_classif
# Flag discrete FEATURES via discrete_features=[True/False, ...]

X = ...          # 2D array of features
y_cont = ...     # continuous target
mi = mutual_info_regression(X, y_cont, discrete_features=[True, False],
                            random_state=0)   # 1st feature discrete, 2nd continuous

# Nominal features must be label-encoded first AND flagged discrete:
codes = LabelEncoder().fit_transform(cat_series).reshape(-1, 1)
mi_cat = mutual_info_classif(codes, y_discrete, discrete_features=[True], random_state=0)
```

**Why MI is not the universal default despite working everywhere:**

| Drawback | Consequence |
|---|---|
| **No natural [0,1] scale** | Raw MI is in nats/bits; thresholds are not comparable across pairs unless you normalize (NMI, adjusted MI). |
| **No sign** | Cannot tell positive from negative association — matters for redundancy reasoning. |
| **Estimation variance** | k-NN / binning estimates are sensitive to sample size, the `n_neighbors` setting, and the curse of dimensionality; small samples give noisy MI. |
| **Discrete-flag dependency** | Wrong `discrete_features` flag silently produces a wrong estimate (treats categories as a numeric scale). |
| **Cost** | Heavier than a vectorized `df.corr()` over a full matrix. |

**Verdict.** Use MI as a **secondary, general-purpose screen** — specifically when you suspect non-monotonic relationships that correlations miss, or when mixing wildly different types and want one consistent (if coarse) number. For the common, type-matched cases, the specialized measures in the matrix (φ, Pearson/Spearman, η², Cramér's V) are cheaper, signed, and better-calibrated. **MI is the safety net, not the first tool.**

---

## Diagnostics & Pitfalls (cross-cutting)

- **Always plot before dropping a feature** — scatter/box/violin reveal shapes a single coefficient hides.
- **Pearson–Spearman (or η²–ε²) gap = diagnostic** of skew/outliers → trust the robust (rank-based) member.
- **`df.corr()` on label-coded nominal data is meaningless** — the value changes if you relabel.
- **High-cardinality categoricals inflate η² and Cramér's V** → use ω² / Bergsma-corrected V, or collapse rare levels.
- **Pairwise is not multivariate:** two features with low pairwise association can still be jointly collinear with a third → check **VIF** after pair-based pruning.
- **r = 0 ≠ independence** — only zero *linear* (Pearson) or *monotonic* (Spearman) association; MI = 0 is the real independence test.
- **Leakage (target case):** compute any target-aware association inside CV folds, never on the full dataset.

---

## Decision rule (quick)

1. Identify each feature's type (binary / continuous / nominal / ordinal) using the definitions table.
2. Look up the pair in the matrix → use the listed measure.
3. Continuous involved + skew/outliers/unknown shape → prefer the **rank-based** option (Spearman / ε²).
4. Nominal involved → **never** `df.corr()`; use Cramér's V (with binary/categorical) or η²/ε² (with continuous).
5. Ordinal involved → **Spearman / Kendall τ** (keeps order, ignores spacing).
6. Suspect non-monotonic dependence, or mixing very different types → add **mutual information** as a second pass.
7. After pairwise pruning → check **VIF** for residual multicollinearity.

---

## References

1. Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection*. CRC Press. https://feat.engineering/
2. pandas — `DataFrame.corr` (pearson / spearman / kendall): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
3. SciPy — `pearsonr` / `spearmanr` / `kendalltau`: https://docs.scipy.org/doc/scipy/reference/stats.html
4. SciPy — `f_oneway` (ANOVA, η²) and `kruskal` (ε²): https://docs.scipy.org/doc/scipy/reference/stats.html
5. SciPy — `contingency.association` (Cramér's V): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html
6. scikit-learn — `mutual_info_regression` / `mutual_info_classif`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html
7. dcor — distance correlation: https://dcor.readthedocs.io/
8. Anscombe, F. J. (1973). "Graphs in Statistical Analysis." *The American Statistician*, 27(1).
