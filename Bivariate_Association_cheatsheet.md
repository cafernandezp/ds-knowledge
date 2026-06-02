# Bivariate Association Cheatsheet — Feature vs Feature

> **Created by:** Claude Opus 4.7
> **Date:** 6 May 2026
> **Purpose:** quick reference for detecting redundancy between predictors (feature vs feature) to decide which one to drop.

---

## Goal

Identify pairs of predictors that are strongly associated → drop one of them (typically the one with weaker relationship to `y`, more missing values, or higher computational cost).

## Key concepts before reading the table

- **Association vs causation:** every method here measures *statistical association* only. None of them prove that one feature causes the other.
- **Symmetry:** correlations (Pearson, Spearman, Kendall) and χ²-based measures (φ, Cramér's V) are symmetric — `cor(A, B) = cor(B, A)`. Use asymmetric measures (Theil's U, η, regression R²) when you care about directional dependence.
- **Linear vs monotonic vs general:** Pearson catches linear, Spearman catches monotonic (linear + non-linear ordered), MI catches *any* dependency including non-monotonic.
- **Effect size vs significance:** p-values shrink with n; with enough data, trivial correlations become "significant." Always evaluate the **magnitude** (|r|, V, MI) alongside any p-value.
- **Threshold ≠ ground truth:** the thresholds in the table (|r| > 0.8, V > 0.7) are conventional starting points. For final decisions, rank pairs by your measure and inspect the top few visually.

---

## Summary table

| Feature A type | Feature B type | Primary measure | Python function (library) | Null handling | Range / interpretation | Typical redundancy threshold |
|---|---|---|---|---|---|---|
| **Binary** | **Binary** | **φ (phi coefficient)** = √(χ²/n) | <ul><li>`df.corr(method="pearson", min_periods=30)` ✅ phi = Pearson on 0/1</li><li>`scipy.stats.chi2_contingency` → φ = √(χ²/n)</li><li>`sklearn.metrics.matthews_corrcoef`</li></ul> | `.corr` → pairwise deletion (OK for EDA). Otherwise → **impute with mode** before calling | [-1, 1] (φ) or [0, 1] (V) | \|φ\| > 0.7 |
| **Binary** | **Continuous** | **Point-biserial correlation** | <ul><li>`df.corr(method="pearson", min_periods=30)` ✅ point-biserial = Pearson with one binary var</li><li>`scipy.stats.pointbiserialr(binary, cont)`</li><li>Significance: `scipy.stats.ttest_ind`</li><li>Non-parametric: `scipy.stats.mannwhitneyu`</li></ul> | `.corr` → pairwise deletion. For scipy → **must drop or impute** (mode for binary, median for continuous) | [-1, 1] | \|r_pb\| > 0.7 |
| **Continuous** | **Continuous** | **Pearson r** (linear) | <ul><li>`df.corr(method="pearson", min_periods=30)` (pandas)</li><li>`scipy.stats.pearsonr` / `spearmanr` / `kendalltau`</li><li>Non-linear: `dcor.distance_correlation` (dcor pkg)</li></ul> | `.corr` → pairwise deletion (default). For downstream PCA / consistent matrix → **impute with median** (or KNN/iterative if >5% missing) | [-1, 1] | \|r\| > 0.8–0.9 |
| **Categorical nominal** | **Continuous** | **η (eta) / η²** from ANOVA | <ul><li>`df.corr(...)` ❌ not applicable directly</li><li>`pingouin.anova(...)` → returns η² directly</li><li>`scipy.stats.f_oneway` (manual η²)</li><li>Non-parametric: `scipy.stats.kruskal`</li><li>MI: `sklearn.feature_selection.mutual_info_regression`</li></ul> | scipy/pingouin → **drop NaN per group** or impute (median for continuous; for categorical NaN can be treated as its own level `"Missing"`) | η ∈ [0,1]; η² = explained variance | η > 0.7 or η² > 0.5 |
| **Categorical nominal** | **Categorical nominal** | **Cramér's V** (χ²-based) | <ul><li>`df.corr(...)` ❌ not applicable</li><li>`scipy.stats.contingency.association(table, method="cramer")` (scipy ≥ 1.7)</li><li>`pingouin.chi2_independence`</li><li>MI: `sklearn.metrics.normalized_mutual_info_score`</li></ul> | Build the contingency table with `pd.crosstab(..., dropna=False)` → **treat NaN as its own category** (often the best signal); otherwise impute with mode | [0, 1] | V > 0.7 |
| **Ordinal** | **Ordinal** | **Spearman ρ** or **Kendall τ** | <ul><li>`df.corr(method="spearman", min_periods=30)` or `method="kendall"` (pandas)</li><li>`scipy.stats.spearmanr` / `kendalltau`</li></ul> | `.corr` → pairwise deletion (OK if MCAR). Otherwise → **impute median rank** or median value | [-1, 1] | \|ρ\| > 0.8 |
| **Ordinal** | **Continuous** | **Spearman ρ** | <ul><li>`df.corr(method="spearman", min_periods=30)`</li><li>`scipy.stats.spearmanr`</li></ul> | `.corr` → pairwise deletion. For scipy → **impute with median** before calling | [-1, 1] | \|ρ\| > 0.8 |
| **Ordinal** | **Nominal** | **Cramér's V** (ignores order) or η if ordinal treated as score | <ul><li>`df.corr(...)` ❌ not applicable</li><li>`scipy.stats.contingency.association(...)` (Cramér's V)</li><li>η-treatment: same tools as nominal-vs-continuous</li></ul> | Same as Cramér's V row → **NaN as its own category** via `pd.crosstab(dropna=False)` | [0, 1] | V > 0.7 |
| **Any** | **Any** (general) | **Mutual Information (MI)** | <ul><li>`df.corr(...)` ❌ not applicable</li><li>`sklearn.feature_selection.mutual_info_regression` (continuous y) / `mutual_info_classif` (discrete y)</li><li>MIC: `minepy.MINE`</li><li>Distance corr: `dcor.distance_correlation`</li></ul> | sklearn estimators **do not accept NaN** → **must impute** (median for numeric, mode for categorical) or use `SimpleImputer`/`IterativeImputer` | MI ≥ 0; normalized [0, 1] | MI_norm > 0.7 |
| **Multivariate** (1 feature vs all others) | — | **VIF** (Variance Inflation Factor) | <ul><li>`df.corr(...)` ❌ not the right tool</li><li>`statsmodels.stats.outliers_influence.variance_inflation_factor`</li></ul> | **Must impute** (median for numeric); VIF fits OLS internally and fails on NaN. Standardize features first | VIF ≥ 1 | VIF > 5–10 → multicollinearity |

---

## Null handling — quick reference

### Imputation strategy by variable type

| Variable type | Recommended strategy | sklearn / pandas call |
|---|---|---|
| **Continuous (symmetric)** | Mean | `SimpleImputer(strategy="mean")` |
| **Continuous (skewed or with outliers)** | **Median** ← default choice | `SimpleImputer(strategy="median")` or `df.fillna(df.median())` |
| **Binary / categorical** | Mode (most frequent) | `SimpleImputer(strategy="most_frequent")` |
| **Categorical with informative missingness** | Treat NaN as its own level `"Missing"` | `df[col].fillna("Missing")` or `pd.crosstab(..., dropna=False)` |
| **>5% missing** in any column | **Iterative / KNN imputation** | `IterativeImputer()`, `KNNImputer(n_neighbors=5)` |
| **>20% missing** or MNAR suspected | Impute + **add binary missing-indicator** | `MissingIndicator()` alongside imputer |

### `min_periods` — what value to use?

Minimum number of valid pairs required for `.corr()` to return a value (otherwise NaN). Educated defaults:

| Sample size n | Suggested `min_periods` | Rationale |
|---|---|---|
| n < 100 | `min_periods=20` | Floor for stable correlation |
| 100 ≤ n < 1000 | `min_periods=30` | Common rule of thumb (CLT kicks in) |
| 1000 ≤ n < 10k | `min_periods=50` | Tighter to flag sparse pairs |
| n ≥ 10k | `min_periods=100` or `0.01 * n` | Avoid trusting correlations from <1% of data |

```python
df.corr(method="spearman", min_periods=30)  # cells with <30 valid pairs → NaN
```

**What `.corr` does with NaN by default:** pairwise deletion — drops rows where either column in the pair is NaN. Fine for EDA / screening, but the resulting matrix may be inconsistent (different cells use different row subsets) and **not positive semi-definite** → impute before using it for PCA, factor analysis, or Mahalanobis distance.

---

## Practical notes

### When `corr(method="pearson")` is mathematically valid

Pandas' `df.corr(method="pearson")` produces a **valid** result in all of these cases — they are all mathematically equivalent to specialized formulas:

| Variable types | Same as | Valid with `corr(method="pearson")`? |
|---|---|---|
| Binary (0/1) ↔ Binary (0/1) | **φ phi coefficient** | ✅ Yes — identical result |
| Binary (0/1) ↔ Continuous | **Point-biserial r** | ✅ Yes — identical result |
| Continuous ↔ Continuous | Pearson r | ✅ Yes — its primary use |
| Categorical (>2 levels) ↔ anything | — | ❌ **No** — requires one-hot, Cramér's V, or η² |
| Ordinal with non-linear spacing | — | ⚠️ Use Spearman/Kendall instead |

**Why the equivalence holds:** Pearson, phi, and point-biserial are all special cases of the same formula `cov(X,Y) / (σ_X · σ_Y)`. The binary case just plugs in 0/1 values, and the math collapses into the named statistics.

**Caveat — encoding matters:** for phi to be correct, binary variables must be coded as 0/1 (or any two distinct numerics). Strings like `"yes"/"no"` won't work — encode first.

---

## Deep dive — Pearson and Spearman across data types

Pearson and Spearman are the two correlations you'll reach for 90% of the time. Knowing **exactly** where each is valid (and where each silently breaks) is essential.

### Pearson r — applicability matrix

**What it measures:** strength and direction of the **linear** relationship between two numeric variables.
**Formula:** `r = cov(X, Y) / (σ_X · σ_Y)` — covariance normalized by standard deviations.
**Assumptions for the parametric interpretation:** linearity, normality of both variables (for CI/p-values), homoscedasticity, no influential outliers.

| Combination | Valid? | Notes |
|---|---|---|
| Continuous ↔ Continuous | ✅ Primary use | Best when relationship is roughly linear and no extreme outliers |
| Binary (0/1) ↔ Binary (0/1) | ✅ = φ coefficient | Encoding must be numeric (0/1); strings fail |
| Binary (0/1) ↔ Continuous | ✅ = Point-biserial | Same caveat — encode binary as 0/1 |
| Ordinal (equally spaced) ↔ Continuous | ⚠️ Acceptable | Only if integer codes reflect true equal spacing (e.g., 1–5 Likert with constant intervals); otherwise Spearman |
| Ordinal ↔ Ordinal (equal spacing both) | ⚠️ Acceptable | Same caveat |
| Ordinal (unequal spacing) ↔ anything | ❌ Misleading | Codes are arbitrary → r reflects encoding, not data |
| Nominal (>2 levels) ↔ anything | ❌ Invalid | Categorical codes have no order; r is meaningless |
| Heavy-tailed / outlier-prone numeric | ⚠️ Unstable | A single outlier can swing r by 0.5+; use Spearman or winsorize |
| Non-linear monotonic relationship | ⚠️ Underestimates | Will return low r even when association is perfect monotonic |

**Killer example:** for `y = x²` over `x ∈ [-1, 1]`, Pearson r ≈ 0 even though `y` is **perfectly determined** by `x`. Pearson doesn't detect non-linear association.

**Anscombe's quartet:** four datasets with identical Pearson r = 0.816 but radically different shapes (one linear, one curve, one outlier-driven, one with leverage). **Lesson: always plot before trusting r.**

---

### Spearman ρ — applicability matrix

**What it measures:** strength and direction of the **monotonic** relationship — does Y consistently increase (or decrease) with X, regardless of whether the increase is linear?
**How it works:** Pearson r computed on **ranks** instead of raw values.
**Assumptions:** monotonicity (if you care about direction); no assumption of linearity, normality, or homoscedasticity.

| Combination | Valid? | Notes |
|---|---|---|
| Continuous ↔ Continuous | ✅ Best general-purpose choice | Robust to outliers and non-normality |
| Ordinal ↔ Ordinal | ✅ Natural domain | Ranks are already what Spearman computes |
| Ordinal ↔ Continuous | ✅ Recommended | Spacing of ordinal codes doesn't matter — only order |
| Binary (0/1) ↔ Continuous | ⚠️ Works but suboptimal | Equivalent to **rank-biserial**; point-biserial or Mann-Whitney are more interpretable |
| Binary (0/1) ↔ Binary (0/1) | ⚠️ Degenerate | Reduces to φ with tie corrections; conceptually "monotonic" doesn't apply to 2 points |
| Nominal (>2 levels) ↔ anything | ❌ Invalid | Ranks of nominal categories are arbitrary |
| Non-monotonic relationship (U-shape, sinusoidal) | ⚠️ Misses it | Spearman ≈ 0 for perfect U-shape; use MI or distance correlation |
| Many tied values | ⚠️ Power loss | Ties get averaged ranks; reduces sensitivity. Use `scipy.stats.spearmanr` which has tie-aware p-values |

**Killer example:** `y = exp(x)` — perfectly monotonic but exponential. Pearson r ≈ 0.7, Spearman ρ = 1.0. Spearman captures it perfectly because it only cares about order.

**Where Spearman fails:** any **non-monotonic** relationship. Example: `y = sin(x)` over `[0, 2π]` — Spearman ρ ≈ 0 despite a clear functional relationship. Reach for **mutual information** or **distance correlation** when you suspect non-monotonic patterns.

---

### Pearson vs Spearman — decision guide

| Situation | Recommendation |
|---|---|
| Data looks roughly linear, no outliers, ≥ interval scale | **Pearson** (slightly more efficient if assumptions hold) |
| Outliers present, skewed distribution, or unknown shape | **Spearman** (default for EDA) |
| Ordinal data | **Spearman** (Pearson assumes equal spacing) |
| Small sample (n < 30) and ties | **Kendall τ** (more robust than Spearman) |
| Non-monotonic relationship suspected | **Mutual Information** or **distance correlation** (neither correlation will detect it) |
| Need interpretable effect size for reporting | Pearson (variance explained = r²) |
| Many variables, fast screening | Spearman — robust default, computed in one call: `df.corr(method="spearman")` |

**Rule of thumb for EDA:** start with Spearman as the default for feature redundancy screening. Drop to Pearson only when you've verified linearity and need r² for reporting.

---

### Common pitfalls (apply to both)

- **r = 0 does not mean independence** — only zero *linear* (Pearson) or *monotonic* (Spearman) association. Variables can be strongly dependent and still show r ≈ 0.
- **Correlation ≠ causation** — even |r| = 0.99 says nothing about which variable causes the other, or whether a confounder drives both.
- **Sample size matters** — with n = 10, r = 0.5 isn't significant; with n = 10,000, r = 0.05 is "significant" but useless. Always separate **statistical significance** from **practical/effect size**.
- **Restricted range attenuates r** — if you subset to a narrow band of X, the observed correlation drops even if the underlying relationship is unchanged.
- **Confounding by a third variable** — high pairwise r between A and B can be entirely driven by C. Use partial correlation (`pingouin.partial_corr`) when you need to control for a covariate.
- **Always visualize** — Anscombe's quartet should be every analyst's screensaver. Pair correlations with scatterplots (`sns.pairplot`, `pd.plotting.scatter_matrix`) before deciding to drop features.

---

### Why redundancy thresholds differ by measure

You'll notice the table uses |r| > 0.8–0.9 for Pearson/Spearman but |φ| > 0.7 or V > 0.7 for χ²-based measures. Reasons:

- **Pearson/Spearman scale linearly with shared variance:** r² = 0.64 (r = 0.8) means 64% of variance is shared; r² = 0.81 (r = 0.9) means 81%.
- **Cramér's V and φ are bounded by the table dimensions:** V = 0.7 in a 2×2 table already implies strong dependence; the upper bound is rarely close to 1.
- **MI has no natural scale** unless normalized; thresholds depend on the normalization (NMI vs adjusted MI).

When in doubt: rank pairs by your measure of choice and inspect the top ~5–10% rather than relying on a fixed cutoff.

---

### Other practical notes

- **Phi vs Cramér's V:** φ is the 2×2 case; Cramér's V generalizes to r×c tables. Both derive from χ². For 2×2 tables, φ = signed version of V.
- **η (eta):** square root of ANOVA's η²; measures how much of the continuous variable's variance is explained by the categorical predictor. Analogous to r² but for categorical → continuous.
- **MI captures non-linearities** that correlations miss; cost: estimation more sensitive to sample size, binning choices, and curse of dimensionality.
- **VIF complements pairwise analysis:** two features may have low pairwise |r| but together create collinearity with a third → check VIF *after* pair-based filtering. Iterate: drop highest-VIF feature, refit, repeat until all VIF < 5–10.
- **Kendall τ vs Spearman ρ:** Kendall is more robust to small samples and ties, has a more direct probabilistic interpretation (P(concordant) − P(discordant)), but is O(n²) vs Spearman's O(n log n). Use Kendall when n < 30 or when ties dominate.
- **Symmetry:** Pearson, Spearman, Kendall, φ, Cramér's V are all **symmetric** — cor(X, Y) = cor(Y, X). Theil's U and η are **asymmetric** — useful when you want directional dependence.

---

## Decision rule for dropping (when a redundant pair is detected)

1. Keep the feature with **stronger association to `y`** (relevance).
2. Tie → keep the one with **lower % missing**.
3. Still tie → keep the **more interpretable** or **cheaper to obtain** feature.
4. If one is derived from the other → drop the derived one (unless it clearly adds non-linear signal).

---

## Quick Python implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

def phi_coefficient(x, y):
    # x, y binary (0/1)
    return np.corrcoef(x, y)[0, 1]

def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def point_biserial(binary, continuous):
    return stats.pointbiserialr(binary, continuous).correlation

def eta_squared(categorical, continuous):
    # ANOVA-based: between-group var / total var
    groups = [continuous[categorical == c] for c in np.unique(categorical)]
    grand_mean = continuous.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((continuous - grand_mean) ** 2).sum()
    return ss_between / ss_total

def vif(df_numeric):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    return pd.Series(
        [variance_inflation_factor(df_numeric.values, i)
         for i in range(df_numeric.shape[1])],
        index=df_numeric.columns
    )
```

---

## Workflow example — mixed-type redundancy screening

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 0) Type inventory ---
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
bin_cols = [c for c in num_cols if df[c].nunique() == 2]

# --- 1) Impute (median for numeric, mode for categorical) ---
df_num = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(df[num_cols]),
    columns=num_cols
)
df_cat = df[cat_cols].fillna("Missing")

# --- 2) Numeric ↔ numeric: Spearman (robust default) ---
corr_num = df_num.corr(method="spearman", min_periods=30)
upper    = corr_num.where(np.triu(np.ones(corr_num.shape), k=1).astype(bool))
pairs    = upper.stack().reset_index()
pairs.columns = ["feat_a", "feat_b", "rho"]
redundant_num = pairs[pairs["rho"].abs() > 0.85]

# --- 3) Categorical ↔ categorical: Cramér's V ---
def cramers_v(x, y):
    confusion = pd.crosstab(x, y, dropna=False)
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

cat_pairs = [
    (a, b, cramers_v(df_cat[a], df_cat[b]))
    for i, a in enumerate(cat_cols) for b in cat_cols[i+1:]
]
redundant_cat = [(a, b, v) for a, b, v in cat_pairs if v > 0.7]

# --- 4) VIF on survivors (after pair-based pruning) ---
def vif_scores(X):
    return pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns
    )

survivors = [c for c in num_cols if c not in to_drop_pair]
vif = vif_scores(df_num[survivors])
# Iterate: drop highest VIF, refit, until all < 10
while vif.max() > 10:
    worst = vif.idxmax()
    survivors.remove(worst)
    vif = vif_scores(df_num[survivors])

# --- 5) Always visualize before final decisions ---
# import seaborn as sns; sns.heatmap(corr_num, cmap="coolwarm", center=0)
# sns.pairplot(df_num[top_pairs])
```

---

## Glossary of abbreviations

| Acronym | Full name |
|---|---|
| ANOVA | Analysis of Variance |
| AUC | Area Under the (ROC) Curve |
| MI | Mutual Information |
| MIC | Maximal Information Coefficient |
| Matthews CC | Matthews Correlation Coefficient |
| VIF | Variance Inflation Factor |
| r_pb | Point-biserial correlation |
| η, η² | Eta, eta-squared (ANOVA effect size) |
| φ | Phi coefficient |
| ρ | Spearman's rho |
| τ | Kendall's tau |
| γ | Goodman-Kruskal gamma |
