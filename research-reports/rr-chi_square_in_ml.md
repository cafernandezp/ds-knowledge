# Chi-Square Test in Machine Learning — Goodness-of-Fit, Independence, Feature Selection, and Drift Detection

> **Context.** χ² tests compare observed vs. expected *frequency counts* across discrete categories.
> Three ML use cases share the same underlying statistic: (1) **goodness-of-fit** — does an empirical
> distribution match a fixed reference (Benford, uniform, a known population)? (2) **independence** —
> are two categorical variables associated (e.g. feature vs. target)? (3) **two-sample comparison** —
> has a categorical/binned feature's distribution shifted between train and production (drift)?
> **Hard constraint: χ² applies to counts over discrete categories, not raw continuous values.**

## TL;DR

- Use χ² whenever you have **counts in discrete bins/categories** — not continuous values directly.
- Same formula, three flavors, different degrees of freedom: goodness-of-fit (1 sample vs. fixed
  distribution), independence (contingency table, r×c), two-sample drift (mathematically a
  contingency test with "period" as one axis).
- `sklearn.feature_selection.chi2` = independence test between a **non-negative** feature and the
  class label; used for filter-based selection, mainly on count/one-hot/text features.
- For drift monitoring: χ² is the default for **categorical or low-cardinality numerical** columns;
  use Kolmogorov-Smirnov / PSI / Wasserstein for continuous features.
- p-value scales with `n` — at large sample sizes even trivial, operationally irrelevant shifts
  become "significant" (p < 0.001). Always pair p with an **effect size** (Cramér's V, PSI, MAD).
- Rule of thumb: expected count ≥ 5 per cell; below that, use Fisher's exact test or a permutation
  p-value instead.

## Comparison

| Variant | Tests | Data shape | df | Typical ML use |
|---|---|---|---|---|
| Goodness-of-fit | Sample vs. fixed distribution | 1 categorical var, k categories | k−1 | Benford analysis, checking synthetic-data realism, class-balance checks |
| Independence | Association between 2 categorical vars | r×c contingency table | (r−1)(c−1) | Feature vs. target dependence, A/B test on categorical outcome |
| Feature selection (`sklearn.feature_selection.chi2`) | Dependence: non-negative feature vs. class | features × classes | per-feature | Filter selection for text/count features before high-dim models |
| Two-sample drift | Reference vs. current distribution | 2×k table | k−1 | Categorical/binned feature monitoring in production |

## 1. Goodness-of-fit test (1-sample)

**What.** Tests whether observed category counts match a fixed, known distribution (Benford, uniform,
a published population split).

**Pros.**
- Closed-form, no need for a second empirical sample — the "expected" side is a formula or table.
- Cheap to compute, interpretable df = k−1.

**Risks.**
- Sensitive to `n`: p-value shrinks toward 0 as sample size grows even for small deviations.
- Requires expected count ≥ 5 per cell (rule of thumb) or the χ² approximation to the true
  sampling distribution breaks down.
- Only tests the categorical marginal — ignores any ordering or magnitude information within a bin.

**How / verify.**
```python
from scipy.stats import chisquare
import numpy as np

observed = np.array([2950, 1700, 1090, 930, 860, 640, 570, 570, 690])  # first-digit counts, e.g. Benford
expected_p = np.log10(1 + 1 / np.arange(1, 10))
expected = expected_p * observed.sum()

chi2, p = chisquare(observed, expected)
mad = np.mean(np.abs(observed / observed.sum() - expected_p))  # effect size, n-independent
```

## 2. Test of independence (contingency table)

**What.** Tests whether two categorical variables are associated — e.g. a binned feature vs. a
class label, or treatment group vs. categorical outcome.

**Pros.**
- Works directly on a cross-tab; no distributional assumption on the underlying continuous variable.
- Foundation for filter-based feature selection and simple A/B analysis on categorical outcomes.

**Risks.**
- Detects *association*, not its direction or strength — always report an effect size (Cramér's V).
- Low-count cells inflate Type-I error; use Yates' continuity correction for 2×2 tables, or Fisher's
  exact test / a permutation-based p-value for small `n`.

**How / verify.**
```python
import pandas as pd
from scipy.stats import chi2_contingency

table = pd.crosstab(df['feature_bucket'], df['target'])
chi2, p, dof, expected = chi2_contingency(table)  # correction=True by default for 2x2 tables
```

## 3. Filter-based feature selection (`sklearn.feature_selection.chi2`)

**What.** Computes the χ² statistic between each **non-negative** feature and the class label;
ranks features by dependence with the target. Primarily used on count/one-hot/text features
(e.g. term counts in document classification). Continuous features must be binned first
(e.g. with `KBinsDiscretizer`) — the function requires non-negative integer-like values.

**Pros.**
- Fast, model-agnostic, scales well as a pre-filter before high-dimensional models (bag-of-words).
- Simple `SelectKBest`/`SelectPercentile` integration in scikit-learn pipelines.

**Risks.**
- Silently wrong on raw continuous or negative-valued features — must discretize first.
- Ignores feature-feature interactions (univariate filter).
- Ranking many features by raw p-value without correction inflates false positives — apply
  Benjamini-Hochberg (`statsmodels.stats.multitest.multipletests`) when selecting from hundreds
  of candidates.

**How / verify (leakage-safe: bin + select fit on train only).**
```python
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('bin', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')),
    ('select', SelectKBest(chi2, k=20)),
])
X_train_sel = pipe.fit_transform(X_train, y_train)   # fit on train only
X_test_sel = pipe.transform(X_test)                  # reuse train bin edges, no leakage
```

## 4. Two-sample drift test (categorical / binned features in production)

**What.** Compares the reference (train) distribution of a categorical or low-cardinality numerical
feature against the current (production) distribution. Mathematically a contingency test with
"period" (reference vs. current) as one axis. This is the default method for categorical and
low-cardinality numerical columns in open-source monitoring libraries (e.g. Evidently AI uses
chi-square for categorical columns and Kolmogorov-Smirnov for numerical columns with more than 5
unique values, by default). <cite index="40-1">It is based on column type: categorical, numerical, text data or embeddings, the number of observations in the reference dataset, and the number of unique values in the column, and for categorical columns or numerical columns with 5 or fewer unique values it uses the chi-squared test.</cite>

**Pros.**
- Well-understood, closed-form, widely implemented default in ML-monitoring tooling.
- Directly reusable from the goodness-of-fit / independence machinery above.

**Risks.**
- With large production batches (tens of thousands of rows/day), p < 0.001 becomes almost
  guaranteed regardless of whether the shift is operationally meaningful — **never gate an alert
  on p-value alone at scale**; pair with Cramér's V or PSI and a business threshold.
- Wrong tool for continuous features unless explicitly binned (binning itself is a modeling choice
  that can hide or manufacture drift depending on bin edges).

**How / verify.**
```python
from scipy.stats import chi2_contingency
import numpy as np

def categorical_drift(ref_counts, cur_counts):
    table = np.array([ref_counts, cur_counts])
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))  # effect size, independent of n
    return chi2, p, cramers_v
```

## Problem-specific considerations

- Benford fraud-detection case (discussed earlier in this thread): goodness-of-fit variant,
  comparing observed first-digit counts of transaction amounts against the fixed log10 distribution
  — a categorical variable with 9 levels, which is exactly what χ² is built for.
- If the actual object of interest is the raw continuous amount (not its first digit), χ² is the
  wrong primary tool — binning loses information; prefer Kolmogorov-Smirnov, Wasserstein distance,
  or PSI on the raw values instead.
- Monitoring many columns at once (dashboard-style) → correct for multiple comparisons
  (Benjamini-Hochberg) before flagging "drifted" features, or false alarms accumulate with the
  number of monitored columns.

## Diagnostics & pitfalls

- Expected count < 5 in any cell → χ² approximation is unreliable; merge categories or switch to
  Fisher's exact test / a permutation-based p-value.
- p-value is not an effect size. Always report Cramér's V (independence/drift) or MAD
  (goodness-of-fit) alongside p — decisions should be threshold-based on the effect size, not on
  "p < 0.05" alone, especially at large `n`.
- χ² assumes independent observations. Repeated records from the same entity (same customer,
  multiple transactions) violate this and understate the true p-value.
- `sklearn.feature_selection.chi2` requires non-negative, effectively integer-like feature values —
  verify dtype/range before calling; pass continuous features through `KBinsDiscretizer` first. <cite index="24-1">The function requires non-negative integer feature values such as booleans or frequencies, and if some features are continuous they need to be binned first, for example using KBinsDiscretizer.</cite>
- Train/val/test separation: fit any discretizer/binner used ahead of χ² feature selection on the
  training fold only; apply the same bin edges to validation/test/production to avoid leakage and
  to keep drift comparisons apples-to-apples.

## Decision rule

1. Counts in discrete categories, 1 sample vs. a known/fixed distribution → goodness-of-fit
   (`scipy.stats.chisquare`).
2. Two categorical variables, want to test association → independence test
   (`scipy.stats.chi2_contingency`).
3. Categorical/count feature vs. class label, ranking many candidate features → `sklearn`'s `chi2`
   inside a `Pipeline` with a discretizer for continuous inputs.
4. Comparing train vs. production distribution of a categorical/low-cardinality feature → two-sample
   χ² + Cramér's V; switch to KS test or PSI for continuous features.
5. Any expected cell count < 5, or `n` large enough that p-values saturate near 0 → make the
   decision on effect size (Cramér's V / MAD / PSI), not on the p-value alone.

## References

1. SciPy — `scipy.stats.chi2_contingency`, SciPy v1.18 Manual. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
2. SciPy — `scipy.stats.chisquare`, SciPy Manual. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
3. scikit-learn — `sklearn.feature_selection.chi2`, scikit-learn stable documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
4. Evidently AI — Data drift algorithm (default test selection by column type). https://docs-old.evidentlyai.com/reference/data-drift-algorithm
5. Evidently AI — Data and prediction drift in ML, ML observability course. https://learn.evidentlyai.com/ml-observability-course/module-2-ml-monitoring-metrics/data-prediction-drift-in-ml
6. Giskard — How to test categorical data drift in ML systems. https://www.giskard.ai/knowledge/how-to-test-ml-models-2-n-categorical-data-drift
