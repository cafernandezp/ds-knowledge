# Binary–Binary Correlation — Phi, Cramér's V, and Spearman Compared

> **Created by:** Claude Sonnet 4.6
> **Date:** June 17 2026

---

> **Problem:** Measure the statistical association between two binary (0/1) features to detect
> redundancy in a predictive-modeling context.
>
> **Assumptions:** both features are encoded as numeric 0/1 integers; no ordering is implied;
> goal is feature-vs-feature redundancy screening (no target involved).
>
> **Constraints:** need a single, correct, efficient call in a standard Python stack.

---

## TL;DR / Recommendation

- **Use phi (φ)** — it is the canonical, mathematically exact measure for 2×2 binary pairs.
- `df.corr(method="pearson")` on 0/1 columns **returns φ exactly** — one efficient call for the whole matrix.
- **Why Pearson is OK here but Spearman is not:** Pearson on 0/1 data is *not* assuming a numeric scale — its formula `cov(X,Y)/(σ_X σ_Y)` collapses algebraically into φ, which measures **co-occurrence** of the two categories, not order. So "use Pearson" really means "use φ." Spearman, by contrast, explicitly assumes a *monotonic order* between values — an assumption that is meaningless for an arbitrary 0/1 label. Same input, different assumptions: Pearson's reduces to a valid categorical statistic; Spearman's does not.
- **Cramér's V** is identical to |φ| for 2×2 tables — valid, but unsigned and slightly more expensive.
- **Spearman ρ** is theoretically applicable but degenerate for binary data: it reduces to φ with tie corrections, loses the sign, and wastes compute. **Do not use it for binary–binary pairs.**
- **Feature origin matters for Spearman, not for φ:** for one-hot / nominal binaries (`province_madrid`), the 0/1 labels are arbitrary so Spearman is *semantically invalid*; for natural binaries (`has_purchased`) it is defensible but still degenerates to φ. Either way → use `df.corr(method="pearson")` (φ).
- Redundancy threshold: **|φ| > 0.7** (equivalent to sharing ≥ 49% of variance).

---

## Comparison Table

| Measure | Captures sign? | Semantically valid for binary? | Python call | Computational cost | Recommended? |
|---|---|---|---|---|---|
| **φ (Pearson on 0/1)** | ✅ Yes | ✅ Yes — designed for 0/1 | `df.corr(method="pearson")` | O(n) | ✅ **Yes — primary** |
| **Cramér's V** | ❌ No (always ≥ 0) | ✅ Yes (= \|φ\| for 2×2) | `chi2_contingency` + formula | O(n) + overhead | ✅ Yes — if unsigned is fine |
| **Spearman ρ** | ✅ Yes | ⚠️ No — monotonicity is meaningless with only 2 distinct values; degenerates to φ with tie-correction overhead | `df.corr(method="spearman")` | O(n log n) rank sort | ⚠️ Avoid for binary–binary |
| **Kendall τ** | ✅ Yes | ⚠️ No — concordance/discordance pairs degenerate entirely under massive ties; O(n²) for no gain | `df.corr(method="kendall")` | O(n²) | ❌ Do not use |
| **Matthews CC (MCC)** | ✅ Yes | ✅ Yes (= φ exactly) | `sklearn.metrics.matthews_corrcoef` | O(n) | ✅ Equivalent alias |

---

## φ (Phi Coefficient) — primary choice

**What.** Pearson correlation applied to two 0/1 variables. The formula collapses algebraically into the phi coefficient, the standard association measure for 2×2 contingency tables.

**Core equations.**

$$\phi = \frac{AD - BC}{\sqrt{(A+B)(C+D)(A+C)(B+D)}}$$

where A, B, C, D are the four cells of the 2×2 table (TN, FP, FN, TP if you think of one column as a "label"). Equivalently:

$$\phi = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y} = r_{XY} \quad \text{(Pearson on 0/1 columns)}$$

**Range.** [−1, +1]. Sign has meaning: +1 = perfect positive agreement, −1 = perfect negative agreement, 0 = independence.

**Pros.**
- Exact — no approximation, no tie correction needed.
- Signed — distinguishes positive from negative association.
- One-line: `df.corr()` computes the full matrix in a single vectorized pass.
- Identical to MCC (Matthews Correlation Coefficient), a well-known ML metric.

**Risks.**
- Columns must be numeric 0/1 (or any two distinct numbers; strings fail silently).
- `df.corr()` uses **pairwise deletion** by default — set `min_periods` appropriately for sparse data.
- φ = ±1 is only achievable when the marginal distributions of X and Y are identical; otherwise the maximum attainable |φ| < 1 even with strong association (see § Problem-specific Considerations).

**How / verify.**

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

rng = np.random.default_rng(42)
n = 1_000
a = rng.integers(0, 2, n)
b = (a ^ rng.integers(0, 2, n, endpoint=False) & rng.integers(0, 2, n)).astype(int)

df = pd.DataFrame({"A": a, "B": b})

# --- Method 1: Pearson on 0/1 (exact phi, fastest) ---
phi_matrix = df.corr(method="pearson", min_periods=30)
phi_ab = phi_matrix.loc["A", "B"]
print(f"phi (Pearson): {phi_ab:.4f}")

# --- Method 2: manual from contingency table (verify) ---
table = pd.crosstab(df["A"], df["B"]).values  # [[TN, FP], [FN, TP]]
A_, B_, C_, D_ = table[0,0], table[0,1], table[1,0], table[1,1]
phi_manual = (A_*D_ - B_*C_) / np.sqrt(
    (A_+B_) * (C_+D_) * (A_+C_) * (B_+D_)
)
print(f"phi (manual):  {phi_manual:.4f}")  # must match

# --- Assertion ---
assert abs(phi_ab - phi_manual) < 1e-10, "Mismatch — check column encoding"
```

---

## Cramér's V — valid alternative, unsigned

**What.** Chi-squared-based effect size generalized to r×c tables. For 2×2 tables: V = |φ|.

$$V = \sqrt{\frac{\chi^2}{n \cdot (k-1)}} \quad k = \min(r, c)$$

For 2×2: k = 2, so V = √(χ²/n) = |φ|.

**Pros.**
- Generalizes cleanly to multi-category variables — same code works for binary and non-binary.
- Always non-negative, [0, 1] — easier to threshold uniformly in a mixed-type pipeline.

**Risks.**
- **Loses sign** — cannot distinguish positive from negative association. This matters for redundancy reasoning (X ≈ Y vs X ≈ 1−Y are both redundant but represent opposite relationships).
- Slightly more expensive: requires building the contingency table and running chi2.
- Chi-squared approximation degrades when expected cell counts < 5 (use Fisher's exact test then).

**How / verify.**

```python
from scipy.stats import chi2_contingency
import pandas as pd

def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.values.sum()
    k = min(table.shape) - 1
    return (chi2 / (n * k)) ** 0.5

# For binary vars, V == |phi|
v = cramers_v(df["A"], df["B"])
print(f"Cramér's V: {v:.4f}")
print(f"|phi|:      {abs(phi_ab):.4f}")
assert abs(v - abs(phi_ab)) < 1e-10
```

---

## Spearman ρ — why it is a poor choice for binary–binary

**What.** Pearson correlation on ranks. For continuous variables: captures monotonic (not just linear) association. For binary 0/1 data: the rank operation produces highly tied ranks, and all tie-correction methods converge back to an approximation of φ.

**Why it degenerates.**

When X ∈ {0, 1} with n₀ zeros and n₁ ones:
- All zeros receive the same average rank: r₀ = (n₀ + 1) / 2
- All ones receive: r₁ = n₀ + (n₁ + 1) / 2
- Computing Pearson on these two-value rank vectors is equivalent to computing φ — but with an extra O(n log n) sorting step and tie-correction overhead.

In practice, `scipy.stats.spearmanr` returns a value very close to φ but not identical due to tie-handling implementation details. The difference is numerical noise, not information.

**Spearman's value adds nothing for binary–binary:**

| Property | Spearman advantage (continuous) | Binary case |
|---|---|---|
| Non-linear monotonic detection | ✅ Captures beyond Pearson | ❌ With 2 values, monotonic = linear — no difference |
| Outlier robustness | ✅ Robust via ranks | ❌ 0/1 values have no outliers by construction |
| Cost | O(n log n) | Unnecessary: φ is O(n) |

**Risks.**
- Misleads readers into thinking a different quantity is being measured.
- Small numerical discrepancies vs φ can cause inconsistencies in automated pipelines.
- If your pipeline mixes binary and continuous columns and you call `df.corr(method="spearman")` uniformly, the binary–binary cells are valid but suboptimal. Correct, but wasteful.

**Critique.** Using Spearman for binary–binary is a common cargo-cult pattern: it works on continuous data, so practitioners apply it everywhere. For this specific case it is technically defensible (the result is asymptotically φ) but conceptually wrong and computationally wasteful. Use φ.

---

## Problem-specific Considerations

### φ\_max — the maximum attainable phi

When the marginals of X and Y differ, |φ| = 1 is not achievable. Example: P(X=1) = 0.9, P(Y=1) = 0.5 → φ_max ≈ 0.33 even for the strongest possible association. This can make fixed thresholds (|φ| > 0.7) misleading.

**Mitigation:** normalize phi by its theoretical maximum.

$$\phi_{\text{norm}} = \frac{\phi}{\phi_{\max}}, \quad \phi_{\max} = \sqrt{\frac{p_X (1-p_X)}{p_Y (1-p_Y)}} \cdot \text{sign}(\phi)$$

```python
def phi_normalized(x, y):
    phi = np.corrcoef(x, y)[0, 1]
    px, py = x.mean(), y.mean()
    # phi_max depends on which marginal is the constraining one
    phi_max = np.sqrt(min(px, 1-px) * min(py, 1-py) /
                      (max(px, 1-px) * max(py, 1-py)))
    return phi / phi_max if phi_max > 0 else 0.0
```

### Screening many binary columns at once

`df.corr(method="pearson")` handles the full matrix in one call — no loop needed.

```python
import numpy as np
import pandas as pd

# Assume df_bin contains only 0/1 integer columns
phi_mat = df_bin.corr(method="pearson", min_periods=30)

# Extract upper triangle pairs above threshold
upper = phi_mat.where(np.triu(np.ones(phi_mat.shape, dtype=bool), k=1))
redundant = (
    upper.stack()
    .reset_index()
    .rename(columns={"level_0": "feat_a", "level_1": "feat_b", 0: "phi"})
    .query("phi.abs() > 0.7")
    .sort_values("phi", key=abs, ascending=False)
)
print(redundant)
```

### Does the semantic origin of the binary feature matter?

**Short answer: yes, it matters for Spearman; it does not matter for φ.**

A binary 0/1 column can arise from two conceptually different origins:

| Origin | Example | Is there an order between 0 and 1? | Correct Python function |
|---|---|---|---|
| **Ordinal / natural binary** | `has_purchased` (no → yes), `is_adult` (no → yes) | ✅ Yes — 1 is "more" than 0 in a meaningful sense | `df.corr(method="pearson")` → φ (do **not** use Spearman: degenerates to φ anyway) |
| **Nominal / one-hot encoded** | `province_madrid` (not Madrid → Madrid), `gender_female` | ❌ No — the encoding is arbitrary; swapping 0↔1 gives an equally valid representation | `df.corr(method="pearson")` → φ, or `scipy.stats.chi2_contingency` → Cramér's V (Spearman is **invalid** here) |

**Why this matters for Spearman:** Spearman measures *monotonic* association — it assumes that higher values of X correspond to higher (or lower) values of Y in a meaningful way. For a nominal binary feature like `province_madrid`, the 0/1 encoding is arbitrary: calling "not Madrid" = 0 and "Madrid" = 1 is a convention, not a measurement scale. There is no monotonic relationship to detect. Spearman is semantically invalid here, regardless of what number it returns.

For a natural binary like `has_purchased`, a monotonic interpretation is at least defensible, but it still degenerates to φ as explained above — so Spearman remains the wrong tool for a different reason (cost/degeneracy).

**Why φ (and Cramér's V) are immune to this distinction:** φ is derived from the 2×2 contingency table — it measures co-occurrence patterns, not numerical order. The labels 0 and 1 are just identifiers for the two categories. Whether the feature is "has product X" or "province = Madrid", the contingency table structure is the same and φ is equally valid.

**Practical rule:**

| Feature type | φ valid? | Spearman valid? | Use |
|---|---|---|---|
| Natural binary (`has_X`, `is_adult`) | ✅ | ⚠️ Degenerates to φ | φ |
| One-hot / nominal binary (`province_madrid`) | ✅ | ❌ Semantically invalid | φ or Cramér's V |

**Consequence for pipelines:** if your binary columns include one-hot-encoded categoricals (which is almost always the case in tabular ML), using `df.corr(method="spearman")` is doubly wrong for those columns — both semantically invalid and computationally wasteful. Use `df.corr(method="pearson")`.

---

### Mixed binary + continuous columns

If your DataFrame contains both types, `df.corr(method="pearson")` is still correct:
- Binary–binary cell → φ (exact)
- Binary–continuous cell → point-biserial r (exact)
- Continuous–continuous cell → Pearson r

One call, one matrix, all cases handled correctly by the same formula.

---

## Diagnostics & Pitfalls

- **Encoding check:** columns must be 0/1 integers (or floats). String `"yes"/"no"` silently produces NaN or wrong results. Always run `df[col].unique()` to verify.
- **NaN handling:** `df.corr()` uses pairwise deletion. Set `min_periods` to avoid reporting correlations from very few valid pairs (see table below).
- **φ_max trap:** imbalanced binary features can show |φ| ≪ 0.7 even when maximally associated. Check marginals (class balance) before applying a fixed threshold.
- **Chi-square cell counts:** for Cramér's V, expected counts < 5 in any cell → chi-square approximation unreliable. Use Fisher's exact test or bootstrap the statistic.
- **Not a leakage risk:** this is feature–feature screening with no target involved — no train/test leakage possible. Apply on the full training set before the CV loop.
- **Symmetry:** φ, V, and ρ are all symmetric (cor(A,B) = cor(B,A)) — the matrix is consistent.

**min_periods guidance:**

| Sample size n | Suggested min_periods |
|---|---|
| n < 100 | 20 |
| 100 ≤ n < 1 000 | 30 |
| 1 000 ≤ n < 10 000 | 50 |
| n ≥ 10 000 | 100 or 0.01·n |

---

## Decision Rule

1. **Two binary (0/1) features, redundancy screening** → use `df.corr(method="pearson")`. The result is φ exactly. Threshold: |φ| > 0.7.
2. **Need unsigned measure or mixing binary with multi-category in one function** → use Cramér's V. For 2×2 it equals |φ|.
3. **Suspect imbalanced marginals (one class < 20%)** → compute φ_norm instead of raw φ.
4. **Spearman or Kendall for binary–binary** → avoid. Redundant compute, no added information.
5. **All binary columns + some continuous columns in the same DataFrame** → `df.corr(method="pearson")` handles every pair type correctly in one call.

---

## References

1. Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
2. Agresti, A. (2002). *Categorical Data Analysis* (2nd ed.). Wiley. — Chapter 2, phi and contingency association.
3. scikit-learn — `matthews_corrcoef` (MCC = φ for binary): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
4. pandas — `DataFrame.corr`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
5. SciPy — `chi2_contingency`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
6. SciPy — `spearmanr` (tie correction behavior): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
7. Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection*. CRC Press. https://feat.engineering/ — Chapter 10 (Feature Selection Overview).
8. Davenport, J. W. & El-Sanhouri, I. (1991). "The Maximum Value of a phi Coefficient." *Journal of the American Statistical Association.*
