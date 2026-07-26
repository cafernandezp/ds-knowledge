# 12 Statistical Tests — When to Use, Risks, Minimal Implementation

> **Scope.** Reference guide for the 12 classical hypothesis tests commonly used in EDA / analyst
> workflows: 3 t-tests, 2 ANOVAs, Chi-Square, 2 correlation tests, and 4 nonparametric tests
> (Mann-Whitney, Wilcoxon, Kruskal-Wallis, Shapiro-Wilk).
> **Assumptions.** Data is tabular (pandas). Significance level α = 0.05 unless stated. Tests are
> assumed to run on data collected **before** any model training split is finalized (see §7 for
> leakage implications when tests feed feature selection or A/B decisions).
> **Stack.** `scipy.stats` (core), `statsmodels` (two-way ANOVA, `anova_lm`).

---

## TL;DR

- **Parametric first choice, if assumptions hold:** t-tests (01–03) and ANOVA (04–05) — more power
  than their nonparametric counterparts when normality + homoscedasticity are reasonable.
- **Nonparametric fallback:** Mann-Whitney U (09) ↔ independent t-test (02); Wilcoxon (10) ↔ paired
  t-test (03); Kruskal-Wallis (11) ↔ one-way ANOVA (04). Use when N is small, data is ordinal, or
  normality fails.
- **Always check normality before choosing parametric vs. nonparametric** — Shapiro-Wilk (12) is a
  diagnostic gate, not a standalone deliverable. Don't over-trust it above N≈5000 (p-value degrades;
  W statistic stays informative).
- **Chi-Square (06)** is for categorical × categorical association, not means — the single most
  common misuse is applying it to continuous data binned "for convenience," which discards
  information and inflates Type I error via arbitrary bin boundaries.
- **Correlation ≠ regression ≠ causation.** Pearson (07) and Spearman (08) quantify association
  strength only; never report them as effect estimates for decision-making without a model behind
  them.
- **Every test above has a multiple-comparisons risk** when run repeatedly across many
  columns/groups (common in EDA sweeps). Correct with Bonferroni/BH-FDR before acting on p-values.

---

## Parametric vs. Nonparametric Tests

**What "parametric" means.** The test assumes the data comes from a distribution describable by a
fixed set of parameters (typically the normal distribution, via its mean and variance). The test
statistic (t, F) is derived analytically from that assumed distribution.

**What "nonparametric" means.** The test makes no assumption about the shape of the underlying
distribution. Most of the nonparametric tests in this report work by converting values to **ranks**
first — which is what makes them distribution-free, but also what makes them discard some
information (the exact magnitude of a difference, not just its direction).

**Trade-off — neither is "better," each wins under different conditions.**

| | Parametric (t-test, ANOVA, Pearson) | Nonparametric (Mann-Whitney, Wilcoxon, Kruskal-Wallis, Spearman) |
|---|---|---|
| **Statistical power** | Higher — smaller sample needed to detect a real effect, **if assumptions hold** | Lower — ranking discards magnitude information |
| **Assumptions** | Normality (+ homoscedasticity for group tests) | Minimal — no distributional shape assumed |
| **Robustness to outliers** | Low — mean/variance are pulled by extreme values | High — ranks cap the influence of any single point |
| **Interpretability of effect** | Direct: mean, mean difference, in original units | Indirect: rank sums, medians, stochastic dominance |
| **Works on ordinal data** | No (Pearson/ANOVA assume interval/ratio scale) | Yes — this is their native use case |
| **Extends to complex designs** | Yes — covariates, interactions, mixed models (regression-based) | Limited — few nonparametric equivalents for multi-factor designs |

**Decision principle:** check assumptions first (normality via Shapiro-Wilk (12), outliers via a
boxplot/histogram), then default to parametric for power **if** they hold; fall back to
nonparametric otherwise. Don't default to nonparametric "to be safe" when data genuinely is normal
and outlier-free — you pay a real power cost for no benefit.

**Examples.**
- **Favors nonparametric:** comparing average transaction amount between two customer segments.
  Amounts are typically right-skewed with a handful of very large outliers — a t-test's mean gets
  distorted by those outliers and the normality assumption fails even at moderate N. Mann-Whitney
  (09) on the same data is unaffected by the magnitude of the outliers, only their rank.
- **Favors parametric:** comparing average processing time (in seconds) of a validation step before
  vs. after a code optimization, measured across many runs. If the distribution is roughly
  symmetric with no extreme outliers, a paired t-test (03) has more power to detect a real
  improvement than Wilcoxon (10), and reports the result in directly actionable units (seconds
  saved) rather than a rank-based statistic.

---

## Comparison Table

| # | Test | Family | Parameters (if parametric) | Design | H₀ | Variable types (see below) | Main risk | Nonparametric alt. |
|---|------|--------|------------------------------|--------|-----|------------------------------|-----------|---------------------|
| 01 | One-Sample t-Test | Parametric | μ, σ² | 1 sample vs. reference | μ = μ₀ | CN/DN vs. μ₀ | Outliers distort mean/variance | Wilcoxon signed-rank (vs. median) |
| 02 | Independent t-Test | Parametric | μ₁, μ₂, σ₁², σ₂² | 2 independent groups | μ₁ = μ₂ | CN/DN by BF or NOM≤6(2) | Pooled-variance form if variances unequal | Mann-Whitney U (09) |
| 03 | Paired t-Test | Parametric | μ_d, σ_d² | 2 matched measurements | μ_d = 0 | CN/DN, paired | Misaligned pairs → silent wrong results | Wilcoxon signed-rank (10) |
| 04 | One-Way ANOVA | Parametric | μ₁...μₖ, σ² | 3+ groups, 1 factor | μ₁=...=μₖ | CN/DN by NOM≤6/ORD | Significant F ≠ which group differs | Kruskal-Wallis (11) |
| 05 | Two-Way ANOVA | Parametric | μ_ij (cell means), σ² | 3+ groups, 2 factors + interaction | no main/interaction effect | CN/DN by NOM≤6/ORD × NOM≤6/ORD | Interaction must be read before main effects | Aligned Rank Transform (not covered) |
| 06 | Chi-Square Test | Nonparametric | — | categorical × categorical | independence | BF/NOM/ORD × BF/NOM/ORD | Sparse cells (expected count < 5) | Fisher's exact (small N) |
| 07 | Pearson Correlation | Parametric | ρ (assumes μₓ, μᵧ, σₓ, σᵧ) | 2 continuous variables | ρ = 0 | CN/DN, or BF↔CN/DN (point-biserial) | Outliers; linear relationships only | Spearman (08) |
| 08 | Spearman Correlation | Nonparametric | — | 2 ordinal/monotonic variables | ρₛ = 0 | CN/DN/ORD | Misses non-monotonic relationships | — |
| 09 | Mann-Whitney U | Nonparametric | — | 2 independent groups | equal distributions | CN/DN/ORD by BF or NOM≤6(2) | Shape difference ≠ median difference | — |
| 10 | Wilcoxon Signed-Rank | Nonparametric | — | 2 matched measurements | median diff = 0 | CN/DN/ORD, paired | Assumes symmetric differences | — |
| 11 | Kruskal-Wallis | Nonparametric | — | 3+ independent groups | equal distributions | CN/DN/ORD by NOM≤6/ORD | Shape difference ≠ median difference | — |
| 12 | Shapiro-Wilk | Nonparametric | — | normality diagnostic | data ~ Normal | CN (DN as approximation) | p-value unreliable at N > 5000 | Anderson-Darling, D'Agostino K² |

---

## Variable Types & Test Validity

> Test validity depends on the **measurement scale** of the variable(s) involved, not on how the
> data happens to be stored — a 0/1 column being numeric-typed doesn't make it fair game for
> Pearson. The codes below (CN, DN, BF, NOM≤6, NOM>6, ORD) tag every test in this report.

| Code | Type | Definition | Example |
|---|---|---|---|
| **CN** | Continuous numeric | Real-valued, interval/ratio scale, effectively unbounded resolution | `transaction_amount`, `account_age_days` |
| **DN** | Discrete numeric | Integer counts, ratio scale but not continuous | `num_failed_logins`, `num_prior_alerts` |
| **BF** | Binary flag (0/1) | 2-level nominal variable, dual-purpose: grouping variable *or* numeric 0/1 indicator | `is_fraud`, `has_prior_alert` |
| **NOM≤6** | Low-cardinality nominal | Unordered categories, few levels (rule of thumb: ≤ 6) | `payment_channel` (4 levels) |
| **NOM>6** | High-cardinality nominal | Unordered categories, many levels | `merchant_id`, `zip_code` |
| **ORD** | Ordinal categorical | Ordered categories, spacing between levels unknown/unequal | `risk_level` (bajo/medio/alto) |

**Why binary flag is kept separate from nominal.** A 0/1 flag is technically a 2-level nominal
variable, but its numeric encoding lets it play two roles plain nominal categories can't: (a) a
grouping variable for a 2-group test (02, 09), and (b) a numeric variable in a correlation — the
point-biserial correlation is literally Pearson's r computed on the 0/1 column (07). Collapsing it
into "nominal" would hide that second use.

**Why cardinality (≤6 vs. >6) matters even though both are nominal.** The test math itself doesn't
change with the number of levels, but two practical constraints do:
- **Chi-Square (06):** more levels → more contingency-table cells → the expected-count-≥5 rule gets
  harder to satisfy without a large N.
- **Group-comparison tests (04, 05, 11):** more groups → more pairwise post-hoc comparisons →
  heavier multiple-testing correction, and each group needs enough rows to estimate its mean/rank
  distribution reliably. A high-cardinality factor like `merchant_id` (thousands of levels) is
  rarely usable directly — group rare levels into "other," bin, or target-encode before testing.

**Why discrete numeric is kept separate from continuous.** The same tests apply to both, but
discrete counts are frequently right-skewed and bounded at 0 (Poisson-like), which violates the
normality assumption behind the parametric tests (01–05, 07, 12) more easily than genuinely
continuous data — run Shapiro-Wilk (12) before defaulting to a parametric test on count data.

**Why ordinal is not just "nominal with order."** Ordinal data supports rank-based tests natively
(Spearman 08, Mann-Whitney 09, Wilcoxon 10, Kruskal-Wallis 11), but is **not** valid for Pearson
(07) — Pearson assumes equal spacing between values, which "bajo / medio / alto" doesn't guarantee
even when coded 1/2/3.

---

## 01. One-Sample t-Test

**What.** Tests whether a sample mean differs from a known/hypothesized value μ₀.

**Family.** Parametric — estimates **μ** (population mean) and **σ²** (population variance) under
an assumed normal distribution.

**Valid for.** One **CN** or **DN** variable compared against a fixed benchmark μ₀. Not valid for
BF, NOM, or ORD (no mean to test).

**Assumptions.**
- Observations independent and identically distributed.
- Population approximately normal, or N ≳ 30 (Central Limit Theorem covers moderate skew).
- μ₀ is a fixed, pre-specified reference — not estimated from the same data.

**H₀ / H₁.**
- H₀: μ = μ₀ (sample mean equals the benchmark).
- H₁: μ ≠ μ₀ (two-sided; use `alternative='less'`/`'greater'` for one-sided).

**p-value → decision.**
- p < 0.05 (typical α) → reject H₀: the sample mean is statistically different from μ₀.
- p ≥ 0.05 → fail to reject H₀: no evidence of a difference (not proof they're equal).
- Practical: e.g. is the average transaction amount in a portfolio different from a regulatory or
  historical benchmark? A reject signals "investigate," not "fraud" — pair with the effect size
  (Cohen's d) to judge whether the gap is large enough to matter operationally.

**Pros.**
- Simple, closed-form, high power for small N when normality holds.
- Directly interpretable effect size (Cohen's d = (x̄−μ₀)/s).

**Risks.**
- Sensitive to outliers (mean/variance not robust).
- Invalid under strong non-normality with small N (< ~30) — check Shapiro-Wilk (12) first.
- Testing many variables against a benchmark without correction inflates false positives.

**How / verify.**
```python
import numpy as np
from scipy import stats

def one_sample_ttest(x, mu0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    stat, p = stats.ttest_1samp(x, popmean=mu0)  # two-sided by default
    return {"t": stat, "p": p, "reject_h0": p < alpha, "n": len(x)}

# usage
x = np.random.normal(5.2, 1.0, 40)
result = one_sample_ttest(x, mu0=5.0)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume the sample mean equals mu0.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence the mean differs from mu0.")
```

---

## 02. Independent Samples t-Test

**What.** Compares means of two independent groups.

**Family.** Parametric — estimates **μ₁, μ₂** (population means of each group) and **σ₁², σ₂²**
(group variances) under an assumed normal distribution per group.

**Valid for.** **CN**/**DN** target split by a **BF** or a 2-level **NOM≤6** grouping variable. For
3+ groups, use ANOVA (04) instead.

**Assumptions.**
- Groups are independent of each other (no shared subjects/rows).
- Each group approximately normal, or large N per group (CLT).
- Equal variances required only for Student's pooled form; Welch's (default here) relaxes this.

**H₀ / H₁.**
- H₀: μ₁ = μ₂ (the two group means are equal).
- H₁: μ₁ ≠ μ₂.

**p-value → decision.**
- p < 0.05 → reject H₀: the two group means differ significantly.
- p ≥ 0.05 → fail to reject H₀: no evidence the means differ.
- Practical: e.g. do flagged vs. non-flagged transactions differ in average amount? Reject supports
  using amount as a discriminating feature — but check the size of the mean difference, not just p,
  before promoting it to a model.

**Pros.**
- Standard, well-understood; Welch variant removes the equal-variance assumption.
- Fast, closed-form CI for the mean difference.

**Risks.**
- **Classic pitfall: using Student's pooled-variance form when variances differ** — inflates Type I
  error. Default to **Welch's t-test** (`equal_var=False`) unless variance equality is verified
  (Levene's test).
- Non-normal + small/unequal N → unreliable p-values; prefer Mann-Whitney U (09).
- Independence violated if groups share subjects, time periods, or clusters (use paired test or
  mixed models instead).

**How / verify.**
```python
from scipy import stats

def independent_ttest(a, b, alpha=0.05, equal_var=False):
    # Welch's t-test by default — safer when group variances are unknown/unequal
    stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
    return {"t": stat, "p": p, "reject_h0": p < alpha}

# verify variance-equality assumption before choosing equal_var=True
levene_stat, levene_p = stats.levene(a := [1,2,3,4,5], b := [2,3,4,5,9])
result = independent_ttest(a, b)
print(result, "levene_p=", levene_p)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume the two group means are equal.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence the group means differ.")
```

---

## 03. Paired Samples t-Test

**What.** Tests mean difference of matched before/after (or matched-pair) observations.

**Family.** Parametric — estimates **μ_d** (mean of the paired differences) and **σ_d²** (variance
of the differences) under an assumed normal distribution of the differences.

**Valid for.** **CN**/**DN** variable measured twice on the same units (matched pairs). Not for
BF/NOM/ORD (no mean to difference meaningfully).

**Assumptions.**
- Pairs are genuinely matched (same subject/unit under two conditions).
- Differences (before − after) approximately normal.
- Pairs independent of each other.

**H₀ / H₁.**
- H₀: μ_d = 0 (no average change between conditions).
- H₁: μ_d ≠ 0.

**p-value → decision.**
- p < 0.05 → reject H₀: a real average change exists between conditions.
- p ≥ 0.05 → fail to reject H₀: no evidence of a change.
- Practical: e.g. did a new AML rule change average review time for the same batch of cases,
  before vs. after rollout? Reject → the rule had a measurable effect; check the sign and size of
  d̄ to know direction and magnitude before reporting it as an improvement.

**Pros.**
- Controls for subject-level variance → more power than an independent test for the same N.
- Simple to compute and interpret (equivalent to a one-sample t-test on the differences).

**Risks.**
- Requires **true pairing** — mismatched row order silently produces wrong results (no error
  raised). Always join on an explicit ID before differencing.
- Assumes differences are approximately normal, not the raw values — check normality on `d = a - b`.
- Regression-to-the-mean can masquerade as a real "before/after" effect; a control group rules this out.

**How / verify.**
```python
import numpy as np
from scipy import stats

def paired_ttest(before, after, alpha=0.05):
    before, after = np.asarray(before), np.asarray(after)
    assert len(before) == len(after), "mismatched pairs"
    stat, p = stats.ttest_rel(before, after)
    return {"t": stat, "p": p, "reject_h0": p < alpha}

before = np.array([70, 72, 68, 75, 71])
after  = np.array([68, 70, 66, 74, 70])
result = paired_ttest(before, after)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume the average difference is zero.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence of a change before/after.")
```

---

## 04. One-Way ANOVA

**What.** Tests whether 3+ group means differ, using one categorical factor.

**Family.** Parametric — estimates **μ₁...μₖ** (population mean of each group) and a **common σ²**
(pooled within-group variance) under an assumed normal distribution per group.

**Valid for.** **CN**/**DN** target grouped by a **NOM≤6** or **ORD** factor (3+ levels). Avoid
running directly on **NOM>6** — group rare levels or bin first.

**Assumptions.**
- Groups independent of each other; normal residuals within each group.
- Homogeneity of variance across groups (verify with Levene's test).

**H₀ / H₁.**
- H₀: μ₁ = μ₂ = ... = μₖ (all group means equal).
- H₁: at least one group mean differs from the rest.

**p-value → decision.**
- p < 0.05 → reject H₀: at least one group differs → run Tukey HSD to find which.
- p ≥ 0.05 → fail to reject H₀: no evidence of group differences.
- Practical: e.g. do risk scores differ across three transaction channels? Reject → channel is
  worth encoding as a model feature; the omnibus test alone doesn't say which channel, only that
  segmentation matters.

**Pros.**
- Single omnibus test instead of many pairwise t-tests → controls family-wise error better than
  uncorrected repeated t-tests.
- `f_oneway` and the equivalent `ols` formula give identical results — the OLS form generalizes to
  two-way and covariate-adjusted designs.

**Risks.**
- **Significant F only tells you "some group differs," not which one** — always follow with a
  post-hoc test (Tukey HSD) before acting.
- Assumes homogeneity of variance across groups (Levene's test) and normal residuals.
- Unbalanced groups + unequal variance is the most common real-world violation — Welch's ANOVA
  (`pingouin.welch_anova`) is the safer default if that's suspected.

**How / verify.**
```python
from scipy import stats

def one_way_anova(*groups, alpha=0.05):
    stat, p = stats.f_oneway(*groups)
    return {"F": stat, "p": p, "reject_h0": p < alpha}

g1 = [23, 25, 22, 24, 26]
g2 = [30, 28, 31, 29, 27]
g3 = [20, 19, 21, 18, 22]
result = one_way_anova(g1, g2, g3)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume all group means are equal.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence of group differences.")

# post-hoc if significant:
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
```

---

## 05. Two-Way ANOVA

**What.** Tests main effects of two categorical factors and their interaction on a continuous target.

**Family.** Parametric — estimates **μ_ij** (population mean of each factor-combination cell) and a
**common σ²** (pooled residual variance) under an assumed normal distribution per cell.

**Valid for.** **CN**/**DN** target crossed by two **NOM≤6**/**ORD** factors, each with few enough
levels that every factor-combination cell keeps sufficient observations.

**Assumptions.**
- Normal residuals; homogeneity of variance across all factor-combination cells.
- Observations independent; reasonably balanced design (unbalanced designs need explicit Type II/III SS).

**H₀ / H₁** (three separate tests in one table).
- Factor A: H₀ — no main effect of A; H₁ — A affects the target.
- Factor B: H₀ — no main effect of B; H₁ — B affects the target.
- Interaction: H₀ — A and B act independently; H₁ — the effect of A depends on the level of B.

**p-value → decision.**
- Check the **interaction row first**: p < 0.05 → factors interact; interpret main-effect p-values
  with caution (they're averaged over interaction levels and can mislead).
- Interaction p ≥ 0.05 → drop it, interpret each main effect's p-value directly.
- Practical: does the effect of transaction channel on risk score depend on customer segment? A
  significant interaction means one global threshold won't work — build segment-specific rules or
  features instead of a single pooled cutoff.

**Pros.**
- Detects **interaction effects** invisible to two separate one-way ANOVAs (e.g., a treatment that
  only works for one subgroup).
- More efficient than running one-way ANOVA per factor: shared residual variance estimate.

**Risks.**
- **Interpretation order matters:** check the interaction term first. A significant interaction
  makes the "main effect" p-values misleading in isolation (they average over interaction levels).
- Requires **Type II or Type III sum-of-squares** choice for unbalanced designs — Type I
  (sequential, statsmodels default `typ=1`) gives order-dependent results; use `typ=2` (no
  higher-order interactions involving the factor) or `typ=3` (interaction present) explicitly.
- Small/empty cells (factor combinations with too few observations) destabilize estimates.

**How / verify.**
```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def two_way_anova(df, y_col, factor_a, factor_b, typ=2):
    formula = f"{y_col} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})"
    model = ols(formula, data=df).fit()
    return sm.stats.anova_lm(model, typ=typ)

rows = [
    ("low", "S", 18), ("low", "S", 20), ("low", "P", 15), ("low", "P", 17),
    ("med", "S", 24), ("med", "S", 26), ("med", "P", 21), ("med", "P", 23),
    ("high", "S", 28), ("high", "S", 30), ("high", "P", 25), ("high", "P", 27),
]
df = pd.DataFrame(rows, columns=["water", "sun", "yield_"])
result = two_way_anova(df, "yield_", "water", "sun")
print(result)

# interpretation (example) — check the interaction row first
interaction_p = result.loc["C(water):C(sun)", "PR(>F)"]
if interaction_p < 0.05:
    print(f"p={interaction_p:.4f} < 0.05 -> Reject H0: cannot assume water and sun act independently (interaction present).")
else:
    print(f"p={interaction_p:.4f} >= 0.05 -> Fail to reject H0: no evidence of an interaction between water and sun.")
```

---

## 06. Chi-Square Test of Independence

**What.** Tests association between two categorical variables via a contingency table.

**Family.** Nonparametric — no distributional parameters are estimated; the statistic compares
observed vs. expected cell counts directly.

**Valid for.** Two categorical variables — any combination of **BF**, **NOM≤6**, **NOM>6**, **ORD**
(order is ignored, so ORD loses information here — Spearman/Kruskal-Wallis preserve it better).
Not valid on raw **CN**/**DN** — binning is a last resort, not a default.

**Assumptions.**
- Observations independent (each row is one unit, counted once).
- Variables are categorical (nominal); expected cell counts ≥ 5 in most cells (rule of thumb).

**H₀ / H₁.**
- H₀: the two variables are independent (no association).
- H₁: the two variables are associated.

**p-value → decision.**
- p < 0.05 → reject H₀: association exists — check Cramér's V for strength before acting.
- p ≥ 0.05 → fail to reject H₀: no evidence of association.
- Practical: is structuring behavior associated with account type? Reject → account type is a
  useful conditioning variable for an AML rule. A significant p with a weak Cramér's V (e.g. < 0.1)
  means the association is statistically real but practically negligible — don't over-engineer a
  rule around it.

**Pros.**
- Works on nominal data with no ordering assumption.
- Extends cleanly to r×c tables (not just 2×2).

**Risks.**
- **Expected cell counts < 5** invalidate the χ² approximation — use Fisher's exact test for small
  samples/sparse tables instead.
- Significance ≠ effect size: report Cramér's V alongside p, since large N makes trivial
  associations "significant."
- **Do not bin continuous variables into categories to force a Chi-Square** — this discards
  information and the result depends on arbitrary bin edges; use Pearson/Spearman/point-biserial
  instead for continuous ↔ continuous or continuous ↔ binary pairs.

**How / verify.**
```python
import numpy as np
from scipy.stats import chi2_contingency

def chi_square_test(contingency_table, correction=True, alpha=0.05):
    # correction=True applies Yates' continuity correction, applied only for 2x2 tables
    chi2, p, dof, expected = chi2_contingency(contingency_table, correction=correction)
    min_expected = expected.min()
    return {"chi2": chi2, "p": p, "dof": dof, "min_expected": min_expected,
             "reject_h0": p < alpha, "valid_approx": min_expected >= 5}

table = np.array([[30, 10], [20, 40]])  # rows=Group, cols=Outcome
result = chi_square_test(table)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume Group and Outcome are independent.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence of association between Group and Outcome.")
```

---

## 07. Pearson Correlation

**What.** Measures linear association strength and direction between two continuous variables.

**Family.** Parametric — estimates **ρ** (population correlation coefficient) under an assumed
bivariate normal distribution with means **μₓ, μᵧ** and variances **σₓ², σᵧ²**.

**Valid for.** Two **CN** (or **DN**, if not too coarse) variables. Also valid, via the identical
point-biserial formula, for one **BF** vs. one **CN**/**DN**. Not valid for **NOM**; not appropriate
for **ORD** (use Spearman 08 instead — Pearson assumes equal spacing between levels).

**Assumptions.**
- Both variables continuous; relationship is linear (check a scatterplot).
- No extreme outliers; bivariate approximate normality assumed for the significance test (not for r itself).

**H₀ / H₁.**
- H₀: ρ = 0 (no linear association).
- H₁: ρ ≠ 0.

**p-value → decision.**
- p < 0.05 → reject H₀: the linear association is statistically real.
- p ≥ 0.05 → fail to reject H₀: no evidence of a linear relationship.
- Practical: is transaction amount linearly correlated with account age? Reject + high |r| (e.g.
  > 0.7) → possible redundant features / multicollinearity risk downstream; reject + low |r| →
  statistically real but practically negligible — don't act on p alone.

**Pros.**
- Interpretable, bounded [-1, 1]; basis for R² in simple linear regression.
- Fast, standard, well-understood confidence intervals (Fisher z-transform).

**Risks.**
- **Captures linear relationships only** — a strong nonlinear (e.g., U-shaped) relationship can
  yield r ≈ 0. Always scatter-plot before trusting r.
- Sensitive to outliers — a single extreme point can flip the sign.
- Assumes bivariate normality for the significance test (not for r itself); for skewed data prefer
  Spearman (08) or report r alongside a robust alternative.
- Correlation is not causation, and is not directional — it says nothing about which variable
  "drives" the other.

**How / verify.**
```python
import numpy as np
from scipy import stats

def pearson_corr(x, y, alpha=0.05):
    r, p = stats.pearsonr(x, y)
    return {"r": r, "p": p, "reject_h0": p < alpha}

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
result = pearson_corr(x, y)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume x and y are linearly uncorrelated.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence of a linear relationship.")
```

---

## 08. Spearman Rank Correlation

**What.** Measures monotonic association using ranked values — Pearson correlation computed on ranks.

**Family.** Nonparametric — no distributional parameters estimated; **ρₛ** is a rank correlation,
not a population parameter of an assumed distribution.

**Valid for.** Any pair of **CN**, **DN**, or **ORD** variables (rank-based, so ordinal is natively
supported — code "bajo/medio/alto" as 1/2/3 and it works correctly). Not valid for **NOM** (no
order to rank).

**Assumptions.**
- Relationship is monotonic (not necessarily linear); variables at least ordinal.
- No distributional assumption on the raw variables (this is the point of using ranks).

**H₀ / H₁.**
- H₀: ρₛ = 0 (no monotonic association).
- H₁: ρₛ ≠ 0.

**p-value → decision.**
- p < 0.05 → reject H₀: a monotonic association is present.
- p ≥ 0.05 → fail to reject H₀.
- Practical: financial amounts are usually right-skewed — a significant ρₛ where Pearson's r was
  weak/non-significant flags a real but nonlinear relationship worth a log-transform or a
  tree-based model that captures it natively.

**Pros.**
- Robust to outliers and to nonlinear-but-monotonic relationships.
- Valid for ordinal data (Likert scales) where Pearson's interval-scale assumption doesn't hold.

**Risks.**
- Still misses non-monotonic relationships (e.g., quadratic) — same scatter-plot-first discipline
  applies.
- Loses information vs. Pearson when the relationship genuinely is linear (lower power).
- Many tied ranks (common with coarse categorical-like data) reduce test validity; scipy handles
  ties via average ranks but power still degrades.

**How / verify.**
```python
from scipy import stats

def spearman_corr(x, y, alpha=0.05):
    rho, p = stats.spearmanr(x, y)
    return {"rho": rho, "p": p, "reject_h0": p < alpha}

x = [1, 2, 3, 4, 5]
y = [5, 3, 4, 2, 1]
result = spearman_corr(x, y)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume x and y have no monotonic association.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence of a monotonic relationship.")
```

---

## 09. Mann-Whitney U Test

**What.** Nonparametric test comparing distributions (typically location/median) of two independent
groups; the rank-based analogue of the independent t-test.

**Family.** Nonparametric — no distributional parameters estimated; based on rank sums.

**Valid for.** **CN**/**DN**/**ORD** target split by a **BF** or a 2-level **NOM≤6** grouping
variable.

**Assumptions.**
- Independent samples; data at least ordinal.
- Similar distribution shape/spread across groups needed to read the result strictly as a median
  comparison — otherwise it's a general "distributions differ" test.

**H₀ / H₁.**
- H₀: the two distributions are equal (P(X > Y) = 0.5).
- H₁: one distribution is stochastically larger than the other.

**p-value → decision.**
- p < 0.05 → reject H₀: the distributions differ.
- p ≥ 0.05 → fail to reject H₀.
- Practical: financial amounts are rarely normal — comparing anomaly scores of flagged vs.
  non-flagged transactions with Mann-Whitney avoids the false confidence a t-test would give on
  skewed data.

**Pros.**
- No normality assumption; robust to outliers via ranking.
- Works on ordinal data where means are not meaningful.

**Risks.**
- Strictly tests **stochastic dominance** (whether one distribution tends to produce larger values),
  not equality of medians — only interpretable as a median comparison **if the two distributions have
  similar shape/spread**; otherwise, differing shapes can trigger significance without a location shift.
- `alternative` parameter default is `'two-sided'` in current SciPy (≥1.7); older code relying on the
  deprecated `alternative=None` behavior halves the p-value — pin the parameter explicitly.
- Low power vs. t-test when data genuinely is normal — don't default to it "to be safe" if normality
  is verified.

**How / verify.**
```python
from scipy import stats

def mann_whitney(a, b, alpha=0.05, alternative="two-sided"):
    stat, p = stats.mannwhitneyu(a, b, alternative=alternative)
    return {"U": stat, "p": p, "reject_h0": p < alpha}

a = [12, 15, 14, 10, 13]
b = [22, 25, 19, 23, 21]
result = mann_whitney(a, b)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume the two distributions are equal.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence the distributions differ.")
```

---

## 10. Wilcoxon Signed-Rank Test

**What.** Nonparametric test for matched-pair/paired-sample differences — rank-based analogue of the
paired t-test.

**Family.** Nonparametric — no distributional parameters estimated; based on signed ranks of the
paired differences.

**Valid for.** **CN**/**DN**/**ORD** variable measured twice on the same units (matched pairs).

**Assumptions.**
- Paired differences are symmetric around the median (weaker than "normal," not assumption-free).
- Pairs independent of each other.

**H₀ / H₁.**
- H₀: median of the differences = 0.
- H₁: median of the differences ≠ 0.

**p-value → decision.**
- p < 0.05 → reject H₀: a real shift exists between conditions.
- p ≥ 0.05 → fail to reject H₀.
- Practical: compare a model's anomaly score for the same transactions before/after a
  feature-engineering change — Wilcoxon avoids assuming the score differences are normally
  distributed, common when scores are bounded or skewed.

**Pros.**
- No normality assumption on the differences.
- Robust to outliers in the difference distribution (uses signed ranks, not raw magnitudes fully).

**Risks.**
- Requires the distribution of differences to be **symmetric** around the median for the test to be
  a valid median test (weaker than "normal," but not assumption-free).
- Ties and zero-differences need explicit handling — `zero_method` parameter (`'wilcox'` drops zeros
  by default, `'pratt'` keeps them ranked); silently dropping zeros can bias results if zeros are
  frequent and meaningful.
- Small N (< ~10) has limited power; check that the sample size can plausibly detect the difference
  of interest.

**How / verify.**
```python
from scipy import stats

def wilcoxon_signed_rank(before, after, alpha=0.05):
    stat, p = stats.wilcoxon(before, after)
    return {"W": stat, "p": p, "reject_h0": p < alpha}

before = [8, 7, 9, 6, 8, 10]
after  = [6, 5, 7, 5, 6, 8]
result = wilcoxon_signed_rank(before, after)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume the median difference is zero.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence of a shift before/after.")
```

---

## 11. Kruskal-Wallis Test

**What.** Nonparametric alternative to one-way ANOVA for 3+ independent groups — extends
Mann-Whitney's rank logic to k groups.

**Family.** Nonparametric — no distributional parameters estimated; based on rank sums across k
groups.

**Valid for.** **CN**/**DN**/**ORD** target grouped by a **NOM≤6** or **ORD** factor (3+ levels).
Avoid directly on **NOM>6** — group/bin first.

**Assumptions.**
- Independent groups; data at least ordinal.
- Similar distribution shape across groups needed to read the result as a median comparison.

**H₀ / H₁.**
- H₀: all groups come from the same distribution.
- H₁: at least one group differs from the rest.

**p-value → decision.**
- p < 0.05 → reject H₀: at least one group differs → follow with Dunn's test (corrected) to find
  which.
- p ≥ 0.05 → fail to reject H₀.
- Practical: compare anomaly scores across several business units without assuming normality —
  common when unit sizes/behaviors vary widely.

**Pros.**
- No normality assumption; robust to outliers.
- Simple omnibus test when comparing several groups on ordinal or non-normal continuous data.

**Risks.**
- Same caveat as Mann-Whitney: strictly a **distribution-equality** test; interpret as "median
  differs" only when group shapes are comparable.
- Significant H says "at least one group differs," not which — pair with Dunn's post-hoc test
  (with multiple-comparison correction), not raw pairwise Mann-Whitney runs.
- Loses power vs. ANOVA when normality genuinely holds.

**How / verify.**
```python
from scipy import stats

def kruskal_wallis(*groups, alpha=0.05):
    stat, p = stats.kruskal(*groups)
    return {"H": stat, "p": p, "reject_h0": p < alpha}

g1 = [4, 6, 5, 7]
g2 = [9, 11, 10, 12]
g3 = [3, 2, 4, 1]
result = kruskal_wallis(g1, g2, g3)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume all groups come from the same distribution.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence of group differences.")
```

---

## 12. Shapiro-Wilk Test

**What.** Tests whether a sample was drawn from a normal distribution — the standard **gate** before
choosing a parametric test.

**Family.** Nonparametric — it's the test that *checks* whether the parametric (normal-distribution)
assumption holds; it has no distributional parameters of its own to estimate.

**Valid for.** **CN** variables; **DN** with many distinct values is a reasonable approximation.
Not meaningful for **BF**, **NOM**, or **ORD** — too many tied values for the statistic to be
informative.

**Assumptions.**
- Sample i.i.d.; variable continuous (not meaningful on discrete/count data with many ties).

**H₀ / H₁.**
- H₀: the sample is drawn from a normal distribution.
- H₁: the sample is not normally distributed.

**p-value → decision.**
- p < 0.05 → reject H₀: evidence against normality → route to the nonparametric branch of the
  decision tree (§ Decision Rule).
- p ≥ 0.05 → fail to reject H₀: no evidence against normality (not proof of it) → a parametric
  test is a reasonable default.
- Practical: this is a gatekeeper, not a deliverable — run it before choosing between the
  parametric (t-test/ANOVA/Pearson) and nonparametric (Mann-Whitney/Wilcoxon/Kruskal-Wallis/
  Spearman) branch for the actual analysis.

**Pros.**
- Among the most powerful normality tests for small-to-moderate N.
- Simple pass/fail signal to route toward parametric vs. nonparametric tests upstream.

**Risks.**
- **N > 5000**: the W statistic stays accurate but the p-value becomes unreliable — don't trust the
  p-value alone at large N; inspect a Q-Q plot and skewness/kurtosis instead.
- **Large N makes trivial deviations "significant"** (any real dataset is never perfectly normal);
  small N has low power to detect real non-normality. Always combine with a visual check (histogram
  / Q-Q plot), not the p-value in isolation.
- A "fail to reject normality" result is not proof of normality — it's absence of evidence against it.

**How / verify.**
```python
from scipy import stats

def shapiro_test(x, alpha=0.05):
    stat, p = stats.shapiro(x)
    return {"W": stat, "p": p, "looks_normal": p >= alpha, "n": len(x),
            "note": "p-value unreliable for n>5000" if len(x) > 5000 else None}

import numpy as np
x = np.random.normal(0, 1, 100)
result = shapiro_test(x)
print(result)

# interpretation (example)
if result["p"] < 0.05:
    print(f"p={result['p']:.4f} < 0.05 -> Reject H0: cannot assume the sample is normally distributed.")
else:
    print(f"p={result['p']:.4f} >= 0.05 -> Fail to reject H0: no evidence against normality.")
```

---

## Problem-Specific Considerations

- **A/B testing / experimentation:** independent t-test (02) or Mann-Whitney (09) are the workhorses;
  always pre-register the metric and test **before** looking at results (peeking inflates Type I
  error — use a sequential-testing correction if checking repeatedly).
- **Feature screening for ML (numeric target ↔ categorical feature):** one-way ANOVA (04) / Kruskal-
  Wallis (11) as a **univariate pre-filter only** — never as the sole selection criterion; they
  ignore feature interactions.
- **Feature screening (categorical ↔ categorical):** Chi-Square (06), report Cramér's V for effect
  size, not just p.
- **Drift / distribution-shift checks in production monitoring:** Kolmogorov-Smirnov or Chi-Square
  (06) between reference and current windows; treat significance as a monitoring signal, not an
  automatic retraining trigger — corroborate with a performance-impact metric.

---

## Diagnostics & Pitfalls

- **Scale mismatch is the most common misuse of all:** running Pearson on ordinal data, ANOVA on a
  high-cardinality nominal factor, or Chi-Square on binned continuous data all produce a p-value
  that *looks* valid but isn't. Check the "Valid for" line of each test against the variable-type
  table (§ Variable Types & Test Validity) before running it.
- **Multiple comparisons:** running any of these tests across many columns (typical EDA sweep)
  inflates the family-wise Type I error. Correct with Bonferroni (conservative) or
  Benjamini-Hochberg FDR (less conservative, preferred for exploratory screening).
- **Leakage in ML pipelines:** if a test result (e.g., "feature X differs significantly by class")
  feeds feature selection or preprocessing decisions, compute it **on the training split only** —
  computing it on train+validation/test data and then filtering leaks target information into
  validation, biasing the reported metric optimistically.
- **Independence violations:** repeated measures on the same subject, clustered/grouped data (e.g.
  multiple rows per customer), or time-series autocorrelation all violate the "independent
  observations" assumption behind every test above except the paired/repeated designs (03, 10) —
  use mixed-effects models or block/cluster-aware tests instead.
- **p-value ≠ effect size ≠ practical significance.** Always report an effect size (Cohen's d,
  Cramér's V, r) alongside p; large N makes negligible effects "significant."
- **Assumption checks are prerequisites, not afterthoughts:** run Shapiro-Wilk (12) + Levene's test
  before committing to a parametric test family; document the check in the analysis, don't just
  assume normality because N "feels large."

---

## Decision Rule / Quick Guide

1. Comparing **1 sample vs. a benchmark** → normal? → **01 (t-test)**; else → Wilcoxon signed-rank vs. median.
2. Comparing **2 independent groups**, continuous target → normal + equal variance? → **02 (Welch t-test)**; else → **09 (Mann-Whitney)**.
3. Comparing **2 matched/paired measurements** → normal differences? → **03 (paired t-test)**; else → **10 (Wilcoxon)**.
4. Comparing **3+ independent groups**, 1 factor → normal + homoscedastic? → **04 (one-way ANOVA)** + Tukey HSD; else → **11 (Kruskal-Wallis)** + Dunn's test.
5. Comparing **3+ groups across 2 factors** (want interaction) → **05 (two-way ANOVA)**.
6. Testing **association between 2 categorical variables** → expected counts ≥ 5? → **06 (Chi-Square)**; else → Fisher's exact.
7. Measuring **strength of association, 2 continuous variables** → linear + normal-ish? → **07 (Pearson)**; monotonic/ordinal/outlier-prone? → **08 (Spearman)**.
8. Unsure if data is normal → run **12 (Shapiro-Wilk)** + Q-Q plot first, then route to 1–5 accordingly.
9. Any test above run **repeatedly across columns/groups** → apply Bonferroni/BH-FDR before acting.

---

## References

1. SciPy — `scipy.stats.mannwhitneyu` (current default `alternative='two-sided'`, `method='auto'`). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
2. SciPy — `scipy.stats.shapiro` (N>5000 p-value caveat). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
3. SciPy — `scipy.stats.ttest_ind` (`equal_var` parameter, Welch's t-test). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
4. SciPy — `scipy.stats.chi2_contingency` (`correction` parameter, Yates' correction). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
5. SciPy — `scipy.stats.wilcoxon` (`zero_method` parameter). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
6. statsmodels — `statsmodels.stats.anova.anova_lm` (Type I/II/III sum of squares, two-way ANOVA via `ols`). https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html
7. statsmodels — ANOVA user guide (two-way formula syntax, interaction terms). https://www.statsmodels.org/stable/anova.html
