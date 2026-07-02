# Feature Engineering — Summary

> **Created by:** Claude Opus 4.7
> **Date:** 6 May 2026
> **Source:** *Feature Engineering and Selection: A Practical Approach for Predictive Models* — Max Kuhn & Kjell Johnson — https://feat.engineering/
> **Scope:** Chapters 1, 3, 5–8 (Introduction, Modeling Process, Categorical Encoding, Numeric Engineering, Interactions, Missing Data)

---

## What is feature engineering?

**Definition:** the process of transforming raw predictors into representations that better expose the predictor–response relationship to a model.

- **Feature engineering** → modify / create predictors (this document)
- **Feature selection** → choose a subset of predictors (separate document)
- **Feature extraction** → derive new components from many predictors (e.g. PCA)

**Why it matters:** even strong models underperform when relationships are hidden in the wrong representation. Engineering exposes:

- transformations of a predictor (log, sqrt, Box-Cox)
- interactions (products, ratios)
- functional relationships (splines, polynomials)
- equivalent re-representations (encodings of categorical, date features)

**Risk:** the more representations you try, the higher the chance of **overfitting** → always evaluate inside a resampling scheme (CV).

---

## Modeling process context (Ch. 3)

Feature engineering sits **inside** the resampling loop, not before it.

```
Train/Test split
  └── For each CV fold (training set):
        1. Fit engineering steps (impute, encode, transform)
        2. Apply to held-out fold
        3. Train model + evaluate
  └── Final: refit pipeline on all training data → score on test
```

**Key principles:**

- **No leakage:** statistics for imputation, encoding, scaling must be learned **only on the training fold**, then applied to the validation/test fold.
- **Reproducibility:** fix random seeds; pipeline as a single object (e.g. `sklearn.Pipeline`).
- **Metric choice:** match the problem (RMSE/MAE for regression; AUC, log-loss, F1 for classification; calibration when probabilities matter).

---

## 1) Encoding Categorical Predictors (Ch. 5)

**Goal:** convert non-numeric predictors into numeric form usable by models.

### 1.1 Basic encodings
- **One-hot encoding** — one binary column per level; drops one as reference if needed (full rank). Best for low-cardinality nominal vars.
- **Dummy encoding** — same as one-hot but always drops a reference level (used in linear models to avoid collinearity).
- **Ordinal encoding** — integer codes that respect a meaningful order (e.g. low/med/high → 1/2/3). Only valid when ordering exists.

### 1.2 High-cardinality strategies
When a variable has many levels (e.g. ZIP code, product ID), one-hot becomes infeasible.

- **Target / mean encoding** — replace level with mean of `y` for that level
  - Risk: **leakage** — must compute on training fold only, ideally with smoothing or out-of-fold encoding
  - Smoothing: `enc = (n·mean_level + α·mean_global) / (n + α)`
- **Frequency / count encoding** — replace level with its frequency in the data
- **Hashing trick** — hash levels into a fixed number of buckets; collisions accepted
  - Pros: fixed memory; handles new levels naturally
  - Cons: not interpretable
- **Embeddings** — learn a dense low-dimensional vector per level (neural networks, entity embeddings)

### 1.3 Special cases
- **Rare levels** — collapse into an `"Other"` bucket (frequency threshold, e.g. < 1%)
- **New / unseen levels at inference** — assign to `"Other"` or use hashing/embedding fallback
- **Ordered factors** — polynomial contrasts (linear, quadratic, cubic trends)

### 1.4 Date / time features
Decompose into multiple engineered predictors:

- year, month, day-of-week, day-of-year, hour, minute
- is_weekend, is_holiday, is_month_end
- **Cyclical encoding** — for periodic variables (hour, month): `sin(2πx/T)`, `cos(2πx/T)` to preserve cyclicity

---

## 2) Engineering Numeric Predictors (Ch. 6)

**Goal:** reshape numeric features so that distributions and relationships better match model assumptions.

### 2.1 Scaling / centering
Required for distance-based models (k-NN, SVM) and regularized linear models (Lasso, Ridge, Elastic Net).

- **Standardization (z-score):** `(x − μ) / σ` → mean 0, std 1
- **Min-max scaling:** `(x − min) / (max − min)` → range [0, 1]
- **Robust scaling:** uses median and IQR; resistant to outliers
- **Trees do not need scaling** (XGBoost, RF) — splits are invariant to monotonic transforms

### 2.2 Distribution transformations
Reduce skewness, stabilize variance, approximate normality.

- **Log transform:** `log(x)` or `log(x + 1)` for right-skewed positive data
- **Square root / cube root:** milder than log
- **Box-Cox transform:** parameterized power family for **positive** data
  - `y(λ) = (x^λ − 1) / λ` if λ ≠ 0; `log(x)` if λ = 0
  - λ estimated via maximum likelihood
- **Yeo-Johnson transform:** like Box-Cox but supports **zero and negative** values
- **Quantile transform:** maps to uniform or normal distribution by ranks

### 2.3 Outlier handling
- **Winsorization / clipping** — cap values at percentiles (e.g. 1st / 99th)
- **Spatial sign transform** — project rows onto unit hypersphere; useful in high-dim with outliers
  - `xᵢ_new = xᵢ / ||x||₂`

### 2.4 Discretization (binning)
Convert numeric → categorical. Generally **not recommended** unless required (loses information).

- **Equal-width** bins — fixed-width intervals
- **Equal-frequency** (quantile) bins — same count per bin
- **Supervised binning** — splits chosen to maximize relationship with `y` (e.g. via decision-tree splits)

### 2.5 Nonlinear basis expansions
Let linear models capture nonlinear relationships.

- **Polynomial features** — `x, x², x³` (use sparingly; explodes dimensionality)
- **Splines** — piecewise polynomials joined at knots
  - **Natural cubic splines** — linear at the boundaries (more stable)
  - **B-splines** — locally supported basis
- **GAM** (Generalized Additive Models) — automatic smooth term fitting per feature

---

## 3) Detecting Interaction Effects (Ch. 7)

**Definition:** an interaction exists when the effect of one predictor on `y` depends on the value of another.

- Linear models capture interactions only if explicitly added: `y ~ x₁ + x₂ + x₁:x₂`
- Trees and tree ensembles capture interactions automatically through their splits.

### 3.1 Manual / domain-driven
- **Product terms:** `x₁ × x₂`
- **Ratios:** `x₁ / x₂` (e.g. BMI = weight / height²)
- **Differences:** `x₁ − x₂` (e.g. profit = revenue − cost)

### 3.2 Statistical detection
- **ANOVA / nested F-tests** — compare model with vs without interaction
- **Hierarchy principle:** if you include an interaction `x₁:x₂`, keep main effects `x₁`, `x₂` even if non-significant — improves interpretability and stability.

### 3.3 Algorithmic detection
- **Tree-based methods** — features appearing on the same path indicate potential interactions
- **H-statistic (Friedman & Popescu)** — measures interaction strength via partial dependence
- **SHAP interaction values** — decompose pairwise contributions in tree models

### 3.4 Risks
- **Combinatorial explosion** — `p` features → `p(p−1)/2` pairwise interactions; restrict via domain knowledge or screening
- **Multicollinearity** — products of correlated features amplify it; center predictors first

---

## 4) Handling Missing Data (Ch. 8)

**Why it matters:** most models cannot accept NaN; how you handle missingness can bias results.

### 4.1 Mechanisms (Rubin's typology)
- **MCAR** (Missing Completely At Random) — missingness independent of all variables; rare in practice. Listwise deletion is unbiased.
- **MAR** (Missing At Random) — missingness depends on **observed** variables only. Imputation can recover unbiased estimates.
- **MNAR** (Missing Not At Random) — missingness depends on the **unobserved** value itself. Hardest case; requires modeling the missingness mechanism.

### 4.2 Simple strategies
- **Listwise deletion** (drop rows) — only safe when MCAR and few rows affected
- **Constant imputation** — fill with 0, "Unknown", or sentinel value (e.g. −999 for trees)
- **Mean / median imputation** — fast; distorts variance and ignores correlations
- **Mode imputation** — for categorical features

### 4.3 Model-based imputation
- **k-NN imputation** — fill with mean/mode of k nearest neighbors (by other features)
- **Iterative / MICE** (Multiple Imputation by Chained Equations) — model each missing column as a function of the others, iterate until convergence
- **Tree-based imputation** — use Random Forest (e.g. `missForest`) to predict missing values
- **Matrix factorization / softImpute** — low-rank reconstruction; effective for many sparse features

### 4.4 Native handling
- **XGBoost / LightGBM / CatBoost** — handle NaN internally by learning the optimal direction at each split
- **Surrogate splits** (CART) — alternative split when value is missing

### 4.5 Missingness as signal
Often the **fact** that a value is missing is informative.

- **Add a binary indicator** `x_missing ∈ {0, 1}` alongside the imputed value
- Particularly useful when MAR/MNAR is plausible (e.g. patient skipped a test)

### 4.6 Best practices
- **Impute inside CV folds** — never compute imputation statistics on the full dataset
- **Pair imputation with indicators** when missingness may carry signal
- **Multiple imputation** when uncertainty quantification matters (m datasets → pool results)

---

## Diagnostics & pitfalls (cross-chapter)

- **Leakage is the #1 risk:** any statistic learned from data (mean for imputation, target encoding, scaling parameters) must come from training folds only.
- **Match transformation to model:** trees ignore monotonic transforms and scaling; linear models benefit greatly from both.
- **Check distributions before/after:** histograms, Q-Q plots, residual diagnostics.
- **Beware of zero-inflation, heavy tails, and bounded variables** — choose transforms accordingly.
- **Document the pipeline:** every engineering decision must be reproducible and applied identically at training and inference time.

---

## Glossary of abbreviations

| Acronym | Full name |
|---|---|
| AUC | Area Under the (ROC) Curve |
| BMI | Body Mass Index |
| CV | Cross-Validation |
| GAM | Generalized Additive Model |
| IQR | Interquartile Range |
| MAE | Mean Absolute Error |
| MAR | Missing At Random |
| MCAR | Missing Completely At Random |
| MICE | Multiple Imputation by Chained Equations |
| MNAR | Missing Not At Random |
| PCA | Principal Component Analysis |
| RMSE | Root Mean Squared Error |
| SHAP | SHapley Additive exPlanations |
| ZIP | (Postal) ZIP Code |
