# Feature Engineering and Selection — Summary

> *A Practical Approach for Predictive Models* — Max Kuhn & Kjell Johnson
> Source: https://feat.engineering/

---

## Topic: Feature Selection Methods — Family Schema

## 1) Filter methods (model-agnostic, pre-modeling)
Score features by intrinsic statistical properties; fast, no model fit.

- **Univariate (target-aware)**
  - Numeric → numeric: Pearson, Spearman, Kendall correlation
  - Numeric → categorical: t-test, ANOVA F-test
  - Categorical → categorical: Chi-square, mutual information
  - General: **Mutual Information (MI)**, Information Gain
- **Unsupervised (target-free)**
  - Near-zero variance / zero variance removal
  - High pairwise correlation pruning
  - Multicollinearity via **VIF** (Variance Inflation Factor)
  - Missingness threshold filtering
- **Multivariate filter**
  - **mRMR** (max-Relevance, min-Redundancy)
  - **ReliefF** family (instance-based weighting)

**Use when:** very wide data, quick screening, pre-step before wrapper/embedded.

---

## 2) Wrapper methods (search + model evaluation)
Search over feature subsets; score each via CV performance of a chosen model.

- **Greedy / sequential**
  - **Forward selection (SFS)**
  - **Backward elimination (SBS)**
  - **Stepwise / bidirectional**
  - **Floating** variants (SFFS, SBFS)
- **Recursive**
  - **RFE** / **RFECV** (rank features by model importance, drop iteratively)
- **Global / stochastic search**
  - Genetic algorithms, simulated annealing, particle swarm
  - Exhaustive search (only feasible for very small p)

**Pros:** captures interactions; **Cons:** costly, overfitting risk → **always wrap in CV**.

---

## 3) Embedded methods (selection inside model fit)
Selection emerges from the learning algorithm's objective.

- **Regularization-based (sparse)**
  - **Lasso (L1)**, **Elastic Net** → coefficient shrinkage to 0
  - L1-penalized logistic regression, L1-SVM
- **Tree-based importance**
  - Random Forest / XGBoost / LightGBM **gain / split / permutation importance**
  - **Boruta** (RF + shadow features, statistical test)
- **Other**
  - Gradient Boosting feature importance with thresholding (`SelectFromModel`)
  - Sparse linear models with stability selection

**Use when:** moderate-to-high p, want speed + decent accuracy; default in modern pipelines.

---

## 4) Hybrid / iterative methods
Combine filter + wrapper/embedded to balance speed and quality.

- Filter → wrapper (pre-screen, then SFS/RFE)
- Filter → embedded (variance/correlation prune → Lasso/RF)
- **Stability selection** (subsampling + Lasso/RF, retain frequently selected)
- **Permutation importance** post-fit (any model)

---

## 5) Dimensionality reduction (related, not strictly selection)
Transforms features rather than selecting subsets — keep separate conceptually.

- **Linear:** PCA, PLS, LDA, ICA
- **Nonlinear:** Kernel PCA, t-SNE, UMAP, autoencoders

**Note:** loses interpretability of original features.

---

## Decision guide (quick)

| Situation | First choice |
|---|---|
| p ≫ n, fast screen | Filter (MI, variance, correlation) |
| Linear signal, want sparsity | Lasso / Elastic Net |
| Nonlinear + interactions | RF / XGBoost importance, Boruta |
| Small p, need best subset | Wrapper (RFECV, SFS) |
| Need stability / inference | Stability selection, permutation importance |

---

## Diagnostics & pitfalls

- **Leakage:** do selection **inside CV folds**, never on full data before split.
- **Correlated features:** importance is shared/diluted → use grouped or permutation-based.
- **Multiple testing:** filter p-values need correction (Bonferroni, BH-FDR).
- **Importance ≠ causality**; permutation > impurity for unbiased ranking.
- **Scale sensitivity:** standardize for L1/L2, distance-based filters.
