# Feature Selection — Summary

> **Created by:** Claude Opus 4.7
> **Date:** 6 May 2026
> **Source:** *Feature Engineering and Selection: A Practical Approach for Predictive Models* — Max Kuhn & Kjell Johnson — https://feat.engineering/
> **Scope:** Chapters 10–12 (Feature Selection Overview, Greedy Search, Global Search)

---

## Topic: Feature Selection Methods — Family Schema

**Goal of feature selection:** choose a subset of input variables that maximizes predictive performance, reduces overfitting, lowers training cost, and improves interpretability.

**Three big families** (plus hybrids and dimensionality reduction):

| Family | Idea | Cost | Uses a model? |
|---|---|---|---|
| Filter | Score features by statistics | Low | No |
| Wrapper | Search subsets, evaluate with a model | High | Yes (external) |
| Embedded | Selection happens during model training | Medium | Yes (internal) |

---

## 1) Filter methods (model-agnostic, pre-modeling)

**What they do:** rank or remove features using statistical properties of the data alone — no predictive model is trained. Fast and scalable; usually a first screening step.

### 1.1 Univariate, target-aware → **feature vs target**
Each feature is scored independently against `y`. Ignores feature-feature interactions.

- **Numeric feature → numeric target** (regression)
  - **Pearson correlation** — linear association, assumes normality, sensitive to outliers
  - **Spearman correlation** — rank-based, captures monotonic (non-linear) relationships
  - **Kendall's τ (tau)** — rank-based, robust on small samples and ties
- **Numeric feature → categorical target** (classification)
  - **t-test** — compares means of feature across 2 classes
  - **ANOVA F-test** (Analysis of Variance) — compares means across ≥ 2 classes
- **Categorical feature → categorical target**
  - **Chi-square (χ²) test** — independence between two categorical variables
  - **Mutual Information (MI)** for discrete vars — bits of info shared
- **General-purpose (any types)**
  - **Mutual Information (MI)** — measures any (linear or non-linear) dependency; MI = 0 ⇔ independence
  - **Information Gain (IG)** — reduction in entropy of `y` given the feature; same family as MI

### 1.2 Unsupervised, target-free → **feature vs feature** (or feature alone)
Removes features without looking at `y`. Useful as a pre-cleaning step.

- **Near-Zero Variance (NZV) / Zero Variance removal** — drops features that are (almost) constant; they carry no information
- **High pairwise correlation pruning** — for highly correlated pairs (e.g. |r| > 0.9), keep one, drop the other
- **Variance Inflation Factor (VIF)** — measures multicollinearity of one feature against *all others* via R²; rule of thumb: VIF > 5–10 → problematic
- **Missingness threshold filtering** — drop features with too many missing values (e.g. > 50%)

### 1.3 Multivariate filter → **feature vs target + feature vs feature**
Considers redundancy among features while measuring relevance to target.

- **mRMR** (minimum Redundancy, Maximum Relevance) — selects features that are highly relevant to `y` but minimally redundant with already-selected features
- **ReliefF family** — instance-based; weights features by how well they distinguish nearest neighbors of the same vs different class

**Use when:** very wide data (p ≫ n), quick screening, or as a pre-step before wrapper / embedded methods.

---

## 2) Wrapper methods (search + model evaluation)

**What they do:** treat feature selection as a **search problem**. Try subsets of features, train a model on each, and score with cross-validation. The "wrapper" wraps around the model.

→ Comparison is **feature subset vs target**, mediated by a chosen model.

### 2.1 Greedy / sequential search
Iteratively add or remove one feature at a time based on CV score.

- **SFS** (Sequential Forward Selection) — start empty, add the feature that most improves the score, repeat
- **SBS** (Sequential Backward Selection / Backward Elimination) — start with all, drop the least useful one, repeat
- **Stepwise / bidirectional** — alternate add/remove steps; classical in statistics
- **Floating variants** (more flexible, less greedy)
  - **SFFS** (Sequential Forward Floating Selection) — forward steps with conditional backward checks
  - **SBFS** (Sequential Backward Floating Selection) — backward steps with conditional forward checks

### 2.2 Recursive elimination
Use the model's own importance to rank features, then drop the weakest.

- **RFE** (Recursive Feature Elimination) — fit model, drop k least important features, refit, repeat until target size
- **RFECV** (RFE with Cross-Validation) — same, but choose the optimal number of features by CV

### 2.3 Global / stochastic search
Explore the combinatorial space of subsets with metaheuristics.

- **Genetic Algorithms (GA)** — evolve populations of subsets via crossover and mutation
- **Simulated Annealing (SA)** — probabilistic acceptance of worse solutions to escape local minima
- **Particle Swarm Optimization (PSO)** — swarm-based search inspired by flocking
- **Exhaustive search** — try all 2^p subsets; only feasible when p is very small (≲ 20)

**Pros:** captures feature interactions; tailored to the chosen model.
**Cons:** computationally expensive; high overfitting risk → **always nest inside cross-validation**.

---

## 3) Embedded methods (selection inside model fit)

**What they do:** the model's training objective itself drives feature selection. Faster than wrappers, more powerful than filters. The model decides which features matter while learning.

→ Comparison is implicit **feature vs target** through the loss function.

### 3.1 Regularization-based (induce sparsity)
Add a penalty to the loss that shrinks coefficients; some go exactly to 0.

- **Lasso** (Least Absolute Shrinkage and Selection Operator) — L1 penalty (`λ Σ|βᵢ|`); produces sparse models
- **Elastic Net** — combination of L1 + L2 penalties; handles correlated groups better than pure Lasso
- **L1-penalized Logistic Regression** — Lasso adapted for binary/multiclass classification
- **L1-SVM** (L1-penalized Support Vector Machine) — sparse linear SVM

### 3.2 Tree-based importance
Tree ensembles produce importance scores as a by-product of training.

- **Random Forest / XGBoost / LightGBM** — importance derived from:
  - **Gain** — total loss reduction contributed by splits on the feature
  - **Split count / frequency** — how often the feature is used to split
  - **Cover** — number of samples affected by splits on the feature
- **Boruta** — wraps Random Forest; creates **shadow features** (random permutations) and keeps only features statistically more important than the best shadow

### 3.3 Other embedded approaches
- **`SelectFromModel`** (scikit-learn) — keep features with importance above a threshold from any fitted model (linear or tree)
- **Stability selection** — repeatedly subsample data + fit Lasso/RF; retain features selected often (more in §4)

**Use when:** moderate-to-high p, want speed + accuracy. Default in modern pipelines.

---

## 4) Hybrid / iterative methods

**What they do:** combine the strengths of filter, wrapper, and embedded methods to balance speed, stability, and predictive performance.

- **Filter → wrapper** — pre-screen with MI/correlation, then run SFS/RFE on survivors
- **Filter → embedded** — prune low-variance / highly correlated features, then fit Lasso or RF
- **Stability selection** — subsample many times, run Lasso/RF on each, retain features chosen above a frequency threshold (e.g. 60% of runs); reduces variance of selection
- **Permutation importance** (post-hoc, model-agnostic) — after fitting any model, shuffle each feature and measure performance drop; works on test data and avoids impurity bias

→ These are typically **feature vs target** evaluations driven by model performance, but use feature-vs-feature pre-filtering.

---

## 5) Dimensionality reduction (related, not strictly selection)

**What it does:** **transforms** the original features into a smaller set of new components. Unlike selection, original features are *not* preserved.

- **Linear methods**
  - **PCA** (Principal Component Analysis) — orthogonal components capturing maximum variance (unsupervised)
  - **PLS** (Partial Least Squares) — supervised version, finds components correlated with `y`
  - **LDA** (Linear Discriminant Analysis) — supervised, maximizes class separability
  - **ICA** (Independent Component Analysis) — components statistically independent (used in signal separation)
- **Nonlinear methods**
  - **Kernel PCA** — PCA in a kernel-induced feature space
  - **t-SNE** (t-distributed Stochastic Neighbor Embedding) — preserves local structure; for visualization, not modeling
  - **UMAP** (Uniform Manifold Approximation and Projection) — faster than t-SNE, preserves more global structure
  - **Autoencoders** — neural networks that learn compressed representations

**Trade-off:** loses interpretability of original features. Use when raw inputs are highly redundant or for visualization / preprocessing.

---

## Decision guide (quick)

| Situation | First choice |
|---|---|
| p ≫ n, fast screen | Filter (MI, variance, correlation) |
| Linear signal, want sparsity | Lasso / Elastic Net |
| Nonlinear + interactions | RF / XGBoost importance, Boruta |
| Small p, need best subset | Wrapper (RFECV, SFS) |
| Need stability / inference | Stability selection, permutation importance |
| Highly redundant inputs / visualization | Dimensionality reduction (PCA, UMAP) |

---

## Diagnostics & pitfalls

- **Data leakage:** perform feature selection **inside CV folds**, never on the full dataset before splitting — otherwise test scores are optimistic.
- **Correlated features:** importance gets shared/diluted between them → prefer **permutation importance** or grouped methods over impurity-based scores.
- **Multiple testing:** when running many univariate filters, correct p-values (**Bonferroni**, **Benjamini–Hochberg FDR**) to control false discoveries.
- **Importance ≠ causality:** a feature ranked important may be a confounder or proxy; don't infer causal effects from selection.
- **Impurity bias:** tree-based impurity importances favor high-cardinality and continuous features → use **permutation importance** for fair ranking.
- **Scale sensitivity:** standardize features (zero mean, unit variance) before applying L1/L2 regularization or distance-based filters (ReliefF, MI estimates with k-NN).
- **Stability:** small data perturbations can change selected features; use stability selection or repeated CV to assess robustness.

---

## Glossary of abbreviations

| Acronym | Full name |
|---|---|
| ANOVA | Analysis of Variance |
| FDR | False Discovery Rate |
| GA | Genetic Algorithm |
| ICA | Independent Component Analysis |
| IG | Information Gain |
| LDA | Linear Discriminant Analysis |
| Lasso | Least Absolute Shrinkage and Selection Operator |
| MI | Mutual Information |
| mRMR | minimum Redundancy, Maximum Relevance |
| NZV | Near-Zero Variance |
| PCA | Principal Component Analysis |
| PLS | Partial Least Squares |
| PSO | Particle Swarm Optimization |
| RF | Random Forest |
| RFE | Recursive Feature Elimination |
| RFECV | Recursive Feature Elimination with Cross-Validation |
| SA | Simulated Annealing |
| SBFS | Sequential Backward Floating Selection |
| SBS | Sequential Backward Selection |
| SFFS | Sequential Forward Floating Selection |
| SFS | Sequential Forward Selection |
| SVM | Support Vector Machine |
| t-SNE | t-distributed Stochastic Neighbor Embedding |
| UMAP | Uniform Manifold Approximation and Projection |
| VIF | Variance Inflation Factor |
| XGBoost | Extreme Gradient Boosting |
