---
title: "Conformal Prediction vs Quantile Regression for Regression Uncertainty"
slug: conformal-vs-quantile
topic: regression-uncertainty
domain: machine-learning
tags:
  - conformal-prediction
  - quantile-regression
  - cqr
  - prediction-intervals
  - uncertainty-quantification
  - heteroskedasticity
  - homoskedasticity
  - regression
methods:
  - quantile-regression
  - split-conformal-prediction
  - conformalized-quantile-regression
difficulty: intermediate
prerequisites:
  - regression
  - gradient-boosting
  - quantiles
language: python
libraries:
  - scikit-learn
  - numpy
  - matplotlib
related_libraries:
  - mapie
last_updated: 2026-05-07
status: stable
---

# Conformal Prediction vs Quantile Regression

> **TL;DR** — Quantile Regression (QR) gives **adaptive** intervals but no coverage guarantee. Split Conformal Prediction (CP) gives a **finite-sample coverage guarantee** but a fixed-width interval. Conformalized Quantile Regression (CQR) combines both. **Under heteroskedasticity, CQR clearly wins. Under homoskedasticity, CQR offers no benefit over plain CP.**

---

## 1. When to use each method

| Criterion                   | Quantile Regression          | Conformal Prediction          | CQR                            |
| --------------------------- | ---------------------------- | ----------------------------- | ------------------------------ |
| Coverage guarantee          | Asymptotic, model-dependent  | Finite-sample, distribution-free | Finite-sample, distribution-free |
| Captures heteroskedasticity | Yes (natively)               | No (fixed width)              | Yes                            |
| Requires calibration set    | No                           | Yes                           | Yes                            |
| Output                      | Interval `[lo, hi]`          | `point ± q̂`                  | Interval `[lo, hi]`            |
| Best when                   | Need conditional quantiles   | Need rigorous coverage        | Heteroskedastic data           |

### Decision rule

```
Need guaranteed coverage?
├── No  → Quantile Regression (richer, interpretable quantiles)
└── Yes → Is residual variance constant in x?
         ├── Yes → Split Conformal Prediction (simpler, equally good)
         └── No  → Conformalized Quantile Regression (CQR)
```

---

## 2. Mathematical foundations

### Quantile Regression

Minimizes the **pinball loss** for quantile $\tau$:

$$
L_\tau(y, \hat{q}) = \tau (y - \hat{q})_+ + (1-\tau)(\hat{q} - y)_+
$$

Train one model per quantile (e.g. $\tau = 0.05$ and $\tau = 0.95$ for a 90% interval).

### Split Conformal Prediction

Given a trained point estimator $\hat{f}$ and a calibration set of size $n$:

1. Compute nonconformity scores: $s_i = |y_i - \hat{f}(x_i)|$
2. Compute $\hat{q} = \text{Quantile}_{(1-\alpha)(1+1/n)}(\{s_i\})$
3. Predict: $\hat{C}(x) = [\hat{f}(x) - \hat{q},\ \hat{f}(x) + \hat{q}]$

**Marginal coverage guarantee:** $\mathbb{P}(y \in \hat{C}(x)) \geq 1 - \alpha$ — distribution-free, finite-sample.

### CQR (Conformalized Quantile Regression)

Combines QR with conformal calibration:

1. Train quantile models $\hat{q}_{lo}$, $\hat{q}_{hi}$
2. Compute conformity scores: $s_i = \max(\hat{q}_{lo}(x_i) - y_i,\ y_i - \hat{q}_{hi}(x_i))$
3. Compute $q_{cqr} = \text{Quantile}_{(1-\alpha)(1+1/n)}(\{s_i\})$
4. Predict: $\hat{C}(x) = [\hat{q}_{lo}(x) - q_{cqr},\ \hat{q}_{hi}(x) + q_{cqr}]$

---

## 3. Setup

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

ALPHA = 0.10  # target 90% coverage

# Heteroskedastic: noise grows with x
def make_hetero(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0, 10, n)).reshape(-1, 1)
    y = 2 * X.ravel() + rng.normal(0, 0.5 + 0.3 * X.ravel(), n)
    return X, y

# Homoskedastic: constant noise
def make_homo(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0, 10, n)).reshape(-1, 1)
    y = 2 * X.ravel() + rng.normal(0, 1.5, n)
    return X, y
```

---

## 4. Methods — implementation

### Quantile Regression

```python
lo_m = GradientBoostingRegressor(loss="quantile", alpha=0.05,
                                 n_estimators=100, random_state=42)
hi_m = GradientBoostingRegressor(loss="quantile", alpha=0.95,
                                 n_estimators=100, random_state=42)
lo_m.fit(X_train, y_train)
hi_m.fit(X_train, y_train)

qr_lo = lo_m.predict(X_test)
qr_hi = hi_m.predict(X_test)
```

### Split Conformal Prediction

```python
pt_m = GradientBoostingRegressor(n_estimators=100, random_state=42)
pt_m.fit(X_train, y_train)

scores = np.abs(y_cal - pt_m.predict(X_cal))
q_hat  = np.quantile(scores,
                     np.ceil((1-ALPHA)*(len(scores)+1)) / len(scores))

cp_pt = pt_m.predict(X_test)
cp_lo = cp_pt - q_hat
cp_hi = cp_pt + q_hat
```

### Conformalized Quantile Regression (CQR)

```python
cal_scores = np.maximum(lo_m.predict(X_cal) - y_cal,
                        y_cal - hi_m.predict(X_cal))
q_cqr = np.quantile(cal_scores,
                    np.ceil((1-ALPHA)*(len(cal_scores)+1)) / len(cal_scores))

cqr_lo = lo_m.predict(X_test) - q_cqr
cqr_hi = hi_m.predict(X_test) + q_cqr
```

---

## 5. Results — Heteroskedastic data

> $y = 2x + \mathcal{N}(0,\ 0.5 + 0.3x)$ — variance grows linearly with $x$.

### Empirical results

| Method | Coverage | Avg Width |
| ------ | -------- | --------- |
| QR     | 0.860    | 6.20      |
| CP     | 0.930    | 8.09      |
| CQR    | 0.920    | **6.90**  |

Constants: `q_hat = 4.0468`, `q_cqr = 0.3514`.

### Visualisation

![QR vs CP — Heteroskedastic](assets/hetero_qr_vs_cp.png)

QR's interval **adapts** — narrow near $x \approx 0$, wide near $x \approx 10$. CP's band is **uniformly wide** because $\hat{q}$ is a single scalar.

![CQR — Heteroskedastic](assets/hetero_cqr.png)

CQR keeps the adaptive shape of QR while inheriting CP's coverage guarantee.

![Metrics — Heteroskedastic](assets/hetero_metrics.png)

### 15-sample inspection

![15 samples — Heteroskedastic](assets/hetero_15_samples.png)

White circles = true $y$. Coloured bars = predicted interval. Red bars = misses. CP bars are all the same length; QR and CQR vary per sample.

---

## 6. Results — Homoskedastic data

> $y = 2x + \mathcal{N}(0,\ 1.5)$ — constant variance.

### Empirical results

| Method | Coverage | Avg Width |
| ------ | -------- | --------- |
| QR     | 0.840    | 4.78      |
| CP     | 0.940    | **5.40**  |
| CQR    | 0.930    | 5.87      |

Constants: `q_hat = 2.7013`, `q_cqr = 0.5446`.

> **Note:** Under homoskedasticity, **CQR is slightly *wider* than CP** while delivering the same coverage. The adaptive machinery has nothing to adapt to and adds a small calibration overhead.

### Visualisation

![QR vs CP — Homoskedastic](assets/homo_qr_vs_cp.png)

Both bands now have **roughly constant width**. QR's adaptiveness disappears because the conditional spread is constant.

![CQR — Homoskedastic](assets/homo_cqr.png)

![Metrics — Homoskedastic](assets/homo_metrics.png)

### 15-sample inspection

![15 samples — Homoskedastic](assets/homo_15_samples.png)

All three methods produce **visually similar bar lengths**. CP's simplicity becomes an advantage.

---

## 7. Side-by-side comparison

| Scenario        | Method | Coverage | Width | Verdict                       |
| --------------- | ------ | -------- | ----- | ----------------------------- |
| Heteroskedastic | QR     | 0.860    | 6.20  | Under-covers                  |
| Heteroskedastic | CP     | 0.930    | 8.09  | Covers, wide                  |
| Heteroskedastic | **CQR**| 0.920    | 6.90  | **Best — covers + adaptive**  |
| Homoskedastic   | QR     | 0.840    | 4.78  | Under-covers                  |
| Homoskedastic   | **CP** | 0.940    | 5.40  | **Best — simplest, narrowest**|
| Homoskedastic   | CQR    | 0.930    | 5.87  | OK, but no advantage over CP  |

### Key insight

> **The case for CQR over plain CP is essentially the case for heteroskedasticity.**
>
> If your residuals look constant-variance, plain CP is the right call. Diagnostic check: plot residuals vs predictions or vs each feature. If spread is roughly constant, prefer CP for simplicity.

---

## 8. Predicting on new data — step-by-step

For each method, calibration constants ($\hat{q}$, $q_{cqr}$) are computed **once after training** and reused at inference.

### QR

1. Train two GBMs: $\tau = 0.05$ and $\tau = 0.95$ on `X_train, y_train`.
2. At inference: call both models on `X_new`.
3. Output = `[lo_m.predict(X_new), hi_m.predict(X_new)]`.

```python
X_new = np.array([[3.5], [7.2], [1.0]])
qr_lo_new = lo_m.predict(X_new)
qr_hi_new = hi_m.predict(X_new)
```

### CP

1. Train point estimator on `X_train, y_train`.
2. Compute nonconformity scores on `X_cal`: $s_i = |y_i - \hat{f}(x_i)|$.
3. Compute $\hat{q} = \text{Quantile}_{(1-\alpha)(1+1/n_{cal})}$ of $\{s_i\}$ — store it.
4. At inference: `point = pt_m.predict(X_new)`, then `lo = point - q_hat`, `hi = point + q_hat`.

```python
cp_point_new = pt_m.predict(X_new)
cp_lo_new = cp_point_new - q_hat
cp_hi_new = cp_point_new + q_hat
```

### CQR

1. Train `lo_m` ($\tau=0.05$) and `hi_m` ($\tau=0.95$).
2. Compute conformity scores on `X_cal`: $s_i = \max(\hat{q}_{lo}(x_i) - y_i,\ y_i - \hat{q}_{hi}(x_i))$.
3. Compute $q_{cqr}$ — store it.
4. At inference: `lo = lo_m.predict(X_new) - q_cqr`, `hi = hi_m.predict(X_new) + q_cqr`.

```python
cqr_lo_new = lo_m.predict(X_new) - q_cqr
cqr_hi_new = hi_m.predict(X_new) + q_cqr
```

---

## 9. Summary — outputs at inference

| Method | Output                                    | Point estimate | Adaptive width | Coverage guarantee |
| ------ | ----------------------------------------- | -------------- | -------------- | ------------------ |
| QR     | `[lo_m(x), hi_m(x)]`                      | No             | Yes            | No                 |
| CP     | `pt_m(x) ± q_hat`                         | Yes            | No             | Yes                |
| CQR    | `[lo_m(x) − q_cqr, hi_m(x) + q_cqr]`      | No             | Yes            | Yes                |

---

## 10. Failure modes & diagnostics

- **Calibration / test exchangeability** — All conformal guarantees assume calibration and test data are exchangeable. Distribution shift breaks this; use weighted or adaptive conformal variants.
- **Calibration set too small** — $n_{cal} < 100$ produces unstable $\hat{q}$. Aim for $n_{cal} \geq 200$.
- **QR crossing** — Independently fit quantile models can produce $\hat{q}_{lo}(x) > \hat{q}_{hi}(x)$ for some $x$. Monotone-quantile methods or post-hoc sorting fix this.
- **Marginal vs conditional coverage** — CP / CQR guarantee *marginal* coverage (averaged over $x$). They may under-cover in specific subgroups. Use Mondrian or local conformal for group-conditional guarantees.

---

## 11. Reproducing this report

```bash
python conformal-vs-quantile.py
# → regenerates all assets/*.png and prints metrics
```

Splits used: `n_train=600`, `n_cal=200`, `n_test=200`. Model: `GradientBoostingRegressor(n_estimators=100, random_state=42)`. Target coverage: $1 - \alpha = 0.90$.

## 12. References

- Romano, Patterson, Candès (2019). *Conformalized Quantile Regression.* NeurIPS.
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World.*
- Angelopoulos, Bates (2023). *A Gentle Introduction to Conformal Prediction.* [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
- MAPIE library — production-ready CQR / CP wrappers: [github.com/scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE)
