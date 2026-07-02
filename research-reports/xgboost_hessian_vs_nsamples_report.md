# XGBoost — Hessian vs Number of Samples: Why min_child_weight Means Different Things for Different Objectives

> **Scope.** Deep technical explanation of how XGBoost's Newton-step optimization links the Hessian of the loss function to the `min_child_weight` parameter, and why the same numerical value of `min_child_weight` implies radically different effective sample-size floors depending on the objective chosen.
> **Assumed background.** Familiarity with gradient boosting at a high level; comfort with first/second derivatives.
> **Motivation.** The XGBoost docs say `min_child_weight` is the "minimum sum of instance weight (hessian) needed in a child." For linear regression (`reg:squarederror`), this equals the number of instances. For logistic-loss objectives, it does **not** — and using the same numeric values across objectives leads to drastically different effective regularization without any warning.

---

## TL;DR

- `min_child_weight` is a **Hessian-sum threshold**, not an instance-count threshold. Only for `reg:squarederror` are the two identical (because `h_i = 1` always).
- For `reg:logistic`, `h_i = p_i(1 − p_i) ≤ 0.25`. A value of `min_child_weight = 60` requires **≥240 instances** at `p=0.5` and **≥667 instances** at `p=0.1` — meaning extreme-prediction leaves can never form on small datasets.
- The correct formula is: `min_child_weight_intended = N_target_samples × mean(h_i)`.
- For `reg:logistic` with `mean(y) ≈ 0.55`: to require at least 30 real instances per leaf, set `min_child_weight ≈ 7–8`, not 60.
- Every objective has a different `h_i`; **always compute the objective-specific scaling factor before setting `min_child_weight`**.

---

## 1. Newton Boosting Foundations

### 1.1 The Taylor expansion objective

At round `t`, XGBoost adds a new tree `f_t` to minimize the second-order Taylor approximation of the loss:

```
Obj(t) ≈ Σᵢ [ gᵢ · f_t(xᵢ) + ½ hᵢ · f_t(xᵢ)² ] + Ω(f_t)
```

where:
- `gᵢ = ∂ℓ(yᵢ, ŷᵢ)/∂ŷᵢ` — first derivative (gradient)
- `hᵢ = ∂²ℓ(yᵢ, ŷᵢ)/∂ŷᵢ²` — second derivative (Hessian, scalar per instance)
- `Ω(f_t) = γT + ½λ Σⱼ wⱼ²` — tree complexity penalty (T leaves, weights wⱼ)

This is a separable sum: **each leaf is an independent quadratic in its weight**. The key insight is that `hᵢ` controls the **curvature** of the local loss — how "steep" the parabola is for the weight of the leaf that instance `i` lands in.

### 1.2 Optimal leaf weight

For a leaf `j` containing instance set `Iⱼ`, minimizing the quadratic gives:

```
w*ⱼ = − Gⱼ / (Hⱼ + λ)

where  Gⱼ = Σᵢ∈Iⱼ gᵢ     (sum of gradients in leaf j)
       Hⱼ = Σᵢ∈Iⱼ hᵢ     (sum of Hessians in leaf j)
       λ   = reg_lambda   (L2 regularization)
```

Three things to read from this formula:

1. **The Hessian is the denominator.** Larger `Hⱼ` → smaller `|w*ⱼ|` → predictions shrink toward zero (in margin space, i.e. toward `base_score` in prediction space). This is the regularization effect of the Hessian itself, separate from `λ`.

2. **`λ` adds to `Hⱼ`.** So `reg_lambda` and `min_child_weight` interact multiplicatively: a large `min_child_weight` forces large `Hⱼ`, but then `λ` is added on top. Both inflate the denominator.

3. **If `hᵢ` is small per instance, `Hⱼ` grows slowly with `n`.** You need many instances to accumulate enough Hessian to form a leaf.

### 1.3 The split gain formula

A split of node `P` into children `L` and `R` is executed only if the gain exceeds the regularization threshold:

```
Gain = ½ [ GL² / (HL + λ) + GR² / (HR + λ) − GP² / (HP + λ) ] − γ
```

**The split is rejected if either child has `HL < min_child_weight` or `HR < min_child_weight`.**

This is the exact mechanism: `min_child_weight` is a **Hessian-sum floor**, not a count floor. The tree cannot learn a region if it cannot accumulate enough Hessian there.

---

## 2. The min_child_weight = n_samples Equivalence (and when it breaks)

### 2.1 Why it works for reg:squarederror

For MSE loss: `ℓ = ½(y − ŷ)²`

```
gᵢ = ŷᵢ − yᵢ          (residual)
hᵢ = 1                  (constant — always 1, for every instance)
```

Because `hᵢ = 1`:

```
Hⱼ = Σᵢ∈Iⱼ hᵢ = |Iⱼ| = n_leaf
```

So `min_child_weight ≡ min_samples_leaf` exactly. This is the **only common objective** where this equivalence holds.

### 2.2 Why it breaks for reg:logistic

For cross-entropy loss with sigmoid link: `ℓ = −[y log p + (1−y) log(1−p)]` where `p = sigmoid(m)` and `m` is the raw margin:

```
gᵢ = pᵢ − yᵢ           (predicted prob minus label)
hᵢ = pᵢ(1 − pᵢ)        (variance of a Bernoulli(pᵢ))
```

`hᵢ ∈ (0, 0.25]`:

- Maximum at `pᵢ = 0.5`: `hᵢ = 0.25`
- At `pᵢ = 0.1` or `0.9`: `hᵢ = 0.09`
- At `pᵢ = 0.01` or `0.99`: `hᵢ = 0.0099`

The Hessian sum for a leaf is:

```
Hⱼ = Σᵢ∈Iⱼ pᵢ(1 − pᵢ) ≤ n_leaf × 0.25
```

So the minimum number of instances to satisfy `Hⱼ ≥ min_child_weight` is:

```
n_min_real = min_child_weight / mean(hᵢ in leaf)
           ≥ min_child_weight / 0.25
           = 4 × min_child_weight
```

And for leaves with extreme predictions:

```
n_min_real → ∞  as  pᵢ → 0 or pᵢ → 1
```

**This is the collapse mechanism.** The model tries to form leaves with high or low predicted values (to capture extreme targets), but those leaves — by definition — have `hᵢ → 0`, so they accumulate Hessian very slowly. On a small dataset, `Hⱼ` never reaches `min_child_weight`. The tree cannot split further. All predictions stay near `base_score`.

---

## 3. Hessian Formulas by Objective

The table below gives the gradient, Hessian, and effective n_min per objective. All Hessians are with respect to the **raw margin** `m` (before any link function is applied), which is what XGBoost optimizes internally.

| Objective | Loss | Gradient `gᵢ` | Hessian `hᵢ` | Max `hᵢ` | n_min for mcw=60 at mean pred |
|---|---|---|---|---|---|
| `reg:squarederror` | ½(y−ŷ)² | ŷᵢ−yᵢ | **1** (constant) | 1.00 | **60** (exact) |
| `reg:logistic` | −[y log p + (1−y)log(1−p)] | pᵢ−yᵢ | **pᵢ(1−pᵢ)** | 0.25 | **≥240** |
| `binary:logistic` | same as reg:logistic | pᵢ−yᵢ | **pᵢ(1−pᵢ)** | 0.25 | **≥240** |
| `multi:softprob` | multiclass cross-entropy | pₖ−1{yᵢ=k} per class | **pₖ(1−pₖ)** per class | 0.25 | **≥240** |
| `count:poisson` | Poisson log-likelihood | ŷᵢ−yᵢ (log link) | **ŷᵢ** (predicted count) | unbounded | ŷ-dependent |
| `reg:gamma` | Gamma log-likelihood | 1−yᵢ/ŷᵢ (log link) | **1** (in log-link space) | 1.00 | **60** (near exact) |
| `reg:tweedie` | Tweedie log-likelihood | see below | **ŷᵢ^(2−p)** | depends on `p` | ŷ-dependent |
| `reg:absoluteerror` | \|y−ŷ\| (smoothed) | ≈sign(ŷ−y) | **≈1** (smooth approx) | ≈1 | **≈60** |
| `reg:pseudohubererror` | Pseudo-Huber(δ) | residual/√(1+(e/δ)²) | 1/(1+(e/δ)²)^(3/2) | 1.00 | **≈60** at small residuals |

### Notes on specific objectives

**`reg:logistic` vs `binary:logistic`:** mathematically identical Hessian. The difference is only in what values `y` can take and how the output is interpreted.

**`multi:softprob`:** the Hessian per class is `p_k(1−p_k)`. For K=10 classes with uniform predictions, `h ≈ K × 0.09 = 0.9` (still less than 1 for fewer classes or unbalanced predictions).

**`count:poisson`:** `hᵢ = ŷᵢ` (the predicted count). If counts are small (e.g. ŷ ≈ 0.5 events/period), `hᵢ < 1` → same under-accumulation as logistic. If counts are large (ŷ ≈ 100), `hᵢ` is large → `min_child_weight` is satisfied with fewer instances.

**`reg:tweedie`:** `hᵢ = ŷᵢ^(2−p)` where `p ∈ (1,2)` is `tweedie_variance_power`. For `p=1.5` (common): `hᵢ = ŷᵢ^0.5` (square root of prediction). Hessian is data-magnitude dependent.

**`reg:absoluteerror`:** MAE has `hᵢ = 0` analytically (non-differentiable). XGBoost implements this via an internal smoothed approximation where the effective Hessian is approximately 1 for most instances; `min_child_weight` behaves close to `reg:squarederror`.

**`reg:pseudohubererror`:** `hᵢ = 1/(1+(eᵢ/δ)²)^(3/2)` where `eᵢ = ŷᵢ−yᵢ`. For residuals near 0 (well-fit instances), `hᵢ ≈ 1`. For large residuals (`|eᵢ| >> δ`), `hᵢ → 0` — meaning poorly-fit instances accumulate almost no Hessian, making those regions hard to split further (self-regularizing behavior).

---

## 4. The Collapse Mechanism — Step by Step

This section traces exactly why `reg:logistic` with a high `min_child_weight` collapses predictions to `base_score`.

### Step 1: Initialization

XGBoost sets `base_score ≈ sigmoid⁻¹(mean(y))`. All initial predictions are `p₀ = mean(y)`.

### Step 2: First Hessian computation

At round 0, all `pᵢ = p₀`. Therefore all `hᵢ = p₀(1−p₀)` — a single constant. For `mean(y) = 0.55`, `h₀ = 0.2475`.

To split a node of `n` instances, both children must have `Hⱼ ≥ min_child_weight`. With uniform `h₀`, the minimum node size to allow a split is:

```
n_min_split = 2 × min_child_weight / h₀
            = 2 × 60 / 0.2475
            ≈ 485 instances per parent node
```

### Step 3: What the first tree can do

With ~5,000 training rows and the root requiring 485+ rows to split: the tree **can** split at the root (5,000 >> 485). But each subsequent split needs the child to have ≥242 instances. With `max_depth=4`, at most 16 leaves, averaging ~312 rows/leaf at a `max_depth=4` balanced tree. Many splits near the leaves will be blocked.

### Step 4: Round 2+ — the vicious cycle for extreme predictions

After round 1, some predictions move slightly away from `p₀`. The key asymmetry: **high-target instances** (e.g. `y ≈ 0.9`) get predictions that slowly rise, but as `p` increases, `h = p(1−p)` **decreases**. The nodes that need to learn extreme values are the hardest to split because their Hessian accumulates most slowly.

Concretely: a node of 300 instances all predicting `p=0.1` has:

```
H = 300 × 0.1 × 0.9 = 27  <  min_child_weight=60  →  cannot split
```

A node of 300 instances all predicting `p=0.5` has:

```
H = 300 × 0.25 = 75  ≥  min_child_weight=60  →  can split
```

The center is learnable; the tails are not.

### Step 5: Leaf weight shrinkage

Even for leaves that do form, the weight formula `w* = −G/(H+λ)` further shrinks the correction:

```
With H=60 (barely allowed), λ=9:
   denominator = 60 + 9 = 69
   weight correction ≈ G/69

With H=600 (10x as many instances), λ=9:
   denominator = 600 + 9 = 609
   weight correction ≈ G/609  (much smaller per-step correction)
```

`reg_lambda=9` contributes relatively little when `H` is already large, but for borderline leaves (`H` just above threshold), it further shrinks the update toward zero. The result is that even the leaves that form produce small corrections and the model converges to predictions clustered around `base_score`.

---

## 5. The General Correction Formula

For any objective, the formula to translate "I want at least N_target real instances per leaf" into the correct `min_child_weight` is:

```
min_child_weight_correct = N_target × mean_leaf_hessian
                         = N_target × E[hᵢ]
```

where `E[hᵢ]` is the expected Hessian per instance, which depends on the typical prediction values in that region. In practice, use the **global mean prediction** (≈ `mean(y)` at initialization) as the approximation:

### Per-objective formula and worked example

Assume `mean(y) = 0.55`, `N_target = 30` (the real desired minimum), `N_target_original = 60` (the intent when setting mcw=60):

| Objective | `E[hᵢ]` formula | `E[hᵢ]` at `mean(y)=0.55` | `mcw` for `N_target=30` | `mcw` for `N_target=60` | Your `mcw=60` → real `n_min` |
|---|---|---|---|---|---|
| `reg:squarederror` | 1 | 1.000 | 30 | 60 | **60** ✓ |
| `reg:logistic` | `ȳ(1−ȳ)` | 0.2475 | **~7.4** | **~14.9** | **~242** ✗ |
| `binary:logistic` | `ȳ(1−ȳ)` | 0.2475 | **~7.4** | **~14.9** | **~242** ✗ |
| `count:poisson` | `mean(ŷ)` | depends | N/A | N/A | ŷ-dep. |
| `reg:gamma` | ≈1 | ≈1.000 | ≈30 | ≈60 | **≈60** ✓ |
| `reg:tweedie (p=1.5)` | `mean(ŷ)^0.5` | depends | N/A | N/A | ŷ-dep. |
| `reg:absoluteerror` | ≈1 | ≈1.000 | ≈30 | ≈60 | **≈60** ✓ |

### Python utility function

```python
import numpy as np

def recommended_min_child_weight(objective: str, y_train, N_target: int = 30,
                                  tweedie_power: float = 1.5) -> dict:
    """
    Compute the correct min_child_weight for a given XGBoost objective
    given the desired minimum real instances per leaf (N_target).
    Returns the value and the key diagnostic numbers.
    """
    y = np.asarray(y_train, dtype=float)
    ym = y.mean()

    if objective in ("reg:squarederror", "reg:linear", "reg:absoluteerror",
                     "reg:pseudohubererror", "reg:gamma"):
        h_avg = 1.0
        note = "h_i ≈ 1 (constant); mcw ≡ n_samples"

    elif objective in ("reg:logistic", "binary:logistic"):
        h_avg = ym * (1.0 - ym)
        note = f"h_i = p(1-p); at mean(y)={ym:.3f} → h_avg={h_avg:.4f}"

    elif objective == "multi:softprob":
        # For K balanced classes, E[p_k] ≈ 1/K → h_avg per class ≈ (1/K)(1-1/K)
        # User should supply n_classes; here we approximate from y distribution
        K = len(np.unique(y))
        p_k = 1.0 / K
        h_avg = p_k * (1.0 - p_k) * K   # sum over classes per instance
        note = f"h_i = Σ_k p_k(1-p_k); K={K} balanced → h_avg={h_avg:.4f}"

    elif objective == "count:poisson":
        h_avg = ym   # E[ŷ] ≈ mean(y) at initialization
        note = f"h_i = ŷ_i; at mean(y)={ym:.3f} → h_avg={h_avg:.4f}"

    elif objective == "reg:tweedie":
        h_avg = ym ** (2.0 - tweedie_power)
        note = (f"h_i = ŷ^(2-p); p={tweedie_power}, mean(y)={ym:.3f} "
                f"→ h_avg={h_avg:.4f}")
    else:
        h_avg = 1.0
        note = "Unknown objective — defaulting to h_avg=1 (squarederror assumption)"

    mcw_correct  = N_target * h_avg
    mcw_original = 60   # common default people use
    n_min_if_60  = mcw_original / h_avg if h_avg > 0 else float("inf")

    return {
        "objective":        objective,
        "h_avg":            round(h_avg, 4),
        "N_target":         N_target,
        "min_child_weight": round(mcw_correct, 2),
        "n_min_if_mcw_60":  round(n_min_if_60, 1),
        "note":             note,
    }

# Example usage:
result = recommended_min_child_weight("reg:logistic", y_tr, N_target=30)
print(result)
# → {'objective': 'reg:logistic', 'h_avg': 0.2475,
#    'N_target': 30, 'min_child_weight': 7.43,
#    'n_min_if_mcw_60': 242.4, 'note': '...'}
```

---

## 6. The Hessian as a Natural Weighting Mechanism

Beyond `min_child_weight`, the Hessian plays two additional roles that interact with regularization.

### 6.1 Hessian as Fisher information / effective sample weight

For generalized linear models, the Hessian of the log-likelihood is the **Fisher information**. An instance with low `hᵢ` carries less information about the model parameters — it is "less informative" about the correct leaf weight, so XGBoost (correctly) demands more of them to form a reliable estimate. This is not a bug; it is the correct Bayesian behavior.

For `reg:logistic`: instances near the decision boundary (`p ≈ 0.5`) carry maximum information (`h = 0.25`); instances the model is already very confident about (`p ≈ 0 or 1`) carry almost none. The model appropriately learns more from uncertain instances.

### 6.2 The leaf weight denominator — implicit shrinkage

The leaf weight `w* = −G/(H+λ)` can be rewritten as:

```
w* = − (mean gradient) × n_leaf / (Σhᵢ + λ)
```

For `reg:squarederror` (`hᵢ = 1`): `w* = −(mean residual) × n / (n + λ)` → a standard ridge-shrunk mean.

For `reg:logistic`: `w* = −(mean gradient) / (mean(hᵢ) + λ/n)` → the effective learning rate is scaled by `1/mean(hᵢ)`. This means:
- At `p=0.5` (center): step size is scaled by `1/0.25 = 4×`.
- At `p=0.1` (tail): step size is scaled by `1/0.09 = 11×`.

**The model takes larger steps in the tails but can't form leaves there — the two effects cancel.** That is the collapse: the optimizer wants to take big steps in extreme-prediction zones, but `min_child_weight` prevents those zones from ever being created.

### 6.3 Hessian and sample_weight interaction

If `sample_weight` is provided, XGBoost multiplies `hᵢ` and `gᵢ` by the weight. This means:

```
min_child_weight threshold is compared to Σᵢ∈leaf (wᵢ × hᵢ)
```

For `reg:logistic` with inverse-density weights (upweighting tails), the effective Hessian for high-weight instances at `p→0 or p→1` is `wᵢ × hᵢ`. Since `hᵢ→0` and `wᵢ→large`, the product can remain non-trivial — this is one mechanism by which sample weighting (Option G in the main report) helps the tails.

---

## 7. Practical Interactions with Other Hyperparameters

### 7.1 reg_lambda and min_child_weight are additive in the denominator

```
w*ⱼ = − Gⱼ / (Hⱼ + λ)
```

The effective denominator is `Hⱼ + λ`. For a leaf with `Hⱼ = min_child_weight` (barely formed), `λ = 9` adds 15% to the denominator for `mcw=60` but 900% for `mcw=1`. So:

- **High `mcw` + high `λ`:** double shrinkage, very conservative.
- **Low `mcw` + high `λ`:** `λ` dominates; the tree can form tiny leaves but their weights are severely shrunk by `λ`.
- **Low `mcw` + low `λ`:** leaves can form and have unshrunken weights; needs `max_depth` or `gamma` to prevent overfitting.

When lowering `min_child_weight`, also consider relaxing `reg_lambda` proportionally to avoid over-shrinking leaf weights.

### 7.2 gamma acts on Hessian-normalized gain

The split gain threshold `γ` is compared against the **gain**:

```
Gain = ½ [ GL²/(HL+λ) + GR²/(HR+λ) − GP²/(HP+λ) ] − γ
```

Since gains scale with `H`, a `γ` calibrated for `reg:squarederror` is proportionally stricter for `reg:logistic` (lower `H` → lower gain for the same structural benefit). To convert:

```
γ_logistic ≈ γ_squarederror × h_avg²   (rough approximation)
```

For your data, `h_avg ≈ 0.25`, so `γ = 0.5` with `reg:logistic` is effectively like `γ ≈ 0.5 / 0.0625 = 8` with `reg:squarederror` — very restrictive. This is another axis of the compression.

### 7.3 colsample and subsample — interaction with small H

When only 60% of instances are sampled (`subsample=0.6`) and features are further subsampled, the effective Hessian available per leaf is:

```
Hⱼ_effective ≈ H_full × subsample
```

For a leaf with `H_full = 70` (just above threshold with full data), after 60% sampling: `H ≈ 42 < 60` → the leaf disappears. Subsampling with `reg:logistic` makes the Hessian shortage worse. With `subsample=0.6` and `min_child_weight=60`, the effective real-instance requirement is:

```
n_min_real = min_child_weight / (h_avg × subsample)
           = 60 / (0.2475 × 0.6)
           ≈ 404 instances
```

---

## 8. Summary Table: Effect of Current Parameters on reg:logistic

Your current configuration (`mcw=60`, `reg_lambda=9.08`, `max_depth=4`, `subsample=0.6`):

| Parameter | Nominal value | Effective meaning for reg:logistic | Verdict |
|---|---|---|---|
| `min_child_weight=60` | 60 Hessian units | ≥242 real instances at center; ≥667 at tails | **Over-restrictive** |
| `reg_lambda=9.08` | L2 on leaf weights | Leaf weight divided by `(H+9)`, large relative to `H_min=60` | **Adds extra shrinkage** |
| `max_depth=4` | 16 max leaves | With n=5k and n_min≈242, ≤20 effective leaves | **Tight but not primary issue** |
| `subsample=0.6` | 60% row sampling | Increases n_min by 1/0.6 → ≈404 real instances required | **Compounds mcw problem** |
| `gamma=0.5` | Split threshold | Effectively ~8× stricter than for squarederror | **Adds extra conservatism** |
| Combined | — | Predicted range ≈ [0.25, 0.80] — confirmed by train marginal plot | **Collapse confirmed** |

---

## 9. Diagnostics and Pitfalls

- **Diagnostic #1 — check `std(pred)` on train with current params.** If `std(pred) < 0.15` on train for a target with `std(y) ≈ 0.35`, collapse is confirmed. The check costs nothing and provides immediate evidence.
- **Diagnostic #2 — compare with `reg:squarederror` on same data.** Train the same hyperparameters with MSE objective. If the MSE model has a wider range, the difference is purely the Hessian scaling of `min_child_weight`.
- **Pitfall: porting hyperparameters across objectives.** A `min_child_weight` that works well for `reg:squarederror` will always be too large for `reg:logistic` by a factor of `1/h_avg ≈ 4×`. Never transfer numeric values across objectives without rescaling.
- **Pitfall: imbalanced classes with binary:logistic.** If `mean(y) = 0.05` (5% positive), then `h_avg = 0.05 × 0.95 = 0.0475`. A `min_child_weight = 1` (the default) already requires `1/0.0475 ≈ 21` instances per child. Extremely imbalanced cases need `min_child_weight` near 0 to allow any splits on the minority class.
- **Pitfall: Hessian collapse in well-calibrated models.** A well-calibrated model that has already placed most predictions near 0 or 1 will have very low `hᵢ` in late rounds — meaning `min_child_weight` becomes progressively harder to satisfy as training proceeds. This is why early stopping or a `n_estimators` search is important: over-training can cause late-round Hessian starvation.
- **Pitfall: `count:poisson` with rare events.** If `mean(ŷ) ≈ 0.1` events/period, `h_avg ≈ 0.1` → same regime as logistic at `p=0.1`. `min_child_weight=1` already requires ~10 instances.
- **`min_child_weight` in Optuna:** when searching, always define the range in **objective-consistent units**, not raw sample counts. Use `h_avg` as a scaling constant per the formula in Section 5.

---

## 10. Decision Rule — Setting min_child_weight Correctly

```
1. Compute h_avg:
   h_avg = mean(y) * (1 − mean(y))          # for reg:logistic / binary:logistic
   h_avg = 1.0                               # for reg:squarederror / reg:absoluteerror
   h_avg = mean(y) ** (2 − tweedie_power)    # for reg:tweedie

2. Decide N_target (real instances you want per leaf):
   Small dataset (n < 5k):  N_target = 20–40
   Medium (5k–50k):         N_target = 30–60
   Large (> 50k):           N_target = 50–150

3. Set: min_child_weight = N_target × h_avg

4. For Optuna search, span: [N_target_low × h_avg, N_target_high × h_avg]
   Example for reg:logistic, mean(y)=0.55:
       low  = 5  × 0.2475 = 1.24
       high = 120 × 0.2475 = 29.7
   trial.suggest_float("min_child_weight", 1.2, 30.0, log=True)

5. When changing objectives, always recompute min_child_weight from scratch.
   Never port numeric values directly across objectives.
```

---

## References

1. XGBoost original paper — Chen & Guestrin (2016): second-order Taylor expansion, leaf weight formula `w* = −G/(H+λ)`, split gain. https://arxiv.org/pdf/1603.02754.pdf
2. XGBoost documentation — `min_child_weight` definition: "Minimum sum of instance weight (hessian) needed in a child." https://xgboost.readthedocs.io/en/stable/parameter.html
3. XGBoost GitHub — Community note confirming Hessian-sum interpretation of `min_child_weight` for logistic loss (h_i=0.25 at p=0.5). https://medium.com/@ryassminh/xgboost-with-a-simple-example-92d5d91789e2
4. XGBoost — Custom Objective and Evaluation Metric (gradient and Hessian computation for built-in and custom losses). https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
5. XGBoost GitHub Issue — `reg:absoluteerror` Hessian behavior and comparison with pseudo-Huber. https://github.com/dmlc/xgboost/issues/7674
6. XGBoost GitHub Issue — Hessian and `min_child_weight` intuition for logistic loss. https://github.com/dmlc/xgboost/issues/2483
7. Everdark (2019) — Demystify Modern Gradient Boosting Trees: Hessian role in split gain and leaf weight. https://everdark.github.io/k9/notebooks/ml/gradient_boosting/gbt.nb.html
