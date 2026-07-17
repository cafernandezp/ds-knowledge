# `min_child_weight` in XGBoost — Hessian Units vs. Instance Counts Across Objectives

> **Problem.** `min_child_weight` is denominated in **sum of Hessians** in a child node, not in number of rows. This is transparent for `reg:squarederror` (Hessian ≡ 1/instance) but opaque for `binary:logistic`/`reg:logistic`, `multi:softprob`, and log-link GLM objectives (`count:poisson`, `reg:gamma`, `reg:tweedie`), where the per-instance Hessian depends on the *current* prediction.
> **Assumptions.** Tree booster (`gbtree`), default `base_score` unless stated, unit instance weights unless stated (`scale_pos_weight`/`sample_weight` noted separately as a modifier). Split validity check: a split is accepted only if `sum(hessian)` in **both** children ≥ `min_child_weight` (source-level behavior, not just doc prose).
> **Scope.** Answers: (1) when to reason in raw instance count `n` vs. sum-of-Hessian `H`, and (2) approximately how many cases `n` correspond to a given Hessian budget under `binary:logistic`, as a function of the predicted probability `p`.

## TL;DR

- **`reg:squarederror`**: Hessian `h_i = 1` → `min_child_weight` **is** the minimum instance count. No translation needed.
- **`binary:logistic` / `reg:logistic`**: `h_i = p_i(1-p_i) ∈ (0, 0.25]`. The same `min_child_weight` corresponds to **anywhere from `4×min_child_weight` instances (at p=0.5) to unboundedly many** as predictions become confident (`p→0` or `p→1`).
- **Rule of thumb for logistic**: to keep the *same* effective minimum-instance behavior you'd get from squared error, start from **`4×`** the `min_child_weight` value you'd use for a regression task — then verify, don't assume.
- **`multi:softprob`** uses an upper-bound diagonal Hessian `h_i = 2·p_i(1-p_i) ∈ (0, 0.5]` (a documented XGBoost implementation detail, not the textbook softmax Hessian) — factor differs from binary logistic by 2×.
- **Log-link GLMs** (`count:poisson`, `reg:gamma`, `reg:tweedie`): Hessian scales with the **predicted mean**, is **unbounded above** and can be near-zero for low counts — there is no fixed multiplier; you must inspect it empirically.
- **Verify, don't derive-and-trust**: XGBoost exposes the actual sum-of-Hessian per node as `Cover` in `booster.trees_to_dataframe()`. Use it to confirm what `min_child_weight` is doing in practice, especially with `sample_weight`/`scale_pos_weight` in play (these multiply directly into the Hessian sum).

## What "per-instance Hessian" and "range" mean (read before the table)

Every row `i` contributes a **gradient** `g_i` (first derivative of the loss w.r.t. the model's raw prediction `F_i`) and a **Hessian** `h_i` (second derivative). XGBoost's split-finding uses both, and the leaf-validity check only looks at the Hessian side:

- **Per-instance Hessian `h_i`** — the curvature of the loss at row `i`'s current prediction. It is *not* a fixed constant per objective (except for squared error); it is a function of where that row currently sits (e.g., its predicted probability `p_i`). Intuitively, `h_i` measures how much *information/confidence* that single row contributes toward satisfying `min_child_weight` — a row with high `h_i` "counts for more" than a row with low `h_i`, even though both are exactly one instance.
- **Range of `h_i`** — the set of values `h_i` can mathematically take for that objective, taken over all possible predictions. This range is what determines the **conversion factor** between `min_child_weight` and raw instance count `n`. A tight, bounded range (e.g., `(0, 0.25]` for logistic) still lets you write `n ≥ min_child_weight / max(h_i)` as a best case. An unbounded or open-ended range (e.g., Poisson) means no such fixed conversion exists — the same `min_child_weight` can require anywhere from a handful to millions of rows depending on the data.

In short: the table below tells you, for each objective, **how much a single row is "worth"** toward the `min_child_weight` threshold, and therefore how to (or whether you can) translate `min_child_weight` back into "number of rows."

## Comparison table — Hessian behavior by objective

| Objective family | Per-instance Hessian `h_i` (curvature contributed by one row) | Range of `h_i` (values it can take) | What `min_child_weight` means in instance-count terms |
|---|---|---|---|
| `reg:squarederror` | `1` | `{1}` (constant) | `min_child_weight` = exact min. # instances per leaf |
| `binary:logistic` / `reg:logistic` | `p_i(1-p_i)` | `(0, 0.25]` | `n ≥ min_child_weight / 0.25` at best (p=0.5); `n → ∞` as `p→0,1` |
| `multi:softprob` | `2·p_i(1-p_i)` (diagonal upper bound used by XGBoost, not the exact softmax Hessian) | `(0, 0.5]` | `n ≥ min_child_weight / 0.5` at best; grows sharply for confident class probabilities |
| `count:poisson` | `exp(F_i + max_delta_step)` (= predicted mean rate, scaled) | `(0, ∞)`, unbounded | No fixed bound — scales with the magnitude of the target; low-count regions need far more rows per unit of `min_child_weight` |
| `reg:gamma` / `reg:tweedie` | mean-dependent (log-link curvature); not constant, not bounded | varies with target scale | Same caveat as Poisson — must be checked per dataset, not assumed |

## `reg:squarederror`

**What.** Loss `L_i = ½(y_i - F_i)²` → `g_i = F_i - y_i`, `h_i = 1` for every row (unit weights).

**Pros.**
- `min_child_weight` is literally "minimum rows per leaf" — this is the documented and web-verified statement: minimum sum of instance weight (hessian) needed in a child, which in a linear regression task simply corresponds to the minimum number of instances needed in each node.
- Zero translation cost; safe to tune directly against dataset size (e.g., `min_child_weight ∈ [1, 20]` for a few thousand rows).

**Risks.**
- None specific to this objective — but people carry this "`min_child_weight` = n" intuition into classification/count objectives, where it silently breaks.

**How / verify.**
```python
import xgboost as xgb

params = dict(objective="reg:squarederror", max_depth=5, eta=0.1, min_child_weight=10)
dtrain = xgb.DMatrix(X_train, label=y_train)
bst = xgb.train(params, dtrain, num_boost_round=200)

cover = bst.trees_to_dataframe().query("Feature == 'Leaf'")["Cover"]
print(cover.min())  # should be >= min_child_weight, and equals leaf row-count here
```

## `binary:logistic` / `reg:logistic`

**What.** `p_i = σ(F_i)`, `g_i = p_i - y_i`, `h_i = p_i(1-p_i)`. This is the standard logistic-loss curvature — confirmed directly in the XGBoost source pattern (`hess = preds * (1 - preds)`) and discussed by maintainers as the source of the `min_child_weight`/instance-count confusion.

**Pros.**
- Hessian weighting is *informative*: rows near the decision boundary (`p≈0.5`, high curvature) count more toward satisfying `min_child_weight` than confidently-classified rows (`p≈0` or `1`, low curvature). This concentrates splitting capacity where the model is still uncertain.

**Risks.**
- The exact same numeric `min_child_weight` is far **more permissive** early in training / near the boundary and far **more restrictive** in already-confident regions and later boosting rounds — so a single global `min_child_weight` value has a *shifting* meaning as `p` moves away from 0.5 across rounds.
- `scale_pos_weight` and per-row `sample_weight` multiply directly into `h_i` (the leaf check is `Σ w_i·h_i ≥ min_child_weight`), so raising `scale_pos_weight` for imbalance effectively lowers the number of raw positive rows needed to satisfy `min_child_weight` — an easy-to-miss interaction.

**How / verify.**
```python
def n_for_hessian(min_child_weight: float, p: float) -> float:
    """Approx. # instances (homogeneous node at probability p) to reach a Hessian budget."""
    h = p * (1 - p)
    return min_child_weight / h

for p in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001]:
    print(f"p={p:<6} h={p*(1-p):.4f}  n(min_child_weight=1)={n_for_hessian(1,p):7.1f}  n(min_child_weight=10)={n_for_hessian(10,p):8.1f}")
```

## `multi:softprob`

**What.** Gradient `g_{i,c} = p_{i,c} - 1[y_i=c]`. XGBoost does **not** use the exact softmax Hessian (which is a full cross-class matrix); it uses a **diagonal upper bound**, implemented as `h_{i,c} = max(2·p_{i,c}(1-p_{i,c})·w_i, eps)` — the factor of 2 is a deliberate implementation choice XGBoost maintainers confirmed is an upper-bound approximation for the diagonal, not the raw softmax second derivative.

**Pros.**
- Same qualitative behavior as binary logistic (boundary rows count more), extended per-class.

**Risks.**
- The 2× factor means a value of `min_child_weight` tuned by "binary-logistic intuition" is **not directly portable** to `multi:softprob` — the ceiling of `h` is `0.5` instead of `0.25`.
- With many classes, most `p_{i,c}` sit far from `0.5` (low Hessian) simply because probability mass is spread thin — `min_child_weight` can become effectively very restrictive as `num_class` grows, independent of any real overfitting signal.

**How / verify.**
```python
df = bst.trees_to_dataframe()  # multi-class booster: one set of trees per class per round
per_class_cover = df.query("Feature == 'Leaf'").groupby(df["Tree"] % num_class)["Cover"]
print(per_class_cover.describe())
```

## Log-link GLM objectives — `count:poisson`, `reg:gamma`, `reg:tweedie`

**What.** Log link `μ_i = exp(F_i)`. For Poisson, the source-level Hessian is `h_i = exp(F_i + max_delta_step)`, i.e. proportional to the **predicted mean count** (confirmed by maintainers discussing the `max_delta_step` scaling term in the Poisson objective). Gamma/Tweedie follow the same qualitative pattern: curvature tracks the mean, not a fixed constant.

**Pros.**
- None specific to `min_child_weight` — this is a structural property of log-link exponential-family losses, not a tunable design choice.

**Risks.**
- Hessian is **unbounded above** for high predicted counts and **near zero** for low ones — there is no universal multiplier analogous to the "×4" logistic heuristic. A fixed `min_child_weight` will implicitly demand very different row counts across the range of the target.
- Do **not** reuse a `min_child_weight` value calibrated on `reg:squarederror` or `binary:logistic` for these objectives without re-checking `Cover`.

**How / verify.**
```python
params = dict(objective="count:poisson", max_depth=5, min_child_weight=5)
bst = xgb.train(params, xgb.DMatrix(X_train, label=y_train), num_boost_round=200)
cover = bst.trees_to_dataframe().query("Feature == 'Leaf'")["Cover"]
cover.describe()  # inspect empirically; no closed-form n-vs-min_child_weight table exists here
```

## Problem-specific: when to think in `n` vs. in Hessian, and the approximate mapping

**When `n` (raw instance count) is the right mental model:**
- `reg:squarederror` (and any custom objective with `h_i ≡ const`).
- A rough *initial* sanity check for `binary:logistic` right at model start with `base_score = 0.5`, where every row has `h_i = 0.25` — here `n ≈ 4×min_child_weight` uniformly, because no row has separated yet.

**When sum-of-Hessian is the only correct model (don't think in `n`):**
- Any round/leaf where predictions have started to separate (`p` away from 0.5) under `binary:logistic`/`multi:softprob`.
- Any log-link GLM objective, at any point in training.
- Whenever `sample_weight` or `scale_pos_weight` ≠ 1, because the check is `Σ w_i h_i ≥ min_child_weight`, not `Σ h_i ≥ min_child_weight`.

**Approximate `n` for a given Hessian budget under `binary:logistic`** (homogeneous node at probability `p`, unit weights, `n ≈ min_child_weight / [p(1-p)]`):

| `p` (or `1-p`) | `h = p(1-p)` | `n` needed, `min_child_weight=1` | `n` needed, `min_child_weight=10` |
|---|---|---|---|
| 0.50 | 0.2500 | 4 | 40 |
| 0.30 / 0.70 | 0.2100 | 5 | 48 |
| 0.20 / 0.80 | 0.1600 | 6 | 63 |
| 0.10 / 0.90 | 0.0900 | 11 | 111 |
| 0.05 / 0.95 | 0.0475 | 21 | 211 |
| 0.02 / 0.98 | 0.0196 | 51 | 510 |
| 0.01 / 0.99 | 0.0099 | 101 | 1,010 |
| 0.001 / 0.999 | 0.0010 | 1,001 | 10,010 |

Read this as: **near the decision boundary, a handful of rows can satisfy `min_child_weight`; in a confidently-classified region, satisfying the same `min_child_weight` can require 100–1,000× more rows.** This is the concrete answer to "how many cases correspond to how much Hessian" for logistic loss.

## Diagnostics & pitfalls

- **Always confirm empirically.** `booster.trees_to_dataframe()["Cover"]` on leaf rows gives the *actual* achieved sum-of-Hessian per leaf for the fitted model — this is ground truth, the table above is only a homogeneous-node approximation.
- **`scale_pos_weight` / `sample_weight` leakage into `min_child_weight`.** Since the split check is on `Σ w_i·h_i`, reweighting for class imbalance silently changes how many *raw* rows are needed. Recompute effective `n ≈ min_child_weight / (mean(w)·mean(h))` before trusting an "instances" intuition.
- **Cross-library confusion.** `min_child_weight` in XGBoost is *not* the same concept as LightGBM's `min_child_samples` (a true row count); it maps to LightGBM's `min_sum_hessian_in_leaf`, which has a very different default scale (LightGBM defaults `min_sum_hessian_in_leaf=1e-3`, `min_data_in_leaf=20`) — porting a tuned value between the two libraries by name alone is a known source of surprising behavior.
- **Round-to-round drift.** As boosting proceeds and predictions become confident, the same `min_child_weight` becomes progressively more conservative in well-separated regions while staying permissive near the boundary — watch for later trees producing shallower splits away from the margin; compensate with `max_depth`/`gamma` if this under-fits confident regions.
- **`base_score` shifts the p=0.5 anchor.** If `base_score` is set to a class prior (common for imbalanced targets) rather than left at the default, initial-round Hessians are `p₀(1-p₀)`, not `0.25` — recompute the "early rounds" reference point accordingly.
- **Train/val/test.** `min_child_weight` is a regularization hyperparameter, not a feature engineering step — the only leakage risk is the generic one of tuning it (or any hyperparameter) by peeking at the test set; select it via CV/held-out validation only.

## Decision rule / quick guide

1. Objective has constant Hessian (`reg:squarederror` and similar) → tune `min_child_weight` directly as a row count.
2. Objective is `binary:logistic`/`reg:logistic` → start near `4×` your regression-scale `min_child_weight` intuition, then confirm with `Cover`; don't assume symmetry once predictions separate.
3. Objective is `multi:softprob` → start near `2×` the binary-logistic `min_child_weight` value (ceiling `0.5` vs `0.25`), and check `Cover` **per class**, since class count dilutes average confidence.
4. Objective is a log-link GLM (`count:poisson`/`reg:gamma`/`reg:tweedie`) → don't transfer a `min_child_weight` value from another objective; fit once, inspect the `Cover` distribution, then set it from an observed percentile.
5. Any `scale_pos_weight` ≠ 1 or custom `sample_weight` in use → recompute effective instance count before reasoning about `min_child_weight` in "rows" terms.
6. Tune `min_child_weight` jointly with `max_depth`/`gamma` via CV — they trade off against the same conservatism budget; don't grid-search `min_child_weight` in isolation.

## References

1. XGBoost Parameters documentation — `min_child_weight` definition and default. https://xgboost.readthedocs.io/en/stable/parameter.html
2. dmlc/xgboost GitHub Issue #2483 — "Intuition behind hessian and min_child_weight." https://github.com/dmlc/xgboost/issues/2483
3. dmlc/xgboost GitHub Issue #5987 — mapping `min_child_weight` (XGBoost) to `min_sum_hessian_in_leaf`/`min_data_in_leaf` (LightGBM). https://github.com/dmlc/xgboost/issues/5987
4. dmlc/xgboost GitHub Issue #5769 — "Hessians of `multi:softmax` and `multi:softprob` appear to be incorrect" (confirms the 2× diagonal-upper-bound factor). https://github.com/dmlc/xgboost/issues/5769
5. dmlc/xgboost GitHub Issue #10258 — "Is the hessian right in `multi:softprob`?" (independent re-derivation matching #5769). https://github.com/dmlc/xgboost/issues/10258
6. dmlc/xgboost GitHub Issue #2661 — "`exp(p+max_delta_step)` in poisson regression hessian: what is the purpose?" (confirms Poisson Hessian = scaled predicted mean). https://github.com/dmlc/xgboost/issues/2661
7. XGBoost Custom Objective and Evaluation Metric tutorial — diagonal-Hessian approximation for `multi:softprob`. https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
8. `xgb.model.dt.tree` (R package docs) — definition of `Cover` as the Hessian/observation metric collected per node/leaf. https://rdrr.io/cran/xgboost/man/xgb.model.dt.tree.html
9. dmlc/xgboost test suite — `trees_to_dataframe()["Cover"]` verified equal to the sum of per-node Hessian from the raw model dump. https://github.com/dmlc/xgboost/blob/master/tests/python/test_parse_tree.py
