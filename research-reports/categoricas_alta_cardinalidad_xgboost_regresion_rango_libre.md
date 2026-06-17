# High-cardinality categorical variables — strategies for XGBoost (regression, arbitrary-range continuous target)

> **Scope:** this report is scoped to **regression with a continuous target of arbitrary range** (unbounded, possibly skewed, with outliers or negatives) using `XGBRegressor`. The encoding strategies are target-agnostic; what changes vs the bounded `[0,1]` case is the `objective`, the optional **target transform**, the range of the target-encoded feature, the metrics, and the (non-)applicability of WoE. For target ∈ [0,1] or binary targets see the companion reports.

> **Problem context:** XGBoost regressor, continuous `y` over an arbitrary range. High-cardinality categoricals (ZIP, product ID, city, SKU…), tens to thousands of levels. Goal: encode them without exploding dimensionality, without leakage, with sane handling of rare/unseen levels.

---

## TL;DR — recommendation

1. **Default:** XGBoost native categorical support (`enable_categorical=True`). No manual encoding, no leakage, optimal partitioning.
2. **Strong level ↔ y signal:** *target/mean encoding* with **cross-fitting** + smoothing (`sklearn.TargetEncoder(target_type="continuous")`). Encoded values land in the **target's own range** (not [0,1]).
3. **Cheap, leakage-free complement:** *frequency/count encoding*.
4. **Extreme cardinality / streaming:** *hashing trick*. **Always:** collapse rare levels into `"Other"`.
5. **Skew/outliers:** fix them via the **objective** (`reg:absoluteerror`, `reg:pseudohubererror`) or a **target transform**, not via the encoding.

> WoE does **not** apply here (it is defined for binary targets) — see §6. Avoid one-hot on high-cardinality columns.

---

## Comparison table

| Strategy | Dim. | Leakage risk | Unseen levels | Cost | XGBoost fit | When |
|---|---|---|---|---|---|---|
| **Native XGBoost** | 1 col (internal) | None | Native | Low | ★★★★★ | Default |
| **Target/Mean enc.** | 1 col | **High** (needs CV) | → global mean | Med | ★★★★☆ | Level ↔ y strong |
| **Frequency/Count** | 1 col | Low | → 0 / default | Low | ★★★★☆ | Popularity is signal |
| **Ordinal/Label** | 1 col | None | → −1 | Low | ★★★☆☆ | Real order / native base |
| **Hashing** | k cols | None | Natural | Low | ★★★☆☆ | Extreme cardinality / online |
| **WoE** | 1 col | — | — | — | ☆☆☆☆☆ | **N/A** (binary only) |
| **Rare grouping** | — (pre) | None | → `"Other"` | Low | ★★★★★ | Universal pre-step |
| **Entity embeddings** | d cols | Med | → OOV vector | **High** (NN) | ★★☆☆☆ | Lots of data + complex signal |

---

## 1) Native XGBoost support  *(recommended default)*

**What.** XGBoost ≥ 1.5 (current 3.x) handles pandas `category` columns directly via *optimal partitioning* under `tree_method="hist"`: per-category gradient histogram → sort by gradient statistic → evaluate contiguous partitions (Fisher) → `value ∈ {set}` splits. Tree splits are invariant to monotonic target transforms, so the encoding behaves identically whatever the target range.

**Pros.**
- Zero leakage; partitions learned inside tree training.
- Scales to high cardinality far better than one-hot; unaffected by target scale.
- `max_cat_to_onehot` (default 4) and `max_cat_threshold` (default 64) tune one-hot-vs-partition and limit overfit.

**Risks.**
- Near-unique levels can overfit; mitigate with `max_cat_threshold` and rare grouping.
- Requires `tree_method="hist"` and `category` dtype at inference too.
- Less control/interpretability than an explicit scheme.

**How / verify.**
```python
import pandas as pd
from xgboost import XGBRegressor

for c in cat_cols:
    df[c] = df[c].astype("category")

model = XGBRegressor(
    enable_categorical=True, tree_method="hist",
    objective="reg:squarederror",     # see §"specific considerations" for alternatives
    max_cat_to_onehot=4, max_cat_threshold=64,
    n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42,
)
model.fit(X, y)   # no clipping; predictions live on the target's natural range
```

---

## 2) Target / Mean encoding  *(value lands in the target's range)*

**What.** Replaces each level by the cross-fitted, smoothed mean of `y` for that level. With an arbitrary-range target, the encoded column inherits **the target's units and range** (not [0,1]) — fine for trees, but be aware of it.

**Pros.**
- One column captures the level ↔ `y` signal directly; ideal for high cardinality.
- Bayesian smoothing toward the global mean stabilizes rare levels.
- Trees don't need the feature scaled, so the inherited range is harmless.

**Risks.**
- **Leakage / catastrophic overfit** without out-of-fold encoding; near-unique levels memorize `y`.
- A **heavy-tailed target** makes per-level means unstable and outlier-sensitive; consider encoding on a transformed target or using a robust objective (§specific).
- Can dominate impurity-based importances → use permutation importance.

**How / verify.**
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor

pre = ColumnTransformer(
    [("te", TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=42), cat_cols)],
    remainder="passthrough",
)
pipe = Pipeline([("pre", pre),
                 ("xgb", XGBRegressor(objective="reg:squarederror", n_estimators=500,
                                      learning_rate=0.05, max_depth=6, random_state=42))])
# cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")  -> no leakage
pipe.fit(X_train, y_train)
```

---

## 3) Frequency / Count encoding

**What.** Replace level by its train frequency/count.

**Pros.** No leakage, cheap, monotonic-friendly; useful when popularity is signal; orthogonal complement to target encoding.

**Risks.** Equal-frequency levels collide; weak if frequency is unrelated to `y`; frequency drift between train and inference.

**How / verify.**
```python
def fit_frequency(s):            return s.value_counts(normalize=True)
def apply_frequency(s, f, d=0.): return s.map(f).fillna(d)
freq = fit_frequency(X_train[col])
X_train[col + "_freq"] = apply_frequency(X_train[col], freq)
X_test[col + "_freq"]  = apply_frequency(X_test[col],  freq)
```

---

## 4) Ordinal / Label encoding

**What.** One integer per level.

**Pros.** Trivial, cheap, 1 column; correct when a real order exists; good base for native handling.

**Risks.** Artificial order when none exists (native handling is strictly better at high cardinality); needs explicit unseen-level policy.

**How / verify.**
```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
X_test[cat_cols]  = enc.transform(X_test[cat_cols])
```

---

## 5) Hashing trick

**What.** Hash levels into `k` fixed buckets.

**Pros.** Fixed memory regardless of cardinality; handles new levels without retraining; no leakage.

**Risks.** Collisions lose information; not interpretable; `k` trade-off.

**How / verify.**
```python
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
def hash_column(s, n=16):
    M = FeatureHasher(n_features=n, input_type="string").transform(s.astype(str).map(lambda v: [v]))
    return pd.DataFrame(M.toarray(), columns=[f"{s.name}_h{i}" for i in range(n)], index=s.index)
```

---

## 6) Weight of Evidence (WoE)  *(does it apply here? — critique)*

**What.** WoE replaces each level by `ln(%events / %non-events)`; the standard in credit scoring, paired with Information Value (IV).

**Critique for arbitrary-range regression.**
- WoE is **defined for a binary target** (it needs event vs non-event counts). With a general continuous target it **does not apply**.
- Binarizing `y` at a threshold to force WoE is **arbitrary and discards information** — don't, unless the threshold has a concrete business meaning (and then it's a different, classification problem).
- Unlike the `[0,1]` case (where `Σy` can be read as expected events → a pseudo-WoE ≈ logit of the per-level mean), an **unbounded** target has no such interpretation. The right continuous analog is **target encoding** (§2).
- Verdict: **use target encoding**; reserve WoE/IV for a genuinely binary version of the problem (see the binary-classification report).

**Pros.** (only when target is binary) interpretable, monotonic in log-odds, IV screens features.

**Risks.** Not applicable to continuous regression without lossy binarization; same leakage as target encoding; `ln(0)` needs regularization.

---

## 7) Rare-level grouping  *(universal pre-step)*

**What.** Collapse infrequent levels into `"Other"` by frequency threshold.

**Pros.** Cuts noise/variance on rare levels; natural destination for unseen levels; combines with any encoder; cheap.

**Risks.** May merge heterogeneous levels; threshold is a hyperparameter.

**How / verify.**
```python
def fit_rare(s, min_freq=0.01):
    p = s.value_counts(normalize=True); return set(p[p >= min_freq].index)
def apply_rare(s, keep, other="__OTHER__"): return s.where(s.isin(keep), other)
```

---

## 8) Entity embeddings  *(high cost)*

**What.** Learn dense per-level vectors with a neural net, then feed them to XGBoost.

**Pros.** Captures complex similarity between levels; compresses huge cardinality.

**Risks.** Needs a NN, lots of data, heavier pipeline; leakage if trained with `y` outside CV; low interpretability.

---

## Arbitrary-range regression specific considerations

- **Objective — match the target's shape:**
  - `reg:squarederror` — default; sensitive to outliers.
  - `reg:absoluteerror` — L1, robust to outliers, targets the median *(added 1.7)*.
  - `reg:pseudohubererror` — Huber, robust and differentiable.
  - `reg:squaredlogerror` — positive targets, penalizes under-prediction; reduces skew effect.
  - `reg:gamma`, `reg:tweedie` — positive / zero-inflated targets (costs, insurance, demand).
  - `reg:quantileerror` — pinball loss for quantile/prediction intervals *(added 2.0; set `quantile_alpha`)*.
- **Target transform (alternative to changing objective):** `log1p`, **Box-Cox** (positive only), **Yeo-Johnson** (any sign) reduce skew/stabilize variance. Apply via `TransformedTargetRegressor` so the inverse is handled automatically and metrics are reported on the original scale.
- **Feature scaling not needed** for trees (splits are invariant to monotonic transforms) — only the target's distribution matters here.
- **Metrics:** RMSE / MAE / R²; **RMSLE** or **MAPE** for positive multiplicative targets (avoid MAPE with zeros); MAE/quantile loss when outliers dominate. **No clipping** of predictions.

```python
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

# Right-skewed positive target: model log(y), report on original scale
ttr = TransformedTargetRegressor(
    regressor=XGBRegressor(objective="reg:squarederror", tree_method="hist",
                           enable_categorical=True, n_estimators=500,
                           learning_rate=0.05, max_depth=6, random_state=42),
    func=np.log1p, inverse_func=np.expm1,
)
ttr.fit(X, y)
```

---

## Diagnostics & pitfalls

- **Leakage (#1):** any `y`-derived encoder (target) must be learned on the train fold only; wrap in a `Pipeline`/CV; fit any target transform inside CV as well.
- **Unseen levels:** define the destination (global mean, `0`, `"Other"`, native). Verify no `NaN` post-encoding.
- **Rare levels / heavy tails:** unstable means → smoothing/grouping and/or a robust objective.
- **One-hot at high cardinality:** avoid with XGBoost.
- **Importance bias:** prefer **permutation importance** on validation.
- **IDs ≈ #rows:** likely identifiers; consider dropping unless CV shows real signal.
- **Reproducibility:** fix `random_state` everywhere; single `Pipeline`.

---

## Pipeline (leakage-safe template)

```python
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def to_frequency(d): return d.apply(lambda s: s.map(s.value_counts(normalize=True)).fillna(0.0))

pre = ColumnTransformer([
    ("te",   TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=42), te_cols),
    ("freq", FunctionTransformer(to_frequency), freq_cols),
    ("num",  "passthrough", num_cols),
], remainder="drop")

base = Pipeline([("pre", pre),
                 ("xgb", XGBRegressor(objective="reg:pseudohubererror", tree_method="hist",
                                      n_estimators=600, learning_rate=0.05, max_depth=6,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42))])
model = TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)  # if positive-skewed
print(-cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error").mean())
```

---

## Quick decision rule

1. Minimal friction / good baseline → **native XGBoost**.
2. Strong level ↔ y signal → **target encoding (continuous) with CV + smoothing**.
3. Popularity is signal → add **frequency encoding**.
4. Extreme cardinality / new levels → **hashing**.
5. Skewed/outlier-prone target → fix via **objective** or **target transform**, not the encoding.
6. Binary target instead? → use WoE/IV (see the binary report).
7. Always: **group rare levels** and evaluate one **Pipeline** under CV.

---

## References

1. XGBoost — *Categorical Data* (`enable_categorical`, optimal partitioning / Fisher, `value ∈ {set}` splits). https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
2. XGBoost — *Parameters* (regression objectives `reg:squarederror`, `reg:absoluteerror` [1.7], `reg:pseudohubererror`, `reg:squaredlogerror`, `reg:gamma`, `reg:tweedie`, `reg:quantileerror` [2.0]; `tree_method="hist"`, `max_cat_to_onehot`, `max_cat_threshold`). https://xgboost.readthedocs.io/en/stable/parameter.html
3. scikit-learn — `TargetEncoder` (`target_type="continuous"`, internal cross-fitting, `smooth`, `cv`). https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
4. scikit-learn — `TransformedTargetRegressor` (target transform with automatic inverse). https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
5. Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems.* SIGKDD Explorations 3(1), 27–32.
6. Kuhn, M. & Johnson, K. *Feature Engineering and Selection: A Practical Approach for Predictive Models.* https://feat.engineering/ (numeric/target transforms, categorical encoding).
7. scikit-learn — `FeatureHasher` (hashing trick). https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
