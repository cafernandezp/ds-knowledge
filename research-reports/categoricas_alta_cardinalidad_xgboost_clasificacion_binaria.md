# High-cardinality categorical variables — strategies for XGBoost (binary classification)

> **Scope:** this report is scoped to **binary classification** with XGBoost (`XGBClassifier`, target ∈ {0, 1}). The encoding strategies themselves are target-agnostic, but the `objective`, `target_type`, metrics, calibration, and the role of **WoE/IV** are specific to the binary case. For regression targets see the companion reports.

> **Problem context:** XGBoost binary classifier, target ∈ {0, 1}, possibly imbalanced. High-cardinality categoricals (ZIP, product ID, city, SKU…), tens to thousands of levels. Goal: encode them without exploding dimensionality, without leakage, with sane handling of rare/unseen levels.

---

## TL;DR — recommendation

1. **Default:** XGBoost native categorical support (`enable_categorical=True`). No manual encoding, no leakage, optimal partitioning.
2. **Strong level ↔ y signal:** *target/mean encoding* with **cross-fitting** + smoothing (`sklearn.TargetEncoder(target_type="binary")`). Encodes shrunk `P(y=1|level)` ∈ [0,1].
3. **Interpretable, finance-style:** **WoE + Information Value (IV)** — now a first-class option (it is *defined* for binary targets). Monotonic in log-odds; IV doubles as a feature screen.
4. **Cheap, leakage-free complement:** *frequency/count encoding*.
5. **Extreme cardinality / streaming:** *hashing trick*. **Always:** collapse rare levels into `"Other"`.

> Avoid one-hot on high-cardinality columns with XGBoost (dimensionality, sparsity, slow `hist`).

---

## Comparison table

| Strategy | Dim. | Leakage risk | Unseen levels | Cost | XGBoost fit | When |
|---|---|---|---|---|---|---|
| **Native XGBoost** | 1 col (internal) | None | Native | Low | ★★★★★ | Default |
| **Target/Mean enc.** | 1 col | **High** (needs CV) | → global mean | Med | ★★★★☆ | Level ↔ y strong |
| **WoE / IV** | 1 col | **High** (needs CV/reg) | → WoE 0 | Med | ★★★★☆ | Binary only; interpretability |
| **Frequency/Count** | 1 col | Low | → 0 / default | Low | ★★★★☆ | Popularity is signal |
| **Ordinal/Label** | 1 col | None | → −1 | Low | ★★★☆☆ | Real order / native base |
| **Hashing** | k cols | None | Natural | Low | ★★★☆☆ | Extreme cardinality / online |
| **Rare grouping** | — (pre) | None | → `"Other"` | Low | ★★★★★ | Universal pre-step |
| **Entity embeddings** | d cols | Med | → OOV vector | **High** (NN) | ★★☆☆☆ | Lots of data + complex signal |

---

## 1) Native XGBoost support  *(recommended default)*

**What.** XGBoost ≥ 1.5 (current 3.x) handles pandas `category` columns directly via *optimal partitioning* under `tree_method="hist"`: it builds the per-category gradient histogram, sorts categories by their gradient statistic, and evaluates only contiguous partitions (Fisher), producing `value ∈ {set}` splits instead of one-hot.

**Pros.**
- Zero leakage; partitions learned inside tree training.
- Scales to high cardinality far better than one-hot.
- `max_cat_to_onehot` (default 4) and `max_cat_threshold` (default 64) tune one-hot-vs-partition and limit overfit.

**Risks.**
- Near-unique levels can overfit; mitigate with `max_cat_threshold` and rare grouping.
- Requires `tree_method="hist"` and `category` dtype at inference too.
- Less control/interpretability than an explicit scheme.

**How / verify.**
```python
import pandas as pd
from xgboost import XGBClassifier

for c in cat_cols:
    df[c] = df[c].astype("category")

model = XGBClassifier(
    enable_categorical=True, tree_method="hist",
    objective="binary:logistic", eval_metric="aucpr",
    max_cat_to_onehot=4, max_cat_threshold=64,
    n_estimators=500, learning_rate=0.05, max_depth=6,
    scale_pos_weight=(neg / pos),     # set for class imbalance; 1.0 if balanced
    random_state=42,
)
model.fit(X, y)   # X must keep 'category' dtype at predict time too
```

---

## 2) Target / Mean encoding  *(binary: shrunk event rate)*

**What.** Replaces each level by the cross-fitted, smoothed `P(y=1 | level)` ∈ [0,1].

**Pros.**
- One column captures the level ↔ event-rate signal directly; ideal for high cardinality.
- `target_type="binary"` makes the intent explicit; output bounded in [0,1].
- Bayesian smoothing stabilizes rare levels.

**Risks.**
- **Leakage / catastrophic overfit** without out-of-fold encoding; near-unique levels memorize `y`.
- Unstable for low-count levels → rely on smoothing.
- Can dominate impurity-based importances and miscalibrate probabilities → calibrate.

**How / verify.**
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

pre = ColumnTransformer(
    [("te", TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=42), cat_cols)],
    remainder="passthrough",
)
pipe = Pipeline([("pre", pre),
                 ("xgb", XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                                       n_estimators=500, learning_rate=0.05, max_depth=6,
                                       random_state=42))])
# cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")  -> no leakage
pipe.fit(X_train, y_train)
```

---

## 3) Weight of Evidence (WoE) + Information Value (IV)  *(native to binary targets)*

**What.** For each level (or bin): `WoE = ln(%events / %non-events)`. The companion **IV = Σ (%events − %non-events) · WoE** quantifies the feature's overall predictive power. This is the credit-scoring standard and is **defined precisely for a binary target** — unlike the regression case, here it is fully applicable.

**Pros.**
- Monotonic in the log-odds; very interpretable; aligns with logistic models.
- **IV gives a built-in feature screen** (rough guide: <0.02 useless, 0.1–0.3 medium, >0.5 strong, and suspiciously high IV may signal leakage).
- Handles high cardinality well with supervised binning.

**Risks.**
- **Binary only** (its strength here, its limit elsewhere).
- Same **leakage** as target encoding → use CV / built-in regularization + noise.
- `ln(0)` undefined for levels with 0 events or 0 non-events → needs Laplace/regularization.

**How / verify.** `category_encoders.WOEEncoder` (supported target: binomial; `regularization` prevents div-by-zero, `randomized`/`sigma` add training-only noise to curb overfit).
```python
from category_encoders import WOEEncoder   # pip install category_encoders
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

woe = WOEEncoder(cols=cat_cols, regularization=1.0, randomized=True, sigma=0.05, random_state=42)
pipe = Pipeline([("woe", woe),
                 ("xgb", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                       n_estimators=500, learning_rate=0.05, max_depth=6,
                                       random_state=42))])
pipe.fit(X_train, y_train)   # fit WoE on TRAIN only; evaluate inside CV to avoid leakage

# Quick IV screen (manual), per level with Laplace alpha:
import numpy as np, pandas as pd
def information_value(s, y, alpha=0.5):
    pos, neg = y.sum(), (1 - y).sum()
    g = pd.DataFrame({"x": s.values, "y": y.values}).groupby("x")["y"].agg(["sum", "count"])
    pe = (g["sum"] + alpha) / (pos + alpha)
    pn = (g["count"] - g["sum"] + alpha) / (neg + alpha)
    woe = np.log(pe / pn)
    return float(((pe - pn) * woe).sum())
```

---

## 4) Frequency / Count encoding

**What.** Replace level by its train frequency/count.

**Pros.** No leakage, cheap, monotonic-friendly; useful when popularity is signal; good complement to target/WoE (orthogonal axis).

**Risks.** Distinct levels with equal frequency collide; weak if frequency is unrelated to `y`; sensitive to train↔inference frequency drift.

**How / verify.**
```python
def fit_frequency(s):            return s.value_counts(normalize=True)
def apply_frequency(s, f, d=0.): return s.map(f).fillna(d)
freq = fit_frequency(X_train[col])
X_train[col + "_freq"] = apply_frequency(X_train[col], freq)
X_test[col + "_freq"]  = apply_frequency(X_test[col],  freq)
```

---

## 5) Ordinal / Label encoding

**What.** One integer per level.

**Pros.** Trivial, cheap, 1 column; correct when a real order exists; good base for native handling.

**Risks.** Imposes artificial order when none exists (native handling is strictly better at high cardinality); needs explicit unseen-level policy.

**How / verify.**
```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
X_test[cat_cols]  = enc.transform(X_test[cat_cols])
```

---

## 6) Hashing trick

**What.** Hash levels into `k` fixed buckets.

**Pros.** Fixed memory regardless of cardinality; handles new levels without retraining (online); no leakage.

**Risks.** Collisions lose information; not interpretable; `k` is a trade-off (few = more collisions, many = more dimensionality).

**How / verify.**
```python
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
def hash_column(s, n=16):
    M = FeatureHasher(n_features=n, input_type="string").transform(s.astype(str).map(lambda v: [v]))
    return pd.DataFrame(M.toarray(), columns=[f"{s.name}_h{i}" for i in range(n)], index=s.index)
```

---

## 7) Rare-level grouping  *(universal pre-step)*

**What.** Collapse infrequent levels into `"Other"` by frequency threshold.

**Pros.** Cuts noise/variance on rare levels; natural destination for unseen levels; combines with any encoder; cheap.

**Risks.** May merge heterogeneous levels; threshold is a hyperparameter (too high erases informative minorities).

**How / verify.**
```python
def fit_rare(s, min_freq=0.01):
    p = s.value_counts(normalize=True); return set(p[p >= min_freq].index)
def apply_rare(s, keep, other="__OTHER__"): return s.where(s.isin(keep), other)
```

---

## 8) Entity embeddings  *(high cost)*

**What.** Learn dense per-level vectors with a neural net, then feed them to XGBoost.

**Pros.** Captures complex similarity between levels; compresses huge cardinality into few dense dims.

**Risks.** Needs a NN, lots of data, heavier pipeline; leakage if trained with `y` outside CV; low interpretability.

---

## Binary-classification specific considerations

- **Objective:** `binary:logistic` (probabilities) or `binary:logitraw`/`binary:hinge` for margins. `multi:softprob` if it later becomes multiclass.
- **Metrics:** ROC-AUC for ranking; **PR-AUC (`aucpr`)** under heavy imbalance; `logloss` when calibrated probabilities matter.
- **Class imbalance:** set `scale_pos_weight ≈ n_neg / n_pos`; consider PR-AUC and threshold tuning over default 0.5.
- **Calibration:** target-encoded/WoE leakage and imbalance distort probabilities. Wrap with `CalibratedClassifierCV` (isotonic or sigmoid) on a held-out fold; check a reliability curve.
- **Threshold selection:** pick the operating point from the business cost matrix or by maximizing F1 / Youden's J on validation, not a fixed 0.5.

---

## Diagnostics & pitfalls

- **Leakage (#1):** any `y`-derived encoder (target, WoE) must be learned on the train fold only; wrap in a `Pipeline`/CV.
- **Unseen levels:** define the destination (global rate, WoE 0, `"Other"`, native). Verify no `NaN` post-encoding.
- **Rare levels:** unstable rates → smoothing/grouping.
- **One-hot at high cardinality:** avoid with XGBoost.
- **Importance bias:** use **permutation importance** on validation rather than impurity importances.
- **IDs ≈ #rows:** likely identifiers; consider dropping unless CV shows real signal.
- **Reproducibility:** fix `random_state` everywhere; single `Pipeline`.

---

## Pipeline (leakage-safe template)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import cross_val_score
from category_encoders import WOEEncoder
from xgboost import XGBClassifier

pre = ColumnTransformer([
    ("te",   TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=42), te_cols),
    ("woe",  WOEEncoder(regularization=1.0, randomized=True, sigma=0.05, random_state=42), woe_cols),
    ("num",  "passthrough", num_cols),
], remainder="drop")

pipe = Pipeline([("pre", pre),
                 ("xgb", XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                                       tree_method="hist", n_estimators=600, learning_rate=0.05,
                                       max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                       scale_pos_weight=(neg / pos), random_state=42))])
print(cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean())
```

---

## Quick decision rule

1. Minimal friction / good baseline → **native XGBoost**.
2. Strong level ↔ event-rate signal → **target encoding (binary) with CV + smoothing**.
3. Need interpretability / feature screen → **WoE + IV**.
4. Popularity is signal → add **frequency encoding**.
5. Extreme cardinality / new levels → **hashing**.
6. Always: **group rare levels** and evaluate one **Pipeline** under CV; **calibrate** if probabilities matter.

---

## References

1. XGBoost — *Categorical Data* (`enable_categorical`, optimal partitioning / Fisher, `value ∈ {set}` splits). https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
2. XGBoost — *Parameters* (`tree_method="hist"`, `max_cat_to_onehot` default 4, `max_cat_threshold` default 64, `objective="binary:logistic"`, `scale_pos_weight`, `eval_metric`). https://xgboost.readthedocs.io/en/stable/parameter.html
3. scikit-learn — `TargetEncoder` (`target_type="binary"`, internal cross-fitting, `smooth`, `cv`). https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
4. category_encoders — `WOEEncoder` (supported target: binomial; `regularization`, `randomized`, `sigma`). https://contrib.scikit-learn.org/category_encoders/woe.html
5. Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems.* SIGKDD Explorations 3(1), 27–32.
6. scikit-learn — `CalibratedClassifierCV` (probability calibration). https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
7. scikit-learn — `FeatureHasher` (hashing trick). https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
