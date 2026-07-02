# Uniform Residuals for a Bounded XGBoost Regressor — Segment-Structured Target

> **Problem.** XGBoost regressor predicting `progress_score ∈ [0,1]` (advancement of a judicial procedure over the next ~18 months). One row = one procedure. Primary metric = **MAE**. Objective = `reg:logistic`. Train ≈ 6,000 rows; ~1,000–2,000 per procedure type.
> **Observed.** (1) Per-*true*-decile plot: prediction curve much flatter than the real curve (range ≈ 0.34–0.64 vs 0.08–0.95). (2) **Marginal distribution plot confirms severe under-dispersion even in TRAIN** — predicted range ≈ [0.25, 0.80] vs real [0.0, 1.0]; the predicted mode sits at ~0.55 ≈ global mean of `y`. (3) The target is **quasi-discrete / spike-structured**: visible mass concentrations at ~0.0, ~0.13, ~0.33, ~0.50, ~0.90 consistent with a fixed number of judicial stages. (4) Target shape changes by procedure type (EJECUTIVO / HIPOTECARIO / MONITORIO / ORDINARIO; KS up to 0.21 also across `person_type`). (5) OOF residuals are multimodal and segment-structured, mirroring the per-segment target shape.
> **Goal.** Make the avg-real vs avg-pred curves more *parallel* → a more *uniform* error distribution across deciles/segments, without giving up much MAE.
> **Assumptions.** CV is segment-stratified or close to it; the decile plot bins by the *true* target; calibration data can be produced OOF; per-segment n (~1k–2k) is enough to fit a 1-D calibrator. Quasi-discrete structure in target = fraction of completed judicial stages (k/N). Stated and proceeding.

---

## TL;DR / Recommendation (ranked)

1. **(C, now #1) Relax regularization before doing anything else.** The marginal histogram confirms the model is *under-fitting* on train itself — the predicted range is hard-capped at [0.25, 0.80] and the spike at ~0.55 = `base_score` collapse. This is driven by `min_child_weight ∈ [60,500]` being far too aggressive for 6k rows. Lower to 5–40, allow `max_depth` 5–7, relax `gamma`/`reg_lambda`. Re-plot the marginal on train: if tails fill in, the compression was pure over-regularization and everything else gets cheaper.
2. **(H, new) Reconsider the target framing.** The quasi-discrete spikes (~0, ~1/7, ~2/7…) indicate the target is probably `stages_completed / N_stages`, not a generic continuous score. A continuous regressor will always produce blurry predictions between spikes and bimodal residuals — not a tuning problem, a representation problem. Evaluate zero-inflated, ordinal, or two-part approaches before investing more in calibration.
3. **(B) Per-segment post-hoc calibration — necessary but not sufficient.** A single global calibrator is ruled out by the segment-structured residuals. Per-segment isotonic works for level bias. **Quantile mapping on a collapsed [0.25,0.80] distribution is cosmetic** — it stretches a low-resolution range; do C first so it has proper spread to map.
4. **(A) Re-read the diagnostic.** The true-target-decile plot has a built-in regression-to-the-mean artifact. Re-plot by predicted decile + per segment (reliability curve) to judge calibration correctly.
5. **(D) Don't switch selection to R² alone.** R² ≡ RMSE-ranking; does not target "uniform error." Keep MAE primary; add a calibration-aware secondary (per-segment bias + `std(pred)/std(y)` ratio) as Optuna constraint.
6. **(F) Distributional if multimodality persists.** If OOF residuals stay bimodal within a segment after C+B, go quantile regression (`reg:quantileerror`) or two-stage. Justified after simpler steps fail, not before.

---

## Decision-driving comparison

| # | Option | What it targets | Cost | MAE impact | Fixes segment-structured residuals? | Use when |
|---|---|---|---|---|---|---|
| A | Diagnostic reframing (predicted-bin + per-segment) | Measurement validity | ~0 | none | Reveals the true gap | Always, first |
| B | **Per-segment calibration** (isotonic / quantile-map) | Parallel curves, per-segment bias | Low | ~neutral | Yes (bias/level) | After C; quantile-map only if range is already wide |
| **C** | **Relax regularization + retune** | **Native de-compression (primary fix)** | Medium | neutral/↑ | **Partially — directly** | **Always. Before B.** |
| D | Calibration-aware selection (composite, not R² alone) | Align HPO with the real goal | Low | controlled | Indirectly | When HPO keeps selecting shrunk models |
| E | Per-segment models | Different target shapes per type | Medium | neutral/↑ | Yes (level + shape) | Big shape gaps, enough n/segment |
| F | Distributional / two-stage (quantile, mixture) | Within-segment multimodality | High | varies | Yes (shape) | Bimodal residuals remain after B–E |
| G | Sample weighting / `logit` target | Widen range, balance deciles | Low | ↑ risk | Partially | Tails still under-fit after C |
| **H** | **Target reframing (ordinal / two-part / stage model)** | **Structural representation of quasi-discrete target** | **Medium–High** | **↑ if design is right** | **Yes (irreducible part)** | **Spikes survive after C+B** |

---

## A. Diagnostic reframing — bin by *predicted*, split by segment

**What.** Stop diagnosing calibration on *true*-target deciles. Bin by the **predicted** value (reliability curve) and compute curves/residuals **per procedure type**.

**Pros.**
- Removes the regression-to-the-mean artifact: under predicted-bins a calibrated model has `E[y | pred∈bin] ≈ pred`, so parallelism is the *correct* success criterion.
- Per-segment curves expose where the real bias lives (level shift vs shape mismatch) and tell you whether B, E, or F is needed.
- Zero modeling cost; prevents over-correcting a non-problem.

**Risks.**
- None methodological. Don't keep reporting the true-decile plot as the KPI.

**How / verify.**
```python
import numpy as np, pandas as pd

def reliability_by_segment(y, pred, seg, n_bins=10):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pred), "seg": np.asarray(seg)})
    out = []
    for s, g in df.groupby("seg"):
        q = np.quantile(g["p"], np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.digitize(g["p"], q[1:-1]), 0, n_bins - 1)
        t = g.groupby(b).agg(avg_pred=("p", "mean"),
                             avg_real=("y", "mean"),
                             n=("y", "size"))
        t["bias"] = t["avg_real"] - t["avg_pred"]   # ~0 per bin ⇒ calibrated
        t["seg"] = s
        out.append(t)
    return pd.concat(out)

# rep = reliability_by_segment(y_oof, pred_oof, df["procedure_type"])
# Calibrated ⇒ avg_real ≈ avg_pred in every bin AND every segment.
```

---

## B. Per-segment post-hoc calibration  *(recommended — but after C)*

**What.** Fit a 1-D monotone map from raw prediction to outcome **separately per procedure type**, on **OOF / held-out** predictions. Two flavours: **isotonic** (corrects level/monotone bias) and **quantile mapping** (forces `dist(pred)=dist(real)` → deciles overlap by construction).

**Pros.**
- Directly attacks per-segment level bias — the exact symptom in the OOF residual plots.
- Cheap, transparent, monotone (preserves ranking), keeps output in `[0,1]`.
- Quantile mapping makes the avg-real/avg-pred curves overlap almost by definition.

**Risks.**
- **Leakage** if fit on the same rows the model trained on → must use OOF or a dedicated calibration split.
- A global calibrator will *not* work here (bias differs by segment).
- **Quantile mapping on a collapsed [0.25, 0.80] range is cosmetic.** It forces the marginal to match but does not add pointwise resolution: a model that can only predict in a 0.55-band, when stretched to [0, 1], will produce the same rank of predictions with different numbers. MAE is unlikely to improve. Do C first so there is real dynamic range to map.
- Calibration corrects level/spread, not within-segment multimodality. If residuals stay bimodal after this, that is the signal for H/F.

**How / verify.**
```python
import numpy as np
from sklearn.isotonic import IsotonicRegression

def fit_isotonic_per_segment(y_cal, p_cal, seg_cal, min_n=200):
    # fallback: global model for thin segments
    glob = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal)
    models = {"__global__": glob}
    for s in np.unique(seg_cal):
        m = seg_cal == s
        if m.sum() >= min_n:
            models[s] = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)\
                        .fit(p_cal[m], y_cal[m])
    return models

def apply_isotonic(models, p_new, seg_new):
    out = np.empty_like(p_new, dtype=float)
    for s in np.unique(seg_new):
        m = seg_new == s
        out[m] = models.get(s, models["__global__"]).transform(p_new[m])
    return np.clip(out, 0.0, 1.0)

def quantile_map_per_segment(p_new, seg_new, p_cal, y_cal, seg_cal, n=1000):
    out = np.empty_like(p_new, dtype=float)
    q = np.linspace(0, 1, n)
    segs = np.unique(seg_new)
    for s in segs:
        m_new = seg_new == s
        m_cal = seg_cal == s
        if m_cal.sum() >= 200:
            out[m_new] = np.interp(p_new[m_new],
                                   np.quantile(p_cal[m_cal], q),
                                   np.quantile(y_cal[m_cal], q))
        else:
            out[m_new] = np.interp(p_new[m_new],
                                   np.quantile(p_cal, q),
                                   np.quantile(y_cal, q))
    return np.clip(out, 0.0, 1.0)

# Verify: compare std(pred_before), std(pred_after), MAE before/after per segment.
# std(pred_after) should approach std(y). MAE change should be small (< 5%).
```

---

## C. Relax regularization and re-tune  *(primary fix — do this first)*

**What.** The marginal distribution histogram proves the model is **collapsing to base_score** on train itself. Mechanism: under `reg:logistic`, XGBoost initializes `base_score` ≈ `sigmoid⁻¹(mean(y))`; leaf weights are `−G/H+λ` (Newton step); with `H` inflated by `min_child_weight` and `λ = reg_lambda = 9`, all leaf outputs shrink toward 0 → all predictions collapse to `sigmoid(0) ≈ 0.5–0.6`. The current Optuna space forces this: `min_child_weight ∈ [60,500]`, `max_depth ∈ {3,4}`, `gamma ∈ [0,30]`, `reg_lambda` up to 500.

**Pros.**
- Recovers native dynamic range without any post-processing.
- Lets trees carve segment-specific structure (depth 5–7 can split on procedure type *and* refine), directly reducing segment-structured residuals.
- Cheap relative to reframing the target.

**Risks.**
- Over-relaxing → overfitting; keep the `gap_overfitting = val_mae − train_mae` constraint as the primary guardrail.
- The marginal spread may only partially recover if features genuinely lack signal for extreme-target rows — that's the residual problem for B and H.

**How / verify.**

First, run a quick diagnostic to distinguish "over-regularization" from "no signal in features":
```python
# Diagnostic: unregularized extreme fit (train only — just to measure range recovery)
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

diag = XGBRegressor(
    objective="reg:logistic", tree_method="hist", enable_categorical=True,
    max_depth=8, min_child_weight=1, reg_lambda=1.0, gamma=0.0,
    learning_rate=0.05, n_estimators=500, subsample=0.9,
    colsample_bytree=0.8, random_state=42, n_jobs=-1,
)
diag.fit(X_tr, y_tr)
p_diag = diag.predict(X_tr)
print(f"range: [{p_diag.min():.3f}, {p_diag.max():.3f}]  std: {p_diag.std():.3f}")
# If range is now [0.05, 0.95] → over-regularization was the cause.
# If range is still [0.3, 0.7] → feature signal ceiling; H is mandatory.
```

### C.1 — How to set min_child_weight correctly for reg:logistic

**The core issue.** `min_child_weight` is denominated in **sum of Hessians**, not in number of instances. For `reg:squarederror`, `h_i = 1` for every row, so the two are identical. For `reg:logistic`, the Hessian per instance is:

```
h_i = p_i × (1 − p_i)
```

where `p_i = sigmoid(raw_margin_i)`. This expression is bounded:

```
h_i ∈ (0, 0.25]    maximum at p=0.5, collapses to 0 at p→0 or p→1
```

**Conversion formula.** The minimum number of real instances that a leaf must contain is:

```
n_min_real = min_child_weight / mean(h_i)
           = min_child_weight / (p̄ × (1 − p̄))
```

where `p̄` is the average prediction in that leaf (approximated by the mean of `y` at the start of training, i.e. `base_score`).

**Worked example with your data.** Your target has `mean(y) ≈ 0.55`, so `base_score ≈ sigmoid⁻¹(0.55) ≈ 0.20` in margin space, and the initial predictions are `p ≈ 0.55`.

```python
import numpy as np

y_mean = y_tr.mean()          # ≈ 0.55 for your target
h_avg  = y_mean * (1 - y_mean)   # = 0.55 × 0.45 = 0.2475

# Your current setting:
mcw = 60
n_min_at_base = mcw / h_avg      # = 60 / 0.2475 ≈ 242 instances

# At extreme predictions (p=0.1 or p=0.9, h_i = 0.09):
n_min_at_tails = mcw / (0.1 * 0.9)   # = 60 / 0.09 ≈ 667 instances

print(f"n_min at base_score (p≈0.55): {n_min_at_base:.0f}")
print(f"n_min at extreme   (p≈0.10): {n_min_at_tails:.0f}")
```

**Effective n_min by prediction value (min_child_weight = 60):**

| Predicted `p` | `h_i = p(1−p)` | Effective n_min (mcw=60) |
|---|---|---|
| 0.50 | 0.2500 | **240** |
| 0.55 (your base_score) | 0.2475 | **242** |
| 0.20 / 0.80 | 0.1600 | **375** |
| 0.10 / 0.90 | 0.0900 | **667** |
| 0.05 / 0.95 | 0.0475 | **1,263** |
| 0.02 / 0.98 | 0.0196 | **3,061** |

With only ~5,000 training rows per fold, a leaf covering `p ≈ 0.10` would need **667 rows** to satisfy `mcw=60`. That leaf can never form → the model cannot produce predictions near the tails → collapse to center.

**How to compute the correct value.** If your actual intent is "require at least N_target real instances per leaf", set:

```python
N_target = 30   # your true desired minimum, e.g. 30 samples
p_bar    = y_tr.mean()                  # ≈ 0.55
h_avg    = p_bar * (1 - p_bar)          # ≈ 0.2475

min_child_weight_correct = N_target * h_avg   # ≈ 7.4  → use 5–10

# For the Optuna search space: center on this and search around it
"min_child_weight": trial.suggest_float("min_child_weight", 1.0, N_target * h_avg * 3, log=True)
```

For `N_target = 30` (a reasonable floor), `min_child_weight ≈ 7`. For `N_target = 60` (your original intent), `min_child_weight ≈ 15`. Your current value of 60 effectively enforces ~242 instances minimum at the center, and **over 600 at the tails you are trying to predict**.

Then widen the Optuna search space:
```python
p_bar = y_tr.mean()
h_avg = p_bar * (1 - p_bar)   # objective-specific scaling factor

"max_depth":        trial.suggest_int("max_depth", 3, 7),
"min_child_weight": trial.suggest_float("min_child_weight",
                        5 * h_avg,           # ≈ 1.2  (5 real samples floor)
                        120 * h_avg,          # ≈ 29.7 (120 real samples ceiling)
                        log=True),
"gamma":            trial.suggest_float("gamma", 0.0, 3.0),
"reg_lambda":       trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
"reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
"subsample":        trial.suggest_float("subsample", 0.7, 1.0, step=0.05),
# Keep colsample_* and learning_rate ranges as-is.
# Keep gap_overfitting constraint (tau) as the guardrail against overfitting.
```

Key check after retuning: re-plot the **train marginal histogram** (Real vs Predicted). The predicted distribution should now have visible mass in the tails (< 0.2 and > 0.8). If it does, calibration in B becomes meaningful; if it doesn't, proceed to H.

---

## D. Calibration-aware model selection  *(align HPO with the real goal)*

**What.** Trials are selected on val-MAE only. MAE rewards the conditional median and is blind to per-segment bias and to spread compression. Add a calibration/dispersion term.

**Pros.**
- Aligns HPO with the actual goal ("uniform errors") before any post-hoc fix.
- Reuses the existing `constraints_func` machinery already in the codebase.

**Risks.**
- Pure R²/RMSE switch only changes median→mean; it does **not** deliver uniform errors. Use as a monitor only.
- Composite weights need held-out validation so you don't game one term at the expense of the other.

**How / verify.**
```python
import numpy as np
from sklearn.metrics import mean_absolute_error

def calib_penalty(y, pred, seg, n_bins=10):
    """Mean |per-(segment,predicted-bin) bias| + spread-compression term."""
    pen, k = 0.0, 0
    for s in np.unique(seg):
        m = seg == s
        q = np.quantile(pred[m], np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.digitize(pred[m], q[1:-1]), 0, n_bins - 1)
        for j in np.unique(b):
            sel = b == j
            pen += abs(y[m][sel].mean() - pred[m][sel].mean()); k += 1
    bias  = pen / max(k, 1)
    spread = max(0.0, 1.0 - pred.std() / (y.std() + 1e-9))  # 0 if spread matches
    return bias + 0.5 * spread

# In the objective function, after computing val predictions:
# trial.set_user_attr("constraint", (calib_penalty(y_val, pred_val, seg_val) - EPS,))
# OR multi-objective:
# study = optuna.create_study(directions=["minimize", "minimize"])
# return val_mae_mean, calib_penalty(y_val, pred_val, seg_val)
```

---

## E. Per-segment models  *(when shape gaps are large and n allows)*

**What.** One XGBoost per procedure type. Each learns its own target shape and gets its own calibrator.

**Pros.**
- Removes cross-segment averaging; the per-type multimodality is handled by dedicated models.
- Easy to monitor and reason about per segment.

**Risks.**
- Less data per model (~1k–2k); thin segments must fall back to the global model.
- A global model with `procedure_type` as a categorical feature + per-segment calibration (B) often captures most of the gain — try B+C first.

**How / verify.** Train inside per-segment CV; compare segment MAE and `reliability_by_segment` against the single-model+per-segment-calibration baseline. Adopt E only where it beats that baseline.

---

## F. Distributional / two-stage modeling  *(for residual multimodality that survives B–E)*

**What.** A MAE-minimizing regressor returns the conditional **median**; on a bimodal conditional target the median lands **between** modes → bimodal residuals. Two principled fixes:
- **Quantile regression** — `reg:quantileerror` with vector `quantile_alpha` (XGBoost ≥ 2.0); requires `tree_method="hist"`. Predict the distribution, choose the summary the business needs.
- **Two-stage / mixture** — classify the regime (e.g. "stalls low" vs "advances high"), regress within regime, combine by expected value.

**Pros.**
- Only family that addresses the *shape* of the error, not just its level.
- Quantile outputs give honest uncertainty intervals (useful for Cobranzas decisions).

**Risks.**
- Highest complexity; quantile models can cross → need monotone post-sorting.
- Two-stage propagates classifier errors; needs its own CV and leakage control.
- May not improve MAE at all — it changes *what you predict*. Justify by business need.

**How / verify.**
```python
import numpy as np, xgboost as xgb

# Multi-quantile in one model (XGBoost >= 2.0)
alphas = np.array([0.1, 0.5, 0.9])
dtrain = xgb.QuantileDMatrix(X_tr, y_tr, enable_categorical=True)
booster = xgb.train(
    {"objective": "reg:quantileerror", "quantile_alpha": alphas,
     "tree_method": "hist", "learning_rate": 0.04, "max_depth": 6,
     "min_child_weight": 10, "seed": 42},
    dtrain, num_boost_round=500,
)
Q = booster.inplace_predict(X_val)      # shape (n, 3)
Q = np.sort(Q, axis=1)                  # enforce non-crossing
med = np.clip(Q[:, 1], 0, 1)            # median point estimate
# Verify: residuals (y_val - med) per segment; compare with reg:logistic residuals.
```

---

## G. Sample weighting / target transform  *(secondary levers)*

**What.** (i) Weight rows by `1/density(target)` so under-represented tails pull predictions outward; (ii) model `logit(y)` with clipping under `reg:squarederror`, invert with sigmoid.

**Pros.**
- Both widen the predicted range; weighting is a one-liner (`sample_weight` in `fit`).

**Risks.**
- Weighting trades global MAE for tail accuracy; tune strength carefully.
- `logit` transform is sensitive to clipping; `reg:logistic` already encodes the bounded-output prior, so marginal gains are small.

**How / verify.**
```python
import numpy as np
h, edges = np.histogram(y_tr, bins=20, range=(0, 1), density=True)
w = 1.0 / (h[np.clip(np.digitize(y_tr, edges[1:-1]), 0, 19)] + 1e-3)
w *= len(w) / w.sum()   # normalize to mean 1
# model.fit(X_tr, y_tr, sample_weight=w)
# Verify: std(pred) should widen; track per-decile |error| and global MAE simultaneously.
```

---

## H. Reconsider the target framing  *(deep treatment)*

### H.0 — Why this section exists

The train marginal histogram shows spikes in the target at approximately **0, ~0.13, ~0.33, ~0.50, ~0.90** (and mass at 1.0). This is the fingerprint of a **ratio of small integers**, specifically:

```
progress_score = stages_completed / N_total_stages
```

where `N_total_stages` is small (likely 6–9 depending on procedure type). If so, the target is not a continuous variable with incidental multimodality: **it is an ordinal / count variable represented as a float**. No continuous regression model can be unbiased at the spike locations *and* between them simultaneously — the bimodal residuals, the median-collapse, and the inability to predict extreme values are all **structural consequences of the wrong representation**, not hyperparameter problems.

This section maps the available approaches from least to most invasive.

---

### H.1 — Confirm the structure (do this first, ~10 lines of code)

```python
import numpy as np, pandas as pd

# Hypothesis: target = k / N for small N
def detect_stage_fractions(y, N_max=12, tol=0.02):
    candidates = {}
    for N in range(2, N_max + 1):
        fracs = np.arange(0, N + 1) / N
        counts = {f: ((np.abs(y - f) < tol).sum()) for f in fracs}
        mass = sum(counts.values()) / len(y)
        candidates[N] = mass
    return sorted(candidates.items(), key=lambda x: -x[1])

top = detect_stage_fractions(y_tr)
print(top[:5])
# If N=7 (or similar) captures > 60–70% of rows within tol → confirmed.
# Then: y_stage = np.round(y_tr * N).astype(int)  →  the real underlying variable.
```

If confirmed, the target has **N+1 ordered integer levels** (0, 1, …, N). The float encoding is a downstream artifact of the business metric, not the natural representation. Everything below becomes available.

---

### H.2 — Option 1: Ordinal regression (predict the stage directly)

**When.** Confirmed discrete stages, ordered, uniform spacing (k/N). Best when the model is used to predict *which stage* the procedure will reach.

**How it changes the problem.** Target becomes `y_stage ∈ {0, 1, 2, …, N}`. Approaches:

- **Ordered logistic / proportional-odds model** — `mord` library or custom; explicit ordering assumption; interpretable coefficients.
- **Binary decomposition (Frank & Hall trick)** — fit N binary classifiers `P(y_stage > k)` for k=0…N-1; aggregate to produce the expected stage; output = `Σ P(y > k)`.
- **XGBoost multi-class** — `objective="multi:softprob"`, `num_class=N+1`; loses ordinal structure but gains nonlinearity.

**Pros.**
- Residuals become integers (exact stage error); MAE in stage space is directly interpretable ("off by 1 stage").
- No bimodal residuals by construction — the model predicts a distribution over discrete outcomes.
- Segment-structured probabilities are natural.

**Risks.**
- Requires knowing `N` per procedure type (may differ: HIPOTECARIO may have more stages than EJECUTIVO).
- The float conversion at inference (`score = predicted_stage / N`) is fine but adds a step.
- Ordinal models assume proportional odds; test with likelihood-ratio test or per-cutpoint calibration.

**Minimal code (binary decomposition, works with XGBoost):**
```python
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error

def fit_frank_hall(X_tr, y_stage, N, xgb_params):
    """Binary decomposition for ordinal regression."""
    models = []
    for k in range(N):  # P(y_stage > k | X) for k = 0, ..., N-1
        y_bin = (y_stage > k).astype(int)
        m = XGBClassifier(**xgb_params)
        m.fit(X_tr, y_bin)
        models.append(m)
    return models

def predict_frank_hall(models, X):
    """Expected stage = sum of P(y > k) over k."""
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    # enforce monotone: P(y>0) >= P(y>1) >= ...
    probs = np.minimum.accumulate(probs, axis=1)
    return probs.sum(axis=1)  # E[stage] as a float; convert to score: / N

# N = 7  (or per-segment N)
# xgb_p = {"objective": "binary:logistic", "max_depth": 5, "learning_rate": 0.05,
#           "n_estimators": 300, "min_child_weight": 5, "random_state": 42,
#           "tree_method": "hist", "enable_categorical": True}
# models = fit_frank_hall(X_tr, y_stage_tr, N, xgb_p)
# pred_stage = predict_frank_hall(models, X_val)
# pred_score = pred_stage / N
# print(mean_absolute_error(y_val, pred_score))
```

---

### H.3 — Option 2: Two-part / hurdle model (zero-inflation)

**When.** There is significant mass at **y = 0** ("procedure has not advanced at all"). This is a separate regime that a regressor conflates with "low-but-positive" progress.

**Structure:**
- **Part 1 — binary classifier:** `P(y = 0 | X)` — "will this procedure stall completely?"
- **Part 2 — regressor on y > 0:** `E[y | y > 0, X]` — "given some progress, how much?"
- **Combination:** `ŷ = (1 − P(stall)) × ŷ_continuous`

**Pros.**
- Directly models the business-relevant question: "will it move at all, and if so, how far?"
- Eliminates the spike-at-zero from the regression residuals.
- Each part can be calibrated independently.

**Risks.**
- Part 2 is trained on a subset → smaller n; CV folds must maintain y>0 stratification.
- Classifier errors in Part 1 propagate as scale errors in the final prediction.
- If there is also significant mass at y=1 ("fully resolved"), extend to three-part (zero-inflated + one-inflated + middle).

**Minimal code:**
```python
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_absolute_error

def fit_hurdle(X_tr, y_tr, clf_params, reg_params):
    # Part 1: P(y = 0)
    y_stall = (y_tr == 0).astype(int)
    clf = XGBClassifier(**clf_params)
    clf.fit(X_tr, y_stall)

    # Part 2: regress on positive cases only
    pos = y_tr > 0
    reg = XGBRegressor(**reg_params)
    reg.fit(X_tr[pos], y_tr[pos])

    return clf, reg

def predict_hurdle(clf, reg, X, clip=True):
    p_stall = clf.predict_proba(X)[:, 1]        # P(y = 0)
    y_pos   = reg.predict(X)                     # E[y | y > 0]
    ŷ = (1.0 - p_stall) * y_pos
    return np.clip(ŷ, 0.0, 1.0) if clip else ŷ

# Verify: histogram of predict_hurdle output should show near-zero mass at 0
# (the model assigns mass to P(stall) rather than predicting 0 directly).
# Compare MAE of hurdle vs baseline regressor on the full val set.
```

---

### H.4 — Option 3: Mixture / regime classifier + per-regime regressor

**When.** Residuals are bimodal **within** a segment even after fixing regularization. The two humps indicate two overlapping populations (e.g., "cases that stall early" vs "cases that progress steadily") that share the same feature space.

**Structure:**
- **Step 1 — soft clustering:** fit a GMM or use domain knowledge to define K regimes on the target. `K=2` is usually sufficient for bimodal residuals.
- **Step 2 — regime classifier:** `P(regime=k | X)`.
- **Step 3 — per-regime regressors:** `E[y | regime=k, X]`.
- **Combination:** `ŷ = Σ_k P(regime=k | X) × ŷ_k(X)` — the expected value over regimes.

**Pros.**
- Handles within-segment multimodality that the hurdle model doesn't (the hurdle only separates zero from positive).
- Per-regime regressors see a unimodal target → MAE is meaningful and residuals become approximately Gaussian.

**Risks.**
- Regimes defined post-hoc from the target are supervised; defining them from features alone requires unsupervised clustering (more variance).
- Regime boundaries are fuzzy; the soft-assignment combination is correct but requires careful CV so regimes are defined only on training folds.
- n per regime gets small quickly; prioritize K=2 and validate before K=3.

**Minimal code:**
```python
import numpy as np
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier, XGBRegressor

def fit_mixture_model(X_tr, y_tr, K=2, xgb_base=None):
    xgb_base = xgb_base or {"max_depth": 5, "min_child_weight": 5,
                            "learning_rate": 0.05, "n_estimators": 300,
                            "tree_method": "hist", "enable_categorical": True,
                            "random_state": 42}
    # Step 1: define regimes on target (train only)
    gmm = GaussianMixture(n_components=K, random_state=42)
    regime = gmm.fit_predict(y_tr.reshape(-1, 1))     # 0 or 1

    # Step 2: regime classifier
    clf = XGBClassifier(objective="binary:logistic", **xgb_base)
    clf.fit(X_tr, regime)

    # Step 3: per-regime regressors
    regs = {}
    for k in range(K):
        m = regime == k
        r = XGBRegressor(objective="reg:logistic", **xgb_base)
        r.fit(X_tr[m], y_tr[m])
        regs[k] = r

    return gmm, clf, regs, K

def predict_mixture(clf, regs, X, K):
    p_regime = clf.predict_proba(X)                   # (n, K)
    ŷ = np.zeros(len(X))
    for k in range(K):
        ŷ += p_regime[:, k] * regs[k].predict(X)
    return np.clip(ŷ, 0.0, 1.0)

# Verify: plot residuals per regime separately — each should be approximately unimodal.
```

---

### H.5 — Feature engineering to expose stage structure

**When.** Regardless of which model is chosen, if the underlying target is `k/N`, **features that encode judicial stage history are the most direct signal**. A model that sees "the procedure is currently at stage 3 of 7" does not need to infer the spike at 3/7 from indirect proxies.

Candidate features:
- `current_stage` (integer, if available) — the single most powerful predictor.
- `stages_remaining = N − current_stage` — for time-to-completion framing.
- `stage_velocity = current_stage / days_elapsed` — rate of progression.
- `N_total_stages_for_type` — procedure-type-specific ceiling (encodes that N differs by segment).
- `days_since_last_stage_change` — staleness indicator.
- `frac_stages_completed = current_stage / N` — the feature that directly explains the spike.

If `current_stage` is not available but the procedure type and date are:
```python
# Proxy: encode the most recent known stage milestone from procedure metadata
# (e.g., last court event type → maps to a stage number via a lookup table)
stage_map = {"demanda_presentada": 1, "notificacion": 2, "contestacion": 3,
             "audiencia": 4, "sentencia": 5, "apelacion": 6, "ejecucion": 7}
df["current_stage"] = df["last_event_type"].map(stage_map).fillna(0).astype(int)
df["frac_completed"] = df["current_stage"] / df["procedure_n_stages"]
```

Adding `current_stage` as a feature while keeping continuous regression is a **low-cost, high-reward** intermediate step: the model can now learn the spike positions without changing the prediction framework.

---

### H.6 — Decision guide for target reframing

```
Run H.1 (detect_stage_fractions):
  └─ N confirmed, coverage > 60%?
       ├─ Yes + domain confirms stages →
       │      Add current_stage / frac_completed features (H.5) [always worth it]
       │      + Choose framing:
       │           Business needs "which stage" → H.2 (ordinal / Frank-Hall)
       │           Mass at y=0 is dominant     → H.3 (hurdle / two-part)
       │           Bimodal within-segment after C+B → H.4 (mixture / regime)
       └─ No / coverage < 40% →
              The multimodality is not from stages → stay with C+B+F (distributional)
```

---

## Problem-specific considerations

- **The marginal histogram proves under-dispersion is in TRAIN**, not just OOF/test. This rules out "the model is fine on train, just generalizing poorly." The model genuinely cannot produce extreme values — a consequence of `base_score` collapse plus small-leaf averaging, compounded by the quasi-discrete target structure.
- **`reg:logistic` objective**: trains cross-entropy loss (smooth surrogate for the bounded output), selects on MAE. That mismatch is acceptable and conscious — but it means the booster is not directly minimizing your KPI. After fixing regularization, consider `reg:absoluteerror` only if the train marginal still shows collapse (it sometimes helps spread; sometimes compresses more — test empirically).
- **Two conditioning variables.** Shape varies by `procedure_type` *and* `person_type` (KS up to 0.21). Prefer calibrating/segmenting by the interaction where n permits; otherwise by the stronger factor (procedure type).
- **"Parallel curves / uniform errors" is a calibration objective**, mildly in tension with minimizing MAE. Decide the priority explicitly: if uniformity is the deliverable, accept a small MAE cost (quantile mapping after C); if MAE is sacred, isotonic + C only.
- **Optuna without early stopping is fine** — `n_estimators` is searched. Confirm CV folds are **segment-stratified** so every fold sees every procedure type.

---

## Diagnostics & pitfalls

- **Regression-to-the-mean artifact:** never judge calibration on true-target bins alone; use **predicted-bin reliability curves** (A).
- **base_score collapse diagnostic:** check `pred.std()` on train with current params. If `pred.std() < 0.10` on train itself, the collapse is confirmed; run the unregularized diagnostic in C.
- **`std(pred)/std(y)` monitor:** track this ratio per segment across trials. A shrunk model shows ratios < 0.4; a well-spread model should approach 0.85–1.0 post-calibration.
- **Calibration leakage:** fit any calibrator on OOF/held-out predictions, never on train rows.
- **Segment n for calibration:** thin segments fall back to the global calibrator; never fit isotonic on < 100 points.
- **CV stratification by procedure type:** mandatory; otherwise per-segment metrics are unreliable.
- **Quantile mapping cosmetics:** applying quantile mapping to a collapsed [0.3, 0.7] range produces a wider distribution but not better per-row accuracy. Verify MAE before/after, not just the histogram overlap.
- **Quantile crossing in F:** when using multi-alpha quantile regression, sort predictions per row before use.
- **Stage-feature leakage (H.5):** if `current_stage` encodes progress *at the prediction point*, it is a valid predictor. If it encodes progress *at end of the observation window*, it is the target in disguise — verify the temporal cut.

---

## Decision rule / quick guide

1. **Run `detect_stage_fractions` (H.1).** If N is confirmed → add stage features (H.5) regardless of everything else. Free MAE gain.
2. **Run C diagnostic** (unregularized train fit). If `pred.std()` widens → relax Optuna ranges and retune. Re-plot train marginal.
3. **Re-plot reliability by predicted decile per segment** (A). Re-judge all calibration gaps from there.
4. If per-segment level bias remains → **per-segment isotonic calibration** (B). Want marginal overlap → quantile mapping (B), only after C has widened the range.
5. If HPO keeps selecting shrunk models → add **calibration-aware constraint** (D). Keep MAE primary.
6. Mass at y=0 is dominant → add **hurdle/two-part** (H.3).
7. Bimodal residuals within a segment survive C+B → **mixture / regime** (H.4) or **quantile regression** (F).
8. Big shape gaps per type and enough n → **per-segment models** (E) with individual calibrators.
9. Tails still systematically under-fit → **inverse-density weights** (G).

---

## References

1. XGBoost — Quantile Regression (`reg:quantileerror`, vector `quantile_alpha`, `QuantileDMatrix`). https://xgboost.readthedocs.io/en/stable/python/examples/quantile_regression.html
2. XGBoost — Learning Task Parameters (`reg:logistic`, `reg:absoluteerror`, `base_score`, leaf weight formula). https://xgboost.readthedocs.io/en/stable/parameter.html
3. XGBoost — `reg:quantileerror` availability (introduced in 2.0). https://github.com/dmlc/xgboost/issues/9912
4. scikit-learn — `IsotonicRegression` (`y_min`, `y_max`, `out_of_bounds="clip"`). https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
5. scikit-learn — `GaussianMixture` for regime detection. https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
6. Frank & Hall (2001) — A simple approach to ordinal classification (binary decomposition / Frank-Hall trick). https://link.springer.com/chapter/10.1007/3-540-44795-4_13
7. Cragg (1971) — Some statistical models for limited dependent variables (two-part / hurdle model). https://doi.org/10.2307/1909582
8. Optuna — constrained and multi-objective optimization. https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
9. Kuhn & Johnson — *Feature Engineering and Selection* (modeling process, leakage, calibration, resampling). https://feat.engineering/
