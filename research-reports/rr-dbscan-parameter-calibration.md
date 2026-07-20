# DBSCAN Parameter Calibration — `eps` and `min_samples` for Outlier Detection in Financial Transactions

> **Problem.** Calibrate DBSCAN's two hyperparameters, `eps` and `min_samples`, to flag anomalous financial
> transactions in an unsupervised setting (no reliable fraud labels at calibration time).
> **Assumptions.** Numeric feature matrix (amount, hour, frequency, merchant category, etc.), features
> **standardized** before distance computation, Euclidean metric, moderate dimensionality (~5–15 engineered
> features), scikit-learn ≥ 1.3.
> **Constraint.** The calibration choices must be defensible to a non-technical audience (auditors,
> accountants) — "we tried values until it looked right" is not an acceptable justification.

## TL;DR

1. **Standardize first.** `eps` is a raw distance threshold — without scaling, high-range features (amount)
   dominate and the whole calibration is meaningless.
2. **`eps`**: build the k-distance graph with `k = min_samples`, then find the elbow. Automate elbow
   detection (`kneed`) instead of eyeballing it — reproducible and auditable.
3. **`min_samples`**: start at `dimensionality + 1` (theoretical minimum, Ester et al. 1996); for noisy
   real-world data use `2 × dimensionality` (Sander et al. 1998); increase further for large, noisy
   transaction sets.
4. **Tune jointly, not eps in isolation.** Grid-search a small window around the elbow, score with
   silhouette (computed on non-noise points only) or noise-ratio against operational capacity.
5. **If transaction segments have very different density** (e.g., retail vs. corporate accounts), don't
   force a single `eps` — switch to HDBSCAN, which removes that parameter entirely.
6. **Validate against operational capacity**, not just a statistical score: the number of flagged
   transactions must match what the audit team can realistically review.

## Calibration methods compared

| Method | Cost | Optimizes | When to use |
|---|---|---|---|
| k-distance elbow (manual / `kneed`) | Low | `eps` only | Always — first pass, baseline |
| Dimensionality heuristic (`D+1`, `2D`) | Trivial | `min_samples` only | Starting point, before the elbow step |
| Grid search + silhouette / noise-ratio | Medium–high | `(eps, min_samples)` jointly | When the choice must be defensible / compute allows it |
| HDBSCAN (no `eps`) | Medium | Removes `eps` entirely | Segments with variable density (common in transaction data) |

---

## 1. k-distance graph for `eps` (Kneedle elbow)

**What.** For each point, compute the distance to its k-th nearest neighbor (`k = min_samples`), sort those
distances ascending, and plot them. The curve stays flat where data is dense (normal transactions close to
many neighbors) and rises sharply once points start needing a larger radius to find `k` neighbors — that
break is the elbow, and its y-value is the `eps` to use<cite index="10-1">since the technique calculates the average distance between each point and its k nearest neighbors, where k is the MinPts value selected</cite>.

**Pros.**
- Directly tied to the data's own density — not an arbitrary constant.
- Cheap: one `NearestNeighbors` query, O(n log n) with a tree-based backend.
- Reproducible if elbow detection is automated (see below) instead of read off a chart by eye.

**Risks.**
- A visually "eyeballed" elbow is not reproducible or auditable — two people can read the same chart
  differently.
- Assumes a single, roughly uniform density across the dataset. With mixed segments (retail vs. corporate
  transactions), the k-distance curve may show two elbows or none — signal to use HDBSCAN instead (§4).
- The result depends on `min_samples` chosen beforehand — calibrate `min_samples` first (§2), then `eps`.

**How / verify.**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def calibrate_eps(X, min_samples, plot_curve=False):
    """X must already be standardized. Returns eps at the k-distance elbow."""
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])  # distance to the k-th (farthest) neighbor, ascending

    kneedle = KneeLocator(
        x=np.arange(len(k_dist)), y=k_dist,
        curve="convex", direction="increasing", S=1.0,
    )
    eps = k_dist[kneedle.knee] if kneedle.knee is not None else np.percentile(k_dist, 90)

    if plot_curve:
        kneedle.plot_knee()
    return eps

# usage: X_scaled = StandardScaler().fit_transform(X_train)
# eps = calibrate_eps(X_scaled, min_samples=8)
```
Sanity check: re-run with a small `S` (sensitivity) sweep (`0.5`, `1.0`, `2.0`) — if the detected elbow moves
a lot, the curve doesn't have a clean knee and the joint grid search (§3) should decide instead.

---

## 2. Dimensionality heuristic for `min_samples`

**What.** `min_samples` sets how many neighbors (including the point itself) are required for a point to be
a **core point**. The theoretical floor is `min_samples ≥ D + 1` where `D` is the number of features used
<cite index="10-1">a rule of thumb is to set MinPts to be greater than or equal to the number of dimensions D in the data set, i.e., MinPts ≥ D + 1</cite>, but this floor is too permissive for noisy data —
for datasets with more than two dimensions, <cite index="10-1">choose MinPts = 2*dim, where dim is the dimensionality of the data set (Sander et al., 1998)</cite>.

**Pros.**
- Removes guesswork for the starting value — grounded in the original DBSCAN paper and follow-up work.
- Directly controls sensitivity: <cite index="9-1">larger min_samples leads to more conservative clustering, with fewer clusters and more noise</cite>.
- Cheap to reason about and explain to a non-technical audience ("we require at least twice as many similar
  transactions as variables we're comparing").

**Risks.**
- It's a starting point, not a final answer — the heuristic ignores dataset size and noise level, which
  also matter: <cite index="10-1">the larger the data set, the larger min_samples should be, and if the data set is noisier, choose a larger value</cite>.
- Too low (`min_samples ≤ 2`) degrades DBSCAN toward single-linkage hierarchical clustering, which is overly
  sensitive to chains of nearby points — <cite index="10-1">MinPts must be at least 3</cite>.
- Counting "dimensionality" naively after one-hot encoding categorical variables (merchant category, channel)
  can inflate `D` artificially — count *informative* dimensions, not raw encoded columns.

**How / verify.**
```python
def min_samples_heuristic(n_features, noisy=True):
    """n_features = number of standardized numeric dimensions actually driving the distance metric."""
    base = 2 * n_features if noisy else n_features + 1
    return max(base, 3)  # floor of 3 regardless of dimensionality

# usage: min_samples0 = min_samples_heuristic(n_features=6, noisy=True)  # -> 12
```

---

## 3. Joint grid search with validation metrics

**What.** Treat `(eps, min_samples)` as a small 2D grid around the values from §1–§2, and score each
combination with an internal validation metric plus an operational check (noise ratio).

**Pros.**
- Doesn't rely on a single elbow reading; picks the pair that is jointly stable.
- Silhouette (or DBCV, density-aware) gives a quantitative, comparable score across candidate parameter
  pairs — useful to show the audit/compliance team *why* a configuration was chosen.
- Surfaces the noise ratio directly, which is the number that actually matters operationally (how many
  transactions get flagged for review).

**Risks.**
- Standard silhouette score assumes convex, roughly spherical clusters and penalizes DBSCAN's arbitrarily
  shaped clusters unfairly — treat it as a relative comparison tool between DBSCAN configurations, not an
  absolute quality bar.
- Silhouette requires **excluding noise points** and at least 2 real clusters — if a grid cell returns
  `n_clusters < 2`, skip it rather than error out or force a score.
- Don't tune against known-fraud labels if you have a handful — that leaks supervision into an unsupervised
  step and inflates confidence. Reserve any labels for **post-hoc validation only** (§ Diagnostics).
- Grid search cost grows with grid size × dataset size; keep the grid narrow (±20–30% around the §1/§2
  starting values), not exhaustive.

**How / verify.**
```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def grid_search_dbscan(X, eps_grid, min_samples_grid, max_noise_ratio=0.05):
    results = []
    for eps in eps_grid:
        for min_samples in min_samples_grid:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            noise_ratio = np.mean(labels == -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            sil = np.nan
            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > n_clusters:  # need > 1 point per cluster on average
                    sil = silhouette_score(X[mask], labels[mask])

            results.append(dict(eps=eps, min_samples=min_samples, n_clusters=n_clusters,
                                 noise_ratio=noise_ratio, silhouette=sil))

    import pandas as pd
    df = pd.DataFrame(results)
    # keep configs whose flagged-transaction volume the audit team can realistically review
    feasible = df[df["noise_ratio"] <= max_noise_ratio]
    return feasible.sort_values("silhouette", ascending=False)

# usage:
# eps0 = calibrate_eps(X_scaled, min_samples=12)
# grid = grid_search_dbscan(
#     X_scaled,
#     eps_grid=np.linspace(0.8*eps0, 1.2*eps0, 5),
#     min_samples_grid=range(8, 17, 2),
# )
```

---

## 4. HDBSCAN as an alternative when density varies

**What.** HDBSCAN <cite index="29-1">performs DBSCAN over varying epsilon values and integrates the result to find a clustering that gives the best stability over epsilon, which allows it to find clusters of varying densities unlike DBSCAN</cite>. It replaces `eps` with `min_cluster_size` (and optionally `min_samples`), added to scikit-learn in version 1.3.

**Pros.**
- No `eps` to calibrate — directly addresses the main risk in §1 (mixed-density transaction segments).
- `min_cluster_size` is more intuitive to set with business context: "the smallest group of transactions we'd
  still call a genuine behavioral pattern."
- More robust to parameter choice in general, per scikit-learn's own comparison.

**Risks.**
- Different noise semantics: labels can include `-2`/`-3` for infinite or missing-value samples in addition
  to `-1` for noise — filter explicitly on all of them before counting flagged transactions.
- Still density-based: extremely sparse or extremely small segments (e.g., a merchant with only a handful of
  historical transactions) may be entirely swallowed into noise — expected, not a bug, but should be
  communicated to the audit team.
- Slightly heavier computationally than plain DBSCAN since it explores a range of epsilon values internally.

**How / verify.**
```python
from sklearn.cluster import HDBSCAN

def run_hdbscan(X, min_cluster_size=15, min_samples=None):
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = hdb.fit_predict(X)
    noise_mask = labels < 0  # covers -1 (noise), -2 (inf), -3 (missing)
    return labels, noise_mask

# usage: labels, is_outlier = run_hdbscan(X_scaled, min_cluster_size=15)
```

---

## Problem-specific considerations (financial transactions)

- **Skewed amount distributions**: transaction amounts are typically heavy right-tailed. Log-transform
  (`log1p`) before standardizing, or Euclidean distance will be dominated by a handful of large-amount
  outliers regardless of `eps`.
- **Mixed variable types**: categorical fields (merchant category, channel, country) need encoding before
  Euclidean distance applies; naive one-hot encoding inflates dimensionality and directly shifts the
  `min_samples` heuristic in §2 — count encoded columns, not just original fields.
- **Heterogeneous account segments**: retail vs. corporate accounts, or high- vs. low-frequency customers,
  rarely share one density regime — this is the practical trigger for HDBSCAN (§4) over plain DBSCAN.
- **Seasonality**: end-of-month, end-of-quarter, and campaign periods shift the "normal" density. Recalibrate
  `eps`/`min_samples` (or `min_cluster_size`) periodically rather than fixing them once.

## Diagnostics & pitfalls

- **No train/test leakage for the scaler**: fit `StandardScaler` (and any log-transform parameters) on the
  reference/training window only, then apply to new transactions — never refit on data that includes the
  period being scored.
- **Don't tune parameters against a handful of known fraud cases.** If limited labels exist, hold them out
  entirely from calibration and use them only to **evaluate** the final configuration's recall — using them
  to pick `(eps, min_samples)` is a form of leakage that overstates confidence.
- **Sanity-check the noise ratio.** A well-calibrated configuration on financial transaction data typically
  flags on the order of 1–5% of transactions as noise; near-0% (nothing flagged) or near-100% (nearly
  everything flagged) both signal miscalibration, not a "clean" or "very risky" dataset.
- **Determinism**: `DBSCAN.fit` is deterministic given fixed input order and parameters, but the input row
  order and floating-point tie-breaking can affect border-point assignment at the margin — don't over-index
  on a single point's exact label near a cluster boundary.
- **Standardization is not optional** for either §1's k-distance graph or the raw grid search in §3 — an
  unscaled `amount` feature in raw currency units will make `eps` effectively driven by that one column.

## Decision rule

1. Standardize (and log-transform skewed columns) → always, before anything else.
2. Compute `min_samples` from the dimensionality heuristic (§2) → starting value.
3. Run the k-distance graph with that `min_samples`, detect the elbow with `kneed` (§1) → starting `eps`.
4. Run the narrow grid search (§3) around both starting values → pick the feasible configuration
   (`noise_ratio ≤ operational capacity`) with the best silhouette.
5. If the k-distance curve shows multiple elbows, or transaction segments are known to differ structurally
   in density (retail vs. corporate, low vs. high frequency) → skip straight to HDBSCAN (§4) instead of
   forcing a single `eps`.
6. Recalibrate on a rolling schedule (e.g., quarterly) to account for seasonality, not once at deployment.

## References

1. scikit-learn — `sklearn.cluster.DBSCAN`. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
2. scikit-learn — `sklearn.cluster.HDBSCAN`. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
3. scikit-learn — Demo of HDBSCAN clustering algorithm. https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html
4. Sefidian, A. M. — "How to determine epsilon and MinPts parameters of DBSCAN clustering" (summarizing Ester et al. 1996 and Sander et al. 1998 rules of thumb). https://www.sefidian.com/2022/12/18/how-to-determine-epsilon-and-minpts-parameters-of-dbscan-clustering/
5. NumberAnalytics — "Complete Guide to DBSCAN Density-Based Clustering Techniques" (minPts sensitivity, grid search over eps/minPts). https://www.numberanalytics.com/blog/dbscan-density-clustering-guide
6. `kneed` — Knee-point detection in Python (Kneedle algorithm), documentation and PyPI page. https://kneed.readthedocs.io/ and https://github.com/arvkevi/kneed
7. hdbscan library — Parameter Selection for HDBSCAN*. https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
