# Performance-Window (Target Horizon) Selection for Credit Default Models — Vintage, Hazard and Label-Quality Evidence

> **Context.**
> A supervised binary default model is scored at loan application (`t0`). The target is
> "client defaults within X months of `t0`". X must be chosen from a candidate ladder
> (e.g. 12 / 18 / 24 / 36 months). Every candidate X changes two things at once: the
> **label** (longer X ⇒ more events observed per client) and the **population** (longer X
> ⇒ recent origination cohorts are dropped, because they have not yet matured to X months
> by the data snapshot date). This report consolidates the evidence types that can settle
> the choice, in the order they should be produced, and states which of them can select X
> on their own and which only rule alternatives in or out.
>
> **Assumptions.** Binary event with a known time-to-event per client (`months_to_default`,
> NaN if no event); one global snapshot date; cohorts indexed by origination month; the
> "bad" definition (e.g. 90+ DPD) is already fixed and is **not** what this report chooses;
> features are all observable at `t0`; the model is an application scorecard (ranking +
> calibrated probability), not a behavioural or collections model. Stack: `numpy`/`pandas`,
> `scikit-learn` for metrics/splits, `xgboost` for the label-quality experiment.
>
> **Scope note.** Supersedes and absorbs the relevant parts of
> `rr-composition-seasoning-decomposition-censored-cohorts.md`, which treats the
> composition/seasoning decomposition in isolation. That decomposition appears here as one
> section (a validity gate), not as the whole method. Refer back to that report only for the
> Shapley/Owen non-applicability argument, which is not reproduced in full here.

---

## TL;DR / Recommendation

- **Separate two questions that the candidate-ladder design fuses together.** "Which X gives
  the best label?" is a property of the *hazard curve*. "Which sample do I train on?" is a
  volume/recency trade-off. Answering them on one table of four differently-populated
  scenarios is what makes the decision feel unjustifiable.
- **Choose X on a single closed cohort** — clients with at least `max(candidate X)` months on
  book — read all candidates off that one curve. One population, one curve, composition
  effect zero by construction. This eliminates the confound instead of measuring it.
- **The selection criterion is the conditional hazard, not the cumulative curve.** Cumulative
  increments decay mechanically as the denominator empties; that decay is not evidence of
  maturation. Hazard `h(m) = P(event at m | survived to m−1)` divides by survivors and is
  the quantity whose flattening the industry criterion actually refers to.
- **Decision rule: excess-hazard coverage.** Estimate the plateau `h∞` from the flat tail;
  define excess `e(m) = h(m) − h∞` (origination-attributable risk); pick the smallest
  candidate X whose cumulative excess coverage exceeds a threshold `τ` fixed *before* seeing
  results (τ = 0.90–0.95 typical). Report coverage for every candidate, not just the winner.
- **Composition/seasoning decomposition is a gate, not a selector.** It answers "is the rate
  gap between two windows real or a population artifact?". Passing the gate (composition
  small) licenses reading the ladder as a maturation curve — it never points at a specific X,
  and read naively it argues monotonically for the longest window.
- **The decisive evidence is empirical, not descriptive.** Train the same model on the short
  label and the long label, evaluate both against the matured truth on an out-of-time split.
  If the short label loses no discrimination in the mid-score deciles, the extra maturity is
  not buying anything and recency should win.
- **Roll-rate analysis does not select X.** It selects the DPD cut of the *bad definition*.
  Both are "target definition" decisions and are routinely conflated; they are orthogonal and
  the DPD cut must be fixed first.

---

## The decision, formally

Let `T` be months from `t0` to the default event, `M` the snapshot horizon available for a
given cohort, and `X` the performance window. The label is:

```
y_X = 1[0 <= T <= X]
```

Two distinct error sources drive the choice of X:

**1. Label immaturity (short X).** Clients with `X < T <= M` are labelled good but are
eventual bads. Measured on a matured population, the false-negative rate of the label is:

```
FNR_label(X) = (F(M) − F(X)) / F(M)      where F(m) = P(T <= m)
```

This is not random noise. Clients defaulting late are disproportionately the *marginal*
applicants — the ones near the decision boundary at `t0`. Mislabelling them flattens the
score exactly where cut-offs are set.

**2. Label contamination (long X).** As X grows past the point where risk is attributable to
application-time characteristics, `y_X` starts absorbing events driven by post-origination
shocks (job loss, new external debt, macro). Those positives carry no signal recoverable from
`t0` features; the model can only fit them as noise in the positive class.

**3. Sample cost (long X).** With a fixed snapshot date, only cohorts with
`months_on_book >= X` are fully observed. Larger X ⇒ fewer clients, older clients, larger gap
between the newest training cohort and the deployment population.

The optimal X trades (1) against (2)+(3). Sources (1) and (2) are both readable off the
hazard curve; (3) is readable off the coverage table.

---

## Comparison table

| Method | Answers | Censoring-safe | Can select X alone? | Cost | Use it for |
|---|---|---|---|---|---|
| **Cumulative vintage curve** (bad-rate maturation) | Where does the cumulative bad rate level off? | Yes, with cohort masking | Weakly — plateau is visually ambiguous | Low | First look; industry-expected exhibit |
| **Conditional hazard + excess coverage** | Where does risk per surviving client stop decaying? | Yes, with a closed cohort | **Yes** — with a pre-set τ | Low | The primary selection criterion |
| **Label-noise / FNR audit** | How many eventual bads does X mislabel as good? | Yes | No — disqualifies, doesn't rank | Low | Ruling out short candidates |
| **Label-quality experiment** | Does the shorter label actually cost discrimination? | Yes (OOT design) | **Yes** — decisive when descriptive evidence ties | Medium | Breaking a tie between adjacent candidates |
| **Composition/seasoning decomposition** | Is the gap between two windows real or a population artifact? | By construction (forced order) | **No** — validity gate only | Low | Multi-population ladders; unnecessary on a closed cohort |
| **Coverage/recency trade-off table** | What does each X cost in clients and recency? | Yes | No — cost side only | Low | Pairing with hazard evidence |
| **APC / EMV (dual-time dynamics)** | Separating maturation, vintage quality and calendar effects | Yes (designed for the triangle) | No — explains, doesn't select | Medium–High | COVID-spanning data; vintage-quality drift |
| **Discrete-time survival / chain-ladder** | Full term structure; extrapolating immature cohorts | Yes (native) | No — replaces the fixed window instead | High | Production term-structure; recovering recency |
| **Roll-rate analysis** | Which DPD bucket defines "bad" | N/A | **No — wrong question** | Low | Fixing the bad definition *before* this analysis |

---

## Cumulative vintage curve (bad-rate maturation)

**What.** For each origination cohort, cumulative default rate by months-on-book, masked so
no cohort contributes a cell it has not matured into. The standard scorecard exhibit: the
sample window is chosen from a period where the bad rate has stabilised, <cite index="6-1">the development sample being taken from a time period where the bad rate is deemed stable, or where the cohort is deemed to have matured — that is, where the bad rate starts to level off</cite>. Published worked examples run the same logic on a
cumulative bad-rate-vs-MOB chart: <cite index="15-1">the cumulative bad rate grows quickly before 24 months and then stabilises, so 24 months is taken as the performance window; alternatively, the per-month percentage change is checked and the window is set where that change is minimal</cite>.

**Pros.**
- Universally expected by validators and model-risk reviewers — produce it regardless.
- Cohort-by-cohort form doubles as a vintage-quality diagnostic (do recent cohorts sit above
  older ones at the same MOB?).
- No modelling assumptions; three lines of `groupby`.

**Risks.**
- **The plateau is not identifiable from the cumulative curve.** Cumulative increments shrink
  as the surviving denominator empties, so the curve flattens even when per-client risk is
  constant. Reading "levels off" directly off `F(m)` systematically selects too short an X.
  The "% change per month is minimal" variant inherits the same defect.
- Aggregating cohorts hides vintage heterogeneity; a flat pooled curve can be the average of
  a rising and a falling vintage.
- Coarse bins destroy the shape. A first bin spanning 0–12 months averages the ramp-up, the
  peak and the start of the decay into one number, hiding the region where most of the
  origination-attributable risk lives.

**How / verify.**

```python
import numpy as np
import pandas as pd

def vintage_matrix(df, loan_date, default_date, snapshot_date, max_m=36):
    """Cohort x months-on-book cumulative default rate, right-censored cells blanked.

    Args:
        df: one row per client, must contain 'CohortMonth' (period start timestamp).
        loan_date, default_date: datetime Series aligned to df (default_date NaT if none).
        snapshot_date: single as-of date for the whole extract.
        max_m: max seasoning month to tabulate.

    Returns:
        (matrix, cohort_min_mob) where matrix cells beyond a cohort's maturity are NaN.
    """
    months_to_default = (
        (default_date.dt.year - loan_date.dt.year) * 12
        + (default_date.dt.month - loan_date.dt.month)
    )

    # theoretical worst case per cohort: a loan originated on the cohort's last calendar day
    cohorts = pd.Series(pd.to_datetime(sorted(df["CohortMonth"].unique())))
    cohort_end = cohorts + pd.offsets.MonthEnd(0)
    raw = ((snapshot_date.year - cohort_end.dt.year) * 12
           + (snapshot_date.month - cohort_end.dt.month))
    cohort_min_mob = pd.Series(
        (raw - (snapshot_date.day < cohort_end.dt.day).astype(int)).values,
        index=cohorts.values,
    )

    rates = {m: months_to_default.between(0, m).groupby(df["CohortMonth"]).mean()
             for m in range(1, max_m + 1)}
    matrix = pd.DataFrame(rates).sort_index()
    mask = np.greater.outer(matrix.columns.to_numpy(), cohort_min_mob.reindex(matrix.index).to_numpy()).T
    return matrix.mask(mask), cohort_min_mob
```

Sanity check: the theoretical worst-case maturity must never exceed the actual minimum
`months_on_book` observed in that cohort. Assert it before trusting the mask.

---

## Conditional hazard + excess-hazard coverage  *(the primary criterion)*

**What.** Discrete-time hazard on a **closed cohort** (all clients matured to `max_m`, so the
denominator never thins):

```
h(m) = (F(m) − F(m−1)) / (1 − F(m−1))          F(0) = 0
S(m) = Π_{u<=m} (1 − h(u))                      F(m) = 1 − S(m)
```

Three regimes carry the decision:

- **Ramp + peak** (early months): the worst applicants surface. Origination-attributable.
- **Decay**: the surviving pool improves month over month — latent `t0` risk is still
  working itself out. Still learnable from application features.
- **Plateau `h∞`**: the pool stops improving. A constant hazard is a stationary arrival
  process of *new* shocks, independent of what was visible at `t0`. Not learnable.

The plateau is the separator. Define excess hazard and coverage:

```
e(m)  = max(h(m) − h∞, 0)                       origination-attributable component
C(X)  = Σ_{m<=X} e(m) / Σ_{m<=max_m} e(m)       fraction of it captured by window X
X*    = min { X ∈ candidates : C(X) >= τ }      τ fixed before computing
```

`τ` is the only judgement call, and it is explicit and auditable — which is the point.

**Pros.**
- Removes the denominator artifact that makes the cumulative curve unreadable; the flattening
  the industry criterion refers to is genuinely visible here.
- Turns "where does it level off" from a visual call into a number with a stated threshold.
- Reports a coverage figure for *every* candidate, so a rejected candidate is rejected on a
  quantity, not on absence of argument.
- Estimating it on a closed cohort makes composition effects exactly zero — the entire
  decomposition machinery below becomes unnecessary for the selection step.

**Risks.**
- **`h∞` must be identifiable.** It needs a visibly flat tail of several months. With only two
  or three coarse late bins, the plateau estimate is noise and every downstream number
  inherits that noise. If no flat tail exists within the observable horizon, the data cannot
  support a maturity-based choice — say so rather than inventing a plateau.
- Monthly hazard on small cohorts is noisy; smooth for reading (3-month rolling) but compute
  coverage on raw values.
- A closed cohort is older and smaller than the full sample. That is acceptable for *choosing*
  X (a one-time design decision needing an unbiased curve, not maximum n) but the chosen X
  must then be re-applied to the full sample for training.
- Seasonality and payment-schedule artifacts (e.g. 12-month term products) create hazard
  spikes at contract boundaries that are structural, not maturation. Inspect before fitting
  a plateau through them.

**How / verify.**

```python
import numpy as np
import pandas as pd

def conditional_hazard(pop, max_m=36, months_col="months_to_default"):
    """Discrete-time hazard on a CLOSED cohort (every client matured to max_m).

    Args:
        pop: one row per client; months_col = months to event, NaN if none.
        max_m: horizon to tabulate.
        months_col: column with months-to-event.

    Returns:
        DataFrame with month, events, at_risk, hazard, survival, cum_rate.
    """
    t = pop[months_col]
    at_risk, rows = len(pop), []
    for m in range(1, max_m + 1):
        n_ev = int((t == m).sum())
        rows.append({"month": m, "events": n_ev, "at_risk": at_risk,
                     "hazard": n_ev / at_risk if at_risk else np.nan})
        at_risk -= n_ev
    out = pd.DataFrame(rows)
    out["survival"] = (1 - out["hazard"]).cumprod()
    out["cum_rate"] = 1 - out["survival"]
    return out


def excess_coverage(hz, candidates, plateau_from=30, max_m=36):
    """Excess-hazard coverage per candidate window, with plateau estimated from the tail.

    Args:
        hz: output of conditional_hazard.
        candidates: list of candidate windows, e.g. [12, 18, 24, 36].
        plateau_from: first month of the assumed-flat tail used to estimate h_inf.
        max_m: horizon.

    Returns:
        (table, h_inf). Coverage is undefined if the tail is not flat -- check tail_cv.
    """
    tail = hz.loc[hz["month"] >= plateau_from, "hazard"]
    h_inf = float(tail.median())
    tail_cv = float(tail.std() / tail.mean())          # >0.15 => plateau not credible

    excess = (hz["hazard"] - h_inf).clip(lower=0)
    total = excess.loc[hz["month"] <= max_m].sum()
    table = pd.DataFrame({
        "X": candidates,
        "cum_rate": [float(hz.loc[hz["month"] == x, "cum_rate"].iloc[0]) for x in candidates],
        "coverage": [float(excess.loc[hz["month"] <= x].sum() / total) for x in candidates],
    })
    table.attrs["h_inf"], table.attrs["tail_cv"] = h_inf, tail_cv
    return table, h_inf
```

Verification: `(1 − h).cumprod()` must reproduce `1 − F(m)` from the direct cumulative
calculation to floating tolerance. If it does not, events and risk sets are inconsistent
(usually a censored client left in the denominator).

---

## Label-noise / false-negative audit

**What.** On a population matured to `M`, quantify what each candidate X mislabels:

```
FNR_label(X) = (F(M) − F(X)) / F(M)
```

Optionally split by score decile or by application-feature strata to check *where* the
mislabelled bads sit. The disqualification argument for short windows lives here.

**Pros.**
- Directly interpretable to a credit committee: "a 12-month label calls N% of eventual bads
  good."
- Stratified by score band, it shows the concentration in mid-deciles — the operationally
  relevant harm, which the aggregate number understates.
- Symmetric: applying it to every candidate prevents the common asymmetry of rigorously
  disqualifying one candidate while adopting its neighbour untested.

**Risks.**
- `M` is itself a choice. `FNR_label` is measured *relative to* a reference horizon; state it
  ("FNR against a 36-month truth"), never as an absolute.
- The metric is monotone in X: it always favours the longest window. It can disqualify a
  candidate against a stated tolerance; it cannot select among survivors. Pair it with the
  hazard evidence or it will silently push you to the longest horizon.
- Requires a matured population, which is older — check that its vintage quality is not
  systematically different from the deployment population before generalising the rate.

**How / verify.**

```python
def label_fnr(pop, candidates, reference_m=36, months_col="months_to_default"):
    """False-negative rate of each candidate label against a matured reference horizon."""
    t = pop[months_col]
    f_ref = t.between(0, reference_m).mean()
    return pd.DataFrame({
        "X": candidates,
        "target_rate": [t.between(0, x).mean() for x in candidates],
        "fnr_vs_reference": [(f_ref - t.between(0, x).mean()) / f_ref for x in candidates],
    })
```

---

## Label-quality experiment  *(decisive when descriptive evidence ties)*

**What.** Descriptive curves answer "how much event mass does X capture". They do not answer
"does the model get worse". Train the same model on each candidate label and evaluate all of
them against a single matured truth, out-of-time.

Design:
1. Fix one population matured to `M` (the reference horizon).
2. Time-based split **on client identity**, not loan identity: train on origination months
   `< cut`, test on `>= cut`.
3. Train model_X on `y_X` for each candidate X, using only `t0` features.
4. Evaluate every model against `y_M` on the same test set.
5. Compare AUC/KS globally **and** false-negative rate within mid-score deciles.

**Pros.**
- Measures the thing that actually matters — discrimination against eventual truth — instead
  of a proxy for it.
- Naturally exposes the mid-decile degradation that aggregate metrics wash out.
- Breaks ties between adjacent candidates (the 18-vs-24 type impasse) that no descriptive
  statistic can resolve, because the two labels differ by a few percentage points of event
  mass whose *learnability* is precisely the open question.

**Risks.**
- **Leakage via client identity.** With repeat borrowers, a client-keyed split is mandatory;
  a loan-keyed split puts the same client on both sides and inflates every model equally,
  hiding the comparison.
- **Evaluation-set censoring.** The test set must also be matured to `M`, which pushes the
  time cut back. Verify `min(months_on_book) >= M` in the test set explicitly.
- Confounded by hyperparameters if tuned per label. Fix the configuration across labels; the
  experiment is about the label, not the model.
- Base rates differ across labels, so compare rank-based metrics (AUC, KS) and decile FNR;
  raw log-loss across different `y` definitions is not comparable.

**How / verify.**

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def label_quality_experiment(pop, features, candidates, reference_m=36,
                             cut_month="2019-01-01", months_col="months_to_default",
                             cohort_col="CohortMonth", seed=42):
    """Train on each candidate label, evaluate all against the matured truth, OOT.

    Args:
        pop: one row per CLIENT, matured to reference_m, with t0 features + cohort + outcome.
        features: list of application-time feature columns.
        candidates: candidate windows, e.g. [12, 18, 24].
        reference_m: matured horizon used as evaluation truth.
        cut_month: origination month splitting train (before) from test (on/after).
        months_col, cohort_col: outcome and cohort columns.
        seed: fixed random_state.

    Returns:
        DataFrame with AUC vs truth and mid-decile FNR per candidate label.
    """
    cut = pd.Timestamp(cut_month)
    tr, te = pop[pop[cohort_col] < cut], pop[pop[cohort_col] >= cut]
    assert set(tr.index).isdisjoint(te.index), "client-keyed split violated"

    y_true_te = te[months_col].between(0, reference_m).astype(int).to_numpy()
    rows = []
    for x in candidates:
        y_tr = tr[months_col].between(0, x).astype(int)
        clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="auc", random_state=seed)
        clf.fit(tr[features], y_tr)
        p = clf.predict_proba(te[features])[:, 1]

        # FNR among eventual bads sitting in the middle score deciles (deciles 4-7)
        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
        mid = np.isin(dec, [3, 4, 5, 6])
        mid_bads = y_true_te[mid].sum()
        rows.append({
            "label_X": x,
            "auc_vs_truth": roc_auc_score(y_true_te, p),
            "mid_decile_bad_share": mid_bads / max(y_true_te.sum(), 1),
        })
    return pd.DataFrame(rows)
```

Read it as: if `auc_vs_truth` is flat across candidates and `mid_decile_bad_share` does not
worsen for the shorter label, the extra maturity buys nothing and the recency argument wins.
`early_stopping_rounds` and `eval_metric` belong in the constructor in current XGBoost, not
in `fit` [14].

---

## Composition-vs-seasoning decomposition  *(validity gate, not a selector)*

**What.** When candidates are evaluated on *different* populations — the usual scenario-table
design, where each X drops the cohorts too recent to have matured — an observed rate gap
mixes two causes: same clients given more time (**seasoning**) versus a different client mix
(**composition**). For nested populations `pop_l ⊆ pop_s` with `l > s`:

```
A = rate(pop_s @ s)     B = rate(pop_l @ s)     C = rate(pop_l @ l)
composition = B − A     (horizon fixed, population changed)
seasoning   = C − B     (population fixed, horizon changed)
composition + seasoning = C − A            exact
```

**The order is forced, not chosen.** The reverse bridge — `pop_s` measured at horizon `l` —
is uncomputable: `pop_s` only guarantees `>= s` months on book, and its most recent vintage
has not reached `l` by the snapshot. Measuring it there silently undercounts pending events.
This is the same failure mode as target leakage, along the time axis instead of the feature
axis, and it should be checked with the same rigour.

**Pros.**
- Exact — no residual term, no modelling assumption, three group means.
- Licenses the key interpretive move: if composition is small everywhere, the ladder of target
  rates can be read as a maturation curve rather than as four unrelated books.
- Cheap enough to run on every pair.

**Risks / critique.**
- **It cannot select X, and it is routinely misread as if it could.** Small composition ⇒ the
  gaps are genuine seasoning ⇒ longer windows really do capture more default. Taken alone
  that argument is monotone and terminates at the longest candidate. It is a necessary
  condition for a defensible comparison, never a sufficient one.
- **Path-dependence for ladders of ≥3 groups.** Adjacent-step sums and direct non-adjacent
  pairs share the same `total_gap` (both equal `rate_l − rate_s`) but split it differently,
  because they freeze a different bridge population at a different horizon. Both are correct;
  they answer different questions. Never sum or compare shares across the two families, and
  always label which family a reported number belongs to.
- **Made unnecessary by a better design.** On a closed cohort, all candidates share one
  population — composition is zero by construction and there is nothing to decompose. Prefer
  fixing the design over measuring the confound.
- Silently wrong on non-nested populations; assert the subset relation before computing.

**How / verify.**

```python
def decompose_pair(pop_small, pop_large, s, l, id_col="PartyId",
                   months_col="months_to_default"):
    """Forced-order composition/seasoning split; pop_large must be a subset of pop_small."""
    assert set(pop_large[id_col]).issubset(set(pop_small[id_col])), "nesting violated"
    a = pop_small[months_col].between(0, s).mean()
    b = pop_large[months_col].between(0, s).mean()   # safe bridge: >= l implies >= s
    c = pop_large[months_col].between(0, l).mean()
    return {"pair": f"{s}m -> {l}m", "A": a, "B": b, "C": c,
            "composition": b - a, "seasoning": c - b, "total_gap": c - a}
```

Assertion worth keeping in the notebook: the adjacent-step `total_gap` values must sum to the
direct pair's `total_gap`. Composition shares must **not** be expected to agree.

For the full treatment — including why Shapley/Owen averaging is structurally unavailable
here (roughly half the orderings require an uncomputable quantity) — see
`rr-composition-seasoning-decomposition-censored-cohorts.md` [15].

---

## Coverage / recency trade-off table

**What.** The cost side. Per candidate X: clients retained, loans retained, newest usable
origination month, and the gap between that month and the deployment date.

**Pros.**
- Makes the real constraint visible: with a fixed snapshot, X and recency trade one-for-one.
  A 36-month window means the newest training cohort is three years old at deployment.
- Recency is not a nice-to-have. Underwriting policy, product mix and applicant composition
  drift; a model trained on cohorts three years stale is fitted to a population that no longer
  applies. Post-2020 this is acute — <cite index="4-1">since the pandemic most institutions have either excluded 2020–2021 from modelling or included it without adjustment as noise, with validators divided on which to require</cite>.
- Surfaces the event-count floor: a short window on a low-default portfolio can leave too few
  positives for stable estimation, which is an independent constraint on going shorter.

**Risks.**
- Reading this table alone produces the mirror-image error of the FNR audit: it is monotone in
  the opposite direction and always favours the shortest window.
- Client counts are not the binding constraint — *event* counts are. Report positives, not
  just n.
- With repeat borrowers, "clients" and "loans" diverge sharply; state which population each
  column is built on and never mix them within a row without labelling.

**How / verify.**

```python
def coverage_table(pop, cohort_min_mob, candidates, floor_month,
                   cohort_col="CohortMonth", months_col="months_to_default"):
    """Population, events and recency retained per candidate window."""
    rows = []
    for x in candidates:
        safe = cohort_min_mob[cohort_min_mob >= x].index
        sub = pop[pop[cohort_col].between(floor_month, safe.max())]
        y = sub[months_col].between(0, x)
        rows.append({"X": x, "max_cohort": safe.max().date(), "n_clients": len(sub),
                     "n_events": int(y.sum()), "target_rate": y.mean()})
    return pd.DataFrame(rows)
```

---

## APC / EMV decomposition (dual-time dynamics)

**What.** Decompose a vintage-level rate into three nonparametric components: maturation
(months-on-book), exogenous (calendar date), and vintage quality (origination date).
<cite index="31-1">Breeden's dual-time dynamics is a nonparametric exogenous-maturity-vintage decomposition indexed on two time scales, taking a form identical to Age-Period-Cohort models, where Age, Period and Cohort correspond to Maturity, Time and Vintage</cite>. <cite index="8-1">Additively, `r(v,a,t) = f_m(a) + f_g(t) + f_q(v)`, with `f_m` the maturation curve in months-on-books, `f_g` the exogenous curve in calendar date, and `f_q` the vintage quality curve</cite>.

**Pros.**
- The maturation component `f_m` is exactly the curve the hazard criterion wants, with
  calendar shocks stripped out. On COVID-spanning data this is the principled way to keep a
  macro period from being read as maturation.
- `f_q` quantifies vintage-quality drift directly — the composition question, answered
  structurally instead of pairwise.
- Standard in retail credit and recognised by validators; not an exotic choice.

**Risks.**
- **Identifiability.** Age, period and cohort are exactly collinear (`period = cohort + age`),
  so the three-way split is not unique without a constraint. <cite index="34-1">Identifiability conflict arises because time, maturity and vintage effects cannot be uniquely separated, which requires careful constraints to obtain meaningful parameter estimates</cite>. Different constraints give different-looking maturation curves; the constraint must be stated.
- Needs a full triangle (rate at every cohort × age cell), not four scenario points — a
  materially larger data-prep step.
- Additive/multiplicative forms assume no cohort-by-age interaction, which is exactly what a
  policy change or a pandemic-era cohort may violate.
- Explains the curve; still does not choose X. Feed `f_m` into the hazard criterion.

**How / verify.** Fit as a linear model on the long-format triangle with one constraint —
e.g. drop the first vintage level and treat it as the reference — and inspect the maturation
term:

```python
import numpy as np
import pandas as pd

def emv_design(long_df, rate_col="rate", age_col="age",
               period_col="period", vintage_col="vintage"):
    """Long-format triangle -> additive EMV design matrix with a reference-vintage constraint.

    Args:
        long_df: one row per (vintage, age) observed cell.
        rate_col, age_col, period_col, vintage_col: column names.

    Returns:
        (X, y) ready for a least-squares fit. Identifiability constraint: first vintage
        level dropped as reference -- state this alongside any reported decomposition.
    """
    d = long_df.copy()
    X = pd.concat([
        pd.get_dummies(d[age_col], prefix="age", drop_first=False),
        pd.get_dummies(d[period_col], prefix="per", drop_first=True),
        pd.get_dummies(d[vintage_col], prefix="vin", drop_first=True),
    ], axis=1).astype(float)
    return X, d[rate_col].to_numpy()
```

---

## Discrete-time survival and chain-ladder extrapolation

**What.** Two ways to stop treating X as a fixed cut. Discrete-time survival models the hazard
directly with covariates, yielding `P(default by m)` for any `m` off one fitted model —
<cite index="9-1">the standard approach for modelling the term-structure of default risk under IFRS 9</cite>, and extensible to APC structure with flexible learners [10]. Chain-ladder instead projects immature cohorts forward using age-to-age development factors estimated on matured cohorts.

**Pros.**
- Survival: censoring is handled natively (censored clients simply leave later risk sets) —
  no window choice, no discarded cohorts, and every candidate X becomes a read-off rather
  than a separate model.
- Chain-ladder: recovers the recency lost to censoring. The cohorts excluded by a long window
  can be projected rather than dropped, directly attacking the trade-off's cost side.
- Both make the fixed-window decision reversible instead of baked into the label.

**Risks.**
- Survival: a new production component with its own calibration burden. Long-horizon
  cumulative probabilities on thin risk sets can rank well and calibrate badly — check
  calibration per cohort before treating survival output as a target-rate substitute.
- Chain-ladder: development factors assume stable development patterns across cohorts (no
  cohort-by-age interaction). Projecting COVID-era originations with pre-COVID factors is
  exactly the assumption most likely to fail here.
- Chain-ladder introduces *modelled* labels. Training on projected outcomes couples label
  error to the projection's error — acceptable for portfolio forecasting, dangerous as a
  supervised target. Do not use projected cells as `y`.
- Both are the right answer to a different question than "what fixed X do I ship next
  quarter". Scope them as a follow-up, not a blocker.

**How / verify.** Person-period expansion turns the problem into ordinary binary
classification, which is the cheapest honest entry point:

```python
def person_period(pop, max_m, months_col="months_to_default",
                  mob_col="months_on_book", id_col="PartyId"):
    """Expand one row per client into one row per (client, month at risk).

    A client contributes months 1..min(event_month, months_on_book); the event row is
    flagged 1. Censored clients simply stop contributing -- no imputation.
    """
    out = []
    for r in pop.itertuples():
        t = getattr(r, months_col)
        last = int(min(t if pd.notna(t) else np.inf, getattr(r, mob_col), max_m))
        for m in range(1, last + 1):
            out.append({id_col: getattr(r, id_col), "month": m,
                        "event": int(pd.notna(t) and m == t)})
    return pd.DataFrame(out)
```

Fit any binary classifier on `event ~ month + t0 features`; `1 − Π(1 − p̂(m))` reconstructs
the cumulative curve. Verify against the life-table `conditional_hazard` output on the same
population before adding covariates.

---

## Roll-rate analysis  *(popular, and the wrong tool for this question)*

**What.** Transition rates between delinquency buckets (current → 30 → 60 → 90+ DPD) over a
fixed interval, used to decide which bucket is effectively absorbing. <cite index="13-1">It answers, with quantitative reasoning, whether 60, 90, 120 or higher DPD should identify bad customers: at a bucket where very few debtors roll back to lower buckets, the definition is stable, subject to acquiring enough bad cases</cite>.

**Pros.**
- The correct and standard way to fix the **bad definition**, which must be settled before any
  window analysis — the vintage curve's shape depends on it.
- Identifies **indeterminates**: accounts between the good and bad cuts that <cite index="18-1">roll-rate analysis shows are just as likely to cure as to roll</cite>, and which are conventionally excluded from the development sample rather than forced into either class.
- Cheap and interpretable; expected in documentation alongside the vintage exhibit.

**Risks / critique.**
- **It does not select the performance window.** Roll rates are computed over a fixed
  observation interval and answer "which DPD state is absorbing", not "how long must I watch a
  new client". The two are frequently merged under "target definition" and then one is used to
  justify the other. Fix the DPD cut first; then run the window analysis conditional on it.
- The result is sensitive to the interval chosen, so a roll-rate matrix computed at 12 MOB and
  one at 24 MOB can support different DPD cuts — circular if the window is what you are trying
  to decide.
- Regulatory floor: for EU institutions the definition of default is anchored, not free.
  <cite index="22-1">Under Article 178(1)(b) CRR a default occurs when the obligor is past due more than 90 (180) days on any material credit obligation, with both the absolute and relative components of the materiality threshold breached for 90 consecutive days</cite>. <cite index="20-1">The EBA amended these Guidelines in 2026, confirming that the 1% NPV-loss threshold for debt restructuring remains appropriate and extending the invoice-level past-due treatment from 30 to 90 days for factoring</cite>. An internal roll-rate-derived cut may be used for modelling but must be reconciled with the regulatory definition where the model feeds regulatory processes.

**How / verify.**

```python
def roll_rate_matrix(df, from_col="bucket_t0", to_col="bucket_t1"):
    """Transition shares between delinquency buckets over one fixed interval."""
    return (pd.crosstab(df[from_col], df[to_col], normalize="index")
              .reindex(index=["0", "1-29", "30-59", "60-89", "90+"],
                       columns=["0", "1-29", "30-59", "60-89", "90+"]))
```

---

## Problem-specific considerations

**Regulatory horizon vs. modelling horizon.** IRB capital rests on a one-year PD. A scorecard
with `X > 12` is a legitimate *ranking* model, but its raw output is not a one-year PD;
calibration to the regulatory horizon is a separate step. Decide up front whether the model
feeds capital — if it does, `X = 12` may be imposed regardless of what the hazard curve says,
and the maturity argument becomes an argument about calibration, not about the label.

**Client-level vs loan-level target.** With material repeat business, `t0` must be the
client's *global* first loan and the population must be built from the unfiltered loan history
— rebuilding it from a floor-filtered frame silently corrupts each client's `t0`. Keep the
loan-level frame for volume diagnostics and the client-level frame for the outcome, and label
every column with which population it came from.

**The candidate ladder is four different books, not one book at four horizons.** Because
`max_cohort` shrinks as X grows, comparing target rates across candidates compares different
client mixes. This is the entire reason the composition/seasoning gate exists — and the reason
the closed-cohort design is preferable: it removes the problem instead of measuring it.

**Bin resolution changes the conclusion.** A ladder of 12/18/24/36 gives at most five hazard
intervals, one of which spans a full year. That resolution can hide the early peak entirely
and leave only two or three points from which to estimate the plateau. Compute the hazard
monthly; use the candidate windows only as read-off points.

**Censoring robustness.** Any hazard computed on a shrinking-denominator population mixes
maturation with survivorship: later intervals are estimated on an older, smaller subset. Always
produce the closed-cohort variant alongside, and treat a plateau that disappears under cohort
closure as an artifact, not a finding.

**COVID.** Cohorts originated 2020-03 → 2021-06 sit inside most current European retail
samples and carry both a period effect (payment moratoria, income support) and a vintage-quality
effect (changed applicant mix). If a candidate window's newest cohorts fall inside that band,
the recency advantage it appears to offer may be negative. This is the case where the EMV
period term earns its cost.

**Event count, not client count, is the floor.** Check positives per candidate. On a portfolio
with a low base rate, the shortest windows can fail a minimum-event constraint before any
maturity argument applies.

---

## Diagnostics & pitfalls

- **Cumulative-curve flattening is not maturation.** The single most common error in this
  analysis. Always divide by survivors before reading a plateau.
- **Plateau credibility.** Report the coefficient of variation of the hazard over the assumed-flat
  tail. Above ~0.15, `h∞` is not estimated and no coverage number derived from it should be
  quoted.
- **Censoring is leakage.** Evaluating an immature population at a horizon it has not reached
  undercounts pending events. Treat the maturity assertion (`cohort_min_mob >= X`) with the same
  seriousness as a train/test leakage check, and assert it per pair, not once.
- **Split on client identity.** With repeat borrowers, a loan-keyed time split leaks the same
  client across train and test and inflates every candidate equally — which destroys the
  comparison rather than one side of it.
- **Fix the model configuration across labels.** Tuning per label confounds label quality with
  hyperparameter luck.
- **Pre-register `τ`.** Choosing the coverage threshold after seeing which X it selects converts
  the criterion back into the judgement call it was meant to replace.
- **Full precision until display.** Round only at the final table. Rounding intermediate
  composition/seasoning components before summing produces visible but spurious mismatches.
- **Never sum across decomposition families.** Adjacent-step and direct-pair splits are different
  quantities. Label every reported share with its family.
- **Report `n` and event counts beside every rate.** In a nested-population comparison, `B` and
  `C` share `n` and only `A` differs; without `n` the "population changed" story is invisible.
- **Do not train on projected labels.** Chain-ladder cells are forecasts. Using them as `y`
  couples label error to projection error.
- **Bad definition first.** Every curve in this report is conditional on the DPD cut. Changing it
  invalidates all of them.

---

## Decision rule

1. **Fix the bad definition first** (roll-rate + regulatory floor). If it is not settled, stop —
   nothing downstream is stable.
2. **Check the regulatory constraint.** Model feeds IRB capital → the horizon may be imposed;
   the remaining question is calibration, not label length.
3. **Build a closed cohort** matured to `max(candidates)`. Compute the monthly conditional hazard
   on it. This is the primary artifact.
4. **Is there an identifiable plateau?** Tail CV ≤ ~0.15 → yes; compute excess coverage per
   candidate and pick the smallest X clearing a pre-registered `τ`. No plateau → the data cannot
   support a maturity-based choice; fall through to step 6 and decide empirically.
5. **Audit the survivors symmetrically.** Report `FNR_label` against the matured reference for
   *every* candidate, not only the ones you intend to reject.
6. **If two adjacent candidates remain close** (coverage within a few points, FNR difference not
   decisive), run the label-quality experiment. Flat AUC and no mid-decile degradation → take the
   shorter window and keep the recency.
7. **Only then look at the full ladder** with its different populations, to size the training
   sample at the chosen X. Run the composition/seasoning gate here; if composition is material at
   any pair, the ladder cannot be read as a maturation curve and the closed-cohort answer stands
   alone.
8. **If recency loss is the binding cost**, scope discrete-time survival (term structure, no fixed
   window) or chain-ladder (project immature cohorts) as a follow-up — and do not train on
   projected labels in the meantime.
9. **Record the choice as an ADR** with: the plateau estimate, `τ`, the coverage table, the FNR
   table for all candidates, and the trigger condition for revisiting (materially changed hazard
   shape).

---

## References

1. Siddiqi, N. *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*
   (Wiley/SAS, 2006) — bad-rate maturation and sample/performance window definition.
   https://books.google.com/books/about/Intelligent_Credit_Scoring.html?id=q-qoDQAAQBAJ
2. "Credit Risk: Vintage Analysis" — ListenData. Cumulative bad-rate-vs-MOB construction and the
   stabilisation criterion for the performance window.
   https://www.listendata.com/2019/09/credit-risk-vintage-analysis.html
3. "Roll Rate Analysis" — ListenData. Bucket transitions and the DPD-cut decision.
   https://www.listendata.com/2019/09/roll-rate-analysis.html
4. "The very basics of scorecards" — How to Lend Money to Strangers. Good/bad/indeterminate
   definition and the role of roll rates in excluding indeterminates.
   https://www.howtolendmoneytostrangers.show/articles/the-very-basics-of-scorecards
5. "Bad definition and its impact on credit scorecards" — arXiv:1106.4513. Flattening of ever-DPD
   trend curves as the maturity signal. https://arxiv.org/pdf/1106.4513
6. Regulation (EU) No 575/2013 (CRR) Article 178 — regulatory definition of default.
   https://www.eba.europa.eu/regulation-and-policy/single-rulebook/interactive-single-rulebook/16022
7. EBA Guidelines on the application of the definition of default (EBA/GL/2016/07).
   https://www.eba.europa.eu/sites/default/files/documents/10180/1597103/004d3356-a9dc-49d1-aab1-3591f4d42cbb/Final%20Report%20on%20Guidelines%20on%20default%20definition%20(EBA-GL-2016-07).pdf
8. EBA Q&A 2019_4504 — counting past-due days against the materiality threshold.
   https://www.eba.europa.eu/single-rule-book-qa/qna/view/publicId/2019_4504
9. EBA, "The EBA amends Guidelines on the definition of default" (2026) — NPV threshold confirmed;
   invoice-level past-due extended to 90 days for factoring.
   https://www.eba.europa.eu/publications-and-media/press-releases/eba-amends-guidelines-definition-default
10. Forster, J. & Sudjianto, A. "Modelling time and vintage variability in retail credit portfolios:
    the decomposition approach" (EMV / APC, identifiability). https://arxiv.org/pdf/1305.2815
11. Breeden, J., Thomas, L. & McDonald, J. "Stress-testing retail loan portfolios with dual-time
    dynamics", *Journal of Risk Model Validation* 2(2).
    https://ideas.repec.org/a/pal/jorsoc/v61y2010i3d10.1057_jors.2009.105.html
12. "Approaches for modelling the term-structure of default risk under IFRS 9: a tutorial using
    discrete-time survival analysis" — arXiv:2507.15441. https://arxiv.org/pdf/2507.15441
13. "Discrete-Time Survival Models with Neural Networks for Age-Period-Cohort Analysis of Credit
    Risk" — *Risks* 12(2):31, MDPI. https://www.mdpi.com/2227-9091/12/2/31
14. "Using the Scikit-Learn Estimator Interface" — XGBoost documentation (stable). Confirms
    `early_stopping_rounds` / `eval_metric` are constructor parameters.
    https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
15. Internal — `docs/research-reports/rr-composition-seasoning-decomposition-censored-cohorts.md`
    (forced-order decomposition, path-dependence, Shapley non-applicability).
16. Internal — `notebooks/gen_target_definition.py` Parts A–I; `docs/adr/adr-2026-07-29-target-window-24m-ladder-revalidation.md`.
