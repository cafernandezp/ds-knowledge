# DS Presentation Skeleton — Customer Decks & Technical Interviews

## Context

> Reusable slide skeleton for data-science presentations, stress-tested against two of the user's own past decks, prepared ahead of an upcoming data-science technical test.

- **Sources analyzed:** *BBVA Challenge* (May 2024) — supervised-modeling case, XGBoost classification (contract default) + XGBoost regression (salary estimation), metrics-heavy. *Evalueserve* (Feb 2020) — descriptive-segmentation case, no model, cross-tab analysis → targeted campaign recommendations.
- **Driver:** an upcoming DS technical test, format not yet known.
- **Assumption:** since the format is unknown, the skeleton below is built to flex between a predictive-modeling test and a descriptive/segmentation-analytics test rather than assuming one.
- **Scope:** presentation structure and delivery technique. Underlying model correctness in the source decks is out of scope except for two flagged inconsistencies (see *Diagnostics & Pitfalls*).

## TL;DR

- One 15-row master skeleton covers both customer decks and interviews — toggle depth per row, don't build two decks.
- The BBVA deck is a literal precedent, not just a reference: it already carries a graded "must" / "nice to have" Q&A annex — the same format likely to repeat.
- Single highest-leverage addition: a **literal Q&A annex** — verbatim brief questions, answered directly, separate from the narrative.
- Every results row needs an explicit **baseline/heuristic comparison** — a metric alone convinces no one.
- Business-facing rows read better as **Vision → Actual state → Gap → Segmented action → Next steps** than a generic recommendation slide.
- Once the real test lands, fill in and paste the prompt at the end of this report to generate the deck directly.

## Generic Skeleton

| # | Section | Customer-facing depth | Interview depth |
|---|---|---|---|
| 1 | Title | show | show |
| 2 | Agenda | show | show |
| 3 | Business context & objective | Full, plain language | Brief — 1 slide |
| 4 | Data overview + lineage diagram | Summary only | Full, incl. quality/leakage checks |
| 5 | EDA → insight bullets | Key insights only | Full — nulls, distributions, quality flags |
| 6 | Methodology, incl. decisions NOT taken + why | High-level rationale only | Full — alternatives rejected and why |
| 7 | Model detail (objective/loss, hyperparameters) | Skip or 1 slide | Full — this is the core of the talk |
| 8 | Results, incl. explicit baseline/heuristic comparison | Business-framed ($, impact) | Full metrics + calibration |
| 9 | Interpretability (feature importance / SHAP) | 2–3 key drivers | Full — expect the deepest probing here |
| 10 | Validation & robustness (OOT, CV spread, split consistency) | Light — "we tested for X" | Full — signals rigor |
| 11 | Business recommendation: Vision → Gap → segmented action table → next steps | Full — ROI, thresholds | Brief |
| 12 | Future ideas (new data sources, alternate models) | Optional | Optional |
| 13 | Monitoring plan (PSI, standard 0.10 / 0.25 thresholds) | Light | Full |
| 14 | Technical appendix (full hyperparameters, extra diagnostics) | Not shown | Keep ready — pull up on demand |
| 15 | Q&A annex (verbatim brief questions, answered directly) | N/A | **Mandatory whenever the brief poses explicit questions** |

**Why.**
- Same funnel — context → data → method → results → validation → impact — works for both audiences; only depth shifts per row, not the structure. One deck to maintain, not two.
- Rows 7, 9, 10 are where a panel probes deepest (methodology choices, interpretability, robustness) — thin for customers, loaded for interviews.
- Row 14 is non-negotiable for interviews: the buffer for "why not X instead" without cluttering the main narrative.
- Row 15 exists because the BBVA deck shows this exact test format already grades against explicit questions — treat it as mandatory insurance, not a nice-to-have.

**How to apply.**
- Build once as a master deck with all 15 rows; maintain a customer / interview visibility toggle (hide/show the appendix, adjust depth on rows 7/9/10).
- Pre-load row 14 for interviews with: loss-function derivation, confusion matrix / PR curve, feature-importance plot, one failure-mode example — these get asked most.
- The moment the brief arrives, copy every literal question into row 15 before building anything else; build the rest of the deck around making sure each one is covered somewhere.

## Reference Deck Analysis

### BBVA Challenge (May 2024) — supervised-modeling precedent
Contract-default classifier + salary-estimation regressor, both XGBoost, with an explicit comparison against three existing risk scores and a graded Q&A annex.

- Literal Q&A annex mapped to the test's own "must" / "nice to have" questions.
- Every results slide paired with a baseline comparison (new model's AUC/KS next to the existing scores').
- Methodology slides document what was deliberately **not** done, and why (e.g. no outlier/null treatment because the model is tree-based).
- Target rate held constant (8.12%) across every split in the classification case — confirms stratified sampling.

### Evalueserve (Feb 2020) — descriptive-segmentation precedent
No model — cross-tab / cohort-migration analysis translated into a targeted-campaign recommendation.

- Data-lineage diagram in the backup: raw table → join → clean → final, with row/column counts at each stage.
- Business narrative arc: Vision (quoted) → Actual state (data) → Gap (0%-growth categories flagged) → Segmented action table (rule / message / addressable N) → stakeholder next steps.
- Delivery technique: the same chart reused across 2–3 slides, annotations layered progressively (color deltas → highlight boxes → numbered callouts) instead of shown all at once.

### Upgrades to fold in

| Upgrade | Source | Skeleton row | Why it matters |
|---|---|---|---|
| Literal Q&A annex | BBVA | 15 | Insurance against a rubric item buried in the brief |
| Baseline/heuristic comparison | BBVA | 8 | Reframes metrics as value added, not abstract numbers |
| "Decisions not taken" documentation | BBVA | 6 | Reads as a controlled choice, not an omission |
| Split-rate / target-mean consistency check | BBVA | 10 | Cheap, concrete rigor signal |
| Cost-based metric framing | BBVA | 8 | Stronger than "maximize recall" when the brief mentions cost |
| Data-lineage diagram | Evalueserve | 4 | Communicates ETL rigor in one glance |
| Vision → Gap → Action arc | Evalueserve | 11 | Sharper than a generic recommendation slide |
| Progressive chart-annotation delivery | Evalueserve | — (delivery technique) | Walks a live panel through complexity without losing them |

## Problem-Specific Considerations

- **Predictive-modeling test (BBVA-style):** lead with rows 5–10; have row 15 ready from the moment the brief arrives; use cost-based framing (not recall alone) if the brief mentions cost or business impact anywhere.
- **Descriptive/segmentation test (Evalueserve-style):** lead with row 11's Vision→Gap→Action arc; use the progressive chart-annotation technique for the core visual; skip rows 5, 7, 9 — there's no model to detail.
- **Format unknown until the brief arrives:** prepare all 15 rows, decide emphasis only after reading the literal questions. Row 15 is the only row that's non-negotiable regardless of which mode it turns out to be.

## Diagnostics & Pitfalls

- Bare chart with no interpreted bullets underneath — every analytical slide needs 2–4 bullets translating the chart into a decision.
- Metric reported with no baseline — "AUC 75%" says nothing to a business audience or a panel without a reference point next to it.
- Reusing old slide text as boilerplate without checking it: the BBVA salary-model slide still reads *"Variable objetivo binaria: TOTAL_INCOME"* — leftover from the classification-slide template; TOTAL_INCOME is continuous, not binary.
- PSI alert thresholds as written in the BBVA monitoring slide (≥10% yellow, ≥100% red) are non-standard against the commonly cited convention (<0.10 stable, 0.10–0.25 moderate shift, >0.25 significant shift) — align before reuse.
- No explanation ready for legitimate cross-split variation: a continuous target's mean can (and did, 2011→2023 EUR) drift across splits, since continuous targets aren't typically stratified the way a binary class rate is. Expected — but have the explanation on hand if asked.
- Treating the appendix as optional for an interview — it's the buffer against "why not X instead," not a nice-to-have.

## Decision Rule

1. Brief contains explicit graded questions (must / nice-to-have or similar) → build the Q&A annex (row 15) first, before the narrative.
2. Brief mentions cost, budget, or business impact anywhere → frame the metric/threshold choice around expected cost, not recall or precision alone.
3. Case includes a trained model → lead with rows 5–10; that's where a panel probes.
4. Case is descriptive/segmentation only (no model) → lead with row 11; skip rows 5, 7, 9.
5. Reusing any slide from a past deck → run it through *Diagnostics & Pitfalls* before reuse.

## The Prompt Recommendation

Fill in the brackets once the real test lands, paste into a fresh chat:

```
ROLE: Senior DS presentation architect. Build my technical-interview deck (.pptx) from the case below.

TEST TYPE: [predictive modeling / descriptive-segmentation analytics / unclear — prep modular]

CASE BRIEF
- Business context: [paste]
- Literal questions from the brief, verbatim (keep must/nice-to-have tags if given):
  1. ...
  2. ...

DATA
- Source(s), size, target definition, base rate/distribution: [...]
- Known data-quality issues (nulls, gaps, staleness): [...]

MY WORK (paste from notebook / rr- report)
- EDA findings: [...]
- Feature engineering + what I deliberately did NOT treat and why: [...]
- Models compared, final choice, why: [...]
- Metrics train/val/test, vs. baseline if one exists: [...]
- Robustness checks run (OOT, CV spread, split-rate consistency): [...]

BUILD WITH THIS STRUCTURE
1. Title
2. Agenda
3. Business context & objective
4. Data overview + lineage diagram (rows/cols per stage)
5. EDA → insight bullets, never a raw stats dump
6. Methodology, incl. decisions NOT taken + why
7. Model detail (objective/loss, hyperparameters)
8. Results, incl. explicit baseline/heuristic comparison
9. Interpretability (feature importance / SHAP)
10. Validation & robustness (OOT, CV spread, split consistency)
11. Business recommendation: Vision → Gap → segmented action table (rule / message / addressable N) → next steps
12. Future ideas (new data sources, alternate models)
13. Monitoring plan (PSI, standard 0.10/0.25 thresholds unless told otherwise)
14. Technical appendix (full hyperparameters, extra diagnostics)
15. Q&A ANNEX (mandatory): every literal question above, verbatim, answered directly in 2–4 sentences

RULES
- Every analytical slide = chart/table + 2–4 interpreted bullets, never a bare chart
- Every content slide opens with a one-line "Objective"
- If cost is mentioned anywhere in the brief, frame metric choice around expected cost, not recall/precision alone
- If dual depth is needed (customer vs. interview version), flag depth per slide — don't build two decks
- No filler; don't restate the brief anywhere except the Q&A annex

Before export: confirm every literal question has a direct answer somewhere, every chart has interpretation attached, every metric comes from an actual run.
```

## References

1. *BBVA Challenge* — internal case-study presentation, Cristián Fernández Pugin, May 2024 (user-provided).
2. *"Customers Financial Needs and Company's Vision"* — Evalueserve case study, Cristián Andrés Fernández Pugin, Feb 20, 2020 (user-provided).
