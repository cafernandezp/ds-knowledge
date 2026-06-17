# Designing PRDs for Data Science & Machine Learning Projects
### A practical, opinionated guide using binary classification as the running example

## TL;DR
- A DS/ML PRD is **not** a software PRD with a model bolted on: because ML systems are data-dependent, probabilistic, and decay over time, the document must specify the *data contract, success metrics, validation strategy, and monitoring/retraining plan* up front — the absence of these is the single biggest reason most ML projects never reach or survive production.
- This report gives you a complete, copy-pasteable Markdown PRD template (12 sections), a canonical ML repo layout showing where the PRD lives (`docs/prd.md`) and how it links to the README, model card, data card, and experiment log, and a full worked example for a **customer-churn binary classifier**.
- It also shows how to use the PRD as the foundational context document for AI coding agents (Claude Code's `CLAUDE.md` + explore→plan→code→commit, Codex's `AGENTS.md`, and GitHub Spec Kit's spec→plan→tasks→implement), with the critical caveat that context files must stay short — a controlled ETH Zurich study found that auto-generated context files *reduce* agent success rates.

## Key Findings

1. **Problem framing is where ML projects die, not modeling.** Weak problem framing plus the prototype-to-production gap are recurring root causes of ML project failure (InfoQ, *Why Most Machine Learning Projects Fail to Reach Production*). A PRD's job is to force these decisions early, when changing them is cheap. (Note: the oft-cited "85% of AI projects fail" figure is frequently misquoted — see Caveats.)
2. **Metrics must be designed before the model exists.** Google's *Rules of Machine Learning* (Rule #2) explicitly says "make metrics design and implementation a priority." A DS/ML PRD must separate the *business KPI* (e.g., reduction in monthly churn, net revenue retained) from the *ML metric* (e.g., PR-AUC, recall at a fixed precision) and define the mapping between them.
3. **The data contract is the heart of the document.** Training–serving skew and data leakage are the dominant silent failure modes; a PRD must specify schema, point-in-time correctness, labeling strategy, and the train/serve feature parity requirement.
4. **The PRD belongs in the repo as a living, version-controlled document** (`docs/prd.md`), not in a wiki that rots. It is the parent document from which the model card, data card, experiment log, and README all descend.
5. **AI coding agents work dramatically better from a structured spec** — but the spec they consume must be decomposed and concise. GitHub Spec Kit's `/specify → /plan → /tasks → /implement` and Anthropic's explore→plan→code→commit are the two workflows to anchor on.

## Details

### 1. Context and motivation: why DS/ML PRDs are different

A traditional software PRD describes deterministic behavior: given input X, the system must produce output Y, and you can write an acceptance test that passes or fails. Machine learning breaks every part of that assumption. As Chip Huyen argues in *Designing Machine Learning Systems*, ML systems are "complex because they consist of many different components and involve many different stakeholders" and "unique because they're data dependent, with data varying wildly from one use case to the next." Three structural differences drive everything about how the PRD must change:

**(a) The behavior is learned from data, not specified in code.** You cannot write `assert predict(x) == y` for all x. The system's behavior is an emergent property of the training data, so the data itself becomes a first-class requirement. This is why the academic literature (Serban et al., *Adoption and Effects of Software Engineering Best Practices in Machine Learning*) distinguishes "traditional" practices, "modified" practices, and entirely "new" practices designed for ML — data versioning, training-serving skew tests, and model documentation have no analog in standard software.

**(b) The output is probabilistic and the success criterion is statistical.** A churn classifier does not "work" or "not work" — it has a precision-recall tradeoff that you tune to a business cost structure. The PRD must therefore encode *thresholds* and a *validation strategy*, not a binary pass/fail.

**(c) The system decays.** Traditional software is static until you change it; ML models exist in a state of continuous silent degradation as the world drifts away from the training distribution. The PRD must specify monitoring and a retraining plan as launch requirements, not afterthoughts.

**Common failure modes when the PRD is absent or weak:**

- **Solving the wrong problem.** Teams spend three months training a model that solves a question nobody asked, or that the business cannot act on. The InfoQ analysis of why ML projects fail to reach production names "weak problem framing" and late changes to business goals as primary culprits: "Late changes to business goals require adjustments to data, objective functions, and pipelines, which may result in the loss of work."
- **Metric mismatch.** Optimizing accuracy on an imbalanced churn dataset yields a model that predicts "no churn" for everyone and scores 80%+ while being useless. Without a metric specification, this passes unnoticed.
- **Data leakage / target leakage.** The classic Google example: a cancer-detection model learns "hospital name" as a feature because certain hospitals only treat cancer patients — a feature unavailable at true prediction time. A PRD's data section is where you catch this.
- **Training-serving skew.** Features computed one way in the training notebook (pandas, full history) and another way in production (a Java service, a truncated real-time window) silently diverge. Industry write-ups (Nubank, JFrog/Qwak, Google's ML crash course) cite this as one of the most common production failure modes, with feature mismatches frequently estimated to drive a large share of production ML issues.
- **The prototype-to-production gap.** A notebook that scores 0.79 AUC never becomes a service because nobody specified latency, throughput, explainability, or rollback requirements.

> **Opinion:** The DS/ML PRD's highest-value function is not documentation — it is *forcing function*. It makes the team commit to a falsifiable business hypothesis, a data contract, and a kill criterion before burning a quarter of GPU budget. If you only adopt one section from this guide, adopt the success-metrics section.

### 2. A concrete binary classification example: subscription churn prediction

We will use a single running example throughout: **"ChurnGuard," a customer-churn prediction model for a B2C subscription business (a meal-kit / streaming-style monthly subscription).**

**Business context.** The company has ~2M active monthly subscribers. Monthly logo churn is ~5.5%. The retention team can run a targeted intervention (a discount offer + concierge outreach) but the intervention has a cost (~$12 per contacted customer in discount + labor) and there is limited capacity (~40,000 outreach slots per month). The goal is to identify, each month, the subscribers most likely to cancel in the next 30 days so the retention team spends its fixed budget on the customers where intervention has the highest expected value.

**ML framing.** Binary classification. For each active subscriber at the start of a month, predict `P(churn within next 30 days)`. Positive class = churned. This is an imbalanced problem (~5.5% positive rate).

**Data landscape:**
- *Subscription/billing data* (warehouse): plan type, tenure, price, payment failures, contract type.
- *Product engagement events* (event stream → warehouse): logins, sessions, orders/streams, skips, support tickets.
- *Customer profile*: acquisition channel, geography, demographics (sensitive — handled carefully for fairness).
- *Labels*: derived from billing — did the subscriber cancel within the 30-day window? Labels mature with a 30-day lag, which is a key timing constraint.

**Success metrics.** Business KPI: net revenue retained per month and reduction in monthly churn rate among the contacted cohort, measured against a holdout control group. ML metric: because the team can only act on 40,000 customers, the operating metric is **precision and recall at top-K (K=40,000)** and **PR-AUC** as the threshold-independent summary. Accuracy is explicitly rejected as a metric.

**Stakeholders:** Retention/Marketing PM (owns the business KPI and outreach budget), Data Science (model), ML/Platform Engineering (pipeline + serving), Data Engineering (upstream tables), Legal/Privacy (use of demographic data), and Finance (cost model behind the $12 intervention and revenue figures).

This example is realistic. In a peer-reviewed study on the public Telco churn dataset (n=7,043; *Explainable AI-driven customer churn prediction*, PMC12929532), gradient boosting was the strongest family — "XGBoost attaining the best discriminative ability (AUC-ROC: 0.932)" with an F1 around 0.84 — and threshold tuning mattered: "Threshold optimization at 0.528 balanced precision (0.90) and recall (0.91) while reducing false negatives by 15%." This grounds the targets we set below (and is a reminder that the right operating threshold is rarely 0.5).

### 3. Model ML project repository structure

The layout below is a synthesis of the **Cookiecutter Data Science** standard (DrivenData), Goku Mohandas's **Made With ML**, and conventions from Chip Huyen's work. The guiding principle, in DrivenData's words, is "a logical, reasonably standardized, but flexible project structure."

```
churnguard/
├── README.md                  # Entry point: what/why, quickstart, links to all docs
├── pyproject.toml             # Package metadata + tool config (ruff, black, pytest)
├── Makefile                   # `make data`, `make train`, `make test`, `make serve`
├── .pre-commit-config.yaml    # Linters/formatters run before commit
├── .gitignore
├── CLAUDE.md                  # AI-agent context (symlinked to AGENTS.md) — see §6
├── AGENTS.md
│
├── docs/
│   ├── prd.md                 # ⭐ THE PRD — source of truth for the project
│   ├── model_card.md          # Model documentation (Google model card format)
│   ├── data_card.md           # Dataset documentation (Google data card format)
│   ├── experiment_log.md      # Append-only log of experiments + decisions
│   └── adr/                   # Architecture decision records
│
├── data/
│   ├── raw/                   # Immutable source extracts (never edited)
│   ├── interim/               # Intermediate transformed data
│   ├── processed/             # Final feature sets for modeling
│   └── external/              # Third-party data
│
├── notebooks/                 # EXPLORATION ONLY. Naming: 1.0-jdoe-eda-churn.ipynb
│
├── src/churnguard/
│   ├── config.py              # Config dataclasses / Hydra configs
│   ├── data/                  # make_dataset.py, validate.py (schema/quality checks)
│   ├── features/              # build_features.py — SHARED by train AND serve
│   ├── models/                # train.py, predict.py, evaluate.py
│   └── serve/                 # FastAPI app, batch_inference.py
│
├── configs/                   # YAML: model params, thresholds, feature lists
├── models/                    # Serialized model artifacts + metadata (gitignored/DVC)
├── tests/                     # Unit + data tests + train/serve parity tests
├── reports/figures/           # Generated evaluation plots
└── .github/workflows/         # CI/CD: lint, test, train, deploy
```

**Why the PRD lives in `docs/prd.md` inside the repo.** The PRD must be version-controlled alongside the code it governs, so that a `git blame` on a threshold change in `configs/` can be traced to a PRD revision, and so that a pull request can update code and the spec atomically. A PRD in Confluence or Google Docs rots the moment the code diverges from it; a PRD in the repo can be enforced in code review. This is the same logic behind GitHub Spec Kit's claim that "the PRD isn't a guide for implementation; it's the source that generates implementation."

**How the PRD connects to the other documents** — this is a parent→child hierarchy of decreasing abstraction and increasing specificity:

- **README.md** is the *front door*: a short orientation that links to the PRD and explains how to run things. The PRD answers *why and what*; the README answers *how do I start*.
- **PRD (`docs/prd.md`)** is the *contract*: what we're building, for whom, with what data, to what bar. Written largely before implementation; updated as decisions change.
- **Data card (`docs/data_card.md`)** is the *realized* description of the dataset that the PRD's data-requirements section *asked for*. Google's Data Cards Playbook defines these as "structured summaries of essential facts about various aspects of ML datasets needed by stakeholders." The PRD says "we need 24 months of billing history with these quality bars"; the data card documents what was actually assembled, including biases and known caveats.
- **Model card (`docs/model_card.md`)** is the *realized* description of the trained model that the PRD's model-requirements section asked for. Model cards were introduced by Google researchers (Mitchell et al., 2019) and document intended use, evaluation metrics sliced across groups, limitations, and ethical considerations. It is the post-hoc answer to the PRD's pre-hoc requirements.
- **Experiment log (`docs/experiment_log.md`)** is the *audit trail* connecting the two: each entry records a hypothesis (traceable to a PRD success metric), the configuration, the result, and the decision. This is what makes the difference between "a magic model from last April that no one can recreate" and a reproducible system.

The mental model: **PRD = specification (forward-looking) → experiment log = process (the journey) → model card + data card = realized documentation (backward-looking).**

### 4. The DS/ML PRD template (full Markdown)

Below is the complete template. Each section is preceded by a short note on *why it exists*. Copy this into `docs/prd.md` and fill it in. It draws on Eugene Yan's ML design-doc framework (the Why/What/How structure), Google's *Rules of ML*, and standard PRD conventions (overview, objectives, success metrics, scope).

````markdown
# PRD: [Project Name] — [One-line description]

| Field | Value |
|---|---|
| Status | Draft / In Review / Approved / In Production |
| Author(s) | |
| Reviewers | DS lead, ML eng, Product, Legal |
| Last updated | YYYY-MM-DD |
| Target launch | |

## 1. Project overview & business context
*Why this section exists: anchors the whole project to a business reason and a "why now." Prevents ML-for-ML's-sake.*
- **Problem statement:** What business problem are we solving? Who has it?
- **Why now / why ML:** Why is a heuristic insufficient? (Per Google Rule #1, justify ML over a simple baseline.)
- **Business hypothesis:** "We believe that [predicting X] will let us [take action Y] to achieve [outcome Z]."
- **Stakeholders & RACI:** Who owns the KPI, the model, the pipeline, the data, sign-off.

## 2. Problem framing (ML task definition)
*Why: the same business problem can be framed many ways; this fixes the framing so data and metrics follow.*
- **ML task type:** (e.g., binary classification.)
- **Unit of prediction:** What entity, at what point in time? (e.g., "each active subscriber, scored on the 1st of each month.")
- **Input (X):** Feature space at a high level + the **point-in-time** at which features are available.
- **Output (y):** Label definition and **prediction horizon** (e.g., "churn within 30 days").
- **Label definition & maturation lag:** Exactly how the label is computed and how long until it's known.
- **Baseline:** The non-ML heuristic we must beat (e.g., "contact lowest-engagement decile").

## 3. Success metrics
*Why: per Google Rule #2, design metrics first. Separates business value from model math and defines "done."*
- **Business KPI(s):** The metric the business cares about + how it's measured (ideally a controlled experiment vs. holdout).
- **ML metric(s):** The offline metric(s) optimized, chosen to reflect the operating point (e.g., precision@K, recall@K, PR-AUC). State why accuracy is/ isn't appropriate.
- **Operating point / threshold:** The decision threshold and the cost rationale (false-positive cost vs. false-negative cost).
- **Launch bar:** Minimum metric values required to ship.
- **Kill criteria:** Metric values below which we stop the project.
- **Validation strategy:** Data split (prefer **temporal** split for time-dependent data), cross-validation scheme, and the holdout/golden set.

## 4. Data requirements
*Why: the data contract is the core of an ML PRD. Most failures originate here.*
- **Sources & owners:** Each source table/stream, its owner, freshness, and SLA.
- **Schema:** Key fields, types, and the entity/timestamp keys for point-in-time joins.
- **Volume & history:** How much data, over what period.
- **Quality requirements:** Null-rate ceilings, allowed ranges, expected distributions, referential integrity. (These become automated data tests.)
- **Labeling strategy:** Where labels come from, label lag, label noise, and class balance.
- **Leakage controls:** Explicit list of features that must use **only** pre-prediction-time information; point-in-time correctness requirement.
- **Privacy/compliance:** PII handling, consent, retention, which sensitive attributes may/may not be used.

## 5. Feature engineering requirements
*Why: defines the feature contract and — critically — the train/serve parity requirement.*
- **Feature groups:** Categories of features and rationale.
- **Train/serve parity:** Requirement that training and serving compute features from the **same code path** (e.g., a shared transform module or feature store).
- **Freshness:** How current each feature must be at serving time.
- **Forbidden features:** Anything that leaks the label or is unavailable at inference.

## 6. Model requirements
*Why: constrains the solution space against real-world deployment limits.*
- **Algorithm constraints:** Allowed/preferred model families; start-simple mandate (Google Rule #4).
- **Latency / throughput:** p99 latency for online; batch window for batch.
- **Explainability:** Required level (e.g., per-prediction SHAP for retention agents; global feature importance for stakeholders).
- **Fairness:** Protected groups, the chosen fairness metric (e.g., equal opportunity / demographic-parity ratio), and the acceptable disparity bound (e.g., the four-fifths / 80% rule).
- **Calibration:** Whether probability calibration is required (important when probabilities drive expected-value decisions).

## 7. Training & evaluation pipeline requirements
*Why: makes reproducibility and evaluation rigor a requirement, not a hope.*
- **Reproducibility:** Versioning of code, data, and config; seeds; experiment tracking.
- **Evaluation protocol:** Offline metrics, **sliced** evaluation (by segment, tenure, geography), and robustness/sanity checks.
- **Retraining cadence (design):** How often models are retrained and on what data window.
- **Acceptance gate:** Automated checks a candidate model must pass before promotion.

## 8. Deployment & serving requirements
*Why: closes the prototype-to-production gap.*
- **Serving mode:** Batch vs. online vs. streaming; integration points.
- **Infrastructure:** Where it runs; scaling expectations.
- **Rollout strategy:** Shadow mode → canary → full; A/B vs. holdout control.
- **Rollback:** How to revert; the previous-model fallback.

## 9. Monitoring & maintenance plan
*Why: ML decays. Monitoring + retraining are launch requirements.*
- **Operational monitoring:** Latency, error rates, throughput.
- **Data/feature monitoring:** Drift detection (e.g., PSI thresholds), train/serve skew checks, null spikes.
- **Model-quality monitoring:** Live metric tracking once labels mature; prediction-distribution monitoring before labels arrive.
- **Retraining triggers:** Time-based and/or drift-based; who is paged and when.
- **Ownership:** On-call, dashboards, alerting hygiene.

## 10. Risks, assumptions & out-of-scope
*Why: surfaces blind spots and prevents scope creep.*
- **Assumptions:** What must be true (e.g., label lag stable, upstream tables maintained).
- **Risks & mitigations:** Data, model, ethical, and operational risks.
- **Out of scope (non-goals):** Explicitly list what this project will NOT do.

## 11. Milestones & timeline
- Staged plan; prefer stages ≤ 2–3 weeks, each delivering end-to-end utility.

## 12. Revision history
*Why: the PRD is a living document; track how requirements evolved and why.*
| Date | Author | Change | Reason |
|---|---|---|---|
````

#### Worked snippets for ChurnGuard (selected sections filled in)

To make the template concrete, here are the most decision-heavy sections completed for our example:

**§3 Success metrics (ChurnGuard):**
- Business KPI: ≥ 8% relative reduction in 30-day churn within the contacted cohort vs. a randomized holdout control, and positive net revenue retained after the $12/contact cost.
- ML metric: **precision@40K and recall@40K** (the team's monthly capacity), with **PR-AUC** as the primary threshold-independent summary. Accuracy is explicitly rejected because the base rate is ~5.5%.
- Launch bar: precision@40K ≥ 0.30 and recall@40K ≥ 0.25 on the temporal holdout (so that of 40K contacts, ≥12K are true churners, capturing ≥25% of all churners).
- Kill criterion: if PR-AUC ≤ the engagement-decile baseline + 0.03 after two iterations, stop.
- Validation: **temporal split** — train on months t-18…t-2, validate on t-1, test on t. No random splits (they leak future information).

**§6 Fairness (ChurnGuard):** Protected attribute = age band and geography. Metric = **equal opportunity** (equal true-positive rate across groups, so eligible churners are equally likely to receive a retention offer). Acceptable bound: TPR disparity within the **four-fifths (80%) rule** (the common operationalization in Fairlearn via the demographic-parity ratio). Demographic attributes are used for *auditing*, not as model features, pending Legal sign-off.

### 5. How the PRD maps to the ML workflow

Each PRD section is the *specification* for a phase of the ML lifecycle. The lifecycle is iterative (data ↔ model ↔ monitoring loops feed back), as Made With ML and ml-ops.org both stress, but the mapping is clean:

| ML lifecycle phase | Driven by PRD section(s) | What the PRD pre-commits |
|---|---|---|
| **Problem definition** | §1 Overview, §2 Framing | Business hypothesis, task type, baseline to beat |
| **Data collection** | §4 Data requirements | Sources, schema, quality bars, labeling, leakage controls |
| **Feature engineering** | §5 Feature requirements | Feature groups, train/serve parity, forbidden features |
| **Model training** | §6 Model req., §7 Pipeline | Algorithm constraints, reproducibility, retraining design |
| **Evaluation** | §3 Metrics, §7 Pipeline | Metrics, thresholds, sliced eval, validation split, acceptance gate |
| **Deployment** | §8 Serving | Serving mode, rollout, rollback |
| **Monitoring** | §9 Monitoring, §3 Metrics | Drift/skew checks, live metric tracking, retraining triggers |

The key insight is that **the metrics section (§3) appears twice** — at evaluation and at monitoring — because the same metric that gates the launch must be tracked in production. Similarly, the **data and feature sections (§4–§5) are the contract that both training and serving must honor**, which is precisely how you prevent training-serving skew: by writing the parity requirement down once and enforcing it in both code paths.

A practical sequencing rule, borrowed from Google's *Rules of ML* and Made With ML: build the **end-to-end pipeline with a dumb baseline first** (Rule #4: "Keep the first model simple and get the infrastructure right"), prove the plumbing and the metric instrumentation, then iterate on the model. The PRD's milestones section (§11) should reflect this — milestone 1 is "end-to-end with a logistic-regression or heuristic baseline," not "best possible model." Made With ML's design principles echo this: "every iteration should deliver minimum end-to-end utility," and "manual before ML."

### 6. Using the PRD with Claude Code, Codex, and Spec Kit

The PRD is the ideal foundational context document for AI coding agents — but you must use it correctly, because **dumping the whole PRD into the agent's always-on context file is counterproductive.**

#### The two-file pattern: persistent context vs. the spec

AI coding agents read a small **persistent context file** at the start of every session — `CLAUDE.md` for Claude Code, `AGENTS.md` for OpenAI Codex (now an open standard adopted across tens of thousands of repos). This file is loaded into *every* prompt, so it must be short. The PRD, by contrast, is a long reference document the agent should *read on demand*.

The correct pattern:
- Keep `CLAUDE.md` / `AGENTS.md` **short** — a lean orientation that points to the PRD rather than inlining it:
  ```markdown
  # CLAUDE.md
  This is ChurnGuard, a churn binary-classifier.
  The authoritative spec is docs/prd.md — read it before planning any task.
  Key rules:
  - Features MUST be computed via src/churnguard/features/ (shared train+serve path).
  - Use temporal splits only; never random splits.
  - Run `make test` (includes train/serve parity tests) before any commit.
  ```
- Reference the PRD explicitly so the agent fetches the relevant section when needed. (Claude Code supports `@import` syntax; a common multi-tool pattern is to symlink `CLAUDE.md` to `AGENTS.md` so both agents read one source of truth.)

**Why short matters — the evidence.** Anthropic's own guidance is that "CLAUDE.md is loaded every session, so only include things that apply broadly," and practitioner analysis (HumanLayer, *Writing a good CLAUDE.md*) estimates frontier thinking models follow only "~150–200 instructions with reasonable consistency" — with the sharp corollary that "Claude Code's system prompt contains ~50 individual instructions... nearly a third of the instructions your agent can reliably follow already," leaving roughly 100–150 usable slots; HumanLayer keeps its own root CLAUDE.md "under 60 lines." On the Codex side, the file is silently truncated past `project_doc_max_bytes` (32 KiB by default), and Codex concatenates files from the repo root downward, with deeper (closer) files overriding shallower ones.

Most importantly, a controlled study from ETH Zurich's SRI Lab (Gloaguen et al., *Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?*, arXiv:2602.11988, Feb 2026) found that **LLM-generated context files actually *reduce* task success rates** — "by 0.5% on SWE-BENCH LITE and 2% on AGENTBENCH" — "while increasing inference cost by over 20% on average," and adding "an average of 2.45 to 3.92" extra steps per task. Developer-written files gave only about a 4% gain at up to +19% cost. The study (run across 300 SWE-bench Lite tasks plus a 138-task AGENTbench) concludes that context files should "include only minimal requirements (e.g., specific tooling to use with this repository)." The practical lesson for ML teams: the context file should be minimal and *point to* the PRD; the PRD itself should be read deliberately during planning, not pasted into `CLAUDE.md`.

#### Decomposing the PRD into agent tasks

Three converging workflows tell you how to turn the PRD into executable agent work:

1. **GitHub Spec Kit** formalizes spec-driven development with the command chain `/specify → /plan → /tasks → /implement` (newer versions namespace as `/speckit.*`, with optional `/speckit.constitution`, `/speckit.clarify`, `/speckit.analyze` quality gates). It scaffolds `spec.md`, `plan.md`, and a `tasks/` folder, and — crucially — decomposes the plan into tasks with dependency ordering ("models before services, services before endpoints"), parallel-execution markers (`[P]`), exact file paths per task, and optional test-first structure. Your `docs/prd.md` *is* the `spec.md`; you run `/plan` and `/tasks` against it. Spec Kit's philosophy is the same as this guide's: "Specifications don't serve code—code serves specifications."

2. **Anthropic's Claude Code workflow** is **explore → plan → code → commit**: enter *plan mode* (Claude "reads files and answers questions without making changes"), have it produce a detailed implementation plan you can edit (Ctrl+G opens it in your editor), then switch to implementation, then commit. Anthropic's own rule for when to skip planning is memorable: "If you could describe the diff in one sentence, skip the plan." For ChurnGuard, you'd point Claude at `docs/prd.md` §4–§5 in plan mode and ask it to plan the data-validation and feature-build modules before writing any code. Anthropic stresses giving the agent a verifiable check — "tests, a build, a screenshot to compare" — which maps directly to the PRD's acceptance-gate and parity tests. (This explore-then-plan discipline matters: DataCamp's summary of Anthropic guidance notes that unguided attempts succeed only about a third of the time, which is why separating exploration from execution is the headline recommendation.)

3. **Amazon Kiro** (AWS's spec-driven agentic IDE) uses a three-document spec workflow — `requirements.md` (in EARS notation), `design.md`, and `tasks.md` — and lets developers trigger tasks "one step at a time." This mirrors the PRD → plan → tasks decomposition and shows the pattern is now an industry convention, not a single vendor's idea.

A concrete decomposition for ChurnGuard, derived section-by-section from the PRD:
- **Task 1 (from §4):** Implement `src/churnguard/data/validate.py` enforcing the schema and quality bars; write tests. `[depends on: none]`
- **Task 2 (from §5):** Implement `src/churnguard/features/build_features.py` as the single shared transform; write a train/serve parity test. `[depends on: 1]`
- **Task 3 (from §3, §7):** Implement temporal-split training + `evaluate.py` computing precision@40K, recall@40K, PR-AUC, sliced by age/geography. `[depends on: 2]`
- **Task 4 (from §8):** Wrap the model in a FastAPI service + batch inference, calling the *same* feature module. `[depends on: 2,3]`
- **Task 5 (from §9):** Add drift (PSI) and skew monitoring + alerts. `[depends on: 4]`

Tasks 1–5 respect the dependency ordering Spec Kit recommends, and each maps to a falsifiable acceptance criterion already written in the PRD.

#### How the PRD evolves with the project

The PRD is a **living document**, governed by its revision-history section. As experiments resolve open questions (e.g., the temporal split reveals label lag is actually 45 days, not 30), you update the PRD in the same pull request that changes the code, bump the revision table, and let the agent re-read the updated spec on its next planning pass. This keeps the spec and the implementation in lockstep — the entire point of keeping the PRD in the repo. In Spec Kit / Kiro terms, you re-run `/plan` or regenerate `tasks.md` from the amended spec; in Claude Code terms, you re-enter plan mode against the updated section.

## Recommendations

**Stage 1 — Before any modeling (week 0–1).** Write `docs/prd.md` using the §1–§3 sections only, and get sign-off on the **business hypothesis, the baseline to beat, and the success metric with its threshold and kill criterion.** Do not proceed until the PM and Finance agree on the cost model (the $12/contact and the revenue-retained definition for ChurnGuard). *Benchmark that changes the plan:* if you cannot define a business KPI measurable via a controlled holdout, stop — this is not yet an ML project.

**Stage 2 — Data contract (week 1–2).** Complete §4–§5. Stand up the data card and automated data-quality tests *before* feature engineering. Implement the shared feature module and a train/serve parity test on day one. *Benchmark:* if leakage controls or point-in-time joins can't be guaranteed from the available tables, escalate to Data Engineering before modeling — leakage discovered post-launch invalidates everything.

**Stage 3 — End-to-end baseline (week 2–3).** Per Google Rule #4, ship the full pipeline (data → features → logistic-regression baseline → evaluation → shadow-mode serving) before optimizing the model. Populate the experiment log. *Benchmark:* if the baseline already clears the launch bar, you may not need a complex model — ship it.

**Stage 4 — Iterate the model (week 3+).** Move to gradient-boosted trees (XGBoost/LightGBM are the empirically strong choice for tabular churn), tune the threshold to the precision@K operating point (recall the Telco study's ~0.53 optimum, not 0.5), run sliced + fairness evaluation, and produce the model card. *Benchmark:* promote only if the acceptance gate (§7) and fairness bound (§6) pass.

**Stage 5 — Deploy + monitor (ongoing).** Roll out shadow → canary → full with a holdout control to measure the *business* KPI, not just the ML metric. Wire up drift/skew monitoring and retraining triggers from §9. *Benchmark that changes the plan:* a PSI above ~0.2 on a critical feature or a sustained live-metric drop triggers retraining or rollback.

**For AI-agent-driven builds:** keep `CLAUDE.md`/`AGENTS.md` short (well under a couple hundred lines, ideally pointing to the PRD rather than inlining it); decompose the PRD with Spec Kit's `/plan` + `/tasks` (or Claude's plan mode) into dependency-ordered tasks with explicit file paths and the PRD's acceptance criteria as the agent's verifiable checks.

## Caveats

- **The "85% of AI projects fail" statistic is widely repeated but is usually misquoted.** Gartner's original 2018 forecast was that "through 2022, 85% of AI projects will deliver erroneous outcomes due to bias in data, algorithms, or the teams managing them" — a prediction about *erroneous outcomes*, not wholesale project failure. A separate Gartner figure holds that only ~53% of AI prototypes reach production. Use these to motivate rigor, not as precise measured benchmarks; the trustworthy, well-evidenced claim is that **weak problem framing and the prototype-to-production gap are leading causes of ML project failure** (InfoQ).
- **The specific ChurnGuard numbers (2M subscribers, 5.5% churn, $12/contact, 40K capacity) are illustrative**, constructed to make the metric and cost tradeoffs concrete; they are not drawn from a single real company. The XGBoost/LightGBM performance figures (AUC-ROC 0.932, F1 ~0.84, threshold 0.528) are from one published Telco-churn study (PMC12929532) and will not transfer directly to a different dataset.
- **There is no single canonical ML PRD template.** Eugene Yan explicitly warns that "there's no perfect template" and that templates followed blindly lead to "fill-in-the-blanks" thinking. Treat the 12-section structure as a checklist to provoke thinking, prune sections that don't apply, and add ones that do.
- **AI-agent tooling is moving fast.** Command names (Spec Kit's `/speckit.*` namespacing), default limits (Codex's 32 KiB `project_doc_max_bytes`), and model versions change frequently; verify against current docs before relying on exact syntax. The ETH Zurich finding that context files can *reduce* agent performance is from an early (Feb 2026) controlled study with a strong corroborating practitioner consensus, but the field is young and results may evolve.
- **Fairness metrics are mutually incompatible in general.** You usually cannot satisfy demographic parity and equalized odds simultaneously (a well-established impossibility result); the PRD must pick the metric that matches the harm being prevented and document the tradeoff, rather than claiming the model is "fair" in the abstract.