# MLflow vs CML — Experiment Tracking/Registry vs CI/CD for ML, on GitHub

> **Problem.** Decide how MLflow and CML (Continuous Machine Learning, `iterative/cml`) differ, whether they compete, and how to combine them on **GitHub** (GitHub Actions).
>
> **Assumptions (stated, proceeding):**
> - Tabular binary classifier (scikit-learn/XGBoost-style), metric = AUC + calibration (Brier); time-based (out-of-time) validation available.
> - Cloud = AWS (S3 for artifacts, EC2 for compute); training data is **sensitive/regulated** (no public hosting of outputs).
> - Heavy training may run on a Spark/PySpark cluster, not on a CI runner.
> - Team of >1 person; PR-based review is the control point for model changes.
>
> **Constraints.** Only claims verified against official docs on 2026-09-30 are asserted; everything else is marked as reasoning. Versions: MLflow 3.x docs, `iterative/setup-cml` v2, CML v0.20.3 (latest release I could verify).

---

## TL;DR / Recommendation

- **They are different layers, not substitutes.** MLflow = *system of record* (runs, metrics, artifacts, model registry). CML = *CI/CD glue* (run on git events, provision runners, post a report to the PR) [1][2][11].
- **Default architecture on GitHub:** GitHub Actions runs `train.py` → logs to a **shared MLflow server** → CML (or plain Actions) posts a **metrics diff vs. the current champion** to the PR → on merge to `main`, register the model and move the `champion` **alias** [11][12].
- **Use CML only for what it adds:** PR reports (`cml comment`), and on-demand cloud/self-hosted runners (`cml runner launch`). If you only need a summary, `$GITHUB_STEP_SUMMARY` is native and needs no extra tool [10].
- **Regulated data: change CML's default.** `cml comment create` uploads local images to `asset.cml.dev` by default (`--publish=true`). Use `--publish=false` or a self-hosted `--publish-url` [4][5].
- **Do not** use MLflow as CI (it has no trigger/runner concept) and do not use CML as a registry (it has no run database or model lifecycle) [2][11].
- **Maintenance caveat for CML:** upstream README examples still reference `setup-cml@v1` and Ubuntu 20.04 images, while `setup-cml` documents `v2`; TensorBoard support is deprecated. Pin versions and test [2][3][8].

---

## Comparison table

| Option | Primary role | State/metadata | Compute provisioning | Trigger | Output/UX | Extra infra | Main risk | Use when |
|---|---|---|---|---|---|---|---|---|
| **MLflow only** | Tracking + registry (+ serving) | Backend store (SQL) + artifact store [13][14] | None (runs where code runs) | Manual/script | UI, run comparison, model versions/aliases [11][12] | Tracking server (DB + S3) for teams | No automated gate on PRs | Exploration; central model catalog |
| **CML only** | CI/CD for ML | Git (+ DVC for data/models) [1][2] | `cml runner launch` (AWS/Azure/GCP/K8s, spot) [2][6] | Git events | Markdown report on PR/commit [4] | None beyond GitHub + cloud creds | No run history/registry; public image hosting default [4] | Small team, Git-centric, DVC already used |
| **MLflow + CML** | Registry + PR automation | MLflow (runs/models) + Git (code/config) | CML runner or existing runner | PR/push | PR comment linking MLflow run | MLflow server + GitHub Actions | Two tools to secure/upgrade | Team needs gated promotion with audit trail |
| **MLflow + plain Actions (no CML)** | Registry + native CI | MLflow | Your own runner setup | PR/push | Job summary or custom PR comment [10] | MLflow server | You write the report glue | CML maintenance/hosting concerns |

---

## MLflow — tracking, registry, serving

**What.** Open-source platform for experiment tracking, artifact/model storage, and a **Model Registry** (centralized store, versions, aliases, tags, lineage to the producing run) [11][12]. A tracking server has a **backend store** (metadata) and an **artifact store** [13][14].

**Pros.**
- Run database: params, metrics, tags, artifacts; UI for comparison [13].
- Registry with **aliases** (mutable named pointer, e.g. `models:/Name@champion`) — promotion and rollback = reassign alias [11][12][17].
- Server modes: proxied artifact access (`--artifacts-destination`) or direct client-to-store (`--no-serve-artifacts --default-artifact-root`) [13][19].
- Built-in basic-auth app with per-experiment/model permissions (`mlflow server --app-name basic-auth`) [15].

**Risks.**
- SQLite is the default backend when nothing is configured; use PostgreSQL/MySQL for teams. A self-run server needs a **database-backed** backend store to use the registry [12][14].
- No triggers, runners, or PR feedback — it records, it does not orchestrate.
- Direct artifact mode requires each client (including CI) to have credentials to the artifact store [13].
- API drift: in MLflow 3, `log_model` takes `name` (not `artifact_path`); model artifacts are no longer stored as run artifacts [16]. Pin versions.
- Basic auth needs a Flask secret key (`MLFLOW_FLASK_SERVER_SECRET_KEY`) and an admin password env var on first start [15].

**How / verify.**
```python
# Leakage-safe: scaler inside Pipeline, fit on train only; gate on val; test untouched.
import mlflow, mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=20000, weights=[0.9], random_state=42)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

mlflow.set_experiment("demo-classifier")           # MLFLOW_TRACKING_URI from env
for C in (0.1, 1.0, 10.0):
    with mlflow.start_run(run_name=f"logreg-C{C}"):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=C, max_iter=1000, random_state=42)).fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        mlflow.log_params({"C": C, "seed": 42})
        mlflow.log_metrics({"auc_val": roc_auc_score(y_va, p),
                            "brier_val": brier_score_loss(y_va, p)})
        info = mlflow.sklearn.log_model(pipe, name="model")   # MLflow 3 API [16]

# Promote: register + move alias (models:/demo-classifier@champion) [12][17][18]
from mlflow import MlflowClient
mv = mlflow.register_model(info.model_uri, "demo-classifier")
MlflowClient().set_registered_model_alias("demo-classifier", "champion", mv.version)
```
Verify: run `mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1` locally, set `MLFLOW_TRACKING_URI`, confirm runs appear and `mlflow.sklearn.load_model("models:/demo-classifier@champion")` loads.

---

## CML — CI/CD for ML

**What.** Open-source CLI (Apache-2.0) for CI/CD in ML projects: automate training/evaluation, compare experiments across project history, provision machines, and post Markdown reports to PRs. Principles: GitFlow for data science, auto reports per PR, "no additional services" beyond GitHub/GitLab/Bitbucket and a cloud [1][2].

**Pros.**
- `cml comment create report.md` posts a GitHub-flavored Markdown report (tables, images) to the PR [2][4].
- `cml runner launch --cloud={aws,azure,gcp,kubernetes}` provisions a self-hosted runner (spot supported); `--idle-timeout`, `--single`, `--reuse` control lifecycle; it can restart jobs after spot interruption or workflow timeout [2][6][7].
- `iterative/setup-cml@v2` installs CML from pre-packaged binaries on `ubuntu`/`macos`/`windows` runners; the default `GITHUB_TOKEN` suffices for most functions [3].
- Pairs with DVC (`dvc pull`, `dvc repro`, `dvc metrics diff main --show-md`, `dvc plots diff`) to bring data and diff metrics between commits [2].

**Risks.**
- **Data egress by default:** `--publish` defaults to `true`; local images referenced in the report are uploaded to `https://asset.cml.dev` unless `--publish-url` (self-hosted, e.g. minroud-s3) is set. `--publish-native` is **not available on GitHub** [4][5].
- **PAT required** for `cml runner launch` (register runners) and for `cml comment update`; the default token fails with a `commit_id has been locked` error on update [3][4][7].
- Repeated `cml comment create` produces many comments; use `update` with `--watermark-title` (needs the PAT) [4].
- Documentation drift: README examples use `setup-cml@v1`, `actions/checkout@v3`, and Ubuntu 20.04/Python 3.8 base images, while `setup-cml` documents `v2` [2][3].
- `cml tensorboard` is deprecated (exits 1 with a deprecation notice from v0.20.x) — avoid [8].
- No run database, no model registry, no serving: comparisons across history rely on Git/DVC state [2].

**How / verify.**
```yaml
# .github/workflows/cml.yaml — minimal PR report (no image upload)
name: cml-report
on: [pull_request]
permissions: {contents: read, pull-requests: write}   # verify minimal scopes for your org [3]
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - uses: iterative/setup-cml@v2
      - run: pip install -r requirements.txt && python train.py   # writes metrics.md
      - env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          echo "## Metrics" > report.md
          cat metrics.md >> report.md
          cml comment create --publish=false report.md   # no asset.cml.dev upload [4]
```
```yaml
# On-demand GPU runner (PAT + AWS creds required) — adapted from upstream docs [2][6][7]
jobs:
  launch-runner:
    runs-on: ubuntu-latest
    steps:
      - uses: iterative/setup-cml@v2
      - uses: actions/checkout@v4
      - env:
          REPO_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          cml runner launch --cloud=aws --cloud-region=us-west \
            --cloud-type=g4dn.xlarge --cloud-spot --idle-timeout=5min --labels=cml-gpu
  train:
    needs: launch-runner
    runs-on: [self-hosted, cml-gpu]
    steps:
      - uses: actions/checkout@v4
      - run: python train.py
```
Verify: open a PR from a branch that changes a hyperparameter; confirm the bot comment appears and that no request goes to `asset.cml.dev` (network egress logs) when `--publish=false`.

**Critique.** CML is popular in DVC-centric stacks, but for a team that already runs MLflow on GitHub Actions it mostly adds (a) PR comment formatting and (b) runner provisioning. If you only need (a), native job summaries or a custom PR comment are simpler and remove a dependency with uneven maintenance and a public-hosting default [4][8][10].

---

## MLflow + CML on GitHub *(recommended pattern)*

**What.** MLflow holds runs and the registry; GitHub Actions/CML executes and reports. Git holds code and config; MLflow tags each run with the commit SHA for lineage.

```
PR opened ──► Actions runner ──► train.py ──► MLflow server (runs, artifacts)
                    │                              │
                    │      champion re-scored on the SAME gate set
                    ▼                              ▼
             cml comment (metrics diff + MLflow run link) ──► PR review
merge to main ──► train.py --promote ──► register model + set alias "champion"
```

**Pros.**
- Audit trail: PR ↔ commit SHA ↔ MLflow run ↔ model version.
- Promotion and rollback are alias moves, no redeploy of code [12][17].
- Apples-to-apples gate: candidate and champion are scored on the identical held-out set in the same job (avoids comparing metrics computed on different splits).

**Risks.**
- The runner must reach the MLflow server; a private server implies a self-hosted runner inside the network (reasoning).
- Two credential sets in CI (MLflow, cloud) — least privilege, no secrets in reports.
- Gate set reuse: repeated PR iterations against the same gate set leak into model selection (see Diagnostics).

**How / verify.**
```python
# train.py — gate vs champion; report to Markdown; optional promotion (procedural).
import os, sys, mlflow, mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME, TOL = "demo-classifier", 0.005          # tolerated AUC regression
promote = "--promote" in sys.argv

X, y = make_classification(n_samples=20000, weights=[0.9], random_state=42)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_gate, X_test, y_gate, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                                  stratify=y_tmp, random_state=42)  # test unused in CI

def score(m):
    p = m.predict_proba(X_gate)[:, 1]
    return roc_auc_score(y_gate, p), brier_score_loss(y_gate, p)

mlflow.set_experiment(NAME)
with mlflow.start_run() as run:
    mlflow.set_tag("git_sha", os.environ.get("GITHUB_SHA", "local"))
    cand = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=42)).fit(X_tr, y_tr)
    auc, brier = score(cand)
    mlflow.log_metrics({"auc_gate": auc, "brier_gate": brier})
    info = mlflow.sklearn.log_model(cand, name="model")

    try:                                          # champion may not exist yet
        champ = mlflow.sklearn.load_model(f"models:/{NAME}@champion")
        c_auc, c_brier = score(champ)
    except Exception:
        c_auc, c_brier = None, None

    ok = c_auc is None or auc >= c_auc - TOL
    with open("report.md", "w") as f:
        f.write("## Candidate vs champion (gate set)\n\n| metric | candidate | champion |\n|---|---|---|\n")
        f.write(f"| AUC | {auc:.4f} | {c_auc if c_auc is None else round(c_auc, 4)} |\n")
        f.write(f"| Brier | {brier:.4f} | {c_brier if c_brier is None else round(c_brier, 4)} |\n")
        f.write(f"\nGate: {'PASS' if ok else 'FAIL'} (tol={TOL})  \nMLflow run: `{run.info.run_id}`\n")

    if ok and promote:
        mv = mlflow.register_model(info.model_uri, NAME)
        MlflowClient().set_registered_model_alias(NAME, "champion", mv.version)
sys.exit(0 if ok else 1)
```
```yaml
# .github/workflows/train.yaml (excerpt)
on:
  pull_request:
  push: {branches: [main]}
jobs:
  train:
    runs-on: ubuntu-latest        # use [self-hosted, ...] if MLflow is private
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}   # [15]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - uses: iterative/setup-cml@v2
      - run: pip install -r requirements.txt
      - run: python train.py ${{ github.ref == 'refs/heads/main' && '--promote' || '' }}
      - if: github.event_name == 'pull_request' && always()
        env: {REPO_TOKEN: "${{ secrets.GITHUB_TOKEN }}"}
        run: cml comment create --publish=false report.md
```
Verify: (1) PR with a worse model → job fails, comment shows FAIL; (2) merge a better model → `champion` alias moves in the MLflow UI; (3) reassign alias to the previous version → rollback works [12].

---

## MLflow + plain GitHub Actions *(CML-free alternative)*

**What.** Same as above, but the report is written to `$GITHUB_STEP_SUMMARY` (job summary) or posted with your own PR-comment step; runners provisioned by your own tooling [10].

**Pros.**
- One fewer dependency; no third-party image hosting; native UI.
- Job summaries are built into Actions (append Markdown to the file at `$GITHUB_STEP_SUMMARY`) [10].

**Risks.**
- You own the glue: PR comment updates, runner lifecycle, idle shutdown.
- No `cml runner` conveniences (spot restart, idle timeout) unless you replace them.

**How / verify.**
```bash
cat report.md >> "$GITHUB_STEP_SUMMARY"    # appears on the workflow run summary page
```
Verify: open the run page and confirm the Markdown table renders.

---

## Problem-specific considerations

- **Regulated/financial data:** treat any report as data egress. Default CML image upload goes to `asset.cml.dev`; disable it or self-host [4][5]. Report aggregate metrics only; no row-level samples, no feature values.
- **Heavy training on Spark/PySpark:** a CI runner is a poor place for full-scale training. Run a smoke/sample training in CI, or have CI submit the cluster job and poll for the MLflow run ID; keep the gate scoring step reproducible (same data snapshot ID logged as a tag).
- **Job limits:** GitHub-hosted jobs cap at 6 hours; self-hosted jobs at 5 days; a workflow run at 35 days [9]. Long jobs need self-hosted runners or an external cluster.
- **Metrics for gating (binary risk model):** rank metric (AUC/KS or PR-AUC for imbalance) **and** calibration (Brier, reliability); gate on out-of-time data; log PSI/feature drift separately. A single AUC tolerance can pass a miscalibrated model.
- **Data and artifact source of truth:** if using DVC for data and MLflow for models, define which store is authoritative for model files; avoid duplicating the same artifact in both.
- **Access control:** MLflow basic auth gives per-experiment/model permissions, but it is HTTP basic auth on a remote server; front it with TLS and your network controls [15].

---

## Diagnostics and pitfalls

- **Train/val/test discipline:** fit scalers/encoders inside a `Pipeline` on train only. Use validation/gate for CI decisions; reserve the **test** set for final sign-off, not for every PR.
- **Gate-set overfitting:** many PR iterations against the same gate set = selection on that set. Refresh or rotate a fresh out-of-time holdout periodically; keep the final test untouched.
- **Champion drift:** re-score the champion on the current gate set in the same job rather than reading its stored metrics (different data/split invalidates the comparison).
- **Non-determinism:** fix seeds (`random_state`), log library versions and the data snapshot ID; use a tolerance (`TOL`) rather than strict `>=`.
- **Leakage via time:** random splits on time-dependent targets inflate metrics; use out-of-time validation for the gate.
- **Comment spam:** repeated `cml comment create` on every push; use `cml comment update` + `--watermark-title` with a PAT [4].
- **Secrets:** never print `MLFLOW_TRACKING_PASSWORD` or tokens into the report; the report is visible to everyone with PR access.
- **Orphaned cloud instances:** always set `--idle-timeout` and consider `--single`; verify the instance is terminated after the job [6][7].
- **Version drift:** pin `setup-cml`, CML, MLflow, and action major versions; upstream CML examples lag [2][3]. MLflow 3 changed `log_model` (`name`) and artifact locations [16].
- **MLflow server basics:** SQLite for local only; PostgreSQL/MySQL for teams; decide proxied vs direct artifact access and grant CI the matching permissions [13][14][19].

---

## Decision rule / quick guide

1. Solo exploration, no CI → **MLflow only**, local SQLite backend [14].
2. Team needs shared runs and model versions → **MLflow server** (PostgreSQL + S3 + basic auth) [13][14][15].
3. Need automated checks and reviewer-visible metrics on every PR → add **GitHub Actions**; use **CML** for the comment, or `$GITHUB_STEP_SUMMARY` if a summary is enough [4][10].
4. Need on-demand GPU or large instances from CI → **`cml runner launch`** (PAT + cloud creds), or your own runner provisioning [6][7].
5. Sensitive data → `--publish=false` (or `--publish-url` self-hosted), self-hosted runner inside your network, aggregate metrics only [4][5].
6. Promote on merge to `main` by moving the `champion` **alias**; roll back by reassigning it [12][17].
7. Job longer than 6 h → self-hosted runner (limit 5 days per job) or submit to your cluster and poll [9].
8. Already standardized on DVC pipelines and Git-only state → CML alone can suffice; add MLflow once you need run history or a registry [2].
9. Avoid `cml tensorboard`; avoid building new flows on unpinned CML/action versions [3][8].

---

## References

1. CML documentation home — https://cml.dev/doc
2. `iterative/cml` README (functions, runner arguments, DVC usage, examples) — https://github.com/iterative/cml
3. `iterative/setup-cml` (v2, inputs, token notes) — https://github.com/iterative/setup-cml
4. CML command reference: `comment` (`--publish`, `--publish-url`, `--watermark-title`, GitHub FAQ) — https://github.com/iterative/cml.dev/blob/master/content/docs/ref/comment.md
5. CML command reference: `publish` (asset hosting, `--url`) — https://cml.dev/doc/ref/publish
6. CML command reference: `runner` — https://github.com/iterative/cml.dev/blob/master/content/docs/ref/runner.md
7. CML self-hosted runners guide — https://github.com/iterative/cml.dev/blob/master/content/docs/self-hosted-runners.md
8. CML v0.20.3 release notes (TensorBoard deprecation) — https://github.com/iterative/cml/releases/tag/v0.20.3
9. GitHub Actions limits — https://docs.github.com/actions/reference/limits
10. GitHub Actions workflow commands (`GITHUB_STEP_SUMMARY`) — https://docs.github.com/enterprise-server@3.11/actions/using-workflows/workflow-commands-for-github-actions
11. MLflow Model Registry (concepts, aliases, tags, lineage) — https://www.mlflow.org/docs/2.8.1/model-registry.html
12. MLflow Model Registry workflows (aliases, `models:/name@alias`, DB-backed store requirement) — https://mlflow.org/docs/latest/ml/model-registry/workflow
13. MLflow Tracking Server architecture (backend/artifact stores, proxied artifacts) — https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
14. MLflow backend stores (SQLite default, PostgreSQL/MySQL) — https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/
15. MLflow basic HTTP authentication — https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/
16. MLflow 3 migration guide (`name` vs `artifact_path`, model artifact location) — https://mlflow.org/docs/latest/ml/mlflow-3/
17. MLflow model version aliases introduced in 2.8 (`get_model_version_by_alias`, `models:/...@champion`) — https://github.com/mlflow/mlflow/issues/10336
18. MLflow `register_model` API source and example — https://www.mlflow.org/docs/latest/_modules/mlflow/tracking/_model_registry/fluent.html
19. MLflow Tracking Server (`--no-serve-artifacts`, `--artifacts-destination`, `--artifacts-only`) — https://mlflow.org/docs/latest/ml/tracking/server
