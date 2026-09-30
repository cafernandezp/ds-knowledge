# DVC — Tutorial and Research Report for Data, Model and Experiment Versioning

> **Context**
> - **Problem:** train several models on different versions of a large tabular dataset and be able to compare, later, which model performed how on which dataset version.
> - **Assumptions:** Git repo + GitHub Actions; binary classification, target column `target`, numeric features; XGBoost/scikit-learn; S3-style object storage as remote; Python 3.
> - **Constraints:** data too large for Git; results must be reproducible from a commit; comparison must survive team turnover (no manual bookkeeping).
> - **Scope:** open-source DVC CLI + Python API. Cloud versioning, Studio and DVCLive internals are not covered.
> - **Sources verified:** official DVC docs, consulted 2026-09-30. See References.

---

## TL;DR

- **DVC does not version anything by itself.** It writes small metafiles (`.dvc`, `dvc.lock`) containing content hashes; **Git versions those metafiles**, DVC stores the heavy content in a cache/remote. [1]
- **Recommended workflow (answers the "which dataset version trained which model" question):**
  1. `dvc add` the dataset → commit the `.dvc` file with Git.
  2. Define `prepare → train → evaluate` in `dvc.yaml`; `dvc repro` → commit `dvc.lock` (hashes of data, code, params). [2]
  3. `dvc push` + annotated Git tag per dataset/model run.
  4. Compare later with `dvc metrics diff <tagA> <tagB>`, `dvc params diff`, `dvc diff`. [3]
- **Use `params.yaml` + `dvc exp run -S` for hyperparameter/model-config search**; only promote winners to Git commits. [4][5]
- **CI:** `dvc pull` → `dvc repro` → `dvc metrics diff origin/main --md` in GitHub Actions. [12][13]
- **Reject** for this goal: spreadsheets, Git LFS + commit messages, one DVC remote per dataset version. They rely on manual mapping and give no pipeline/params/metrics linkage.
- **Fix the holdout set** across dataset versions, otherwise metric comparisons are confounded (see Problem-specific considerations).

---

## Comparison table

| Approach | Versions data | Links model ↔ data version | Reproducible run | Manual bookkeeping | Verdict |
|---|---|---|---|---|---|
| **DVC-tracked dirs + Git metafiles + `dvc.yaml`/`dvc.lock`** | Yes (content hash) | Yes, via `dvc.lock` in the same commit | Yes (`dvc repro`) | None | **Use** |
| Git LFS, one archive per version, commit messages as index | Yes (files) | Only by convention (text) | No pipeline/params layer | High | Reject |
| Single dataset version + spreadsheet of options/configs | No | Human-maintained | No | Very high, drifts | Reject |
| One DVC remote per dataset version + text file of URLs | Storage only | Human-maintained | No | High | Reject |

---

## 1. Project setup and data tracking (`dvc init`, `dvc add`)

**What.** Initialize DVC inside a Git repo, then track a file or directory. DVC moves the data into its cache, links it back into the workspace, writes a `.dvc` metafile and adds the data path to `.gitignore`. [1]

**Pros.**
- Git history holds only tiny metafiles; the data file itself is git-ignored. [1]
- Cache path is derived from the content hash (`.dvc/cache/files/md5/<2 chars>/<rest>`), so identical content is stored once. [1]
- Works on files and directories; pipeline outputs are tracked automatically (no manual `dvc add`). [2]

**Risks.**
- Forgetting `git add <file>.dvc data/.gitignore` → the version exists only in your cache, not in history.
- DVC docs position it for local Git-centered ML projects; for data-lake-scale or tens of thousands of files, they point to lakeFS. [1]

**How / verify.**
```bash
uv tool install dvc            # or: pipx install dvc
git init && dvc init
git commit -m "Initialize DVC"

dvc add data/raw.csv
git add data/raw.csv.dvc data/.gitignore
git commit -m "Add raw data"

cat data/raw.csv.dvc           # verify: outs -> md5 + path
```

---

## 2. Remote storage and sharing (`dvc remote`, `dvc push`, `dvc pull`)

**What.** A *remote* is where cached data lives so others (and CI) can retrieve it. `push` uploads cache to the remote; `pull` downloads it, usually after `git pull`/`git clone`. [1]

**Pros.**
- Supports S3, Azure Blob, SSH, HDFS, Google Drive, NFS and a local directory. [1]
- Separates *where bytes live* (remote) from *which version is which* (Git metafiles).

**Risks.**
- **`dvc push` forgotten after `git push`** → collaborators and CI get metafiles but cannot fetch data.
- Credentials must be provisioned in CI (env secrets); auth failures surface only at `dvc pull` (see Diagnostics).

**How / verify.**
```bash
dvc remote add -d storage s3://mybucket/dvcstore
dvc push
# elsewhere (fresh clone / CI):
git clone <repo-url> && cd <repo> && dvc pull
```
Verify: `dvc status --cloud` compares cache against the default remote. [11]

---

## 3. Switching between dataset versions (`git checkout` + `dvc checkout`)

**What.** A version switch = restore the `.dvc` file (or whole commit) from Git, then let DVC sync the workspace with the matching cached data. [1]

**Pros.**
- Switching a large file/dir is a metadata operation plus a link/copy from cache. [1]
- Same mechanism for rollback: `git checkout <rev> <file>.dvc` then `dvc checkout`. [1]

**Risks.**
- `dvc checkout` fails to materialize data that was never pushed or is missing from cache; run `dvc pull` first if needed.
- Checking out only the `.dvc` file while code stays at HEAD mixes versions. Prefer whole-commit checkout when reproducing a run.

**How / verify.**
```bash
# whole state (code + params + data pointers) at a tagged run
git checkout data-v1-xgb && dvc checkout

# only the dataset pointer
git checkout data-v1-xgb -- data/raw.csv.dvc && dvc checkout
```

---

## 4. Pipelines (`dvc.yaml`, `dvc.lock`, `dvc repro`)

**What.** A pipeline is a DAG of *stages*; each stage declares `cmd`, `deps`, `params`, `outs`. `dvc repro` reruns only stages whose inputs changed; `dvc.lock` records the hashes and param values actually used. [2]

**Pros.**
- **`dvc.lock` is the model↔data link**: it stores md5 of every dependency (data and code), the param values, and the md5 of outputs (the model). [2]
- Run cache: an already-seen combination of inputs is restored instead of retrained. [2]
- `dvc dag` visualizes the graph. [2]

**Risks.**
- Undeclared inputs (files read but not in `deps`) are invisible to DVC → stale results.
- Non-deterministic training changes output hashes on every run even with identical inputs; fix seeds and thread settings.
- `dvc.lock` not committed → the run is not reproducible from Git.

**How / verify.** Minimal, leakage-safe tabular pipeline (split happens before any fitting):

`params.yaml`
```yaml
prepare:
  test_size: 0.2
  seed: 42
train:
  n_estimators: 300
  max_depth: 4
  learning_rate: 0.05
  seed: 42
```

`dvc.yaml`
```yaml
stages:
  prepare:
    cmd: python src/prepare.py data/raw.csv data/prepared
    deps: [src/prepare.py, data/raw.csv]
    params: [prepare.test_size, prepare.seed]
    outs: [data/prepared]
  train:
    cmd: python src/train.py data/prepared models/model.pkl
    deps: [src/train.py, data/prepared]
    params: [train.n_estimators, train.max_depth, train.learning_rate, train.seed]
    outs: [models/model.pkl]
  evaluate:
    cmd: python src/evaluate.py models/model.pkl data/prepared eval/metrics.json
    deps: [src/evaluate.py, models/model.pkl, data/prepared]
    metrics:
      - eval/metrics.json:
          cache: false
```

`src/prepare.py`
```python
import sys, pathlib, yaml, pandas as pd
from sklearn.model_selection import train_test_split

raw, out = sys.argv[1], pathlib.Path(sys.argv[2])
p = yaml.safe_load(open("params.yaml"))["prepare"]
df = pd.read_csv(raw)
train, test = train_test_split(df, test_size=p["test_size"],
                               random_state=p["seed"], stratify=df["target"])
out.mkdir(parents=True, exist_ok=True)
train.to_csv(out / "train.csv", index=False)
test.to_csv(out / "test.csv", index=False)
```

`src/train.py`
```python
import sys, os, pickle, yaml, pandas as pd
from xgboost import XGBClassifier

src, model_path = sys.argv[1], sys.argv[2]
p = yaml.safe_load(open("params.yaml"))["train"]
train = pd.read_csv(f"{src}/train.csv")
X, y = train.drop(columns="target"), train["target"]
model = XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                      learning_rate=p["learning_rate"], random_state=p["seed"],
                      eval_metric="logloss")
model.fit(X, y)
os.makedirs(os.path.dirname(model_path), exist_ok=True)
pickle.dump(model, open(model_path, "wb"))
```

`src/evaluate.py`
```python
import sys, os, json, pickle, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

model_path, src, out = sys.argv[1:4]
model = pickle.load(open(model_path, "rb"))
test = pd.read_csv(f"{src}/test.csv")
X, y = test.drop(columns="target"), test["target"]
proba = model.predict_proba(X)[:, 1]
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"roc_auc": float(roc_auc_score(y, proba)),
           "brier": float(brier_score_loss(y, proba))}, open(out, "w"))
```

```bash
dvc repro
dvc dag
git add dvc.yaml dvc.lock params.yaml src .gitignore data/.gitignore
git commit -m "Pipeline + baseline run"
dvc status          # verify: "Data and pipelines are up to date"
```
Change `train.n_estimators` and rerun `dvc repro`: only `train` (and downstream) runs. [2]

---

## 5. Params, metrics and plots (`dvc params/metrics/plots diff`)

**What.** Params (from `params.yaml`, also JSON/TOML/Python) are declared as stage dependencies; metrics/plots are files declared in `dvc.yaml` and compared across Git revisions. [3]

**Pros.**
- Comparison is tied to Git revisions: `dvc metrics diff main workspace`, `dvc metrics diff <tagA> <tagB>`. [3]
- Params diff shows *why* metrics moved; data diff shows *what data* changed (`dvc diff`).

**Risks.**
- **Metrics file ignored by Git** (e.g. whole folder in `.gitignore`) → blank values for the compared revision. [16] Either keep metrics small in Git (`cache: false`, as above) or ensure they are DVC-tracked and pushed. [3]
- Metrics on a test set that changes between dataset versions are not comparable (see Problem-specific considerations).

**How / verify.**
```bash
dvc metrics show
dvc metrics diff data-v1-xgb data-v2-xgb
dvc params  diff data-v1-xgb data-v2-xgb
dvc diff         data-v1-xgb data-v2-xgb     # which tracked data changed
```

---

## 6. Experiments (`dvc exp run/show/apply`)

**What.** Experiments run the pipeline with modified params and are tracked without creating Git commits or branches; you later keep the winner. [4][5]

**Pros.**
- `-S/--set-param` overrides params on the fly; `--queue` + `dvc queue start` batches runs. [4][5]
- `dvc exp show` tabulates params and metrics; `dvc exp apply` restores a chosen experiment to the workspace; `dvc exp branch` persists it. [4]
- Same `dvc.yaml` used in CI and locally.

**Risks.**
- **Only Git- or DVC-tracked files are saved** into the experiment; untracked files are lost. [4][5]
- `dvc exp run` commits changed data dependencies into the DVC cache first, which can be slow on large data. [4]
- Experiments are local until pushed (`dvc exp push`); they are not team-visible by default.

**How / verify.**
```bash
dvc exp run -n depth6 -S train.max_depth=6
dvc exp run --queue -S train.learning_rate=0.03
dvc exp run --queue -S train.learning_rate=0.10
dvc queue start
dvc exp show
dvc exp apply depth6
git add dvc.lock params.yaml && git commit -m "Adopt depth6"
```
To compare model *families*, expose a param (e.g. `train.algo`) that `train.py` dispatches on, or keep one branch per family.

---

## 7. Programmatic access (`dvc.api`, `dvc get`, `dvc import`)

**What.** Read tracked data/models at any Git revision from Python or the CLI, without checking out the workspace. [6][7][9]

**Pros.**
- `dvc.api.open()`/`read()` accept `repo`, `rev` (commit, branch, tag, experiment name) and `remote`; data is streamed from remote storage and uses no disk. [6][7]
- `DVCFileSystem` gives an fsspec-compatible read-only view (`url`, `rev`). [8]
- `dvc get` downloads without tracking; `dvc import` also records the source (creates a `.dvc` file) so it can be updated. [9]

**Risks.**
- `dvc.api.open()` works only as a context manager. [6]
- `read()` loads the full content into memory. [7]
- `rev` must point to a commit whose data was pushed to the remote.

**How / verify.**
```python
import pickle, pandas as pd, dvc.api

with dvc.api.open("data/raw.csv", rev="data-v1-xgb") as f:
    df_v1 = pd.read_csv(f)

model_v1 = pickle.loads(dvc.api.read("models/model.pkl", rev="data-v1-xgb", mode="rb"))
# evaluate model_v1 on the SAME fixed holdout as model_v2 for a fair comparison
```

---

## 8. CI/CD with GitHub Actions

**What.** On each PR: install DVC, pull data from the remote, reproduce the pipeline, report metric changes against `main`. [12][13]

**Pros.**
- Automated data/model validation on code or data changes; metrics reported per PR. [12]
- `dvc pull data --run-cache` also fetches the run cache. [13]
- `iterative/setup-dvc` installs DVC on Ubuntu, macOS and Windows runners. [13]

**Risks.**
- Shallow checkout breaks revision comparison → use `fetch-depth: 0`.
- Secrets/credentials for the remote must exist in the workflow env.
- Retraining in CI on large data needs suitable runners (CML covers provisioning). [12]

**How / verify.**
```yaml
name: ml-ci
on: [pull_request]
jobs:
  train-and-report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - uses: iterative/setup-dvc@v1
      - name: Reproduce and report
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          pip install -r requirements.txt
          dvc pull --run-cache
          dvc repro
          echo "## Metrics vs main" >> "$GITHUB_STEP_SUMMARY"
          dvc metrics diff origin/main --md >> "$GITHUB_STEP_SUMMARY"
```
Verify: the job fails at `dvc pull` if credentials or `dvc push` are missing. Pin action versions per your org policy.

---

## 9. Storage hygiene (`dvc gc`)

**What.** Removes cache (and, with `--cloud`, remote) files not referenced by the scope you choose. [10]

**Pros.**
- Does nothing unless a scope flag is given (`--workspace`, `--all-branches`, `--all-tags`, `--all-commits`, `--all-experiments`). [10]
- Without `--cloud`, collected files can be restored with `dvc fetch` if previously pushed. [10]

**Risks.**
- **`--cloud` deletion is irreversible** unless another remote/backup exists. [10]
- Shared cache across projects: `gc` in one project breaks links in others unless `--projects` is used. [10]
- Scoping to `--workspace` only drops every older dataset version you may need for future comparisons.

**How / verify.**
```bash
dvc gc --all-tags --all-branches          # local cache only, keeps tagged runs
dvc gc --all-tags --all-branches --cloud  # also remote: irreversible
```

---

## 10. Approaches that look reasonable but do not fit the goal

### 10a. Single DVC-tracked dataset + spreadsheet of preprocessing/configs
**What.** One dataset version; a manual sheet records options per experiment.
**Pros.** Zero tooling.
**Risks.**
- Does not version datasets, so future comparison on "specific dataset versions" is impossible.
- Sheet drifts from reality; not linked to code, params or data hashes.
**Verify failure.** Try to reproduce a row from 3 months ago without the sheet author.

### 10b. Git LFS archive per dataset version + commit messages as index
**What.** Store compressed archives in Git LFS; encode model↔archive mapping in messages.
**Pros.** Files are versioned; familiar Git workflow.
**Risks.**
- Mapping is free text, not machine-readable.
- No stage graph, param tracking or metric diff; those are DVC features. [2][3]
- Whole-archive granularity: any row change is a new archive; DVC hashes files/directories in a content-addressed cache. [1]
- DVC positions itself as Git/Git-LFS plus Makefile-style pipelines, compatible with many storage backends. [14]
**Verify failure.** Ask "which archive trained model X?" using only `git log`.

### 10c. One DVC remote per dataset version + text file of URLs
**What.** New remote per version; text file maps model → remote URL.
**Pros.** Physical separation of versions.
**Risks.**
- A remote is a storage location, not a version. Versions are already identified by hashes in metafiles. [1]
- Duplicates storage that content-addressing would deduplicate; adds credentials/config per remote.
- Manual mapping again.
**Verify failure.** `dvc remote list` grows with every dataset revision.

---

## Problem-specific considerations

- **Goal:** "compare how each model performed on specific dataset versions in the future."
  - Each run = one Git commit containing `.dvc` files (data pointers), `params.yaml`, `dvc.yaml`, `dvc.lock`, code. Tag it (annotated) and push data. [1][2]
  - Retrieval later: `git checkout <tag> && dvc checkout`, or `dvc.api` with `rev=<tag>`. [6]
- **Recipe (dataset v1 → v2, same model family):**
  ```bash
  # v1
  dvc add data/raw.csv && git add data/raw.csv.dvc data/.gitignore
  dvc repro
  git add dvc.lock params.yaml && git commit -m "Data v1 + XGB"
  dvc push && git tag -a data-v1-xgb -m "raw data v1, XGB baseline"
  git push --follow-tags

  # v2 (new snapshot replaces data/raw.csv)
  dvc add data/raw.csv
  dvc repro
  git add data/raw.csv.dvc dvc.lock && git commit -m "Data v2 + XGB"
  dvc push && git tag -a data-v2-xgb -m "raw data v2, XGB baseline"
  git push --follow-tags

  dvc metrics diff data-v1-xgb data-v2-xgb
  dvc diff data-v1-xgb data-v2-xgb
  ```
- **Fixed holdout (critical).** `prepare.py` above splits raw data randomly; when the dataset changes, the test rows change, so `roc_auc` deltas mix data-quality and test-set effects.
  - Fix: track a frozen `data/holdout.csv` (`dvc add`, never modified), make it a dependency of `evaluate`, and train only on `raw` minus holdout.
  - For temporal data, use an out-of-time holdout rather than a random split.
- **Metric choice:** `roc_auc` alone is rank-based; also log a calibration-sensitive metric (`brier` above) if probabilities are consumed.
- **Model families:** either parametrize the family in `params.yaml` or keep separate branches; do not encode the family only in a tag name.

---

## Diagnostics and pitfalls

- **Leakage / split discipline**
  - Split in `prepare` before any fitting; fit encoders/scalers on train only (inside the `train` stage or a `Pipeline`), never on `raw`.
  - Test/holdout must not influence hyperparameter selection: if you pick winners with `dvc exp show` on the holdout metric, you are tuning on it. Use a validation split (or CV) for selection and report the holdout once.
- **Reproducibility checks**
  - `dvc status` → "Data and pipelines are up to date". [11]
  - Re-run on a clean clone: `git clone … && dvc pull && dvc repro` should skip all stages (or reproduce identical hashes if seeds are fixed).
  - Undeclared dependency symptom: code change with no `dvc repro` re-run.
- **Sharing failures**
  - Metafiles pushed, data not (`dvc push` skipped) → `dvc pull` fails for others; `dvc status --cloud` detects it. [11]
  - CI auth errors appear at `dvc pull`; `dvc doctor` prints environment/version info for diagnosis. [17]
- **Metrics diff blank** → metrics file ignored by Git. [16]
- **Experiments** save only Git/DVC-tracked files. [4]
- **`dvc gc`** requires an explicit scope; `--cloud` is irreversible; shared caches need `--projects`. [10]

---

## Decision rule / quick guide

1. Data too large for Git and you need to know which data trained which model → **`dvc add` + commit metafiles**.
2. More than one processing/training step → **`dvc.yaml` stages**; commit `dvc.lock` after every meaningful run.
3. Anyone else (or CI) must read the data → **configure a remote and `dvc push` before `git push`**.
4. Comparing runs across time → **annotated Git tag per run + `dvc metrics/params/diff <tagA> <tagB>`**.
5. Hyperparameter or config search → **`dvc exp run -S` (queue for batches)**; apply and commit only the winner.
6. Need data/models inside Python code at a given revision → **`dvc.api.open/read(rev=...)`**.
7. Automated validation on PRs → **GitHub Actions: `dvc pull` → `dvc repro` → `dvc metrics diff`**.
8. Storage growing → **`dvc gc` with explicit scope**; never `--cloud` without a backup.
9. Comparing across dataset versions → **freeze the holdout first**.
10. Tempted by spreadsheet / LFS-plus-messages / remote-per-version → **don't; they require manual mapping.**

---

## References

1. DVC — Get Started (init, add, remotes, push/pull, checkout, cache layout): https://doc.dvc.org/start
2. DVC — Get Started: Data Pipelines (`dvc stage add`, `dvc.yaml`, `dvc.lock`, `dvc repro`, run cache, `dvc dag`): https://doc.dvc.org/start/data-pipelines/data-pipelines
3. DVC — Get Started: Metrics, Plots, and Parameters: https://doc.dvc.org/start/data-pipelines/metrics-parameters-plots
4. DVC — `dvc exp run` command reference: https://doc.dvc.org/command-reference/exp/run
5. DVC — Running Experiments (`--set-param`, `--queue`): https://doc.dvc.org/user-guide/experiment-management/running-experiments
6. DVC — `dvc.api.open()`: https://doc.dvc.org/api-reference/open
7. DVC — `dvc.api.read()`: https://doc.dvc.org/api-reference/read
8. DVC — `DVCFileSystem`: https://dvc.org/doc/api-reference/dvcfilesystem
9. DVC — `dvc get`: https://doc.dvc.org/command-reference/get
10. DVC — `dvc gc`: https://dvc.org/doc/command-reference/gc
11. DVC — `dvc status`: https://dvc.org/doc/command-reference/status
12. DVC — CI/CD for Machine Learning: https://doc.dvc.org/example-scenarios/ci-cd-for-machine-learning
13. CML with DVC (`iterative/setup-dvc`, `dvc pull data --run-cache`, `dvc repro`): https://dvc.org/doc/cml/cml-with-dvc
14. DVC README mirror (Git/Git-LFS + Makefiles positioning): https://github.com/whysage/dvc
15. DVC CLI installation via `uv`/`pipx`: covered in [1].
16. Demo repo noting blank `dvc metrics diff` when metrics file is git-ignored: https://www.github.com/Michael95-m/simple_demo_dvc
17. GitHub issue showing `dvc doctor` used to diagnose `dvc pull` failure in Actions: https://github.com/treeverse/dvc/issues/6899
