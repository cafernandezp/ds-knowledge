# Cloud Developer Workspace for ML on AWS — Codespaces, ECR & Batch

**Last updated:** 2026-09-20

> **Context.** Select a workspace architecture to develop an end-to-end ML exercise with an industry-realistic workflow.
> - **Constraint:** no managed notebook platforms (SageMaker Studio Lab, Colab).
> - **Assumptions:** tabular models first (LR / SVM / RF / XGBoost); CNN/RNN possible later; provider-agnostic in the Introduction, AWS-only in the Packaging and Remote GPU Training sections; single developer, Git-based workflow.
> - **Selection criteria:** reproducibility, parity with deploy environment, disposability, CI/CD integration, portability, cost.
> - **Status:** complete. Facts verified on 2026-09-20; section order revised on 2026-09-20. GitHub Action major versions and cloud GPU availability drift, so re-verify before reuse.


**Reading guide.**
1. **Objective** — what the exercise must achieve.
2. **Introduction** — why Codespaces, and the baseline architecture as a map.
3. **Cloud Access** — how your code talks to the cloud (CLI, SDK, API). Read this before the operational sections; the concepts appear everywhere that follows.
4. **Dev Containers** — set up the reproducible environment.
5. **Packaging** — build and push an immutable container image (AWS).
6. **Remote GPU Training** — run a GPU job on AWS Batch from Codespaces.
7. **Diagnostics & Pitfalls** — cross-cutting risks and the first place to check when something fails.
8. **Decision Rule** — if X → do Y.

## TL;DR

- **Workspace:** GitHub Codespaces with a dev container (`devcontainer.json` + multi-stage Dockerfile). The environment rebuilds from the repository alone; the same file runs locally and in Codespaces.
- **Cloud access:** CLI (`aws`) for provisioning and CI glue; SDK (`boto3`) inside training code. Both resolve credentials through the same chain — SSO for humans, OIDC for CI, job role for containers. No static keys anywhere.
- **CI:** GitHub Actions — lint, test, smoke train on push; build and push an immutable Docker image to Amazon ECR, tagged with the git SHA.
- **GPU training:** AWS Batch on EC2 GPU instances, submitted from Codespaces or CI. Codespaces stays CPU-only; GPU cost runs only while jobs execute. Checkpoints to S3 for Spot resilience.
- **End-to-end path:** edit in Codespaces → PR → Actions builds image → submit Batch job → artifacts in S3 → logs in CloudWatch. Every run maps to one git SHA, one image tag, one job definition revision.

## Objective

- Define a reproducible, industry-standard development workspace for an ML exercise and justify it against the alternatives in the reference diagram ("Cloud Developer Workspace Advantages").
- Deliver a minimal, verifiable stack: environment as code → CI → containerized job in the cloud.
- Proposed acceptance criteria:
  - Environment rebuilds from the repository alone (no manual setup).
  - CI runs lint, tests, and a smoke training on every push/PR.
  - Cloud access uses short-lived credentials (no static keys in the repo or in secrets).
  - Dev, CI, and remote jobs are built from Dockerfiles in the repository with pinned dependencies (CPU: one shared multi-stage file; GPU: a variant that adds the CUDA build of the framework).
  - Fixed train/validation/test split and `random_state`; no leakage across splits.

## Introduction

**Problem.** An ML exercise is only as credible as its environment. A laptop setup is non-deterministic, costlier, and differs from the deploy environment (per diagram), so results and workflow do not transfer to production practice.

**Workspace classes in the diagram.**
- Laptop / workstation.
- Cloud IDEs: GitHub Codespaces, AWS Cloud9, GCP Cloud IDE, Azure Cloud IDE.
- CloudShells (AWS / GCP / Azure).
- GPU + Jupyter notebooks (SageMaker Studio Lab, Colab) — excluded by constraint.

**Screening.**

| Option | Profile (per diagram) | Verdict |
|---|---|---|
| Laptop / workstation | Local | Rejected: non-deterministic, higher cost, differs from deploy env |
| **GitHub Codespaces** | Powerful, disposable, pre-loaded; links to GitHub Actions / Copilot | **Selected.** Environment defined in-repo via `.devcontainer/` [2]; machine types from 2 to 32 cores, not always all available [3] |
| AWS Cloud9 | Cloud IDE | Rejected: closed to new customers since 2024-07-25 [1] |
| GCP / Azure Cloud IDE | Deep cloud integration, SDK, co-located in network | Conditional: only if in-network access to data or services is required |
| CloudShell (AWS / GCP / Azure) | Lightweight, CLI/SDK loaded | Rejected as project environment (assessment): suited to resource administration, not development |
| SageMaker Studio Lab / Colab | GPU + Jupyter | Excluded by constraint |

**Baseline architecture.**
- **Workspace:** Codespaces + dev container (`devcontainer.json`, Dockerfile, dependency lockfile) [2].
- **CI:** GitHub Actions — `ruff`, `pytest`, smoke training on a data sample.
- **Cloud access:** OIDC from Actions to a cloud role, without long-lived secrets; the trust policy must restrict by repo/branch through the `sub` condition [4].
- **Data:** object storage (S3 / GCS / Blob), outside the repository.
- **Tracking:** MLflow (parameters, metrics, artifacts).
- **Execution:** container image in a registry, run as a remote job. GPU workloads (CNN/RNN) are offloaded there: Codespaces no longer offers GPU machine types [5].

**Why it is industry-realistic.** It mirrors the standard delivery path: PR → CI → image → cloud job, with the environment versioned as code.

## Cloud Access: CLI, SDK, and APIs

**What.**
- Every cloud service is controlled through a web **API**: a contract that defines which requests the service accepts, how they are authenticated, and what they return. AWS requests are signed with SigV4 [38]; all Google Cloud APIs expose a JSON/REST interface, and some also gRPC [46]; the Azure CLI and Python SDK sit on top of the Azure REST API [51].
- The CLI and the SDK are two clients of that same API. Both send authenticated requests so you do not write HTTP by hand.
- "Connected to your cloud by CLI/SDK" therefore means: from the dev environment, a terminal command or a Python program authenticates as an identity and calls the provider's API.

```text
 what you want        client                        on the wire                cloud
 "submit a job" --> CLI command or SDK call  --> signed HTTPS request  --> service API
                    aws / gcloud / az            (REST; some gRPC)          (AWS / GCP / Azure)
                    boto3 / google-cloud-* / azure-*
```

**Question 1. What is the difference between accessing by CLI and by SDK?**

| Aspect | CLI | SDK |
|---|---|---|
| Form | Executable run from a terminal or script: `aws`, `gcloud`, `az` [45][48] | Library imported in code: `boto3`, Google Cloud Client Libraries, Azure SDK for Python [37][46][49] |
| Best for | Setup, one-off operations, CI glue: log in, create a repository, submit a job | Application logic: read data from S3 inside `train.py`, call services from a pipeline |
| Input / output | Arguments in, text out (JSON or table) that you parse | Function calls in, native objects and exceptions out |
| Control flow | Shell: pipes, `&&`, exit codes | Full language: loops, functions, tests |
| Authentication | Login command plus profile or environment: `aws sso login`, `gcloud auth login`, `az login` | Credentials resolved automatically when the client is created (credential chain) [39][47][49] |
| Relationship | A program built on the same core as the SDK: `botocore` is the foundation of both the AWS CLI and boto3 [37] | The reusable layer that CLIs and applications are built on |

- Because they share a core, cross-cutting behavior is the same: configuration files, credential resolution, and retry behavior apply to the AWS CLI and to the SDKs alike [41][42].
- The same API operation, both ways (`DescribeJobs` on AWS Batch):

```bash
# CLI: one operation, text result
aws batch describe-jobs --jobs <job-id> --query 'jobs[0].status' --output text \
  --profile ml-dev --region eu-west-1
```

```python
# SDK: same operation, result is a Python dict
import boto3
batch = boto3.Session(profile_name="ml-dev", region_name="eu-west-1").client("batch")
status = batch.describe_jobs(jobs=["<job-id>"])["jobs"][0]["status"]
```

**Question 2. What is an SDK and how is it used?**
- An SDK (software development kit) is a set of platform-specific tools for building software on a platform; it typically includes libraries and may add documentation, samples, debuggers, and compilers [35][36]. In cloud work, "SDK" usually means the provider's language-specific client library. That is the meaning used in this report.
- What it does for you:
  - Finds credentials through a defined chain [39][47][49].
  - Builds and signs the request; with an SDK or the CLI you can skip the manual signing process [38].
  - Handles the low-level details of communication, including authentication [46].
  - Retries failed requests: standard mode uses exponential backoff with jitter and a retry quota, configurable through environment variables, the shared config file, or code [42].
- How it is used: install the package, create a client (no keys in code; select an AWS profile with `AWS_PROFILE` [52]), call a method (one API operation), read the result.

```python
# pip install boto3 google-cloud-storage azure-storage-blob azure-identity
# Identity comes from the credential chain (select an AWS profile with AWS_PROFILE).

# AWS
import boto3
print([b["Name"] for b in boto3.client("s3").list_buckets()["Buckets"]])

# GCP
from google.cloud import storage
print([b.name for b in storage.Client(project="<project-id>").list_buckets()])

# Azure
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
svc = BlobServiceClient("https://<account>.blob.core.windows.net",
                        credential=DefaultAzureCredential())
print([c.name for c in svc.list_containers()])
```

**Question 3. How does communication through a software's API relate to an SDK?**
- They are related but sit at different layers. The API is the service's contract; the SDK is client-side code that speaks that contract for you. You can call the API without an SDK; an SDK cannot work without the API.
- The word "API" is overloaded, which causes the confusion:
  - **Web API of a service:** requests to an HTTP(S) endpoint.
  - **Library API:** the functions of a package you import.
  - An SDK exposes a library API to your code and internally makes web API calls [35].

| Layer | Owner | Example | Job |
|---|---|---|---|
| 1. Service web API | Cloud provider | AWS `DescribeJobs`; Google Cloud JSON/REST or gRPC [46]; Azure REST [51] | Defines operations, parameters, authentication, responses |
| 2. SDK | Provider (or community) | `boto3`, `google-cloud-*`, `azure-*` | Builds and signs requests, retries, parses responses [38][42][46] |
| 3. CLI | Provider | `aws`, `gcloud`, `az` | Parses commands, reuses the same core, prints output [37][45][48] |
| 4. Your code | You | `train.py`, workflows | Business logic; calls layer 2 or layer 3 |

- Three ways to make the same call: CLI (layer 3), SDK (layer 2), raw HTTP (layer 1). With raw HTTP you implement authentication, retries, and parsing yourself. AWS advises always using an SDK or the CLI unless you have a good reason, such as no SDK for your language or a need for full control [38]. Google Cloud also allows direct REST calls with your own HTTP client [46].
- The same relationship holds for software your team builds: a model-serving service exposes a web API (layer 1); a client library for it is an optional convenience that someone must build and maintain; callers may use raw HTTP.

**Cross-cloud map.**

| Item | AWS | GCP | Azure |
|---|---|---|---|
| CLI | `aws` | `gcloud` [45] | `az` [48] |
| Python SDK | `boto3` on `botocore` [37] | Cloud Client Libraries [46] | Azure SDK for Python [49] |
| Service API | HTTPS; requests signed with SigV4 [38] | JSON/REST; some gRPC [46] | Azure REST API [51] |
| Credential resolution in code | boto3 credential chain [39] | Application Default Credentials (ADC) [47] | `DefaultAzureCredential` chain [49][50] |
| Human login | `aws sso login` [12] | `gcloud auth login`; SDKs need `gcloud auth application-default login` [47] | `az login`; `DefaultAzureCredential` includes the Azure CLI login [50] |
| Workload identity (CI, jobs) | OIDC role [4]; job and instance roles [39] | Workload Identity Federation; attached service account [47] | Workload identity; managed identity [49] |

**Authentication: one mechanism, three places (AWS example).** The same command or code runs unchanged in each place because the credential chain resolves the identity.

| Where the code runs | Identity source | How the CLI/SDK find it |
|---|---|---|
| Codespaces or local terminal | IAM Identity Center session from `aws sso login` [12] | Shared config and credentials files, used by SDKs and tools [40][41]; boto3 has an IAM Identity Center provider [39] |
| GitHub Actions | OIDC federation into an IAM role [4] | Temporary credentials for the following steps, resolved through the same chain [4][39] |
| AWS Batch job container | Job role [17] | Container credential provider [39]; ECS sets `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` for tasks [40] and Batch runs on ECS container instances [22] (expected for Batch jobs; confirm with the verification step) |
| EC2 instance | Instance role | Instance metadata provider, last in the chain [39] |

**Recommendation.**
- CLI for provisioning, one-off operations, and CI glue (login, create the repository, register and submit jobs, as in the following sections).
- SDK inside code that runs in jobs and pipelines (data access, checkpoints, as in `train.py`).
- No shelling out to the CLI from Python for data access; no hand-written HTTP unless no SDK exists [38].
- Credentials always come from the chain, never from code.

**Pros.**
- One identity model across dev, CI, and jobs: no code changes between environments.
- No custom signing or retry code [38][42].
- CLI commands are scriptable and reviewable in workflows.

**Risks.**
- Precedence surprises: the chain stops at the first source that returns credentials. In boto3, environment variables rank before shared files, the container provider, and instance metadata [39]; a stray `AWS_ACCESS_KEY_ID` overrides a job role.
- GCP: CLI credentials and ADC credentials are distinct, so a command can work in the terminal and the same code can fail in Python [47].
- Static keys: avoid them; Google states that service account keys create a security risk and are not recommended [47]. Use federation and roles.
- Debug traces expose data: boto3 warns that the botocore wire trace can contain sensitive payloads and should not be used in production [44]; redact CLI `--debug` output before sharing it [43].
- Profile and region mismatch between CLI flags and SDK configuration, and CLI/SDK version drift (assessment): pin `boto3` in `requirements.txt` and set the profile and region explicitly.

**How / verify.**

```bash
aws sts get-caller-identity --profile ml-dev
# request URL, request contents, and raw response (redact before sharing) [43]
aws sts get-caller-identity --profile ml-dev --debug 2>&1 | less
```

```python
import boto3
s = boto3.Session(profile_name="ml-dev", region_name="eu-west-1")
print(s.client("sts").get_caller_identity()["Arn"])   # must equal the CLI identity
boto3.set_stream_logger("botocore")                   # wire trace: local debugging only [44]
s.client("sts").get_caller_identity()
```

- Expected result: the CLI and SDK print the same identity. If they differ, the chain is resolving different sources.
- Same check in each place: run `python -c "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'])"` in Codespaces, in a GitHub Actions step, and as a Batch smoke job (command override). Expect the SSO role, the OIDC role, and the job role respectively.
- Confirm the API call is the same one: with `--debug` and the botocore wire trace, both show a request to the same service endpoint [43][44].

## Dev Containers

**What.**
- A dev container is a container used as a full development environment. A `devcontainer.json` file in the project tells the tool how to access or create it, with a defined tool and runtime stack [6].
- The format is an open specification: any tool or service that supports it can build the same environment from the same file [7]. VS Code with local Docker and GitHub Codespaces are both supported [6][2].
- The configuration lives in the repository, typically under `.devcontainer/` [2]. Editor extensions install and run inside the container, so tooling matches the runtime [6].

**Pros.**
- Reproducible: the environment is defined by the repository, not by a machine; codespaces created from the repository share the same configuration [2].
- Disposable: change the config and rebuild the container instead of repairing a drifted machine [2].
- Portable across hosts: moving to a bigger host changes the machine, not the setup (see "Scaling up" below).
- Near-zero onboarding: clone, open, wait for the build.

**Example configuration.** CPU workflow for the tabular exercise, with AWS CLI and Docker available for Packaging and Remote GPU Training.

```jsonc
// .devcontainer/devcontainer.json
{
  "name": "ml-exercise",
  "build": { "dockerfile": "../Dockerfile", "context": "..", "target": "dev" },
  "features": {
    "ghcr.io/devcontainers/features/aws-cli:1": {},
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  },
  "hostRequirements": { "cpus": 4, "memory": "16gb", "storage": "32gb" },
  "containerEnv": { "PYTHONHASHSEED": "0" },
  "postCreateCommand": "pip install --no-cache-dir -r requirements-dev.txt",
  "customizations": {
    "vscode": { "extensions": ["ms-python.python", "ms-toolsai.jupyter"] }
  }
}
```

```dockerfile
# Dockerfile (repository root): one file, three stages.
# base    = OS + Python + pinned dependencies (shared by dev, CI, and jobs)
# dev     = target used by the dev container (extra tools come from Features)
# runtime = target built by CI and executed by remote jobs
FROM python:3.12-slim AS base
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS dev
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

FROM base AS runtime
COPY src/ src/
CMD ["python", "src/train.py"]
```

- `build.dockerfile` is relative to `devcontainer.json`; `build.context` and `build.target` select the build context and stage [7].
- `features` add tooling without editing the Dockerfile: AWS CLI [9] and Docker-in-Docker [10].
- `hostRequirements` declares minimum CPU, memory, and storage [7].
- Lifecycle commands (`postCreateCommand`) run after the container is created [7].

**Where it runs and where to configure it.**

| Question | Answer |
|---|---|
| Is a dev container tied to an IDE? | No. It is a file in the repository. Any supporting tool starts the container; the IDE is only a client attached to it [6][7]. |
| Local option | VS Code with the Dev Containers extension and local Docker: run `Dev Containers: Reopen in Container` [6]. Limits: your machine's CPU and RAM. |
| Codespaces option | GitHub starts the same container on a remote VM. Open it in the browser, in VS Code desktop, or in JupyterLab [8]. |
| Where to edit `devcontainer.json` | Anywhere: it is plain JSON with comments in the repo. Edit it locally or inside a codespace, then rebuild the container to apply changes [2]. Commit it so new codespaces pick it up. |
| Can you configure locally and move to Codespaces? | Yes. The same file is used in both; no conversion is needed [2][6]. |

**Scaling up (CPU, RAM, GPU).**

| Need | Action |
|---|---|
| More CPU / RAM | Create the codespace on a larger machine type (2 to 32 cores; availability varies and org policy or a repository minimum can restrict it) [3]. `hostRequirements` sets the minimum; cloud services may default to the best available option, otherwise you get a warning [7]. |
| GPU inside the dev environment | Not available in Codespaces: the GPU machine type was deprecated by 2025-08-29 [5]. Locally it works with an NVIDIA GPU, the NVIDIA Container Toolkit, and `"runArgs": ["--gpus=all"]` (example configuration in [11]). |
| GPU for real training | Run a remote AWS job (see Remote GPU Training). |

Recommendation: work in Codespaces by default (no local Docker, same host type as the target). Use a local dev container only when Docker is already available or when a local NVIDIA GPU is needed.

**Risks.**
- Lifecycle scripts: if one fails, later ones are skipped [7]. `onCreateCommand` and `updateContentCommand` may run in prebuilds without user-scoped secrets; put anything that needs them in `postCreateCommand` [7].
- `initializeCommand` runs on the host, which for a cloud service is the cloud machine [7].
- Docker-in-Docker requires a privileged container [7]; the Feature configures it, but verify `docker version` works after the build.
- Two divergent configs (local vs Codespaces) defeat the purpose. Keep one `.devcontainer/devcontainer.json`.
- Codespaces machine types have different billing tiers [3]; stop or delete idle codespaces.

**Verify.**
- Inside the container: `python --version && aws --version && docker version`.
- Rebuild from scratch (`Rebuild Container`) and confirm the environment comes up with no manual steps [2].
- CI parity: `docker build --target dev .` succeeds on a clean runner.

## Packaging

**What.**
- Packaging turns code and pinned dependencies into one immutable, versioned container image that runs identically in CI and in remote jobs.
- AWS services involved:
  - **Amazon ECR**: image registry; the hand-off point between code and compute.
  - **AWS IAM**: roles, plus OIDC federation for CI [4].
  - **Amazon S3**: data and artifacts (outside the image).
  - **AWS Batch**: runtime that pulls the image (next section).

**Who operates what, and from where.**

| Task | Canonical place | Also possible from | Credentials |
|---|---|---|---|
| Build and push the image on merge | GitHub Actions | Codespaces or local terminal (needs Docker) | OIDC role, no stored keys [4] |
| One-time AWS setup (ECR repository, IAM, Batch resources) | Codespaces terminal (AWS CLI) | Local IDE terminal | IAM Identity Center session [12][13] |
| Submit and inspect jobs | Codespaces terminal or Actions | Local terminal, AWS console | Same as above |

- The IDE (local or Codespaces) is only a client. What matters is which credentials its terminal holds.
- Use short-lived credentials: IAM Identity Center for humans [13], OIDC for CI [4]. Do not put static access keys in the repository, the image, or Codespaces secrets.
- In a remote codespace there is no local browser, so use the device-code flow: `aws sso login --use-device-code` [12].

**Image design.**
- Stage `runtime` of the shared Dockerfile is what jobs run; `dev` is what the dev container uses. The shared `base` stage guarantees the same OS, Python, and pinned dependencies.
- Tag with the git SHA. With tag immutability enabled, pushing an existing tag fails with `ImageTagAlreadyExistsException` [15].
- Enable image scanning at the registry level: the repository-level `scanOnPush` option is being deprecated in favor of registry-level configuration [16].
- Match architecture: an image only runs on compute with the same processor architecture (ARM images need ARM instances) [21]. Build with `--platform linux/amd64` for x86 instances.

**How / verify.**

One-time setup from a terminal (Codespaces or local):

```bash
# once: create the profile (name it ml-dev when prompted), then log in per session
aws configure sso --use-device-code
aws sso login --profile ml-dev --use-device-code
aws ecr create-repository --repository-name ml-exercise/train \
  --image-tag-mutability IMMUTABLE --profile ml-dev --region eu-west-1
```

CI role trust policy (create the GitHub OIDC identity provider first: URL `https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`) [4]:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
        "token.actions.githubusercontent.com:sub": "repo:<owner>/<repo>:ref:refs/heads/main"
      }
    }
  }]
}
```

Attach least-privilege ECR push permissions on that single repository to the role. Workflow (action majors as of 2026-09-20 [32][33]):

```yaml
# .github/workflows/build-push.yml
name: build-push
on:
  push:
    branches: [main]
permissions:
  id-token: write
  contents: read
env:
  AWS_REGION: eu-west-1
  REGISTRY: <account-id>.dkr.ecr.eu-west-1.amazonaws.com
  REPOSITORY: ml-exercise/train
jobs:
  image:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: aws-actions/configure-aws-credentials@v6
        with:
          role-to-assume: arn:aws:iam::<account-id>:role/gha-ecr-push
          aws-region: ${{ env.AWS_REGION }}
      - name: Login, build, push
        run: |
          aws ecr get-login-password --region "$AWS_REGION" \
            | docker login --username AWS --password-stdin "$REGISTRY"
          IMAGE="$REGISTRY/$REPOSITORY:${{ github.sha }}"
          docker build --platform linux/amd64 --target runtime -t "$IMAGE" .
          docker push "$IMAGE"
```

The same three commands work from a Codespaces terminal for debugging (Docker-in-Docker Feature required); the ECR login token is valid for 12 hours [14].

Verification:
- `aws ecr describe-images --repository-name ml-exercise/train --image-ids imageTag=<sha>` returns exactly one image.
- Pushing the same tag again is rejected (immutability) [15].
- `docker run --rm <image> python -c "import numpy"` succeeds on a different machine.

**Pros.**
- Auditable chain: git SHA to image tag to job definition revision.
- Short-lived credentials end to end.
- The development host and the execution host are decoupled.

**Risks.**
- Architecture mismatch when building on an ARM laptop for x86 instances [21].
- A trust policy without a `sub` condition lets other repositories assume the role [4].
- Mutable tags silently overwrite images, so a rerun is no longer the same run [15].
- Secrets in image layers or plaintext environment variables: AWS advises against plaintext variables for sensitive data [21]; use the job role for AWS access.
- Data in the image: never `COPY` datasets into it. Data and splits live in S3 (see the leakage notes in the next section).
- Image bloat: keep `runtime` free of dev tools.

## Remote GPU Training

**What.**
- Codespaces remains the IDE and control surface (CPU only). GPU training runs as a containerized **AWS Batch** job on EC2 GPU instances. Nothing trains inside Codespaces.
- Reason: Codespaces no longer offers GPU machine types [5], and on AWS a GPU requires EC2 capacity, not Fargate [20][28].

### Choice among AWS options

| Option | Scheduling | Scale to zero | GPU support | Ops effort (assessment) | Verdict |
|---|---|---|---|---|---|
| **AWS Batch on EC2** | Job queue and scheduler; retries, timeouts, dependencies, array jobs [17][24] | Managed compute environment can sit at 0 vCPUs when idle [34] | `resourceRequirements` type GPU [18][20] | Low to medium | **Recommended** |
| Amazon ECS on EC2 | You manage the cluster and its capacity; tasks pin GPUs [27] | You manage it | GPU-optimized AMI [27] | Medium to high | Fits long-running services, not batch training |
| Amazon EC2 directly | None | Manual stop or terminate | You handle AMI, driver, runtime | High | Debugging only |
| AWS Fargate | n/a | n/a | Not supported [20][28] | n/a | Excluded |

### Architecture and flow

```
Codespaces (IDE, CPU)            GitHub Actions                     AWS
 edit, test, CPU smoke  --push--> build image --push--> ECR  (repo:<sha>)
        |                              |                             |
        | aws batch submit-job         | OIDC role                   | pull
        +------------------------------+--> Batch job queue          |
                                                |                    |
                                     Compute environment (managed EC2)
                                     GPU instance, ECS GPU-optimized AMI
                                                |
                     container with GPU pinned --> S3 (checkpoints, model)
                                                --> CloudWatch Logs
```

1. Develop and smoke-test on a tiny sample in Codespaces.
2. Merge; Actions builds and pushes the GPU image to ECR, tagged with the git SHA (see Packaging).
3. Register a job definition revision that points to that tag, then submit a job to the queue (CLI or CI).
4. Batch places the job. If no GPU instance is available, the managed compute environment launches one; for GPU instance types Batch uses the ECS GPU-optimized AMI [19], which ships the NVIDIA driver and Docker GPU runtime [27]. The ECS agent, using the instance role [22], pulls the image from ECR.
5. The container starts with the requested GPUs pinned to it; they are not available to other jobs while it runs [18]. It reads and writes S3 through the job role [17], and stdout/stderr go to CloudWatch Logs by default [25].
6. When the queue is empty, the environment scales back toward `minvCpus` [34].

### Components and roles

| Component | Purpose | Notes |
|---|---|---|
| Compute environment | Managed EC2 capacity | GPU families only (p3, p4, p5, p6, g3, g3s, g4, g5, g6), otherwise jobs can stay in RUNNABLE [18]; `maxvCpus` caps cost |
| Job queue | Holds jobs until scheduled | One or more compute environments, with priorities [17] |
| Job definition | Blueprint: image, vCPU, memory, GPU, roles, environment | Revisions pin the image tag; GPU count is not shared between jobs [18] |
| Job | One submission | Overrides for command and environment; retry strategy and timeout [24] |
| Instance role | Lets the ECS agent on the EC2 instance call AWS APIs | Required to create a compute environment [22] |
| Job role | AWS permissions for the container (S3 access) | Set in the job definition [17] |
| Execution role | Needed for Fargate only | Not required for EC2 jobs [21] |
| CI role (OIDC) | GitHub Actions calls Batch | Trust policy with `sub` condition [4] |
| Service-linked role | Batch manages compute resources | Created automatically with the first managed compute environment [23] |

### Where each step runs

| Step | Runs in |
|---|---|
| Edit code, unit tests, CPU smoke training | Codespaces (dev container) |
| Build and push image | GitHub Actions |
| Register job definition, submit job | Codespaces terminal (SSO) or Actions (OIDC) |
| Training | AWS Batch on an EC2 GPU instance |
| Logs and artifacts | CloudWatch Logs, S3 (and MLflow, if reachable from the job's subnet) |

### Implementation steps

**How / verify.**

Step 0. GPU vCPU quota. Default quotas for GPU families can be 0 [29]; request an increase for "Running On-Demand G and VT instances" in Service Quotas [30]. One `g5.xlarge` has 4 vCPUs [18]. Set `maxvCpus` at or below the approved quota.

Step 1. Image. Reuse the Packaging workflow with `-f Dockerfile.gpu` instead of `--target runtime` (this file has a single stage) and repository `ml-exercise/train-gpu`. Choose the PyTorch wheel index whose CUDA build is compatible with the AMI driver [31][27]; the smoke job below confirms it.

```dockerfile
# Dockerfile.gpu
FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
WORKDIR /workspace
RUN pip install --no-cache-dir torch==2.6.0 --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir boto3
COPY src/ src/
CMD ["python", "src/train.py"]
```

```python
# src/train.py: minimal GPU job, seeded, resumable from an S3 checkpoint, logs to stdout
import os
import boto3
import torch
import torch.nn as nn
from botocore.exceptions import ClientError

SEED = 42


def make_data(n=2048):
    # Synthetic stand-in. Replace with a dataset read from S3 using a fixed, versioned split.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = (x.mean(dim=(1, 2, 3)) > 0).long()
    return x[:1600], y[:1600], x[1600:], y[1600:]  # train / validation


def build():
    return nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(16, 2))


def main():
    bucket, prefix = os.environ["ARTIFACT_BUCKET"], os.environ["ARTIFACT_PREFIX"]
    epochs = int(os.environ.get("EPOCHS", "5"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    torch.manual_seed(SEED)
    xtr, ytr, xva, yva = make_data()
    model = build().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    s3, ckpt, key = boto3.client("s3"), "/tmp/ckpt.pt", f"{prefix}/ckpt.pt"
    start = 0
    try:  # resume after a retry or a Spot interruption
        s3.download_file(bucket, key, ckpt)
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["epoch"] + 1
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
    for ep in range(start, epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            xb, yb = xtr[idx].to(dev), ytr[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xva.to(dev)).argmax(1).cpu() == yva).float().mean().item()
        print(f"epoch={ep} train_loss={loss.item():.4f} val_acc={acc:.3f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep}, ckpt)
        s3.upload_file(ckpt, bucket, key)
    torch.save(model.state_dict(), "/tmp/model.pt")
    s3.upload_file("/tmp/model.pt", bucket, f"{prefix}/model.pt")


if __name__ == "__main__":
    main()
```

Step 2. Roles. Create the instance profile `ecsInstanceRole` [22] and a job role limited to the artifact prefix:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::<bucket>/runs/*" },
    { "Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::<bucket>",
      "Condition": { "StringLike": { "s3:prefix": "runs/*" } } }
  ]
}
```

Step 3. Compute environment and queue (run from a Codespaces terminal with an SSO session; the Batch service-linked role is created automatically [23]):

```bash
aws batch create-compute-environment --compute-environment-name gpu-ce \
  --type MANAGED --state ENABLED --compute-resources '{
    "type": "EC2",
    "allocationStrategy": "BEST_FIT_PROGRESSIVE",
    "minvCpus": 0, "maxvCpus": 8,
    "instanceTypes": ["g5", "g6"],
    "subnets": ["<subnet-id>"], "securityGroupIds": ["<sg-id>"],
    "instanceRole": "arn:aws:iam::<account-id>:instance-profile/ecsInstanceRole"
  }'
aws batch create-job-queue --job-queue-name gpu-queue --state ENABLED --priority 1 \
  --compute-environment-order order=1,computeEnvironment=gpu-ce
```

The subnets must let instances reach ECR, S3, and CloudWatch Logs (NAT gateway or VPC endpoints).

Spot variant (cheaper, interruptible): set `"type": "SPOT"` and `"allocationStrategy": "SPOT_CAPACITY_OPTIMIZED"`, allow several instance types, keep each checkpointed segment short (AWS advises against jobs of an hour or more on Spot), and use automated retries [26]. The training script above already resumes from S3.

Step 4. Job definition, stored in the repo as `batch/gpu-train.json` (`IMAGE_URI` is replaced by CI):

```json
{
  "jobDefinitionName": "gpu-train",
  "type": "container",
  "retryStrategy": { "attempts": 2 },
  "timeout": { "attemptDurationSeconds": 14400 },
  "containerProperties": {
    "image": "IMAGE_URI",
    "jobRoleArn": "arn:aws:iam::<account-id>:role/ml-train-job-role",
    "resourceRequirements": [
      { "type": "VCPU", "value": "4" },
      { "type": "MEMORY", "value": "14000" },
      { "type": "GPU", "value": "1" }
    ],
    "linuxParameters": { "sharedMemorySize": 2048 },
    "environment": [{ "name": "ARTIFACT_BUCKET", "value": "<bucket>" }]
  }
}
```

- Memory is set below the instance's 16 GiB to leave headroom (see the Batch memory-management note in [20]).
- The default 64 MB of shared memory is too small for PyTorch data-loader workers; `sharedMemorySize` raises it [34].

Step 5. Smoke test, then the real run (CLI shown; the CI variant follows):

```bash
SHA=$(git rev-parse HEAD)
REV=$(sed "s|IMAGE_URI|<account-id>.dkr.ecr.eu-west-1.amazonaws.com/ml-exercise/train-gpu:${SHA}|" \
      batch/gpu-train.json > /tmp/jd.json \
      && aws batch register-job-definition --cli-input-json file:///tmp/jd.json \
         --query revision --output text)

# GPU visibility check (expected output: True <GPU name>)
aws batch submit-job --job-name gpu-smoke --job-queue gpu-queue \
  --job-definition gpu-train:${REV} \
  --container-overrides '{"command":["python","-c","import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"]}'

# Real run
aws batch submit-job --job-name train-${SHA:0:8} --job-queue gpu-queue \
  --job-definition gpu-train:${REV} \
  --container-overrides "{\"environment\":[{\"name\":\"ARTIFACT_PREFIX\",\"value\":\"runs/${SHA}\"},{\"name\":\"EPOCHS\",\"value\":\"20\"}]}"
```

CI variant (manual trigger; role with permission to register job definitions and submit jobs):

```yaml
# .github/workflows/train-gpu.yml
name: train-gpu
on:
  workflow_dispatch:
    inputs:
      epochs:
        default: "20"
permissions:
  id-token: write
  contents: read
jobs:
  submit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: aws-actions/configure-aws-credentials@v6
        with:
          role-to-assume: arn:aws:iam::<account-id>:role/gha-batch-submit
          aws-region: eu-west-1
      - run: |
          IMAGE="<account-id>.dkr.ecr.eu-west-1.amazonaws.com/ml-exercise/train-gpu:${{ github.sha }}"
          sed "s|IMAGE_URI|${IMAGE}|" batch/gpu-train.json > /tmp/jd.json
          REV=$(aws batch register-job-definition --cli-input-json file:///tmp/jd.json \
                --query revision --output text)
          aws batch submit-job --job-name "train-${GITHUB_SHA::8}" --job-queue gpu-queue \
            --job-definition "gpu-train:${REV}" \
            --container-overrides "{\"environment\":[{\"name\":\"ARTIFACT_PREFIX\",\"value\":\"runs/${GITHUB_SHA}\"},{\"name\":\"EPOCHS\",\"value\":\"${{ inputs.epochs }}\"}]}"
```

Step 6. Monitor and verify:
- `aws batch describe-jobs --jobs <job-id>` shows the job status; logs are in CloudWatch Logs, default log group `/aws/batch/job` [25].
- The smoke job prints `True` and a GPU name; if not, the wheel's CUDA build and the AMI driver do not match.
- `s3://<bucket>/runs/<sha>/model.pt` exists after the run.
- Idle check: after the queue drains, the compute environment returns to its minimum and no GPU instance stays running [34].
- Resume test: cancel a running job or force a retry and confirm the log shows the epoch counter continuing from the checkpoint.

**Pros.**
- GPU cost only while jobs run; the IDE never needs a GPU.
- Each run maps to one git SHA, one image tag, and one job definition revision.
- The same path scales to larger instances or several jobs (array jobs [24]).
- No static credentials: SSO for humans, OIDC for CI, job role for the container.

**Risks.**
- GPU quota of 0 leaves jobs stuck in RUNNABLE until it is raised [29][30]; so does listing instance families outside the supported GPU set [18].
- CUDA/driver mismatch between the PyTorch wheel and the AMI [27][31]; the smoke job catches it.
- Two jobs that each request 1 GPU need two GPUs, so parallel runs multiply cost [18].
- Runaway cost: cap `maxvCpus`, always set a job `timeout` [24], and add a budget alert (assessment).
- Spot interruption without checkpointing loses progress [26].
- Leakage and reproducibility across retries:
  - The train/validation/test split must be fixed, versioned in S3, and independent of RNG state; a retried job must reload the same split.
  - Use the validation split for checkpoint selection and early stopping; touch the test split once, at the end.
  - Log the git SHA, the split identifier, and the seed with every run.
  - After a resume, shuffling order differs from an uninterrupted run; treat exact bitwise reproducibility as out of scope.

## Diagnostics & Pitfalls

Cross-cutting risks that span multiple sections. Each section has its own Risks subsection with detail; this is the "if something goes wrong, check here first" index.

- **Credential precedence:** the boto3 chain stops at the first source that returns credentials [39]. A stray `AWS_ACCESS_KEY_ID` in the environment overrides a job role or SSO session. Run `aws sts get-caller-identity` in each environment and confirm the expected identity (Cloud Access, verification step).
- **Architecture mismatch:** an image built on an ARM laptop fails on x86 EC2 instances [21]. Always build with `--platform linux/amd64` (Packaging).
- **GPU quota at zero:** default On-Demand vCPU quotas for GPU families can be 0 [29][30]. Jobs stay in RUNNABLE indefinitely. Check the quota before creating the compute environment (Remote GPU Training, step 0).
- **CUDA / driver mismatch:** the PyTorch wheel's CUDA build must match the NVIDIA driver on the ECS GPU-optimized AMI [27][31]. The smoke job catches it; if it prints `False`, rebuild the image with a compatible wheel index.
- **Leakage across retries:** a retried or resumed job must reload the same fixed, versioned train/validation/test split from S3, independent of RNG state. Touch the test split once, at the end. Log the git SHA, the split identifier, and the seed with every run (Remote GPU Training, Risks).
- **Cost runaway:** cap `maxvCpus` on the compute environment, always set a job `timeout` [24], stop or delete idle Codespaces [3], and add a budget alert.
- **GCP credential confusion:** CLI credentials and Application Default Credentials are distinct [47]. A `gcloud` command can succeed while the same Python code fails. Run `gcloud auth application-default login` separately for SDK use (Cloud Access, Risks).
- **Mutable image tags:** without ECR tag immutability, pushing the same tag overwrites the image and a rerun is no longer the same run [15].
- **Secrets in layers:** never `COPY` datasets or credentials into the image. Data lives in S3; AWS access comes from the job role [17][21].

## Decision Rule

1. **Tabular model (LR / SVM / RF / XGBoost), CPU sufficient** → develop and train entirely in Codespaces. Use the multi-stage Dockerfile (`dev` target for the dev container, `runtime` target for CI and jobs).
2. **CNN / RNN / any model requiring GPU** → develop and smoke-test in Codespaces (CPU, tiny sample). Train as an AWS Batch job on EC2 GPU instances. Use `Dockerfile.gpu` with the appropriate PyTorch CUDA wheel.
3. **Need in-network access to cloud data or services (VPC, IAM-restricted)** → consider GCP or Azure Cloud IDE instead of Codespaces. The rest of the stack (CI, packaging, remote jobs) stays the same.
4. **Team onboarding** → share the repository. The dev container rebuilds the environment; no manual setup.
5. **Spot interruption risk** → set `"type": "SPOT"` with `SPOT_CAPACITY_OPTIMIZED`, allow multiple instance families, checkpoint frequently to S3, and enable automated retries [26].
6. **Debugging a GPU issue** → run the smoke job (Remote GPU Training, step 5) before the real training. If `torch.cuda.is_available()` returns `False`, the CUDA wheel and AMI driver do not match.

*Problem-specific considerations are covered inline: the context box states the exercise constraints, and each section addresses them where they apply. A separate section is omitted because there is no model-specific or metric-specific analysis in this report.*

## References

1. AWS Cloud9 User Guide — Document history: https://docs.aws.amazon.com/cloud9/latest/user-guide/history.html
2. GitHub Docs — Introduction to dev containers: https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/introduction-to-dev-containers
3. GitHub Docs — Changing the machine type for your codespace: https://help.github.com/en/codespaces/customizing-your-codespace/changing-the-machine-type-for-your-codespace
4. GitHub Docs — Configuring OpenID Connect in Amazon Web Services: https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services
5. GitHub Changelog — Upcoming deprecation of GPU machine type in Codespaces: https://github.blog/changelog/2025-08-01-upcoming-deprecation-of-gpu-machine-type-in-codespaces/
6. Visual Studio Code Docs — Developing inside a Container: https://code.visualstudio.com/docs/remote/dev-containers
7. Development Containers — Dev Container metadata reference: https://containers.dev/implementors/json_reference/
8. GitHub Docs — Opening an existing codespace: https://docs.github.com/en/codespaces/developing-in-a-codespace/opening-an-existing-codespace
9. devcontainers/features — aws-cli Feature: https://github.com/devcontainers/features/pkgs/container/features%2Faws-cli
10. Microsoft Learn — Dev containers (example using the docker-in-docker Feature): https://learn.microsoft.com/dotnet/aspire/get-started/dev-containers
11. Example NVIDIA GPU dev container using `--gpus=all` (third-party): https://www.github.com/psaboia/devcontainer-nvidia-base
12. AWS CLI Reference — `aws sso login`: https://docs.aws.amazon.com/cli/latest/reference/sso/login.html
13. AWS CLI User Guide — Configuring IAM Identity Center authentication: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.md
14. Amazon ECR User Guide — Pushing a Docker image to a private repository: https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html
15. Amazon ECR User Guide — Preventing image tags from being overwritten: https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-tag-mutability.md
16. AWS CLI Reference (v1) — `aws ecr create-repository` (notes deprecation of repository-level scanning configuration): https://docs.aws.amazon.com/cli/v1/reference/ecr/create-repository.html
17. AWS Batch User Guide — Components of AWS Batch: https://docs.aws.amazon.com/batch/latest/userguide/batch_components.html
18. AWS Batch User Guide — Run GPU jobs: https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html
19. AWS Batch User Guide — Use a GPU workload AMI: https://docs.aws.amazon.com/batch/latest/userguide/batch-gpu-ami.md
20. AWS Batch API Reference — ResourceRequirement: https://docs.aws.amazon.com/batch/latest/APIReference/API_ResourceRequirement.html
21. AWS Batch API Reference — ContainerProperties: https://docs.aws.amazon.com/batch/latest/APIReference/API_ContainerProperties.html
22. AWS Batch User Guide — Amazon ECS instance role: https://docs.aws.amazon.com/batch/latest/userguide/instance_IAM_role.html
23. AWS Batch User Guide — Using roles for AWS Batch (service-linked role): https://docs.aws.amazon.com/batch/latest/userguide/using-service-linked-roles-batch-general.html
24. AWS CLI Reference — `aws batch submit-job`: https://docs.aws.amazon.com/cli/latest/reference/batch/submit-job.html
25. AWS Batch User Guide — Use the awslogs log driver: https://docs.aws.amazon.com/batch/latest/userguide/using_awslogs.html
26. AWS Batch User Guide — Use Amazon EC2 Spot best practices for AWS Batch: https://docs.aws.amazon.com/batch/latest/userguide/bestpractice6.html
27. Amazon ECS Developer Guide — Task definitions for GPU workloads: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html
28. Amazon ECS Developer Guide — Task definition differences for Fargate (AWS China docs site): https://docs.amazonaws.cn/en_us/AmazonECS/latest/developerguide/fargate-tasks-services.html
29. Amazon EC2 — Instance type quotas (AWS China docs site): https://docs.amazonaws.cn/en_us/ec2/latest/instancetypes/ec2-instance-quotas.html
30. AWS re:Post Knowledge Center — Request a vCPU service quota increase for EC2 On-Demand Instances: https://repost.aws/knowledge-center/ec2-on-demand-instance-vcpu-increase
31. PyTorch — Installing previous versions (CUDA wheel indexes): https://docs.pytorch.org/get-started/previous-versions/
32. GitHub — aws-actions/configure-aws-credentials releases: https://github.com/aws-actions/configure-aws-credentials/releases
33. GitHub — actions/checkout releases: https://github.com/actions/checkout/releases
34. OneUptime — How to Configure AWS Batch for GPU Workloads (practitioner guide, 2026-02-12): https://oneuptime.com/blog/post/2026-02-12-configure-aws-batch-for-gpu-workloads/view
35. AWS — What's the difference between SDK and API?: https://aws.amazon.com/compare/the-difference-between-sdk-and-api/
36. AWS — What is an SDK?: https://aws.amazon.com/what-is/sdk/
37. PyPI — botocore (foundation of the AWS CLI and boto3): https://pypi.org/project/botocore/
38. AWS General Reference — Signing AWS API requests (SigV4): https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
39. Boto3 Guide — Credentials: https://docs.aws.amazon.com/boto3/latest/guide/credentials.html
40. AWS SDK for Java 2.x Developer Guide — Default credentials provider chain: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/credentials-chain.md
41. AWS SDKs and Tools Reference Guide — Standardized credential providers: https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.md
42. AWS SDKs and Tools Reference Guide — Retry behavior: https://docs.aws.amazon.com/sdkref/latest/guide/feature-retry-behavior.html
43. AWS CLI User Guide — Troubleshooting (`--debug` option): https://docs.aws.amazon.com/cli/latest/userguide/troubleshooting.html
44. Boto3 source — `set_stream_logger` (wire-trace warning): https://github.com/boto/boto3/blob/f4de2990399f4ccb307ee610763b6a0d95997b6d/boto3/__init__.py
45. Google Cloud — gcloud CLI overview: https://docs.cloud.google.com/sdk/gcloud
46. Google Cloud — Client libraries and Cloud APIs explained: https://docs.cloud.google.com/apis/docs/client-libraries-explained
47. Google Cloud — How Application Default Credentials works: https://docs.cloud.google.com/docs/authentication/application-default-credentials
48. Microsoft Learn — What is the Azure CLI?: https://learn.microsoft.com/en-us/cli/azure/what-is-azure-cli
49. Microsoft Learn — Credential chains in the Azure Identity client library for Python: https://learn.microsoft.com/en-in/azure/developer/python/sdk/authentication/credential-chains
50. Microsoft Learn — DefaultAzureCredential class (azure-identity): https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.aio.defaultazurecredential?view=azure-python
51. Azure CLI repository documentation summary (REST API spec, then Python SDK, then CLI commands; third-party index): https://docsearch.algolia.com/mcp/docs/repo/azure/azure-cli
52. AWS SDK for Rust Developer Guide — Credential provider chain (shared files, `AWS_PROFILE`): https://docs.aws.amazon.com/sdk-for-rust/latest/dg/credproviders.md
