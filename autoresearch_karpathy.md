# karpathy/autoresearch

> **Last updated:** 2026-06-03
> **Repository:** [https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
> **Stars:** ~84k | **License:** MIT | **Requires:** Single NVIDIA GPU, Python 3.10+

---

## What it is

`autoresearch` is a framework that turns an AI coding agent (Claude, Codex, or similar) into an autonomous ML researcher. The agent is given a real LLM training codebase and runs a tight loop: modify code → train for exactly 5 minutes → evaluate → keep or revert → repeat. You set it going before sleeping and wake up to a log of ~100 experiments and (hopefully) a better model.

The training setup is a single-GPU implementation of Karpathy's `nanochat` — a GPT-style character/text model trained on a text corpus. The model, optimizer, and training loop all live in a single file (`train.py`) that the agent freely modifies. Everything else is fixed.

---

## Why it exists / problem it solves

Traditional ML research is bottlenecked by the human iteration cycle: read → hypothesize → code → wait → interpret → repeat. Each loop takes hours or days. `autoresearch` collapses this cycle to ~5 minutes per iteration and removes the human from the inner loop entirely.

The key insight is separating **what to research** (written by the human in `program.md`) from **how to research it** (executed autonomously by the agent on `train.py`). You become the research director; the agent becomes the engineer running experiments.

---

## Repository structure

The repo has three files that actually matter:

| File | Role | Who edits it |
|------|------|--------------|
| `prepare.py` | Constants, data download, BPE tokenizer training, dataloader, evaluation utilities | Nobody — frozen |
| `train.py` | Full GPT model, Muon + AdamW optimizer, training loop | The AI agent |
| `program.md` | Research mandate and context for the agent | The human |

Supporting files: `pyproject.toml` (dependencies via `uv`), `analysis.ipynb` (result analysis), `uv.lock`.

---

## Core design decisions

**Fixed 5-minute time budget.** Every training run lasts exactly 5 minutes of wall-clock time (excluding startup/compilation), regardless of what the agent changes — model depth, batch size, attention pattern, etc. This makes all experiments directly comparable on the same hardware and yields roughly 12 experiments/hour or ~100 experiments overnight. The downside is that results are platform-specific and not comparable across different GPUs.

**Single scalar metric: `val_bpb`.** Validation bits-per-byte is the only signal used to decide keep-or-revert. It is lower-is-better and vocab-size-independent, so the agent can fairly compare runs even when it changes tokenizer vocabulary size or model architecture.

**Single file to modify.** The agent only touches `train.py`. This keeps diffs reviewable, scope manageable, and failure modes contained.

**`program.md` is the lever you control.** The quality of the research mandate you write directly determines the quality of experiments the agent runs. Vague instructions produce random wandering; precise hypotheses produce focused exploration.

---

## Evaluation metric deep dive

```
val_bpb = validation_cross_entropy_nats / log(2)
```

This converts nats to bits and normalizes by the number of bytes (not tokens), making it comparable across different tokenizations. A model with a 256-token byte-level vocabulary and one with an 8192-token BPE vocabulary can be fairly compared. Lower is always better; a random model over a 256-byte vocab scores 8.0 bpb.

---

## Quick start

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/karpathy/autoresearch
cd autoresearch
uv sync

# 3. One-time data prep: downloads FineWeb corpus slice, trains BPE tokenizer (~2 min)
uv run prepare.py

# 4. Sanity check: run one 5-minute training experiment manually
uv run train.py
# Should print val_bpb ~3.x at the end if setup is correct
```

Once `train.py` completes successfully, you are ready to hand control to the agent.

---

## Running the agent

Point any capable coding agent (Claude, Codex CLI, etc.) at the repo root with file-write permissions scoped to `train.py` only. Then send this prompt to kick off the loop:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will read `program.md` for its research mandate, propose a change to `train.py`, run training via `uv run train.py`, read the `val_bpb` output, and decide whether to keep the change or revert. It then repeats.

---

## `program.md` — structure and best practices

`program.md` is the most important file you control. It is the agent's only source of context about what to do, what has been tried, what the rules are, and what the current state of the research is. Think of it as the combination of a research brief, a lab notebook, and a set of operating procedures.

Below is a reference template with explanations of each section.

---

### Reference template

```markdown
# Research Program

## Goal
Minimize val_bpb on the FineWeb training setup within a 5-minute wall-clock budget per run,
using a single H100 GPU. The baseline val_bpb is 3.42 (see Experiment Log).

## Constraints
- Only modify train.py. Do not touch prepare.py or program.md.
- Each experiment must complete within the 5-minute budget. If a run times out or OOMs, revert immediately.
- After every experiment (pass or fail), append one row to the Experiment Log below.
- Keep changes atomic: one hypothesis per experiment. Do not bundle multiple changes.

## Current Best Configuration
- val_bpb: 3.38
- Commit: <paste relevant diff or describe key changes>
- Notable: Replacing ReLU with SwiGLU in FFN layers gave +0.04 bpb improvement.

## Active Hypotheses
These are ranked by expected gain. Start from the top.

1. Try a cosine learning rate schedule with a longer warmup (500 steps instead of 200).
   Rationale: current LR drops too fast; loss is still volatile at step 200.
2. Test replacing standard LayerNorm with RMSNorm (remove mean-centering).
   Rationale: RMSNorm is faster and performs similarly on language tasks.
3. Experiment with a WINDOW_PATTERN of "SSSL" (3 local + 1 global attention).
   Rationale: local attention is cheaper; the global head should preserve long-range info.

## Ruled Out (do not retry without new evidence)
- Increasing DEPTH from 8 to 12: caused OOM on H100 with TOTAL_BATCH_SIZE=2**17. (Run #004)
- Dropout > 0.1: consistently hurt val_bpb. (Runs #006, #007)

## Experiment Log

| # | Description | val_bpb | Δ vs prev best | Kept? | Notes |
|---|-------------|---------|----------------|-------|-------|
| 001 | Baseline | 3.42 | — | ✅ | Initial run, no changes |
| 002 | LR 3e-4 → 1e-3 | 3.51 | +0.09 | ❌ | Unstable loss, high variance |
| 003 | SwiGLU in FFN | 3.38 | −0.04 | ✅ | Clean improvement |
| 004 | DEPTH 8 → 12 | OOM | — | ❌ | Reverted; reduce batch if retrying |

## Operating Procedure
1. Read this file fully before starting.
2. Pick the top hypothesis from Active Hypotheses.
3. Edit train.py to implement it. Keep the diff minimal.
4. Run: uv run train.py
5. Record val_bpb from the final output line.
6. If val_bpb < current best: keep the change, update Current Best Configuration.
   If val_bpb >= current best: git checkout train.py (full revert).
7. Move the hypothesis from Active Hypotheses to Ruled Out (if failed) or remove it (if succeeded).
8. Append a row to Experiment Log.
9. Propose the next hypothesis (add it to Active Hypotheses) based on what you learned.
10. Repeat from step 2.
```

---

### Section-by-section explanation

**Goal.** A single, unambiguous objective with the metric name, direction (lower/higher), hardware context, and a reference baseline number. Without a numeric baseline, the agent cannot know whether it is making progress.

**Constraints.** Hard rules the agent must not break. Explicit file-scope restrictions prevent the agent from accidentally modifying `prepare.py` or the metric computation. The OOM/timeout rule prevents wasted runs from propagating bad state.

**Current Best Configuration.** The agent must always know where "home base" is. Include the val_bpb score and enough description of the current `train.py` state to reconstruct it if needed. Update this section every time a change is kept.

**Active Hypotheses.** A prioritized queue of things to try. Ranked order matters — it prevents the agent from picking randomly or revisiting things implicitly. Each hypothesis should have a one-sentence rationale grounded in ML reasoning, not just "try X to see what happens."

**Ruled Out.** Equally important as Active Hypotheses. This section prevents the agent from looping back to failed ideas. Reference the specific run number so you can cross-check against the log.

**Experiment Log.** The lab notebook. Every run gets a row regardless of outcome. The Δ column relative to the previous best is the key signal — it tells you at a glance whether the research is converging. Include a brief Notes column for anything that would be useful context for the next hypothesis.

**Operating Procedure.** A numbered, unambiguous loop the agent follows. The most important rules here are: (1) one hypothesis per experiment — no bundling, (2) full revert on failure — no partial keeps, (3) always update the log before proposing the next experiment.

---

## Good practices for `program.md`

**Keep hypotheses grounded in theory.** "Try a different learning rate" is weak. "Reduce peak LR from 3e-4 to 1e-4 because the loss curve shows oscillation in the first 100 steps" is actionable and falsifiable.

**Update `program.md` as the agent runs.** After each experiment, the agent should update the log, move hypotheses between sections, and propose new ones. If you periodically review and refine the active hypotheses list yourself, the research quality improves dramatically.

**Keep diffs atomic.** One change per experiment. Bundling changes makes it impossible to attribute improvements or failures to specific modifications.

**Set explicit failure conditions.** OOM, timeout, and NaN loss are all failure modes that should trigger an immediate revert without counting the run as a hypothesis test. Define these in the Constraints section.

**Tune `program.md` itself iteratively.** The meta-insight of the project is that `program.md` is itself a program. Over time, refine the operating procedure, add new hypothesis-generation strategies, and prune sections that create confusion.

---

## Adapting to your own problem

`autoresearch` is intentionally minimal. To apply the same agentic loop to a different ML problem:

1. Replace `train.py` with your training script. The only interface the agent needs is a final output line containing your metric (e.g., `val_bpb: 3.38`).
2. Replace `prepare.py` with your data pipeline. Keep it frozen.
3. Rewrite `program.md` with your metric, baseline, constraints, and initial hypotheses.
4. Adjust the time budget in `train.py` to something appropriate for your problem (5 minutes is a good starting point for single-GPU setups).

---

## Platform notes and forks

The main repo requires a single NVIDIA GPU and is tested on an H100. For other hardware:

| Platform | Fork |
|----------|------|
| macOS (MLX) | [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) |
| macOS (general) | [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) |
| Windows / RTX | [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) |
| AMD | [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) |

For smaller compute (laptops, Macbooks), key tuning knobs are: lower `DEPTH` (try 4), lower `MAX_SEQ_LEN` (try 256–512), lower `TOTAL_BATCH_SIZE` (try `2**14`), use `WINDOW_PATTERN = "L"` (skip banded attention), and switch to a low-entropy dataset like [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).

---

## Related projects

- [`karpathy/nanochat`](https://github.com/karpathy/nanochat) — the parent training codebase `autoresearch` is built on top of.
- [`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT) — earlier, simpler GPT training repo; good reference for understanding the architecture.

---

*Tags: `llm-training`, `ai-agents`, `automl`, `research-automation`, `gpt`, `pytorch`, `single-gpu`*
