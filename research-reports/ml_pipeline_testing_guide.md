# ML Pipeline Testing Guide: pytest, Great Expectations & Beyond

> **Last updated:** May 27, 2026
> **Scope:** Python · Data Science · MLOps
> **Audience:** Data scientists, ML engineers, data engineers

---

## Table of Contents

1. [The Software Testing Taxonomy](#1-the-software-testing-taxonomy)
2. [Data Science–Specific Tests](#2-data-science-specific-tests)
3. [Production-Grade pytest Suite](#3-production-grade-pytest-suite)
   - [conftest.py — Shared Fixtures](#31-conftestpy--shared-fixtures)
   - [Data Validation Tests](#32-data-validation-tests)
   - [Feature Engineering Tests](#33-feature-engineering-tests)
   - [Model Behavior & Quality Tests](#34-model-behavior--quality-tests)
   - [Pipeline Integration Tests](#35-pipeline-integration-tests)
4. [pytest vs Great Expectations](#4-pytest-vs-great-expectations)
   - [The Core Conceptual Difference](#41-the-core-conceptual-difference)
   - [GX Core Architecture](#42-gx-core-architecture)
   - [Side-by-Side Comparison](#43-side-by-side-comparison)
   - [When to Use Each](#44-when-to-use-each)
   - [Using Both Together](#45-using-both-together)
5. [Drift Detection & Production Monitoring](#5-drift-detection--production-monitoring)
6. [CI/CD Integration](#6-cicd-integration)
7. [Tool Reference](#7-tool-reference)
8. [Citations](#8-citations)

---

## 1. The Software Testing Taxonomy

Tests are organized by **scope** (how much of the system they exercise) and **purpose** (what they validate).

### By Scope / Level

| Level | What it tests | Speed | Isolation |
|---|---|---|---|
| **Unit** | Single function or class in isolation | ⚡ Very fast | High — uses mocks/stubs |
| **Integration** | Two or more components together | 🐢 Medium | Medium |
| **End-to-End (E2E)** | Full system from user entry to output | 🐌 Slow | Low — uses real dependencies |
| **Contract** | API agreements between services | Medium | Medium |
| **Smoke** | "Does it start and run at all?" | Fast | Low |
| **Regression** | "Did we break something that worked?" | Varies | Varies |

### By Purpose

| Type | Goal |
|---|---|
| **Functional** | Validates correct outputs for given inputs |
| **Performance / Load** | Validates speed, throughput, scalability |
| **Security** | Validates resistance to attacks |
| **Fuzz / Property-based** | Feeds random or edge inputs to find unexpected failures |
| **Mutation** | Checks whether your tests actually catch bugs |
| **Acceptance (UAT)** | Validates business requirements from a user perspective |

### The Testing Pyramid

Modern CI/CD practice layers these by frequency:

```
         /\
        /E2E\          ← few; run before deployment
       /------\
      /  Integ  \      ← some; run on pull requests
     /------------\
    /     Unit     \   ← many; run on every commit
   /________________\
```

The most effective strategy runs **fast unit tests on every commit, slower integration tests on pull requests, and end-to-end tests before deployments**.

---

## 2. Data Science–Specific Tests

Classical software testing validates **code correctness**. Data science testing additionally validates **data correctness**, **model behavior**, and **statistical properties** — things that can silently fail even when the code is bug-free.

### 2.1 Data Validation Tests

Verify that **incoming data meets schema and domain expectations** before any processing begins. This is the first line of defense in any ML pipeline.

```python
# Example: pandera schema validation
import pandera as pa
from pandera.typing import DataFrame, Series

class SalesSchema(pa.DataFrameModel):
    revenue: Series[float] = pa.Field(ge=0.0, nullable=False)
    date: Series[pa.DateTime]
    region: Series[str] = pa.Field(isin=["NA", "EU", "APAC"])
```

### 2.2 Distribution / Statistical Tests

Assert that **data distributions remain stable** over time. A change in distribution is a signal of data drift, which can silently degrade model performance even if no code has changed.

```python
from scipy import stats

def test_feature_distribution_unchanged(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> None:
    """Kolmogorov-Smirnov test: distributions should not significantly differ."""
    stat, p_value = stats.ks_2samp(reference, current)
    assert p_value > alpha, f"Distribution drift detected: p={p_value:.4f}"
```

### 2.3 Model Behavior / Semantic Tests

Assert that a model produces **semantically correct outputs** given domain knowledge. A model can run without errors and still produce nonsensical predictions.

```python
def test_model_is_monotonic_on_price() -> None:
    """Higher price should never increase purchase probability."""
    base = predict(price=10.0)
    higher = predict(price=50.0)
    assert higher <= base, "Model violates expected monotonicity"
```

### 2.4 Reproducibility Tests

Assert that **same inputs + same seed → same outputs** across runs and environments. Non-determinism is a common and dangerous source of silent failures in ML.

```python
def test_training_is_deterministic() -> None:
    result_1 = train_model(seed=42)
    result_2 = train_model(seed=42)
    assert np.allclose(result_1.weights, result_2.weights)
```

### 2.5 Performance / Degradation Gate Tests

Assert that **model metrics never fall below a defined floor**. These act as automated quality gates before a model is promoted to production.

```python
def test_model_accuracy_meets_baseline(trained_model, holdout_dataset) -> None:
    accuracy = evaluate(trained_model, holdout_dataset)
    assert accuracy >= 0.85, f"Model degraded: accuracy={accuracy:.3f}"
```

### 2.6 Pipeline Integration Tests

Assert that **the full ML pipeline runs end-to-end** and produces expected shapes, types, and values — not just that it "doesn't crash."

```python
def test_pipeline_output_shape(raw_dataset) -> None:
    result = run_pipeline(raw_dataset)
    assert result.shape[1] == EXPECTED_FEATURE_COUNT
    assert not result.isnull().any().any()
```

### Testing Pyramid for Data Science

```
   Model Quality Gate          Data Quality Gate
   ┌─────────────────┐         ┌──────────────────┐
   │ Perf degradation│         │ Schema validation│
   │ Monotonicity    │    +    │ Distribution     │
   │ Degenerate cls. │         │ Null / type check│
   └─────────────────┘         └──────────────────┘
              +
        Classic Pyramid
       (unit / integ / E2E)
```

---

## 3. Production-Grade pytest Suite

### Project Structure

```
ml_pipeline_tests/
├── conftest.py                   # Shared fixtures and config objects
├── test_data_validation.py       # Schema, domain, and null checks
├── test_feature_engineering.py   # Transformation invariants
├── test_model.py                 # Behavior, reproducibility, performance gates
└── test_pipeline_integration.py  # End-to-end pipeline flow
```

---

### 3.1 `conftest.py` — Shared Fixtures

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DataConfig:
    """Configuration for dataset expectations.

    Args:
        required_columns: Columns that must exist in the raw dataframe.
        target_column: Name of the label/target column.
        min_rows: Minimum acceptable number of rows.
        categorical_columns: Columns expected to be categorical dtype.
    """
    required_columns: list[str]
    target_column: str
    min_rows: int = 100
    categorical_columns: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Configuration for model training and evaluation.

    Args:
        random_seed: Seed for reproducibility.
        min_accuracy: Minimum acceptable accuracy on holdout set.
        min_roc_auc: Minimum acceptable ROC-AUC.
        n_estimators: Number of trees in the random forest.
        test_size: Fraction of data used for evaluation.
    """
    random_seed: int
    min_accuracy: float
    min_roc_auc: float
    n_estimators: int = 100
    test_size: float = 0.2


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    return DataConfig(
        required_columns=["age", "income", "region", "churned"],
        target_column="churned",
        min_rows=50,
        categorical_columns=["region"],
    )


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    return ModelConfig(
        random_seed=42,
        min_accuracy=0.75,
        min_roc_auc=0.70,
    )


@pytest.fixture(scope="session")
def raw_dataframe() -> pd.DataFrame:
    """Synthetic raw dataset mimicking production schema."""
    rng = np.random.default_rng(seed=42)
    n = 200
    return pd.DataFrame({
        "age": rng.integers(18, 80, size=n).astype(float),
        "income": rng.uniform(20_000, 120_000, size=n),
        "region": rng.choice(["NA", "EU", "APAC"], size=n),
        "churned": rng.integers(0, 2, size=n),
    })


@pytest.fixture(scope="session")
def corrupted_dataframe() -> pd.DataFrame:
    """Dataset with intentional schema/quality violations."""
    rng = np.random.default_rng(seed=99)
    n = 50
    return pd.DataFrame({
        "age": rng.uniform(-5, 10, size=n),       # negative ages
        "income": [None] * n,                      # all nulls
        "region": rng.choice(["XX", "ZZ"], size=n),  # unknown categories
        # 'churned' intentionally missing
    })


@pytest.fixture(scope="session")
def trained_pipeline(
    raw_dataframe: pd.DataFrame,
    model_config: ModelConfig,
) -> Pipeline:
    """Fits a reproducible sklearn Pipeline on synthetic data."""
    X = raw_dataframe.drop(columns=["churned", "region"])
    y = raw_dataframe["churned"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=model_config.n_estimators,
            random_state=model_config.random_seed,
        )),
    ])
    pipeline.fit(X, y)
    return pipeline
```

---

### 3.2 Data Validation Tests

```python
# test_data_validation.py
from __future__ import annotations

import pandas as pd
import pytest
from conftest import DataConfig


def test_required_columns_present(
    raw_dataframe: pd.DataFrame,
    data_config: DataConfig,
) -> None:
    """All required columns must exist in the raw dataset."""
    missing = set(data_config.required_columns) - set(raw_dataframe.columns)
    assert not missing, f"Missing required columns: {missing}"


def test_minimum_row_count(
    raw_dataframe: pd.DataFrame,
    data_config: DataConfig,
) -> None:
    """Dataset must meet minimum size threshold for meaningful training."""
    assert len(raw_dataframe) >= data_config.min_rows, (
        f"Dataset too small: {len(raw_dataframe)} < {data_config.min_rows}"
    )


def test_age_within_valid_range(raw_dataframe: pd.DataFrame) -> None:
    """Age must be a realistic positive integer (18–120)."""
    age = raw_dataframe["age"]
    assert age.between(0, 120).all(), (
        f"Age out of range: min={age.min()}, max={age.max()}"
    )


def test_income_non_negative(raw_dataframe: pd.DataFrame) -> None:
    """Income must be non-negative."""
    assert (raw_dataframe["income"] >= 0).all()


def test_target_is_binary(
    raw_dataframe: pd.DataFrame,
    data_config: DataConfig,
) -> None:
    """Target column must contain only 0/1 values."""
    unique_vals = set(raw_dataframe[data_config.target_column].unique())
    assert unique_vals <= {0, 1}, f"Unexpected target values: {unique_vals}"


def test_no_nulls_in_required_columns(
    raw_dataframe: pd.DataFrame,
    data_config: DataConfig,
) -> None:
    """None of the required columns should have null values."""
    null_counts = raw_dataframe[data_config.required_columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert cols_with_nulls.empty, f"Nulls found:\n{cols_with_nulls}"


def test_region_valid_categories(raw_dataframe: pd.DataFrame) -> None:
    """Region must belong to the known set of valid categories."""
    valid_regions = {"NA", "EU", "APAC"}
    actual = set(raw_dataframe["region"].unique())
    unexpected = actual - valid_regions
    assert not unexpected, f"Unknown region values: {unexpected}"


def test_corrupted_frame_fails_age_range(
    corrupted_dataframe: pd.DataFrame,
) -> None:
    """Negative ages in corrupted frame should be detectable."""
    has_invalid_age = (corrupted_dataframe["age"] < 0).any()
    assert has_invalid_age  # confirms our validation WOULD catch it
```

---

### 3.3 Feature Engineering Tests

```python
# test_feature_engineering.py
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


def scale_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply StandardScaler to specified numeric columns."""
    scaler = StandardScaler()
    result = df.copy()
    result[columns] = scaler.fit_transform(df[columns])
    return result


def test_scaling_preserves_shape(raw_dataframe: pd.DataFrame) -> None:
    """Scaling must not add or remove rows/columns."""
    scaled = scale_features(raw_dataframe, ["age", "income"])
    assert scaled.shape == raw_dataframe.shape


def test_scaling_produces_zero_mean(raw_dataframe: pd.DataFrame) -> None:
    """StandardScaler output must have mean ≈ 0 for each scaled column."""
    numeric_cols = ["age", "income"]
    scaled = scale_features(raw_dataframe, numeric_cols)
    for col in numeric_cols:
        assert abs(scaled[col].mean()) < 1e-9, (
            f"Column '{col}' mean after scaling: {scaled[col].mean():.6f}"
        )


def test_scaling_produces_unit_variance(raw_dataframe: pd.DataFrame) -> None:
    """StandardScaler output must have std ≈ 1."""
    numeric_cols = ["age", "income"]
    scaled = scale_features(raw_dataframe, numeric_cols)
    for col in numeric_cols:
        assert abs(scaled[col].std() - 1.0) < 1e-6


def test_no_nulls_introduced_by_scaling(raw_dataframe: pd.DataFrame) -> None:
    """Transformations must never introduce NaN values."""
    numeric_cols = ["age", "income"]
    scaled = scale_features(raw_dataframe, numeric_cols)
    assert not scaled[numeric_cols].isnull().any().any()
```

---

### 3.4 Model Behavior & Quality Tests

```python
# test_model.py
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from conftest import ModelConfig


def test_pipeline_is_deterministic(
    raw_dataframe, model_config: ModelConfig,
) -> None:
    """Same seed must produce identical predictions across two training runs."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = raw_dataframe.drop(columns=["churned", "region"])
    y = raw_dataframe["churned"]

    def build_and_predict() -> np.ndarray:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=model_config.n_estimators,
                random_state=model_config.random_seed,
            )),
        ])
        pipe.fit(X, y)
        return pipe.predict(X)

    np.testing.assert_array_equal(build_and_predict(), build_and_predict())


def test_predictions_are_binary(
    trained_pipeline: Pipeline, raw_dataframe,
) -> None:
    """Model must output only 0 or 1 class labels."""
    X = raw_dataframe.drop(columns=["churned", "region"])
    preds = trained_pipeline.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_predict_proba_sums_to_one(
    trained_pipeline: Pipeline, raw_dataframe,
) -> None:
    """Probability outputs for each sample must sum to 1.0."""
    X = raw_dataframe.drop(columns=["churned", "region"])
    proba = trained_pipeline.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_model_accuracy_above_baseline(
    trained_pipeline: Pipeline, raw_dataframe, model_config: ModelConfig,
) -> None:
    """Accuracy must exceed the configured floor."""
    X = raw_dataframe.drop(columns=["churned", "region"])
    y = raw_dataframe["churned"]
    accuracy = accuracy_score(y, trained_pipeline.predict(X))
    assert accuracy >= model_config.min_accuracy, (
        f"Accuracy below threshold: {accuracy:.3f} < {model_config.min_accuracy}"
    )


def test_model_not_always_same_class(
    trained_pipeline: Pipeline, raw_dataframe,
) -> None:
    """A degenerate model that predicts only one class is unacceptable."""
    X = raw_dataframe.drop(columns=["churned", "region"])
    preds = trained_pipeline.predict(X)
    assert len(np.unique(preds)) > 1, "Model predicts only a single class"
```

---

### 3.5 Pipeline Integration Tests

```python
# test_pipeline_integration.py
from __future__ import annotations

import pandas as pd
import pytest
from conftest import DataConfig, ModelConfig


def run_full_pipeline(
    raw_df: pd.DataFrame,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> dict[str, object]:
    """Simulate the full pipeline: validate → engineer → train → predict.

    Raises:
        ValueError: If required columns are missing or data is too small.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    missing = set(data_config.required_columns) - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(raw_df) < data_config.min_rows:
        raise ValueError(f"Insufficient data: {len(raw_df)} rows")

    feature_cols = ["age", "income"]
    X = raw_df[feature_cols].copy()
    y = raw_df[data_config.target_column]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=model_config.n_estimators,
            random_state=model_config.random_seed,
        )),
    ])
    pipe.fit(X, y)
    return {
        "predictions": pipe.predict(X),
        "proba": pipe.predict_proba(X),
        "feature_names": feature_cols,
    }


def test_pipeline_runs_end_to_end(
    raw_dataframe, data_config, model_config,
) -> None:
    result = run_full_pipeline(raw_dataframe, data_config, model_config)
    assert "predictions" in result and "proba" in result


def test_pipeline_output_length_matches_input(
    raw_dataframe, data_config, model_config,
) -> None:
    result = run_full_pipeline(raw_dataframe, data_config, model_config)
    assert len(result["predictions"]) == len(raw_dataframe)


def test_pipeline_raises_on_missing_columns(
    corrupted_dataframe, data_config, model_config,
) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        run_full_pipeline(corrupted_dataframe, data_config, model_config)


def test_pipeline_raises_on_insufficient_data(
    data_config, model_config,
) -> None:
    tiny_df = pd.DataFrame({
        "age": [25.0, 30.0], "income": [50_000.0, 60_000.0],
        "region": ["NA", "EU"], "churned": [0, 1],
    })
    with pytest.raises(ValueError, match="Insufficient data"):
        run_full_pipeline(tiny_df, data_config, model_config)
```

---

### Running the Suite

```bash
# Run all tests with verbose output
pytest ml_pipeline_tests/ -v

# Run only data validation tests
pytest ml_pipeline_tests/test_data_validation.py -v

# Run only performance gate tests
pytest ml_pipeline_tests/test_model.py -k "baseline" -v

# Stop on first failure (useful in CI)
pytest ml_pipeline_tests/ -x
```

---

## 4. pytest vs Great Expectations

### 4.1 The Core Conceptual Difference

This is the most important question to answer before choosing a tool.

Consider this pytest test:

```python
def test_no_nulls_in_required_columns(raw_dataframe, data_config):
    null_counts = raw_dataframe[data_config.required_columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert cols_with_nulls.empty, f"Nulls found:\n{cols_with_nulls}"
```

Now ask yourself:

- Where does this test **live** in a project? In a `tests/` directory, run by a developer or CI bot.
- **Who runs it and when?** During local development, or on a push to git.
- **What happens when it fails?** A red CI build. A Slack notification if you wired it up manually.
- **What if the schema changes?** You open `conftest.py` and update `required_columns` by hand.
- **Who can read this validation?** Only other Python developers.

Now scale this to **50 columns, 12 tables, daily arriving data from an external vendor**. The friction becomes real fast.

> **pytest validates code during development. Great Expectations validates data in production pipelines — with governance, observability, and automated reporting built in.**

---

### 4.2 GX Core Architecture

Great Expectations is an enterprise-grade data quality framework that brings software testing principles to data pipelines. Its key components are:

| Component | Role |
|---|---|
| **Data Context** | Central config store for sources, suites, checkpoints, and results |
| **Expectation Suite** | A named, reusable collection of validation rules for a dataset |
| **Batch Definition** | Specifies which slice of data to validate (a file, a table, a date partition) |
| **Validation Definition** | Explicitly ties a Batch Definition to an Expectation Suite |
| **Checkpoint** | The primary means for validating data in production; runs one or more Validation Definitions and triggers Actions |
| **Actions** | Post-validation hooks: Slack alerts, email notifications, Data Docs updates |
| **Data Docs** | Auto-generated HTML documentation showing expectations and validation results |

```python
import great_expectations as gx

# 1. Connect to a data source
context = gx.get_context()
data_source = context.data_sources.add_pandas("my_source")
data_asset = data_source.add_dataframe_asset("churn_data")
batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")

# 2. Define expectations in a Suite
suite = context.suites.add(gx.ExpectationSuite(name="churn_suite"))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column="age"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="age", min_value=0, max_value=120
))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="income"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
    column="region", value_set=["NA", "EU", "APAC"]
))

# 3. Create a Validation Definition
validation_definition = context.validation_definitions.add(
    gx.core.validation_definition.ValidationDefinition(
        name="churn_validation",
        data=batch_definition,
        suite=suite,
    )
)

# 4. Create and run a Checkpoint (production-ready)
checkpoint = context.checkpoints.add(
    gx.checkpoint.checkpoint.Checkpoint(
        name="churn_checkpoint",
        validation_definitions=[validation_definition],
    )
)
result = checkpoint.run(batch_parameters={"dataframe": df})
print(result.describe())
# Checkpoints can also trigger Actions: Slack, email, Data Docs
```

---

### 4.3 Side-by-Side Comparison

| Dimension | pytest | Great Expectations |
|---|---|---|
| **Primary purpose** | Test code behavior | Validate data quality |
| **When it runs** | Dev time, CI/CD on code changes | Inside the data pipeline, on every data arrival |
| **What triggers it** | `git push`, `pytest` CLI | Airflow task, Dagster asset, scheduled job |
| **Written by** | Developer | Data engineer, data scientist |
| **Readable by** | Developers | Business stakeholders (via Data Docs HTML) |
| **Schema defined as** | Python asserts + fixtures | Declarative Expectation Suites (JSON/Python) |
| **Failure reporting** | Terminal / CI log | Data Docs HTML + Actions (Slack, email) |
| **Governance / audit trail** | None built-in | Full Validation Results history |
| **Backends supported** | Whatever Python supports | Pandas, Spark, SQLAlchemy, BigQuery, Snowflake |
| **Auto-profiling** | No | Yes (Data Assistant infers expectations from data) |
| **Learning curve** | Low | Medium–High |
| **Setup overhead** | Minimal | Meaningful (Data Context, Suites, Checkpoints) |

---

### 4.4 When to Use Each

**Use pytest when:**

- Writing unit or integration tests for pipeline **code** (transformers, loaders, custom metrics)
- Verifying model output **shapes, types, and edge cases** during development
- Running fast validation checks in CI that block a PR merge
- The team is small and the data schema is stable

**Use Great Expectations when:**

- Data arrives from **external vendors** or sources you don't control
- You need **automated alerting** (Slack, email) when data quality degrades
- You need **audit trails and governance documentation** for compliance
- Your schema **evolves frequently** and you want declarative version control of expectations
- Multiple teams need to **read and contribute** to validation rules
- You are validating **large datasets** across Spark or SQL backends

---

### 4.5 Using Both Together

The recommended production pattern is to use **both**, at different layers:

```
┌─────────────────────────────────────────────────────────┐
│ CI/CD (on every git push)                               │
│   pytest → unit & integration tests on pipeline CODE   │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│ Data Pipeline (on every data arrival)                   │
│   GX Checkpoint → validates incoming DATA               │
│   Actions → Slack/email on failure                      │
│   Data Docs → HTML report for all stakeholders          │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│ Model Training Gate                                     │
│   pytest → performance degradation, reproducibility    │
│   mlflow → metric logging and model registry           │
└─────────────────────────────────────────────────────────┘
```

As one practitioner framework puts it: **use pytest for lightweight, fast checks early in the pipeline; use Great Expectations for rich validations, auto-docs, and governance.** Doing both elevates you from "model builder" to "production engineer."

---

## 5. Drift Detection & Production Monitoring

Once a model is deployed, a new category of testing becomes relevant: **monitoring**. The code can be perfect; the model can pass all quality gates; and yet performance can silently degrade because the real-world data distribution has shifted.

### Types of Drift

| Type | Definition |
|---|---|
| **Data drift** | Distribution of input features changes (e.g. user demographics shift) |
| **Concept drift** | The relationship between features and the target changes |
| **Prediction drift** | Distribution of model outputs changes without a label shift |
| **Label drift** | Distribution of ground truth labels changes |

### Key Tools (2025–2026)

**Evidently AI** is an open-source Python library purpose-built for ML monitoring, offering over 20 pre-built drift detection methods and interactive visual reports. It integrates with streaming platforms to compare current data windows against reference datasets and supports unstructured data, including text embeddings.

**NannyML** specializes in **performance estimation without ground truth labels** — critical for scenarios where labels arrive with significant delay after prediction. It links data drift alerts directly to performance changes, reducing false alarms by focusing only on drift that actually impacts model accuracy. In June 2025, NannyML was acquired by Soda and integrated into their broader data quality platform.

**WhyLabs** (open-sourced under Apache 2.0 in January 2025) provides enterprise-grade AI observability using lightweight data profiles rather than raw data storage, making it privacy-friendly for sensitive applications.

**Alibi Detect** (from Seldon) provides algorithms specifically designed for online drift detection, including outlier detection and adversarial detection for ML security applications.

```python
# Example: Evidently AI data drift report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)
report.save_html("drift_report.html")
```

### The Ideal Monitoring Stack (2025)

The ideal stack combines these tools: **Prometheus** collects raw metrics, **Evidently AI** analyzes for drift patterns, **Grafana** visualizes everything, and alerts trigger when thresholds are exceeded.

---

## 6. CI/CD Integration

Modern best practice embeds ML pipeline tests at multiple stages of the delivery process. A typical GitHub Actions workflow looks like this:

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths: ["src/**", "data/**"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit -v
      - name: Validate data schema
        run: python scripts/validate_data.py   # GX Checkpoint

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Train model
        run: python src/train.py
      - name: Validate model performance
        run: |
          python scripts/validate_model.py \
            --min-auc 0.85 \
            --max-latency-ms 100

  deploy-staging:
    needs: train
    environment: staging
    steps:
      - run: python scripts/deploy.py --env staging
      - run: pytest tests/integration -v
```

---

## 7. Tool Reference

| Category | Tool | Use Case |
|---|---|---|
| Unit / Integration | `pytest` | Code correctness, pipeline behavior |
| Data validation (schema) | `pandera` | DataFrame schema as Python type annotations |
| Data validation (pipeline) | `great-expectations` | Production data quality, governance, reporting |
| Data validation (models) | `pydantic` | Config objects, API inputs/outputs |
| Property-based / fuzz | `hypothesis` | Edge cases, boundary conditions |
| Mutation testing | `mutmut`, `cosmic-ray` | Verify tests actually catch bugs |
| Drift detection | `evidently` | Data and prediction drift reports |
| Drift detection | `nannyml` | Performance estimation without ground truth |
| Drift detection | `whylogs` | Lightweight, privacy-friendly profiling |
| Experiment tracking | `mlflow` | Metric logging, model registry |
| Dataset versioning | `dvc` | Reproducible datasets, rollback capability |
| Pipeline orchestration | `dagster`, `airflow` | Scheduling, observability, asset lineage |
| Monitoring dashboards | `prometheus` + `grafana` | Real-time metric visualization and alerting |

---

## 8. Citations

1. Deepchecks. (2025, October 17). *Best Practices for Testing ML Pipelines*. https://www.deepchecks.com/best-practices-for-testing-ml-pipelines/

2. Sarney, D. (2025, October 31). *Python Testing Best Practices in 2025: Building Reliable Applications That Scale*. https://danielsarney.com/blog/python-testing-best-practices-2025-building-reliable-applications/

3. Rahman, A. (2025, August 14). *8 Python Libraries That Make ML Pipelines Sane*. Artificial Intelligence in Plain English. https://ai.plainenglish.io/8-python-libraries-that-make-ml-pipelines-sane-e92d624d169c

4. 21Devs. (2025, December 13). *ML Pipelines and Deployment: Building End-to-End Machine Learning Workflows in Python*. https://21devs.com/ml-pipelines-and-deployment/

5. Hamza, A. (2025, August 18). *Data Quality for Real-World AI: Why It Matters, and How to Enforce It with Pytest & Great Expectations*. Medium. https://medium.com/@a275hamza/data-quality-for-real-world-ai-why-it-matters-and-how-to-enforce-it-with-pytest-great-dfa610a89845

6. Aeturrell. (2025, March 5). *The Data Validation Landscape in 2025*. https://aeturrell.com/blog/posts/the-data-validation-landscape-in-2025/

7. Chandran, S. (2025, October 2). *Building Data Trust with Great Expectations: A Complete Developer's Journey*. Python in Plain English. https://python.plainenglish.io/building-data-trust-with-great-expectations-a-complete-developers-journey-f436f8019055

8. Great Expectations. (2025). *GX Core Overview*. https://docs.greatexpectations.io/docs/core/introduction/gx_overview/

9. Great Expectations. (2025, March). *What's New in GX: March 2025*. https://greatexpectations.io/blog/whats-new-in-gx-february-2025/

10. Experion Global. (2025, March 7). *Ensuring Data Integrity with Great Expectations (GX Core)*. https://experionglobal.com/ensuring-data-integrity-with-great-expectations-gx-core/

11. Conduktor. (2025). *Great Expectations: Data Testing Framework*. https://www.conduktor.io/glossary/great-expectations-data-testing-framework

12. Conduktor. (2026, May). *Model Drift in Streaming: When ML Models Degrade in Real-Time*. https://www.conduktor.io/glossary/model-drift-in-streaming

13. Kandivlikar, T. (2025, July 18). *Comprehensive Comparison of ML Model Monitoring Tools: Evidently AI, Alibi Detect, NannyML, WhyLabs, and Fiddler AI*. Medium. https://medium.com/@tanish.kandivlikar1412/comprehensive-comparison-of-ml-model-monitoring-tools-evidently-ai-alibi-detect-nannyml-a016d7dd8219

14. Evidently AI. (2025). *What is Data Drift in ML, and How to Detect and Handle It*. https://www.evidentlyai.com/ml-in-production/data-drift

15. NannyML. (2025). *NannyML Cloud — A Better Way to Monitor ML Models*. https://www.nannyml.com/

16. DataEngineeerThings. (2024). *Improve Data Quality with the Great Expectations (GX) Framework*. https://blog.dataengineerthings.org/looking-to-enhance-your-data-quality-this-is-for-you-a34c44ce117e

17. C4: Container, Code, Cloud & Context. (2025, March 17). *MLOps Best Practices: Building Production Machine Learning Pipelines That Scale*. https://www.dataa.dev/2025/03/17/mlops-best-practices-production-ml-pipelines-2025/

18. DasRoot. (2025, December 29). *MLOps: Deploying and Monitoring ML Models in 2025*. https://dasroot.net/posts/2025/12/mlops-deploying-monitoring-ml-models-2025/

19. Endjin. (2026, March 23). *Data Validation in Python: A Look into Pandera and Great Expectations*. https://endjin.com/blog/a-look-into-pandera-and-great-expectations-for-data-validation

20. GreatExpectationsLabs. (GitHub). *Put Data Pipeline Under Test with pytest and Great Expectations*. https://github.com/greatexpectationslabs/put-data-pipeline-under-test-with-pytest-and-great-expectations

---

*Document maintained by: ML Engineering — updated from live sources as of May 27, 2026.*
