# ExFairness - Current State Documentation

**Date:** 2025-12-25
**Version:** 0.4.0
**Repository:** /home/home/p/g/North-Shore-AI/ExFairness

## Overview

ExFairness is a comprehensive library for detecting, measuring, and mitigating bias in AI/ML systems built with Elixir. It provides fairness metrics, bias detection algorithms, and mitigation techniques using Nx tensors for GPU acceleration.

## Architecture

```
ExFairness
├── ExFairness (main module - facade)
├── Metrics/
│   ├── DemographicParity
│   ├── EqualizedOdds
│   ├── EqualOpportunity
│   ├── PredictiveParity
│   └── Calibration
├── Detection/
│   └── DisparateImpact (80% rule)
├── Mitigation/
│   └── Reweighting
├── Utils/
│   ├── Utils (core tensor ops)
│   ├── Metrics (confusion matrix, TPR, FPR, PPV)
│   ├── Bootstrap (confidence intervals)
│   └── StatisticalTests (z-test, chi-square, permutation)
├── Report (comprehensive reporting)
├── Stage (CrucibleIR pipeline integration)
├── Validation (input validation)
└── Error (custom exception)
```

## Module Details

### Main Facade: ExFairness (`lib/ex_fairness.ex`)

**Lines 1-319**

The main entry point providing convenience functions:

| Function | Lines | Description |
|----------|-------|-------------|
| `demographic_parity/3` | 73-75 | Delegates to DemographicParity.compute |
| `equalized_odds/4` | 106-108 | Delegates to EqualizedOdds.compute |
| `equal_opportunity/4` | 129-131 | Delegates to EqualOpportunity.compute |
| `predictive_parity/4` | 152-154 | Delegates to PredictiveParity.compute |
| `calibration/4` | 182-184 | Delegates to Calibration.compute |
| `fairness_report/4` | 212-214 | Generates comprehensive report |
| `evaluate/5` | 261-318 | CrucibleIR.Reliability.Fairness evaluation |

### Fairness Metrics

#### DemographicParity (`lib/ex_fairness/metrics/demographic_parity.ex`)

**Lines 1-159**

Statistical parity requiring equal positive prediction rates across groups.

- **Formula:** `P(Y_hat = 1 | A = 0) = P(Y_hat = 1 | A = 1)`
- **Disparity:** `|rate_A - rate_B|`
- **Key function:** `compute/3` (lines 96-134)
- **Default threshold:** 0.1

**Result type:**
```elixir
%{
  group_a_rate: float(),
  group_b_rate: float(),
  disparity: float(),
  passes: boolean(),
  threshold: float(),
  interpretation: String.t()
}
```

#### EqualizedOdds (`lib/ex_fairness/metrics/equalized_odds.ex`)

**Lines 1-205**

Requires equal TPR and FPR across groups.

- **Formula:** Equal TPR and FPR across groups
- **Key function:** `compute/4` (lines 102-158)
- **Default threshold:** 0.1

**Result type:**
```elixir
%{
  group_a_tpr: float(),
  group_b_tpr: float(),
  group_a_fpr: float(),
  group_b_fpr: float(),
  tpr_disparity: float(),
  fpr_disparity: float(),
  passes: boolean(),
  threshold: float(),
  interpretation: String.t()
}
```

#### EqualOpportunity (`lib/ex_fairness/metrics/equal_opportunity.ex`)

**Lines 1-160**

Relaxed version of equalized odds, only checking TPR equality.

- **Formula:** `P(Y_hat = 1 | Y = 1, A = 0) = P(Y_hat = 1 | Y = 1, A = 1)`
- **Key function:** `compute/4` (lines 94-134)
- **Default threshold:** 0.1

#### PredictiveParity (`lib/ex_fairness/metrics/predictive_parity.ex`)

**Lines 1-159**

Requires equal positive predictive value (precision) across groups.

- **Formula:** `P(Y = 1 | Y_hat = 1, A = 0) = P(Y = 1 | Y_hat = 1, A = 1)`
- **Key function:** `compute/4` (lines 93-133)
- **Default threshold:** 0.1

#### Calibration (`lib/ex_fairness/metrics/calibration.ex`)

**Lines 1-443**

Measures whether predicted probabilities are well-calibrated across groups.

- **Metrics:** ECE (Expected Calibration Error), MCE (Maximum Calibration Error)
- **Key functions:**
  - `compute/4` (lines 119-172)
  - `reliability_diagram/4` (lines 185-223)
- **Options:** `n_bins`, `strategy` (:uniform | :quantile), `threshold`

**Result type:**
```elixir
%{
  group_a_ece: float(),
  group_b_ece: float(),
  disparity: float(),
  passes: boolean(),
  threshold: float(),
  group_a_mce: float(),
  group_b_mce: float(),
  n_bins: integer(),
  strategy: :uniform | :quantile,
  interpretation: String.t()
}
```

### Bias Detection

#### DisparateImpact (`lib/ex_fairness/detection/disparate_impact.ex`)

**Lines 1-166**

EEOC 80% rule (4/5ths rule) for legal compliance.

- **Formula:** `min(rate_A, rate_B) / max(rate_A, rate_B) >= 0.8`
- **Key function:** `detect/3` (lines 92-126)

**Result type:**
```elixir
%{
  group_a_rate: float(),
  group_b_rate: float(),
  ratio: float(),
  passes_80_percent_rule: boolean(),
  interpretation: String.t()
}
```

### Mitigation Techniques

#### Reweighting (`lib/ex_fairness/mitigation/reweighting.ex`)

**Lines 1-161**

Pre-processing technique assigning different weights to training samples.

- **Formula:** `w(a, y) = P(Y = y) / P(A = a, Y = y)`
- **Key function:** `compute_weights/3` (lines 74-90)
- **Targets:** `:demographic_parity`, `:equalized_odds`
- Uses Nx.Defn for GPU acceleration

### Utility Modules

#### Utils (`lib/ex_fairness/utils.ex`)

**Lines 1-127**

Core tensor operations using Nx.Defn:

| Function | Lines | Description |
|----------|-------|-------------|
| `positive_rate/2` | 33-41 | Positive prediction rate for masked subset |
| `create_group_mask/2` | 64-66 | Binary mask for specific group |
| `group_count/2` | 89-92 | Count samples in group |
| `group_positive_rates/2` | 118-126 | Rates for both groups |

#### Utils.Metrics (`lib/ex_fairness/utils/metrics.ex`)

**Lines 1-163**

Confusion matrix and derived metrics using Nx.Defn:

| Function | Lines | Description |
|----------|-------|-------------|
| `confusion_matrix/3` | 46-60 | Computes TP, FP, TN, FN |
| `true_positive_rate/3` | 88-94 | TPR = TP / (TP + FN) |
| `false_positive_rate/3` | 122-128 | FPR = FP / (FP + TN) |
| `positive_predictive_value/3` | 156-162 | PPV = TP / (TP + FP) |

#### Utils.Bootstrap (`lib/ex_fairness/utils/bootstrap.ex`)

**Lines 1-293**

Bootstrap confidence interval computation:

- **Key function:** `confidence_interval/3` (lines 108-145)
- **Methods:** `:percentile`, `:basic`
- **Features:** Stratified sampling, parallel computation
- **Defaults:** 1000 samples, 0.95 confidence level

#### Utils.StatisticalTests (`lib/ex_fairness/utils/statistical_tests.ex`)

**Lines 1-558**

Hypothesis testing for fairness metrics:

| Function | Lines | Description |
|----------|-------|-------------|
| `two_proportion_test/3` | 100-154 | Z-test for demographic parity |
| `chi_square_test/4` | 195-249 | Chi-square for equalized odds |
| `permutation_test/3` | 298-353 | Non-parametric test |
| `cohens_h/2` | 378-380 | Effect size for two proportions |

### Reporting

#### Report (`lib/ex_fairness/report.ex`)

**Lines 1-421**

Comprehensive fairness report generation:

| Function | Lines | Description |
|----------|-------|-------------|
| `generate/4` | 86-109 | Multi-metric report |
| `to_markdown/1` | 134-153 | Markdown export |
| `to_json/1` | 179-181 | JSON export |

**Supports:** demographic_parity, equalized_odds, equal_opportunity, predictive_parity, calibration

### Pipeline Integration

#### Stage (`lib/ex_fairness/stage.ex`)

**Lines 1-311**

CrucibleIR pipeline integration stage.

| Function | Lines | Description |
|----------|-------|-------------|
| `run/2` | 128-152 | Main stage execution |
| `describe/1` | 172-174 | Stage description |

**Context requirements:**
- `experiment.reliability.fairness` - CrucibleIR.Reliability.Fairness config
- `outputs` - List of maps with :prediction, :label, sensitive attribute

**Note:** This stage does NOT implement the `@behaviour Crucible.Stage` from crucible_framework. It uses its own ad-hoc interface matching CrucibleIR patterns.

### Validation

#### Validation (`lib/ex_fairness/validation.ex`)

**Lines 1-240**

Input validation utilities:

| Function | Lines | Description |
|----------|-------|-------------|
| `validate_predictions!/1` | 33-39 | Validates predictions tensor |
| `validate_labels!/1` | 65-71 | Validates labels tensor |
| `validate_sensitive_attr!/2` | 99-109 | Validates sensitive attribute |
| `validate_matching_shapes!/2` | 137-155 | Shape compatibility check |

**Validations:**
- Tensor type check
- Binary values (0 or 1)
- Non-empty tensors
- Multiple groups present
- Sufficient samples per group (default: 10)

### Error Handling

#### Error (`lib/ex_fairness/error.ex`)

**Lines 1-14**

Custom exception module:
```elixir
defexception [:message]
```

## Dependencies

From `mix.exs`:
- `crucible_ir` ~> 0.1.1 - IR integration
- `jason` ~> 1.4 - JSON encoding
- `nx` ~> 0.7 - Tensor operations

Dev/Test:
- `ex_doc`, `dialyxir`, `excoveralls`, `credo`, `stream_data`

## Quality Gates

From `mix.exs` (lines 21-33):
- `warnings_as_errors: true`
- Dialyzer with PLT in `priv/plts`
- ExCoveralls for test coverage

## Test Coverage

Test files in `test/ex_fairness/`:
- `metrics/demographic_parity_test.exs`
- `metrics/equalized_odds_test.exs`
- `metrics/equal_opportunity_test.exs`
- `metrics/predictive_parity_test.exs`
- `metrics/calibration_test.exs`
- `detection/disparate_impact_test.exs`
- `mitigation/reweighting_test.exs`
- `utils_test.exs`
- `utils/metrics_test.exs`
- `utils/bootstrap_test.exs`
- `utils/statistical_tests_test.exs`
- `validation_test.exs`
- `report_test.exs`
- `stage_test.exs`
- `ex_fairness_test.exs` (doctests)

## Summary

ExFairness v0.4.0 is a mature fairness library with:
- 5 fairness metrics (demographic parity, equalized odds, equal opportunity, predictive parity, calibration)
- 1 bias detection method (disparate impact 80% rule)
- 1 mitigation technique (reweighting)
- Statistical inference (bootstrap CI, hypothesis tests)
- Comprehensive reporting (Markdown, JSON)
- CrucibleIR integration via Stage module
- Full input validation
- GPU acceleration via Nx.Defn
