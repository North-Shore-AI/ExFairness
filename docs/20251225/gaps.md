# ExFairness - Gap Analysis

**Date:** 2025-12-25
**Version:** 0.4.0

## Critical Gaps

### 1. Missing Crucible.Stage Behaviour Implementation

**Priority: HIGH**

The existing `ExFairness.Stage` module (lines 1-311 in `lib/ex_fairness/stage.ex`) does NOT implement the `@behaviour Crucible.Stage` from `crucible_framework`.

**Current state:**
- Uses ad-hoc interface matching CrucibleIR patterns
- `run/2` returns `{:ok, context} | {:error, String.t()}`
- `describe/1` returns `String.t()`

**Required for Crucible.Stage behaviour:**
- `@behaviour Crucible.Stage`
- `run/2` must accept `%Crucible.Context{}` and return `{:ok, Context.t()} | {:error, term()}`
- `describe/1` must return `map()` (optional callback)

**Impact:** Cannot be used as a pipeline stage in crucible_framework without adaptation.

### 2. Incomplete Calibration Integration in Report

**Priority: MEDIUM**

The `ExFairness.Report` module includes calibration but has specific requirements:
- Lines 202-218: Requires `:probabilities` option explicitly
- Not automatically included in default metrics unless probabilities provided
- Inconsistent with other metrics that only need predictions/labels

### 3. No CrucibleIR Type Checks

**Priority: MEDIUM**

The `ExFairness.Stage` module pattern-matches on CrucibleIR.Reliability.Fairness struct (line 136) but:
- No compile-time verification that crucible_ir is available
- Tests define a mock struct (lines 8-16 in stage_test.exs)
- May fail at runtime if crucible_ir not loaded

### 4. Missing Metrics (Per README "Coming Soon")

**Priority: LOW** (documented as planned)

From README.md lines 50-56:
- Individual Fairness
- Counterfactual Fairness
- Intersectional Analysis (multi-attribute)
- Temporal Monitoring (drift detection)

### 5. Missing Mitigation Techniques

**Priority: LOW** (documented as planned)

From README.md lines 54-55:
- Resampling (oversampling/undersampling)
- Threshold Optimization (group-specific thresholds)

## Incomplete Features

### 1. Incomplete Chi-Square CDF Implementation

**Location:** `lib/ex_fairness/utils/statistical_tests.ex` lines 450-452

```elixir
# Continued fraction for incomplete gamma
@spec cf_gamma(float(), float()) :: float()
# Simplified; full implementation needed for production
defp cf_gamma(_k, _x), do: 1.0
```

This is a stub that returns 1.0, affecting chi-square p-value accuracy for small degrees of freedom.

### 2. Permutation Test Sensitivity Handling

**Location:** `lib/ex_fairness/utils/statistical_tests.ex` lines 311-319

The permutation test only permutes the sensitive attribute, which is correct but:
- No handling for edge cases with very small sample sizes
- No option for stratified permutation

### 3. No Multi-Group Support

**Current limitation:** All metrics assume binary sensitive attributes (0 or 1).

**Affected modules:**
- `ExFairness.Utils` (lines 64-66, 118-126)
- `ExFairness.Validation` (lines 196-214, 216-239)
- All metric modules

### 4. Missing Nx Backend Configuration

**Issue:** No explicit backend configuration for Nx operations.
- Works with default backend (Nx.BinaryBackend)
- No documentation for GPU acceleration setup
- No EXLA/CUDA integration examples

## Code Quality Gaps

### 1. Dialyzer Warnings Potential

**Location:** `lib/ex_fairness/stage.ex` line 234

```elixir
@spec compute_single_metric(atom(), map(), FairnessConfig.t()) :: map()
```

References `FairnessConfig.t()` which is not defined - should be `CrucibleIR.Reliability.Fairness.t()` or a more specific type.

### 2. Inconsistent Return Type Specifications

**Example:** `lib/ex_fairness/utils/metrics.ex` line 11-16

```elixir
@type confusion_matrix :: %{
        tp: Nx.Tensor.t(),
        fp: Nx.Tensor.t(),
        tn: Nx.Tensor.t(),
        fn: Nx.Tensor.t()
      }
```

Uses `@type` instead of `@typedoc`, may not be exported properly.

### 3. Missing Typespecs for Defn Functions

**Location:** `lib/ex_fairness/utils.ex`, `lib/ex_fairness/utils/metrics.ex`, `lib/ex_fairness/mitigation/reweighting.ex`

Nx.Defn functions have `@spec` annotations but they're not validated by Dialyzer for defn.

## Documentation Gaps

### 1. Missing Architecture Diagram

No visual architecture documentation showing module relationships.

### 2. Incomplete Hexdocs Groups

**Location:** `mix.exs` lines 114-142

Groups are well-defined but missing:
- Utils.Bootstrap
- Utils.StatisticalTests

### 3. No Integration Examples

Missing examples showing:
- End-to-end pipeline with crucible_framework
- GPU acceleration setup
- Real-world dataset usage

## Testing Gaps

### 1. Limited Edge Case Coverage

**Observed in tests:**
- No tests for empty tensor edge cases
- No tests for single-group scenarios (should fail validation)
- No tests for floating-point precision issues

### 2. Missing Property-Based Tests

`stream_data` is included as a dependency but no property-based tests found for:
- Metric computation invariants
- Bootstrap convergence properties
- Statistical test power analysis

### 3. No Integration Tests with Crucible Framework

Tests for `ExFairness.Stage` use mock structs, not actual crucible_framework integration.

## Dependency Issues

### 1. crucible_ir Version

**Location:** `mix.exs` line 46

```elixir
{:crucible_ir, "~> 0.1.1"}
```

Pinned to specific minor version - may cause issues with crucible_framework updates.

### 2. Missing Optional Dependency Documentation

No documentation for:
- How ExFairness works without crucible_ir
- Optional EXLA dependency for GPU acceleration

## Summary Table

| Gap | Priority | Effort | Impact |
|-----|----------|--------|--------|
| No Crucible.Stage behaviour | HIGH | Medium | Blocks framework integration |
| Incomplete chi-square CDF | MEDIUM | Low | Statistical accuracy |
| No multi-group support | MEDIUM | High | Feature limitation |
| FairnessConfig type error | LOW | Low | Dialyzer warnings |
| Missing property tests | LOW | Medium | Test coverage |
| No integration examples | LOW | Medium | Documentation |
