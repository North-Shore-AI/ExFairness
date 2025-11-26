# ExFairness Enhancement Design Document
**Date:** November 25, 2025
**Version:** 0.2.0 → 0.3.0
**Author:** North Shore AI Research Team

---

## Executive Summary

This document outlines comprehensive enhancements to ExFairness to advance from v0.2.0 to v0.3.0, focusing on statistical rigor, additional fairness metrics, and enhanced robustness. The enhancements maintain backward compatibility while significantly expanding the library's capabilities for production fairness assessment.

### Enhancement Overview

| Category | Enhancement | Priority | Effort | Status |
|----------|-------------|----------|---------|--------|
| **Statistical Inference** | Bootstrap Confidence Intervals | HIGH | 2 weeks | Planned |
| **Statistical Inference** | Hypothesis Testing | HIGH | 1 week | Planned |
| **Metrics** | Calibration Fairness | HIGH | 2 weeks | Planned |
| **Robustness** | Enhanced Error Handling | MEDIUM | 1 week | Planned |
| **Performance** | Computation Optimization | MEDIUM | 1 week | Planned |
| **Documentation** | Extended Examples | LOW | 1 week | Planned |

**Total Estimated Effort:** 8 weeks (2 months)

---

## Current State Analysis

### Strengths (v0.2.0)

1. **Solid Foundation**
   - 4 core group fairness metrics fully implemented
   - Robust validation and error handling
   - Comprehensive testing (134 tests, 100% pass rate)
   - Zero compiler warnings
   - GPU-accelerated via Nx.Defn

2. **Production-Ready Features**
   - Disparate Impact detection (EEOC 80% rule)
   - Reweighting mitigation
   - Multi-format reporting (Markdown/JSON)
   - Clear interpretations

3. **Excellent Documentation**
   - 1,437-line README with academic citations
   - Complete API documentation
   - Working examples in all modules
   - Theoretical background explanations

### Identified Gaps

1. **Statistical Rigor**
   - ❌ No confidence intervals for fairness metrics
   - ❌ No hypothesis testing for statistical significance
   - ❌ No effect size measures
   - ❌ No power analysis

2. **Missing Metrics**
   - ❌ Calibration fairness (critical for probability-based decisions)
   - ❌ Conditional demographic parity
   - ❌ Treatment equality
   - ❌ Individual fairness measures

3. **Limited Mitigation**
   - ✅ Reweighting implemented
   - ❌ Resampling techniques
   - ❌ Threshold optimization
   - ❌ Adversarial debiasing

4. **Advanced Features**
   - ❌ Intersectional analysis (multi-attribute fairness)
   - ❌ Temporal drift detection
   - ❌ Multi-class fairness support
   - ❌ Batch fairness monitoring

---

## Enhancement 1: Statistical Inference

### 1.1 Bootstrap Confidence Intervals

**Rationale:**
- Essential for scientific rigor and publication-quality results
- Required for legal defense of fairness claims
- Industry standard in Python frameworks (AIF360, Fairlearn)
- Enables uncertainty quantification

**Mathematical Foundation:**

Bootstrap resampling provides non-parametric confidence intervals without distributional assumptions.

**Algorithm:**
```
For metric M (e.g., disparity):
1. Compute observed metric: M_obs = M(data)
2. For i = 1 to B (bootstrap samples):
   a. Sample n datapoints with replacement: data*_i
   b. Compute M*_i = M(data*_i)
3. Sort {M*_1, ..., M*_B}
4. CI_lower = percentile(α/2)
   CI_upper = percentile(1 - α/2)
```

**Stratified Bootstrap:**
To preserve group proportions, sample separately from each group:
```
For groups A and B with sizes n_A and n_B:
1. Sample n_A from group A with replacement
2. Sample n_B from group B with replacement
3. Combine samples and compute metric
```

**Implementation Plan:**

**Module: `ExFairness.Utils.Bootstrap`**

```elixir
defmodule ExFairness.Utils.Bootstrap do
  @moduledoc """
  Bootstrap confidence interval computation for fairness metrics.

  Implements stratified bootstrap to preserve group proportions and
  parallel computation for performance.

  ## References

  - Efron, B., & Tibshirani, R. J. (1994). "An introduction to the
    bootstrap." CRC press.
  - Davison, A. C., & Hinkley, D. V. (1997). "Bootstrap methods and
    their application." Cambridge university press.
  """

  @default_n_samples 1000
  @default_confidence_level 0.95

  @type bootstrap_result :: %{
    point_estimate: float(),
    confidence_interval: {float(), float()},
    confidence_level: float(),
    n_samples: integer(),
    method: :percentile | :bca | :basic
  }

  @doc """
  Computes bootstrap confidence interval for a fairness metric.

  ## Parameters

    * `data` - List of tensors [predictions, labels?, sensitive_attr]
    * `metric_fn` - Function computing the metric on data
    * `opts` - Options:
      * `:n_samples` - Number of bootstrap samples (default: 1000)
      * `:confidence_level` - Confidence level (default: 0.95)
      * `:method` - Bootstrap method (:percentile, :bca, :basic)
      * `:stratified` - Preserve group proportions (default: true)
      * `:parallel` - Use parallel computation (default: true)
      * `:seed` - Random seed for reproducibility

  ## Returns

  Map containing point estimate and confidence interval.

  ## Examples

      iex> predictions = Nx.tensor([...])
      iex> sensitive = Nx.tensor([...])
      iex> metric_fn = fn [preds, sens] ->
      ...>   result = ExFairness.demographic_parity(preds, sens)
      ...>   result.disparity
      ...> end
      iex> result = ExFairness.Utils.Bootstrap.confidence_interval(
      ...>   [predictions, sensitive],
      ...>   metric_fn,
      ...>   n_samples: 1000
      ...> )
      iex> {lower, upper} = result.confidence_interval
      iex> IO.puts "95% CI: [#{lower}, #{upper}]"

  """
  @spec confidence_interval([Nx.Tensor.t()], function(), keyword()) ::
    bootstrap_result()
  def confidence_interval(data, metric_fn, opts \\ [])

  @doc """
  Performs stratified bootstrap sampling.

  Preserves the proportion of each group defined by sensitive attribute.
  """
  @spec stratified_sample([Nx.Tensor.t()], non_neg_integer(), integer()) ::
    [Nx.Tensor.t()]
  defp stratified_sample(data, n, seed)

  @doc """
  Computes percentile bootstrap confidence interval.

  The most straightforward method, directly using percentiles of
  bootstrap distribution.
  """
  @spec percentile_ci([float()], float()) :: {float(), float()}
  defp percentile_ci(bootstrap_values, confidence_level)

  @doc """
  Computes bias-corrected and accelerated (BCa) bootstrap CI.

  More accurate than percentile method but computationally intensive.
  Corrects for bias and skewness in bootstrap distribution.
  """
  @spec bca_ci([float()], float(), [Nx.Tensor.t()], function()) ::
    {float(), float()}
  defp bca_ci(bootstrap_values, observed, data, metric_fn)
end
```

**Key Features:**
- **Stratified Sampling**: Preserves group proportions
- **Parallel Computation**: Uses Task.async_stream for speed
- **Multiple Methods**: Percentile, BCa, basic bootstrap
- **GPU-Accelerated**: Metric computation uses Nx.Defn
- **Reproducible**: Configurable random seed

**Performance Considerations:**
- 1000 bootstrap samples: ~1-2 seconds for simple metrics
- Parallel execution: 4-8x speedup on multi-core systems
- Memory usage: O(n_samples × n_datapoints)

**Testing Strategy:**
1. Unit tests for sampling correctness
2. Verification that CI contains true parameter (simulation)
3. Test stratification maintains group proportions
4. Property-based tests for interval width
5. Performance benchmarks

---

### 1.2 Hypothesis Testing

**Rationale:**
- Distinguish statistical significance from practical significance
- Required for scientific publications
- Legal compliance requires statistical evidence
- Prevents false positive fairness claims

**Statistical Tests:**

**1. Two-Proportion Z-Test (Demographic Parity)**

**Hypotheses:**
```
H₀: p_A = p_B (no disparity between groups)
H₁: p_A ≠ p_B (disparity exists)
```

**Test Statistic:**
```
Z = (p̂_A - p̂_B) / SE

where SE = sqrt(p̂(1-p̂)(1/n_A + 1/n_B))
      p̂ = (n_A·p̂_A + n_B·p̂_B) / (n_A + n_B)
```

**Decision Rule:**
```
Reject H₀ if |Z| > z_(α/2)
P-value = 2·Φ(-|Z|)  [two-tailed]
```

**2. Chi-Square Test (Equalized Odds)**

**Hypotheses:**
```
H₀: Confusion matrices are independent of group
H₁: Confusion matrices differ by group
```

**Test Statistic:**
```
χ² = Σ (O_ij - E_ij)² / E_ij

where O_ij = observed count in cell (i,j)
      E_ij = expected count under independence
```

**Degrees of Freedom:**
```
df = (rows - 1) × (columns - 1) = 3
```

**3. Permutation Test (Non-Parametric)**

**Algorithm:**
```
1. Compute observed metric: M_obs
2. For i = 1 to n_permutations:
   a. Randomly permute sensitive attributes
   b. Compute M_i on permuted data
3. P-value = (# permutations with M ≥ M_obs) / n_permutations
```

**Advantages:**
- No distributional assumptions
- Exact p-values for small samples
- Works with any metric

**Implementation Plan:**

**Module: `ExFairness.Utils.StatisticalTests`**

```elixir
defmodule ExFairness.Utils.StatisticalTests do
  @moduledoc """
  Hypothesis testing for fairness metrics.

  Provides parametric and non-parametric tests to assess statistical
  significance of observed disparities.

  ## References

  - Agresti, A. (2018). "Statistical methods for the social sciences."
  - Good, P. (2013). "Permutation tests: a practical guide to
    resampling methods."
  """

  @default_alpha 0.05
  @default_n_permutations 10000

  @type test_result :: %{
    statistic: float(),
    p_value: float(),
    significant: boolean(),
    alpha: float(),
    effect_size: float() | nil,
    test_name: String.t(),
    interpretation: String.t()
  }

  @doc """
  Two-proportion Z-test for demographic parity.

  Tests whether positive prediction rates differ significantly
  between groups.

  ## Assumptions

    * Large sample sizes (n_A, n_B > 30)
    * Independent observations
    * np and n(1-p) > 5 for both groups

  ## Parameters

    * `predictions` - Binary predictions tensor
    * `sensitive_attr` - Binary sensitive attribute
    * `opts`:
      * `:alpha` - Significance level (default: 0.05)
      * `:alternative` - 'two-sided', 'greater', 'less'

  ## Returns

  Test result map with p-value and significance.
  """
  @spec two_proportion_test(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
    test_result()
  def two_proportion_test(predictions, sensitive_attr, opts \\ [])

  @doc """
  Chi-square test for equalized odds.

  Tests independence of confusion matrix from group membership.
  """
  @spec chi_square_test(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(),
    keyword()) :: test_result()
  def chi_square_test(predictions, labels, sensitive_attr, opts \\ [])

  @doc """
  Permutation test for any fairness metric.

  Non-parametric test that doesn't assume normal distribution.

  ## Algorithm

  1. Compute observed metric
  2. Repeatedly permute sensitive attributes
  3. Count permutations with metric ≥ observed
  4. P-value = proportion of extreme permutations

  ## Parameters

    * `data` - List of data tensors
    * `metric_fn` - Function computing metric
    * `opts`:
      * `:n_permutations` - Number of permutations (default: 10000)
      * `:alpha` - Significance level (default: 0.05)
      * `:alternative` - Test direction
      * `:seed` - Random seed
  """
  @spec permutation_test([Nx.Tensor.t()], function(), keyword()) ::
    test_result()
  def permutation_test(data, metric_fn, opts \\ [])

  @doc """
  Computes Cohen's h effect size for two proportions.

  Effect size measures:
  - Small: h ≈ 0.2
  - Medium: h ≈ 0.5
  - Large: h ≈ 0.8
  """
  @spec cohens_h(float(), float()) :: float()
  def cohens_h(p1, p2)
end
```

**Integration with Metrics:**

All metric functions enhanced with `:statistical_test` option:

```elixir
result = ExFairness.demographic_parity(predictions, sensitive_attr,
  statistical_test: :z_test,
  alpha: 0.05,
  include_ci: true,
  bootstrap_samples: 1000
)

# Enhanced result:
%{
  group_a_rate: 0.50,
  group_b_rate: 0.60,
  disparity: 0.10,
  passes: false,
  threshold: 0.05,

  # NEW: Statistical inference
  confidence_interval: {0.05, 0.15},
  p_value: 0.023,
  statistically_significant: true,
  effect_size: 0.21,  # Cohen's h

  interpretation: "Group A receives positive predictions at 50.0% rate,
    while Group B receives them at 60.0% rate (disparity = 10.0 pp).
    This disparity is statistically significant (p = 0.023, α = 0.05)
    with 95% CI [5.0, 15.0]. Effect size is small (h = 0.21)."
}
```

---

## Enhancement 2: Calibration Fairness Metric

### 2.1 Calibration Definition

**Rationale:**
- Critical for probability-based decisions (risk assessment, medical diagnosis)
- Ensures predictions mean the same thing across groups
- Required when users rely on prediction confidence

**Mathematical Definition:**

**Group Calibration:**
```
For each score range [s, s+δ]:
P(Y=1 | Ŷ ∈ [s, s+δ], A=0) = P(Y=1 | Ŷ ∈ [s, s+δ], A=1)
```

**Expected Calibration Error (ECE):**
```
ECE = Σ_b (n_b / n) · |acc(b) - conf(b)|

where:
  b = bin index
  n_b = number of samples in bin b
  acc(b) = accuracy in bin b
  conf(b) = average confidence in bin b
```

**Group-specific ECE:**
```
Δ_ECE = |ECE_A - ECE_B|
```

**Maximum Calibration Error (MCE):**
```
MCE = max_b |acc(b) - conf(b)|
```

### 2.2 Implementation Plan

**Module: `ExFairness.Metrics.Calibration`**

```elixir
defmodule ExFairness.Metrics.Calibration do
  @moduledoc """
  Calibration fairness metric.

  Measures whether predicted probabilities are well-calibrated across groups.
  A model is calibrated if predictions of p% actually occur p% of the time.

  ## Mathematical Definition

  For predicted probability ŝ(x) and outcome y:

      P(Y = 1 | ŝ(X) = s, A = a) ≈ s  for all s, a

  Fairness requires calibration holds across all groups.

  ## Use Cases

  - Medical risk scores (predicted risk should match actual risk)
  - Credit scoring (approval probability should match default rate)
  - Hiring (interview likelihood should match success rate)

  ## References

  - Kleinberg, J., et al. (2017). "Inherent trade-offs in algorithmic
    fairness."
  - Pleiss, G., et al. (2017). "On fairness and calibration." NeurIPS.
  - Chouldechova, A. (2017). "Fair prediction with disparate impact."
  """

  @default_n_bins 10
  @default_strategy :uniform  # or :quantile
  @default_threshold 0.1

  @type result :: %{
    group_a_ece: float(),
    group_b_ece: float(),
    disparity: float(),
    passes: boolean(),
    threshold: float(),

    # Detailed calibration info
    bins: [calibration_bin()],
    group_a_mce: float(),
    group_b_mce: float(),

    interpretation: String.t()
  }

  @type calibration_bin :: %{
    bin_index: non_neg_integer(),
    bin_range: {float(), float()},
    group_a_accuracy: float(),
    group_a_confidence: float(),
    group_a_count: integer(),
    group_b_accuracy: float(),
    group_b_confidence: float(),
    group_b_count: integer()
  }

  @doc """
  Computes calibration fairness disparity between groups.

  ## Parameters

    * `probabilities` - Predicted probabilities (0.0 to 1.0)
    * `labels` - Binary labels (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute (0 or 1)
    * `opts`:
      * `:n_bins` - Number of probability bins (default: 10)
      * `:strategy` - Binning strategy (:uniform or :quantile)
      * `:threshold` - Max acceptable ECE disparity (default: 0.1)
      * `:min_per_bin` - Minimum samples per bin (default: 5)

  ## Returns

  Map with ECE for each group, disparity, and detailed bin information.

  ## Examples

      iex> probs = Nx.tensor([0.1, 0.3, 0.6, 0.9, ...])
      iex> labels = Nx.tensor([0, 0, 1, 1, ...])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, ..., 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.Calibration.compute(
      ...>   probs, labels, sensitive,
      ...>   n_bins: 10, strategy: :uniform
      ...> )
      iex> result.passes
      true
      iex> result.group_a_ece
      0.05
  """
  @spec compute(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
    result()
  def compute(probabilities, labels, sensitive_attr, opts \\ [])

  @doc """
  Creates probability bins using uniform or quantile strategy.
  """
  @spec create_bins(Nx.Tensor.t(), non_neg_integer(), :uniform | :quantile) ::
    [{float(), float()}]
  defp create_bins(probabilities, n_bins, strategy)

  @doc """
  Computes expected calibration error for a group.
  """
  @spec compute_ece(Nx.Tensor.t(), Nx.Tensor.t(), [{float(), float()}]) ::
    float()
  defp compute_ece(probabilities, labels, bins)

  @doc """
  Computes maximum calibration error for a group.
  """
  @spec compute_mce(Nx.Tensor.t(), Nx.Tensor.t(), [{float(), float()}]) ::
    float()
  defp compute_mce(probabilities, labels, bins)

  @doc """
  Generates reliability diagram data for visualization.

  Returns data suitable for plotting calibration curves.
  """
  @spec reliability_diagram(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(),
    keyword()) :: map()
  def reliability_diagram(probabilities, labels, sensitive_attr, opts \\ [])
end
```

**Binning Strategies:**

1. **Uniform Binning**: Equal-width intervals
   ```
   bins = [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
   ```

2. **Quantile Binning**: Equal-frequency intervals
   ```
   bins defined so each contains n/B samples
   ```

**ECE Computation:**

```elixir
defp compute_ece(probabilities, labels, bins) do
  n_total = Nx.size(probabilities)

  bin_errors = for {bin_low, bin_high} <- bins do
    # Get samples in bin
    in_bin = Nx.logical_and(
      Nx.greater_equal(probabilities, bin_low),
      Nx.less(probabilities, bin_high)
    )

    n_bin = Nx.sum(in_bin) |> Nx.to_number()

    if n_bin > 0 do
      # Average confidence in bin
      bin_probs = Nx.select(in_bin, probabilities, 0)
      avg_confidence = Nx.sum(bin_probs) / n_bin

      # Accuracy in bin
      bin_labels = Nx.select(in_bin, labels, 0)
      accuracy = Nx.sum(bin_labels) / n_bin

      # Weighted calibration error
      (n_bin / n_total) * abs(accuracy - avg_confidence)
    else
      0.0
    end
  end

  Enum.sum(bin_errors)
end
```

### 2.3 Integration

**Add to Main API:**

```elixir
# In ExFairness module
@doc """
Computes calibration fairness metric.
"""
@spec calibration(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
  Calibration.result()
def calibration(probabilities, labels, sensitive_attr, opts \\ []) do
  ExFairness.Metrics.Calibration.compute(
    probabilities, labels, sensitive_attr, opts
  )
end
```

**Update Fairness Report:**

```elixir
# In ExFairness.Report.generate/4
# Add :calibration to default metrics list when probabilities provided
metrics = if probabilities_provided? do
  [:demographic_parity, :equalized_odds, :equal_opportunity,
   :predictive_parity, :calibration]
else
  [:demographic_parity, :equalized_odds, :equal_opportunity,
   :predictive_parity]
end
```

---

## Enhancement 3: Enhanced Error Handling

### 3.1 Input Validation Improvements

**Current Gaps:**
- Limited validation of probability ranges
- No validation for minimum bin sizes in calibration
- No warning for imbalanced datasets

**Enhancements:**

```elixir
defmodule ExFairness.Validation do
  # Add new validations

  @doc """
  Validates probability tensor is in [0, 1] range.
  """
  @spec validate_probabilities!(Nx.Tensor.t()) :: :ok
  def validate_probabilities!(probabilities) do
    min_val = Nx.reduce_min(probabilities) |> Nx.to_number()
    max_val = Nx.reduce_max(probabilities) |> Nx.to_number()

    cond do
      min_val < 0.0 ->
        raise ExFairness.Error, """
        Probabilities must be in [0, 1] range.
        Found minimum value: #{min_val}
        """

      max_val > 1.0 ->
        raise ExFairness.Error, """
        Probabilities must be in [0, 1] range.
        Found maximum value: #{max_val}
        """

      true -> :ok
    end
  end

  @doc """
  Warns if dataset is highly imbalanced.
  """
  @spec check_class_balance(Nx.Tensor.t(), float()) :: :ok
  def check_class_balance(labels, threshold \\ 0.1) do
    pos_rate = Nx.mean(labels) |> Nx.to_number()

    if pos_rate < threshold or pos_rate > (1 - threshold) do
      IO.warn("""
      Dataset appears imbalanced (positive rate: #{Float.round(pos_rate * 100, 1)}%).

      Fairness metrics may be unreliable with imbalanced data.
      Consider:
      - Collecting more balanced data
      - Using appropriate resampling techniques
      - Focusing on metrics robust to imbalance
      """)
    end

    :ok
  end

  @doc """
  Validates sufficient samples per bin for calibration.
  """
  @spec validate_bin_counts!(Nx.Tensor.t(), integer(), integer()) :: :ok
  def validate_bin_counts!(probabilities, n_bins, min_per_bin)
end
```

---

## Enhancement 4: Performance Optimizations

### 4.1 Cached Computations

**Rationale:**
- Many metrics reuse confusion matrix computations
- Fairness reports call multiple metrics on same data
- Optimization opportunity: compute once, reuse

**Implementation:**

```elixir
defmodule ExFairness.Cache do
  @moduledoc """
  Caching layer for expensive computations.

  Caches confusion matrices and other intermediate results
  to avoid redundant computation in comprehensive reports.
  """

  @doc """
  Computes and caches confusion matrix for both groups.
  """
  def get_confusion_matrices(predictions, labels, sensitive_attr) do
    key = cache_key([predictions, labels, sensitive_attr])

    case :persistent_term.get(key, nil) do
      nil ->
        result = compute_confusion_matrices(predictions, labels, sensitive_attr)
        :persistent_term.put(key, result)
        result
      cached ->
        cached
    end
  end

  defp cache_key(tensors) do
    # Generate unique key from tensor hashes
  end
end
```

### 4.2 Parallel Metric Computation

```elixir
# In ExFairness.Report
defp compute_metrics_parallel(predictions, labels, sensitive_attr, metrics) do
  metrics
  |> Task.async_stream(fn metric ->
    {metric, compute_single_metric(metric, predictions, labels, sensitive_attr)}
  end, max_concurrency: System.schedulers_online())
  |> Enum.map(fn {:ok, result} -> result end)
  |> Enum.into(%{})
end
```

---

## Implementation Roadmap

### Phase 1: Statistical Inference (3 weeks)

**Week 1:**
- Implement `ExFairness.Utils.Bootstrap` module
- Basic percentile CI method
- Stratified sampling
- Unit tests

**Week 2:**
- Implement `ExFairness.Utils.StatisticalTests` module
- Two-proportion Z-test
- Chi-square test
- Permutation test
- Unit tests

**Week 3:**
- Integrate statistical inference into all 4 existing metrics
- Update documentation
- Add examples
- Integration tests

### Phase 2: Calibration Metric (2 weeks)

**Week 4:**
- Implement `ExFairness.Metrics.Calibration` module
- Uniform and quantile binning
- ECE and MCE computation
- Unit tests

**Week 5:**
- Add calibration to main API
- Update fairness report
- Reliability diagram generation
- Documentation and examples

### Phase 3: Enhancements (2 weeks)

**Week 6:**
- Enhanced error handling and validation
- Input validation improvements
- Warning system for data quality issues

**Week 7:**
- Performance optimizations
- Caching layer
- Parallel computation
- Benchmarking

### Phase 4: Documentation & Testing (1 week)

**Week 8:**
- Comprehensive testing across all enhancements
- Update README with new features
- Create migration guide from v0.2.0
- Performance benchmarks
- Final integration testing

---

## Testing Strategy

### Unit Tests

**Bootstrap Module:**
- Test stratified sampling maintains proportions
- Verify CI contains true parameter (simulations)
- Test different CI methods (percentile, BCa)
- Edge cases: small samples, extreme proportions

**Statistical Tests:**
- Verify p-values under null hypothesis
- Test power against known alternatives
- Edge cases: perfect separation, no disparity
- Compare with R/Python implementations

**Calibration:**
- Test ECE computation correctness
- Verify binning strategies
- Test with perfectly calibrated model
- Test with completely miscalibrated model

### Integration Tests

- Test full pipeline with statistical inference
- Verify all metrics work with new options
- Test fairness report with calibration
- Performance tests with large datasets

### Property-Based Tests

```elixir
property "bootstrap CI contains true parameter 95% of time" do
  check all n <- integer(100..1000),
            p <- float(min: 0.1, max: 0.9) do
    # Generate data with known disparity
    # Compute bootstrap CI
    # Verify CI contains true disparity
  end
end

property "p-values uniformly distributed under null hypothesis" do
  check all n <- integer(100..1000) do
    # Generate data under H0
    # Compute p-value
    # Collect p-values
    # Test uniformity with KS test
  end
end
```

---

## Version Update Plan

### Version: 0.2.0 → 0.3.0

**Rationale for MINOR version bump:**
- New features (statistical inference, calibration)
- Backward compatible API
- No breaking changes
- Existing code continues to work

### Files to Update

1. **mix.exs**
   - Version: `0.2.0` → `0.3.0`
   - Update dependencies if needed

2. **README.md**
   - Update version badge
   - Add statistical inference section
   - Add calibration metric documentation
   - Update examples

3. **CHANGELOG.md**
   - Add `## [0.3.0] - 2025-11-25` section
   - List all new features
   - Document enhancements

4. **Documentation**
   - Update all module docs
   - Add new examples
   - Update API reference

---

## Success Criteria

### Functional Requirements
- ✅ Bootstrap confidence intervals working for all metrics
- ✅ Hypothesis testing with p-values for all metrics
- ✅ Calibration metric fully implemented
- ✅ All tests passing (target: 200+ tests)
- ✅ Zero compilation warnings
- ✅ Backward compatible API

### Quality Requirements
- ✅ Test coverage > 90%
- ✅ All doctests passing
- ✅ Performance: <10% slowdown with new features off
- ✅ Documentation: 100% public API documented
- ✅ Examples for all new features

### Non-Functional Requirements
- ✅ Clear migration guide from 0.2.0
- ✅ Academic citations for all new methods
- ✅ Production-ready error handling
- ✅ Performance benchmarks

---

## Risk Assessment

### Technical Risks

**Risk 1: Bootstrap Performance**
- Impact: HIGH
- Probability: MEDIUM
- Mitigation: Parallel execution, optional feature, caching

**Risk 2: Statistical Test Correctness**
- Impact: CRITICAL
- Probability: LOW
- Mitigation: Extensive validation against R/Python, property tests

**Risk 3: API Complexity**
- Impact: MEDIUM
- Probability: LOW
- Mitigation: Sensible defaults, clear documentation

### Schedule Risks

**Risk 4: Implementation Timeline**
- Impact: MEDIUM
- Probability: MEDIUM
- Mitigation: Phased approach, can defer Phase 3/4 if needed

---

## References

### Statistical Inference
1. Efron, B., & Tibshirani, R. J. (1994). "An introduction to the bootstrap." CRC press.
2. Davison, A. C., & Hinkley, D. V. (1997). "Bootstrap methods and their application."
3. Good, P. (2013). "Permutation tests: A practical guide to resampling methods."
4. Agresti, A. (2018). "Statistical methods for the social sciences."

### Calibration
5. Pleiss, G., et al. (2017). "On fairness and calibration." NeurIPS.
6. Kleinberg, J., et al. (2017). "Inherent trade-offs in algorithmic fairness." ITCS.
7. Chouldechova, A. (2017). "Fair prediction with disparate impact." Big Data.
8. Guo, C., et al. (2017). "On calibration of modern neural networks." ICML.

### Effect Sizes
9. Cohen, J. (1988). "Statistical power analysis for the behavioral sciences."
10. Ellis, P. D. (2010). "The essential guide to effect sizes."

---

## Appendix A: API Examples

### Example 1: Bootstrap Confidence Intervals

```elixir
# Basic usage
result = ExFairness.demographic_parity(
  predictions,
  sensitive_attr,
  include_ci: true,
  bootstrap_samples: 1000,
  confidence_level: 0.95
)

IO.puts """
Disparity: #{result.disparity}
95% CI: [#{elem(result.confidence_interval, 0)},
         #{elem(result.confidence_interval, 1)}]
"""
```

### Example 2: Hypothesis Testing

```elixir
# With statistical test
result = ExFairness.demographic_parity(
  predictions,
  sensitive_attr,
  statistical_test: :z_test,
  alpha: 0.05
)

if result.statistically_significant do
  IO.puts "Disparity is statistically significant (p = #{result.p_value})"
  IO.puts "Effect size: #{result.effect_size} (#{classify_effect_size(result.effect_size)})"
else
  IO.puts "No significant disparity detected (p = #{result.p_value})"
end
```

### Example 3: Calibration Assessment

```elixir
# Calibration fairness
result = ExFairness.calibration(
  probabilities,
  labels,
  sensitive_attr,
  n_bins: 10,
  strategy: :uniform
)

IO.puts """
Group A ECE: #{result.group_a_ece}
Group B ECE: #{result.group_b_ece}
Disparity: #{result.disparity}
Passes: #{result.passes}
"""

# Generate reliability diagram
diagram_data = ExFairness.Metrics.Calibration.reliability_diagram(
  probabilities,
  labels,
  sensitive_attr
)
```

### Example 4: Comprehensive Report with All Features

```elixir
# Full fairness assessment
report = ExFairness.fairness_report(
  predictions,
  labels,
  sensitive_attr,
  metrics: [:demographic_parity, :equalized_odds, :equal_opportunity,
            :predictive_parity],
  include_ci: true,
  statistical_test: :permutation,
  bootstrap_samples: 1000,
  alpha: 0.05
)

# Export with enhanced information
markdown = ExFairness.Report.to_markdown(report)
File.write!("fairness_report_with_stats.md", markdown)
```

---

## Appendix B: Migration Guide from v0.2.0

### No Breaking Changes

All existing code continues to work without modification.

### New Optional Parameters

```elixir
# Old (still works)
result = ExFairness.demographic_parity(predictions, sensitive_attr)

# New (enhanced)
result = ExFairness.demographic_parity(
  predictions,
  sensitive_attr,
  include_ci: true,              # NEW
  statistical_test: :z_test,     # NEW
  bootstrap_samples: 1000,       # NEW
  confidence_level: 0.95         # NEW
)
```

### New Metric

```elixir
# Add calibration assessment
calibration_result = ExFairness.calibration(
  probabilities,  # New: requires probabilities instead of binary predictions
  labels,
  sensitive_attr
)
```

---

**Document Status:** APPROVED FOR IMPLEMENTATION
**Approved By:** North Shore AI Research Team
**Date:** November 25, 2025
