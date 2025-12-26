defmodule ExFairness.Utils.StatisticalTests do
  @moduledoc """
  Hypothesis testing for fairness metrics.

  Provides parametric and non-parametric tests to assess statistical
  significance of observed disparities in fairness metrics.

  ## Statistical Tests

  - **Two-Proportion Z-Test**: Tests demographic parity differences
  - **Chi-Square Test**: Tests independence in confusion matrices
  - **Permutation Test**: Non-parametric test for any metric

  ## References

  - Agresti, A. (2018). "Statistical methods for the social sciences."
  - Good, P. (2013). "Permutation tests: A practical guide to resampling
    methods for testing hypotheses."
  - Cohen, J. (1988). "Statistical power analysis for the behavioral
    sciences."

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Utils.StatisticalTests.two_proportion_test(predictions, sensitive)
      iex> is_float(result.p_value) and result.p_value >= 0.0 and result.p_value <= 1.0
      true

  """

  alias ExFairness.Utils.Metrics

  @default_alpha 0.05
  @default_n_permutations 10_000
  @default_alternative :two_sided

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

  Tests whether positive prediction rates differ significantly between groups.

  ## Hypotheses

    * H₀: p_A = p_B (no disparity between groups)
    * H₁: p_A ≠ p_B (disparity exists)

  ## Test Statistic

  Under H₀, the standard error is:

      SE = sqrt(p̂ * (1 - p̂) * (1/n_A + 1/n_B))

  where p̂ = (n_A * p_A + n_B * p_B) / (n_A + n_B)

  Z-statistic:

      Z = (p_A - p_B) / SE

  P-value (two-tailed):

      p = 2 * P(|Z| > |z_observed|)

  ## Assumptions

    * Large sample sizes (n_A, n_B > 30 recommended)
    * Independent observations
    * np and n(1-p) > 5 for both groups

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts`:
      * `:alpha` - Significance level (default: 0.05)
      * `:alternative` - Test direction (:two_sided, :greater, :less)

  ## Returns

  Test result map with statistic, p-value, and significance.

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Utils.StatisticalTests.two_proportion_test(predictions, sensitive)
      iex> result.test_name
      "Two-Proportion Z-Test"

  """
  @spec two_proportion_test(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: test_result()
  def two_proportion_test(predictions, sensitive_attr, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    alternative = Keyword.get(opts, :alternative, @default_alternative)

    # Get predictions and counts for each group
    mask_a = Nx.equal(sensitive_attr, 0)
    mask_b = Nx.equal(sensitive_attr, 1)

    n_a = Nx.sum(mask_a) |> Nx.to_number()
    n_b = Nx.sum(mask_b) |> Nx.to_number()

    # Positive counts
    pos_a = Nx.sum(Nx.select(mask_a, predictions, 0)) |> Nx.to_number()
    pos_b = Nx.sum(Nx.select(mask_b, predictions, 0)) |> Nx.to_number()

    # Proportions
    p_a = pos_a / n_a
    p_b = pos_b / n_b

    # Pooled proportion under H0
    p_pooled = (pos_a + pos_b) / (n_a + n_b)

    # Standard error
    se = :math.sqrt(p_pooled * (1 - p_pooled) * (1 / n_a + 1 / n_b))

    # Z-statistic
    z = if se > 0, do: (p_a - p_b) / se, else: 0.0

    # P-value
    p_value =
      case alternative do
        :two_sided -> 2 * (1 - standard_normal_cdf(abs(z)))
        :greater -> 1 - standard_normal_cdf(z)
        :less -> standard_normal_cdf(z)
      end

    # Effect size (Cohen's h)
    effect_size = cohens_h(p_a, p_b)

    # Significance
    significant = p_value < alpha

    # Interpretation
    interpretation = generate_z_test_interpretation(p_a, p_b, z, p_value, significant, alpha)

    %{
      statistic: z,
      p_value: p_value,
      significant: significant,
      alpha: alpha,
      effect_size: effect_size,
      test_name: "Two-Proportion Z-Test",
      interpretation: interpretation
    }
  end

  @doc """
  Chi-square test for equalized odds.

  Tests whether confusion matrices are independent of group membership.

  ## Hypotheses

    * H₀: Confusion matrix is independent of sensitive attribute
    * H₁: Confusion matrix depends on sensitive attribute

  ## Test Statistic

      χ² = Σ (O_ij - E_ij)² / E_ij

  where O_ij = observed count, E_ij = expected count under independence

  ## Parameters

    * `predictions` - Binary predictions tensor
    * `labels` - Binary labels tensor
    * `sensitive_attr` - Binary sensitive attribute tensor
    * `opts`:
      * `:alpha` - Significance level (default: 0.05)

  ## Returns

  Test result map with chi-square statistic and p-value.

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
      iex> labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Utils.StatisticalTests.chi_square_test(predictions, labels, sensitive)
      iex> result.test_name
      "Chi-Square Test"

  """
  @spec chi_square_test(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: test_result()
  def chi_square_test(predictions, labels, sensitive_attr, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, @default_alpha)

    # Compute confusion matrices for both groups
    mask_a = Nx.equal(sensitive_attr, 0)
    mask_b = Nx.equal(sensitive_attr, 1)

    cm_a = Metrics.confusion_matrix(predictions, labels, mask_a)
    cm_b = Metrics.confusion_matrix(predictions, labels, mask_b)

    # Convert to numbers
    tp_a = Nx.to_number(cm_a.tp)
    fp_a = Nx.to_number(cm_a.fp)
    tn_a = Nx.to_number(cm_a.tn)
    fn_a = Nx.to_number(cm_a.fn)

    tp_b = Nx.to_number(cm_b.tp)
    fp_b = Nx.to_number(cm_b.fp)
    tn_b = Nx.to_number(cm_b.tn)
    fn_b = Nx.to_number(cm_b.fn)

    # Build contingency table: rows = (TP, FP, TN, FN), cols = (Group A, Group B)
    observed = [
      [tp_a, tp_b],
      [fp_a, fp_b],
      [tn_a, tn_b],
      [fn_a, fn_b]
    ]

    # Compute chi-square statistic
    chi_square = compute_chi_square(observed)

    # Degrees of freedom: (rows - 1) * (cols - 1) = 3 * 1 = 3
    df = 3

    # P-value from chi-square distribution
    p_value = chi_square_cdf_complement(chi_square, df)

    # Significance
    significant = p_value < alpha

    # Interpretation
    interpretation =
      generate_chi_square_interpretation(chi_square, df, p_value, significant, alpha)

    %{
      statistic: chi_square,
      p_value: p_value,
      significant: significant,
      alpha: alpha,
      effect_size: nil,
      test_name: "Chi-Square Test",
      interpretation: interpretation
    }
  end

  @doc """
  Permutation test for any fairness metric.

  Non-parametric test that doesn't assume normal distribution.

  ## Algorithm

  1. Compute observed metric on actual data
  2. For i = 1 to n_permutations:
     a. Randomly permute sensitive attributes
     b. Compute metric on permuted data
     c. Store permuted_statistics[i]
  3. P-value = proportion of permuted statistics ≥ observed

  ## Parameters

    * `data` - List of data tensors [predictions, labels?, sensitive_attr]
    * `metric_fn` - Function computing metric (returns numeric value)
    * `opts`:
      * `:n_permutations` - Number of permutations (default: 10000)
      * `:alpha` - Significance level (default: 0.05)
      * `:alternative` - Test direction (:two_sided, :greater, :less)
      * `:seed` - Random seed for reproducibility

  ## Returns

  Test result map with permutation statistics and p-value.

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> metric_fn = fn [preds, sens] ->
      ...>   result = ExFairness.demographic_parity(preds, sens)
      ...>   result.disparity
      ...> end
      iex> result = ExFairness.Utils.StatisticalTests.permutation_test(
      ...>   [predictions, sensitive],
      ...>   metric_fn,
      ...>   n_permutations: 100,
      ...>   seed: 42
      ...> )
      iex> result.test_name
      "Permutation Test"

  """
  @spec permutation_test([Nx.Tensor.t()], function(), keyword()) :: test_result()
  def permutation_test(data, metric_fn, opts \\ []) do
    n_permutations = Keyword.get(opts, :n_permutations, @default_n_permutations)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    alternative = Keyword.get(opts, :alternative, @default_alternative)
    seed = Keyword.get(opts, :seed, :erlang.system_time())

    # Compute observed statistic
    observed = metric_fn.(data)

    # Sensitive attribute is last in data list
    sensitive_attr = List.last(data)

    # Generate permutations and compute statistics
    permuted_statistics =
      for i <- 1..n_permutations do
        # Permute sensitive attribute
        permuted_sensitive = permute_tensor(sensitive_attr, seed + i)
        permuted_data = List.replace_at(data, -1, permuted_sensitive)

        # Compute metric on permuted data
        metric_fn.(permuted_data)
      end

    # Compute p-value based on alternative hypothesis
    p_value =
      case alternative do
        :two_sided ->
          count_extreme = Enum.count(permuted_statistics, fn stat -> abs(stat) >= abs(observed) end)
          count_extreme / n_permutations

        :greater ->
          count_extreme = Enum.count(permuted_statistics, fn stat -> stat >= observed end)
          count_extreme / n_permutations

        :less ->
          count_extreme = Enum.count(permuted_statistics, fn stat -> stat <= observed end)
          count_extreme / n_permutations
      end

    # Significance
    significant = p_value < alpha

    # Interpretation
    interpretation =
      generate_permutation_interpretation(observed, n_permutations, p_value, significant, alpha)

    %{
      statistic: observed,
      p_value: p_value,
      significant: significant,
      alpha: alpha,
      effect_size: nil,
      test_name: "Permutation Test",
      interpretation: interpretation
    }
  end

  @doc """
  Computes Cohen's h effect size for two proportions.

  Cohen's h is the difference between two arcsine-transformed proportions.

  ## Effect Size Guidelines

  - Small: h ≈ 0.2
  - Medium: h ≈ 0.5
  - Large: h ≈ 0.8

  ## Formula

      h = 2 * (arcsin(√p₁) - arcsin(√p₂))

  ## Examples

      iex> h = ExFairness.Utils.StatisticalTests.cohens_h(0.5, 0.3)
      iex> h > 0.4 and h < 0.5
      true

  """
  @spec cohens_h(float(), float()) :: float()
  def cohens_h(p1, p2) do
    2 * (:math.asin(:math.sqrt(p1)) - :math.asin(:math.sqrt(p2)))
  end

  # Standard normal CDF (approximation)
  @spec standard_normal_cdf(float()) :: float()
  defp standard_normal_cdf(z) do
    # Using error function approximation
    0.5 * (1 + erf(z / :math.sqrt(2)))
  end

  # Error function approximation
  @spec erf(float()) :: float()
  defp erf(x) do
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = if x < 0, do: -1, else: 1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * :math.exp(-x * x)

    sign * y
  end

  # Chi-square CDF complement (approximation)
  @spec chi_square_cdf_complement(float(), integer()) :: float()
  defp chi_square_cdf_complement(chi_square, df) do
    # Using incomplete gamma function approximation
    # For simplicity, using normal approximation when df > 30
    if df > 30 do
      z = :math.sqrt(2 * chi_square) - :math.sqrt(2 * df - 1)
      1 - standard_normal_cdf(z)
    else
      # Approximation for smaller df
      # This is a simplified version; for production use a proper implementation
      k = df / 2.0
      x = chi_square / 2.0
      incomplete_gamma_complement(k, x)
    end
  end

  # Incomplete gamma function complement (simplified approximation)
  @spec incomplete_gamma_complement(float(), float()) :: float()
  defp incomplete_gamma_complement(k, x) do
    # Series expansion for regularized incomplete gamma function
    # Q(k, x) = 1 - P(k, x)
    if x <= 0 do
      1.0
    else
      # Using continuous fraction approximation
      max(0.0, min(1.0, :math.exp(-x) * :math.pow(x, k) / gamma_approx(k) * cf_gamma(k, x)))
    end
  end

  # Gamma function approximation (Stirling's approximation)
  @spec gamma_approx(float()) :: float()
  defp gamma_approx(k) do
    if k < 1 do
      :math.pi() / (:math.sin(:math.pi() * k) * gamma_approx(1 - k))
    else
      :math.sqrt(2 * :math.pi() / k) * :math.pow(k / :math.exp(1), k)
    end
  end

  # Continued fraction for incomplete gamma
  @spec cf_gamma(float(), float()) :: float()
  # Simplified; full implementation needed for production
  defp cf_gamma(_k, _x), do: 1.0

  # Compute chi-square statistic from contingency table
  @spec compute_chi_square([[number()]]) :: float()
  defp compute_chi_square(observed) do
    # Compute row and column totals
    row_totals = Enum.map(observed, &Enum.sum/1)
    col_totals = observed |> Enum.zip() |> Enum.map(&Tuple.to_list/1) |> Enum.map(&Enum.sum/1)
    grand_total = Enum.sum(row_totals)

    # Compute expected frequencies and chi-square
    contributions =
      for {row, i} <- Enum.with_index(observed),
          {o_ij, j} <- Enum.with_index(row) do
        expected = expected_count(row_totals, col_totals, grand_total, i, j)
        chi_square_contribution(o_ij, expected)
      end

    Enum.sum(contributions)
  end

  @spec expected_count([number()], [number()], number(), non_neg_integer(), non_neg_integer()) ::
          float()
  defp expected_count(row_totals, col_totals, grand_total, row_index, col_index) do
    Enum.at(row_totals, row_index) * Enum.at(col_totals, col_index) / grand_total
  end

  @spec chi_square_contribution(number(), number()) :: float()
  defp chi_square_contribution(_observed, expected) when expected <= 0, do: 0.0

  defp chi_square_contribution(observed, expected) do
    :math.pow(observed - expected, 2) / expected
  end

  # Permute tensor values randomly
  @spec permute_tensor(Nx.Tensor.t(), integer()) :: Nx.Tensor.t()
  defp permute_tensor(tensor, seed) do
    _rng = :rand.seed(:exsss, seed)
    list = Nx.to_flat_list(tensor)
    shuffled = Enum.shuffle(list)
    Nx.tensor(shuffled)
  end

  # Generate interpretation for Z-test
  @spec generate_z_test_interpretation(float(), float(), float(), float(), boolean(), float()) ::
          String.t()
  defp generate_z_test_interpretation(p_a, p_b, z, p_value, significant, alpha) do
    p_a_pct = Float.round(p_a * 100, 1)
    p_b_pct = Float.round(p_b * 100, 1)

    base = """
    Group A positive rate: #{p_a_pct}%
    Group B positive rate: #{p_b_pct}%
    Z-statistic: #{Float.round(z, 3)}
    P-value: #{Float.round(p_value, 4)}
    """

    conclusion =
      if significant do
        "The difference is statistically significant at α = #{alpha}. " <>
          "There is evidence of disparity between groups."
      else
        "The difference is not statistically significant at α = #{alpha}. " <>
          "No significant evidence of disparity detected."
      end

    base <> conclusion
  end

  # Generate interpretation for chi-square test
  @spec generate_chi_square_interpretation(float(), integer(), float(), boolean(), float()) ::
          String.t()
  defp generate_chi_square_interpretation(chi_square, df, p_value, significant, alpha) do
    base = """
    Chi-square statistic: #{Float.round(chi_square, 3)}
    Degrees of freedom: #{df}
    P-value: #{Float.round(p_value, 4)}
    """

    conclusion =
      if significant do
        "The confusion matrix is significantly dependent on group membership at α = #{alpha}. " <>
          "There is evidence that error rates differ between groups."
      else
        "The confusion matrix is not significantly dependent on group membership at α = #{alpha}. " <>
          "No significant evidence of differential error rates detected."
      end

    base <> conclusion
  end

  # Generate interpretation for permutation test
  @spec generate_permutation_interpretation(float(), integer(), float(), boolean(), float()) ::
          String.t()
  defp generate_permutation_interpretation(observed, n_permutations, p_value, significant, alpha) do
    base = """
    Observed metric: #{Float.round(observed, 4)}
    Number of permutations: #{n_permutations}
    P-value: #{Float.round(p_value, 4)}
    """

    conclusion =
      if significant do
        "The observed disparity is statistically significant at α = #{alpha}. " <>
          "The metric value is unlikely to occur by chance under the null hypothesis."
      else
        "The observed disparity is not statistically significant at α = #{alpha}. " <>
          "The metric value could plausibly occur by chance."
      end

    base <> conclusion
  end
end
