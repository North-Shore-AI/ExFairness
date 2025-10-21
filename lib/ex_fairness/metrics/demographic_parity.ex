defmodule ExFairness.Metrics.DemographicParity do
  @moduledoc """
  Demographic Parity (Statistical Parity) fairness metric.

  Demographic parity requires that the probability of a positive prediction
  is equal across groups defined by the sensitive attribute.

  ## Mathematical Definition

      P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)

  The disparity is measured as the absolute difference between positive
  prediction rates:

      Δ_DP = |P(Ŷ = 1 | A = 0) - P(Ŷ = 1 | A = 1)|

  ## When to Use

  - When equal representation in positive outcomes is required
  - Advertising and content recommendation systems
  - When base rates can legitimately differ between groups

  ## Limitations

  - Ignores base rate differences in actual outcomes
  - May conflict with accuracy if base rates differ
  - Can be satisfied by a random classifier

  ## References

  - Dwork, C., et al. (2012). "Fairness through awareness." ITCS.
  - Feldman, M., et al. (2015). "Certifying and removing disparate impact." KDD.

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.DemographicParity.compute(predictions, sensitive)
      iex> result.passes
      true

  """

  alias ExFairness.Utils
  alias ExFairness.Validation

  @default_threshold 0.1
  @default_min_per_group 10

  @type result :: %{
          group_a_rate: float(),
          group_b_rate: float(),
          disparity: float(),
          passes: boolean(),
          threshold: float(),
          interpretation: String.t()
        }

  @doc """
  Computes demographic parity disparity between groups.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options:
      * `:threshold` - Maximum acceptable disparity (default: 0.1)
      * `:min_per_group` - Minimum samples per group for validation (default: 10)

  ## Returns

  A map containing:
    * `:group_a_rate` - Positive prediction rate for group A (sensitive_attr = 0)
    * `:group_b_rate` - Positive prediction rate for group B (sensitive_attr = 1)
    * `:disparity` - Absolute difference between rates
    * `:passes` - Whether disparity is within threshold
    * `:threshold` - Threshold used
    * `:interpretation` - Plain language explanation of the result

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.DemographicParity.compute(predictions, sensitive)
      iex> result.disparity
      1.0

      iex> predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.DemographicParity.compute(predictions, sensitive)
      iex> result.passes
      true

  """
  @spec compute(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: result()
  def compute(predictions, sensitive_attr, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, @default_threshold)
    min_per_group = Keyword.get(opts, :min_per_group, @default_min_per_group)

    # Validate inputs
    Validation.validate_predictions!(predictions)
    # Check shapes before detailed validation of sensitive_attr
    Validation.validate_matching_shapes!([predictions, sensitive_attr], [
      "predictions",
      "sensitive_attr"
    ])

    Validation.validate_sensitive_attr!(sensitive_attr, min_per_group: min_per_group)

    # Compute positive prediction rates for each group
    {rate_a, rate_b} = Utils.group_positive_rates(predictions, sensitive_attr)

    # Convert to numbers
    rate_a_num = Nx.to_number(rate_a)
    rate_b_num = Nx.to_number(rate_b)

    # Compute disparity
    disparity = abs(rate_a_num - rate_b_num)

    # Determine if passes
    passes = disparity <= threshold

    # Generate interpretation
    interpretation = generate_interpretation(rate_a_num, rate_b_num, disparity, passes, threshold)

    %{
      group_a_rate: rate_a_num,
      group_b_rate: rate_b_num,
      disparity: disparity,
      passes: passes,
      threshold: threshold,
      interpretation: interpretation
    }
  end

  # Generate plain language interpretation
  @spec generate_interpretation(float(), float(), float(), boolean(), float()) :: String.t()
  defp generate_interpretation(rate_a, rate_b, disparity, passes, threshold) do
    rate_a_pct = Float.round(rate_a * 100, 1)
    rate_b_pct = Float.round(rate_b * 100, 1)
    disparity_pct = Float.round(disparity * 100, 1)

    base_msg =
      "Group A receives positive predictions at #{rate_a_pct}% rate, " <>
        "while Group B receives them at #{rate_b_pct}% rate, " <>
        "resulting in a disparity of #{disparity_pct} percentage points."

    pass_msg =
      if passes do
        " This is within the acceptable threshold of #{Float.round(threshold * 100, 1)} percentage points. " <>
          "The model demonstrates demographic parity."
      else
        " This exceeds the acceptable threshold of #{Float.round(threshold * 100, 1)} percentage points. " <>
          "The model violates demographic parity."
      end

    base_msg <> pass_msg
  end
end
