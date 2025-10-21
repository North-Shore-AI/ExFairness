defmodule ExFairness.Metrics.EqualOpportunity do
  @moduledoc """
  Equal Opportunity fairness metric.

  Equal opportunity requires that the true positive rate (TPR) is equal
  across groups defined by the sensitive attribute. This is a relaxed
  version of equalized odds that only considers TPR, not FPR.

  ## Mathematical Definition

      P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)

  The disparity is measured as:

      Δ_EO = |TPR_{A=0} - TPR_{A=1}|

  ## When to Use

  - When false negatives are more costly than false positives
  - Hiring (don't want to miss qualified candidates from any group)
  - College admissions
  - Loan approvals where rejecting qualified applicants is the main concern

  ## Limitations

  - Ignores false positive rates (may unfairly burden one group with false positives)
  - Less restrictive than equalized odds
  - May conflict with demographic parity when base rates differ

  ## References

  - Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." NeurIPS.

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.EqualOpportunity.compute(predictions, labels, sensitive)
      iex> result.passes
      true

  """

  alias ExFairness.Utils
  alias ExFairness.Utils.Metrics
  alias ExFairness.Validation

  @default_threshold 0.1
  @default_min_per_group 10

  @type result :: %{
          group_a_tpr: float(),
          group_b_tpr: float(),
          disparity: float(),
          passes: boolean(),
          threshold: float(),
          interpretation: String.t()
        }

  @doc """
  Computes equal opportunity disparity between groups.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options:
      * `:threshold` - Maximum acceptable TPR disparity (default: 0.1)
      * `:min_per_group` - Minimum samples per group for validation (default: 10)

  ## Returns

  A map containing:
    * `:group_a_tpr` - True positive rate for group A
    * `:group_b_tpr` - True positive rate for group B
    * `:disparity` - Absolute difference in TPR
    * `:passes` - Whether disparity is within threshold
    * `:threshold` - Threshold used
    * `:interpretation` - Plain language explanation of the result

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.EqualOpportunity.compute(predictions, labels, sensitive)
      iex> result.passes
      false

  """
  @spec compute(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: result()
  def compute(predictions, labels, sensitive_attr, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, @default_threshold)
    min_per_group = Keyword.get(opts, :min_per_group, @default_min_per_group)

    # Validate inputs
    Validation.validate_predictions!(predictions)
    Validation.validate_labels!(labels)

    Validation.validate_matching_shapes!(
      [predictions, labels, sensitive_attr],
      ["predictions", "labels", "sensitive_attr"]
    )

    Validation.validate_sensitive_attr!(sensitive_attr, min_per_group: min_per_group)

    # Create group masks
    mask_a = Utils.create_group_mask(sensitive_attr, 0)
    mask_b = Utils.create_group_mask(sensitive_attr, 1)

    # Compute TPR for each group
    tpr_a = Metrics.true_positive_rate(predictions, labels, mask_a) |> Nx.to_number()
    tpr_b = Metrics.true_positive_rate(predictions, labels, mask_b) |> Nx.to_number()

    # Compute disparity
    disparity = abs(tpr_a - tpr_b)

    # Determine if passes
    passes = disparity <= threshold

    # Generate interpretation
    interpretation = generate_interpretation(tpr_a, tpr_b, disparity, passes, threshold)

    %{
      group_a_tpr: tpr_a,
      group_b_tpr: tpr_b,
      disparity: disparity,
      passes: passes,
      threshold: threshold,
      interpretation: interpretation
    }
  end

  # Generate plain language interpretation
  @spec generate_interpretation(float(), float(), float(), boolean(), float()) :: String.t()
  defp generate_interpretation(tpr_a, tpr_b, disparity, passes, threshold) do
    tpr_a_pct = Float.round(tpr_a * 100, 1)
    tpr_b_pct = Float.round(tpr_b * 100, 1)
    disp_pct = Float.round(disparity * 100, 1)
    threshold_pct = Float.round(threshold * 100, 1)

    base_msg =
      "Group A has a true positive rate of #{tpr_a_pct}%, " <>
        "while Group B has #{tpr_b_pct}%, " <>
        "resulting in a disparity of #{disp_pct} percentage points."

    pass_msg =
      if passes do
        " This is within the acceptable threshold of #{threshold_pct}pp. " <>
          "The model provides equal opportunity."
      else
        " This exceeds the acceptable threshold of #{threshold_pct}pp. " <>
          "The model violates equal opportunity."
      end

    base_msg <> pass_msg
  end
end
