defmodule ExFairness.Metrics.EqualizedOdds do
  @moduledoc """
  Equalized Odds fairness metric.

  Equalized odds requires that both the true positive rate (TPR) and
  false positive rate (FPR) are equal across groups defined by the
  sensitive attribute.

  ## Mathematical Definition

      P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)  # Equal TPR
      P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)  # Equal FPR

  The disparities are measured as:

      Δ_TPR = |TPR_{A=0} - TPR_{A=1}|
      Δ_FPR = |FPR_{A=0} - FPR_{A=1}|

  ## When to Use

  - When both false positives and false negatives matter
  - Criminal justice (wrongful conviction and wrongful acquittal both harmful)
  - Medical diagnosis (both missed diseases and false alarms matter)
  - When accuracy across all outcomes is important

  ## Limitations

  - May conflict with demographic parity when base rates differ
  - Requires sufficient samples of both positive and negative labels
  - More restrictive than equal opportunity (which only checks TPR)

  ## References

  - Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." NeurIPS.

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
      iex> labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.EqualizedOdds.compute(predictions, labels, sensitive)
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
          group_a_fpr: float(),
          group_b_fpr: float(),
          tpr_disparity: float(),
          fpr_disparity: float(),
          passes: boolean(),
          threshold: float(),
          interpretation: String.t()
        }

  @doc """
  Computes equalized odds disparity between groups.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options:
      * `:threshold` - Maximum acceptable disparity for both TPR and FPR (default: 0.1)
      * `:min_per_group` - Minimum samples per group for validation (default: 10)

  ## Returns

  A map containing:
    * `:group_a_tpr` - True positive rate for group A
    * `:group_b_tpr` - True positive rate for group B
    * `:group_a_fpr` - False positive rate for group A
    * `:group_b_fpr` - False positive rate for group B
    * `:tpr_disparity` - Absolute difference in TPR
    * `:fpr_disparity` - Absolute difference in FPR
    * `:passes` - Whether both disparities are within threshold
    * `:threshold` - Threshold used
    * `:interpretation` - Plain language explanation of the result

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.EqualizedOdds.compute(predictions, labels, sensitive)
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

    # Compute TPR and FPR for each group
    tpr_a = Metrics.true_positive_rate(predictions, labels, mask_a) |> Nx.to_number()
    tpr_b = Metrics.true_positive_rate(predictions, labels, mask_b) |> Nx.to_number()
    fpr_a = Metrics.false_positive_rate(predictions, labels, mask_a) |> Nx.to_number()
    fpr_b = Metrics.false_positive_rate(predictions, labels, mask_b) |> Nx.to_number()

    # Compute disparities
    tpr_disparity = abs(tpr_a - tpr_b)
    fpr_disparity = abs(fpr_a - fpr_b)

    # Determine if passes (both disparities must be within threshold)
    passes = tpr_disparity <= threshold and fpr_disparity <= threshold

    # Generate interpretation
    interpretation =
      generate_interpretation(
        tpr_a,
        tpr_b,
        fpr_a,
        fpr_b,
        tpr_disparity,
        fpr_disparity,
        passes,
        threshold
      )

    %{
      group_a_tpr: tpr_a,
      group_b_tpr: tpr_b,
      group_a_fpr: fpr_a,
      group_b_fpr: fpr_b,
      tpr_disparity: tpr_disparity,
      fpr_disparity: fpr_disparity,
      passes: passes,
      threshold: threshold,
      interpretation: interpretation
    }
  end

  # Generate plain language interpretation
  @spec generate_interpretation(
          float(),
          float(),
          float(),
          float(),
          float(),
          float(),
          boolean(),
          float()
        ) :: String.t()
  defp generate_interpretation(
         tpr_a,
         tpr_b,
         fpr_a,
         fpr_b,
         tpr_disparity,
         fpr_disparity,
         passes,
         threshold
       ) do
    tpr_a_pct = Float.round(tpr_a * 100, 1)
    tpr_b_pct = Float.round(tpr_b * 100, 1)
    fpr_a_pct = Float.round(fpr_a * 100, 1)
    fpr_b_pct = Float.round(fpr_b * 100, 1)
    tpr_disp_pct = Float.round(tpr_disparity * 100, 1)
    fpr_disp_pct = Float.round(fpr_disparity * 100, 1)
    threshold_pct = Float.round(threshold * 100, 1)

    base_msg =
      "Group A: TPR=#{tpr_a_pct}%, FPR=#{fpr_a_pct}%. " <>
        "Group B: TPR=#{tpr_b_pct}%, FPR=#{fpr_b_pct}%. " <>
        "Disparities: TPR=#{tpr_disp_pct}pp, FPR=#{fpr_disp_pct}pp."

    pass_msg =
      if passes do
        " Both disparities are within the acceptable threshold of #{threshold_pct}pp. " <>
          "The model satisfies equalized odds."
      else
        " At least one disparity exceeds the acceptable threshold of #{threshold_pct}pp. " <>
          "The model violates equalized odds."
      end

    base_msg <> pass_msg
  end
end
