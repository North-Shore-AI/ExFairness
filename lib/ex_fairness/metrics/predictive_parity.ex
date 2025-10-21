defmodule ExFairness.Metrics.PredictiveParity do
  @moduledoc """
  Predictive Parity (Outcome Test) fairness metric.

  Predictive parity requires that the positive predictive value (PPV) or
  precision is equal across groups defined by the sensitive attribute.

  ## Mathematical Definition

      P(Y = 1 | Ŷ = 1, A = 0) = P(Y = 1 | Ŷ = 1, A = 1)

  The disparity is measured as:

      Δ_PP = |PPV_{A=0} - PPV_{A=1}|

  ## When to Use

  - When the meaning of a positive prediction should be consistent across groups
  - Risk assessment tools (positive prediction should mean similar risk)
  - Credit scoring (approved applicants should have similar default rates)
  - When users rely on predictions to make decisions

  ## Limitations

  - Ignores true positive rates and false negative rates
  - May conflict with equalized odds when base rates differ
  - Less restrictive than equalized odds

  ## References

  - Chouldechova, A. (2017). "Fair prediction with disparate impact." Big Data, 5(2), 153-163.

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
      iex> labels = Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.PredictiveParity.compute(predictions, labels, sensitive)
      iex> result.passes
      true

  """

  alias ExFairness.Utils
  alias ExFairness.Utils.Metrics
  alias ExFairness.Validation

  @default_threshold 0.1
  @default_min_per_group 10

  @type result :: %{
          group_a_ppv: float(),
          group_b_ppv: float(),
          disparity: float(),
          passes: boolean(),
          threshold: float(),
          interpretation: String.t()
        }

  @doc """
  Computes predictive parity disparity between groups.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options:
      * `:threshold` - Maximum acceptable PPV disparity (default: 0.1)
      * `:min_per_group` - Minimum samples per group for validation (default: 10)

  ## Returns

  A map containing:
    * `:group_a_ppv` - Positive predictive value for group A
    * `:group_b_ppv` - Positive predictive value for group B
    * `:disparity` - Absolute difference in PPV
    * `:passes` - Whether disparity is within threshold
    * `:threshold` - Threshold used
    * `:interpretation` - Plain language explanation of the result

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
      iex> labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.PredictiveParity.compute(predictions, labels, sensitive)
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

    # Compute PPV for each group
    ppv_a = Metrics.positive_predictive_value(predictions, labels, mask_a) |> Nx.to_number()
    ppv_b = Metrics.positive_predictive_value(predictions, labels, mask_b) |> Nx.to_number()

    # Compute disparity
    disparity = abs(ppv_a - ppv_b)

    # Determine if passes
    passes = disparity <= threshold

    # Generate interpretation
    interpretation = generate_interpretation(ppv_a, ppv_b, disparity, passes, threshold)

    %{
      group_a_ppv: ppv_a,
      group_b_ppv: ppv_b,
      disparity: disparity,
      passes: passes,
      threshold: threshold,
      interpretation: interpretation
    }
  end

  # Generate plain language interpretation
  @spec generate_interpretation(float(), float(), float(), boolean(), float()) :: String.t()
  defp generate_interpretation(ppv_a, ppv_b, disparity, passes, threshold) do
    ppv_a_pct = Float.round(ppv_a * 100, 1)
    ppv_b_pct = Float.round(ppv_b * 100, 1)
    disp_pct = Float.round(disparity * 100, 1)
    threshold_pct = Float.round(threshold * 100, 1)

    base_msg =
      "Group A positive predictions are correct #{ppv_a_pct}% of the time, " <>
        "while Group B positive predictions are correct #{ppv_b_pct}% of the time, " <>
        "resulting in a disparity of #{disp_pct} percentage points."

    pass_msg =
      if passes do
        " This is within the acceptable threshold of #{threshold_pct}pp. " <>
          "The model satisfies predictive parity."
      else
        " This exceeds the acceptable threshold of #{threshold_pct}pp. " <>
          "The model violates predictive parity."
      end

    base_msg <> pass_msg
  end
end
