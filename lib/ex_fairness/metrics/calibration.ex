defmodule ExFairness.Metrics.Calibration do
  @moduledoc """
  Calibration fairness metric.

  Measures whether predicted probabilities are well-calibrated across groups.
  A model is calibrated if predictions of p% actually occur p% of the time.

  ## Mathematical Definition

  For predicted probability ŝ(x) and outcome y:

      P(Y = 1 | ŝ(X) = s, A = a) ≈ s  for all s, a

  Fairness requires calibration holds across all groups.

  ## Expected Calibration Error (ECE)

  ECE measures the weighted average of calibration error across bins:

      ECE = Σ_b (n_b / n) · |acc(b) - conf(b)|

  where:
    - b = bin index
    - n_b = number of samples in bin b
    - acc(b) = accuracy in bin b
    - conf(b) = average confidence in bin b

  ## Group Fairness

  Calibration fairness requires similar ECE across groups:

      Δ_ECE = |ECE_A - ECE_B|

  ## Use Cases

  - Medical risk scores (predicted risk should match actual risk)
  - Credit scoring (approval probability should match default rate)
  - Hiring (interview likelihood should match success rate)
  - Any application where users rely on prediction confidence

  ## References

  - Kleinberg, J., et al. (2017). "Inherent trade-offs in algorithmic fairness."
  - Pleiss, G., et al. (2017). "On fairness and calibration." NeurIPS.
  - Chouldechova, A. (2017). "Fair prediction with disparate impact."
  - Guo, C., et al. (2017). "On calibration of modern neural networks." ICML.

  ## Examples

      iex> # Perfect calibration example
      iex> probs = Nx.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
      iex> labels = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.Calibration.compute(probs, labels, sensitive, n_bins: 5)
      iex> is_float(result.disparity)
      true

  """

  alias ExFairness.Validation

  @default_n_bins 10
  @default_strategy :uniform
  @default_threshold 0.1
  @default_min_per_group 5

  @type result :: %{
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

  @doc """
  Computes calibration fairness disparity between groups.

  ## Parameters

    * `probabilities` - Predicted probabilities (0.0 to 1.0)
    * `labels` - Binary labels (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute (0 or 1)
    * `opts`:
      * `:n_bins` - Number of probability bins (default: 10)
      * `:strategy` - Binning strategy (:uniform or :quantile, default: :uniform)
      * `:threshold` - Max acceptable ECE disparity (default: 0.1)
      * `:min_per_group` - Minimum samples per group (default: 5)

  ## Returns

  Map with ECE for each group, disparity, and detailed calibration metrics:
    * `:group_a_ece` - Expected Calibration Error for group A
    * `:group_b_ece` - Expected Calibration Error for group B
    * `:disparity` - Absolute difference in ECE
    * `:passes` - Whether disparity is within threshold
    * `:threshold` - Threshold used
    * `:group_a_mce` - Maximum Calibration Error for group A
    * `:group_b_mce` - Maximum Calibration Error for group B
    * `:n_bins` - Number of bins used
    * `:strategy` - Binning strategy used
    * `:interpretation` - Plain language explanation

  ## Examples

      iex> probs = Nx.tensor([0.1, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.8, 0.5, 0.3, 0.1, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.8, 0.5, 0.3])
      iex> labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Metrics.Calibration.compute(probs, labels, sensitive, n_bins: 5)
      iex> result.n_bins
      5

  """
  @spec compute(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: result()
  def compute(probabilities, labels, sensitive_attr, opts \\ []) do
    n_bins = Keyword.get(opts, :n_bins, @default_n_bins)
    strategy = Keyword.get(opts, :strategy, @default_strategy)
    threshold = Keyword.get(opts, :threshold, @default_threshold)
    min_per_group = Keyword.get(opts, :min_per_group, @default_min_per_group)

    # Validate inputs
    validate_probabilities!(probabilities)
    Validation.validate_predictions!(labels)

    Validation.validate_matching_shapes!([probabilities, labels, sensitive_attr], [
      "probabilities",
      "labels",
      "sensitive_attr"
    ])

    Validation.validate_sensitive_attr!(sensitive_attr, min_per_group: min_per_group)

    # Create bins
    bins = create_bins(probabilities, n_bins, strategy)

    # Compute ECE and MCE for each group
    mask_a = Nx.equal(sensitive_attr, 0)
    mask_b = Nx.equal(sensitive_attr, 1)

    group_a_ece = compute_ece(probabilities, labels, mask_a, bins)
    group_b_ece = compute_ece(probabilities, labels, mask_b, bins)

    group_a_mce = compute_mce(probabilities, labels, mask_a, bins)
    group_b_mce = compute_mce(probabilities, labels, mask_b, bins)

    # Compute disparity
    disparity = abs(group_a_ece - group_b_ece)

    # Determine if passes
    passes = disparity <= threshold

    # Generate interpretation
    interpretation =
      generate_interpretation(group_a_ece, group_b_ece, disparity, passes, threshold)

    %{
      group_a_ece: group_a_ece,
      group_b_ece: group_b_ece,
      disparity: disparity,
      passes: passes,
      threshold: threshold,
      group_a_mce: group_a_mce,
      group_b_mce: group_b_mce,
      n_bins: n_bins,
      strategy: strategy,
      interpretation: interpretation
    }
  end

  @doc """
  Generates reliability diagram data for calibration plotting.

  Returns bin-level accuracy, confidence, and counts per group using the same
  binning strategy as `compute/4`.
  """
  @spec reliability_diagram(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: %{
          bins: [map()],
          n_bins: integer(),
          strategy: :uniform | :quantile
        }
  def reliability_diagram(probabilities, labels, sensitive_attr, opts \\ []) do
    n_bins = Keyword.get(opts, :n_bins, @default_n_bins)
    strategy = Keyword.get(opts, :strategy, @default_strategy)
    min_per_group = Keyword.get(opts, :min_per_group, @default_min_per_group)

    validate_probabilities!(probabilities)
    Validation.validate_predictions!(labels)

    Validation.validate_matching_shapes!([probabilities, labels, sensitive_attr], [
      "probabilities",
      "labels",
      "sensitive_attr"
    ])

    Validation.validate_sensitive_attr!(sensitive_attr, min_per_group: min_per_group)

    bins = create_bins(probabilities, n_bins, strategy)

    mask_a = Nx.equal(sensitive_attr, 0)
    mask_b = Nx.equal(sensitive_attr, 1)

    bin_stats =
      bins
      |> Enum.with_index()
      |> Enum.map(fn {{bin_low, bin_high}, idx} ->
        %{
          bin_index: idx,
          range: {bin_low, bin_high},
          group_a: bin_group_stats(probabilities, labels, mask_a, bin_low, bin_high),
          group_b: bin_group_stats(probabilities, labels, mask_b, bin_low, bin_high)
        }
      end)

    %{
      bins: bin_stats,
      n_bins: n_bins,
      strategy: strategy
    }
  end

  # Validate probabilities are in [0, 1] range
  @spec validate_probabilities!(Nx.Tensor.t()) :: :ok
  defp validate_probabilities!(probabilities) do
    min_val = Nx.reduce_min(probabilities) |> Nx.to_number()
    max_val = Nx.reduce_max(probabilities) |> Nx.to_number()

    cond do
      min_val < 0.0 ->
        raise ExFairness.Error, """
        Probabilities must be in [0, 1] range.
        Found minimum value: #{min_val}

        Ensure your model outputs probabilities (e.g., use sigmoid activation).
        """

      max_val > 1.0 ->
        raise ExFairness.Error, """
        Probabilities must be in [0, 1] range.
        Found maximum value: #{max_val}

        Ensure your model outputs probabilities (e.g., use sigmoid activation).
        """

      true ->
        :ok
    end
  end

  # Create probability bins
  @spec create_bins(Nx.Tensor.t(), non_neg_integer(), :uniform | :quantile) ::
          [{float(), float()}]
  defp create_bins(probabilities, n_bins, strategy) do
    case strategy do
      :uniform ->
        create_uniform_bins(n_bins)

      :quantile ->
        create_quantile_bins(probabilities, n_bins)
    end
  end

  # Create uniform-width bins
  @spec create_uniform_bins(non_neg_integer()) :: [{float(), float()}]
  defp create_uniform_bins(n_bins) do
    bin_width = 1.0 / n_bins

    for i <- 0..(n_bins - 1) do
      {i * bin_width, (i + 1) * bin_width}
    end
  end

  # Create quantile-based bins (equal frequency)
  @spec create_quantile_bins(Nx.Tensor.t(), non_neg_integer()) :: [{float(), float()}]
  defp create_quantile_bins(probabilities, n_bins) do
    sorted = probabilities |> Nx.to_flat_list() |> Enum.sort()
    n = length(sorted)

    quantiles =
      for i <- 0..n_bins do
        idx = min(floor(i * n / n_bins), n - 1)
        Enum.at(sorted, idx)
      end

    # Create bins from quantiles
    quantiles
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [low, high] -> {low, high} end)
  end

  # Compute Expected Calibration Error
  @spec compute_ece(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), [{float(), float()}]) ::
          float()
  defp compute_ece(probabilities, labels, mask, bins) do
    n_total = Nx.sum(mask) |> Nx.to_number()

    if n_total == 0 do
      0.0
    else
      bins
      |> Enum.map(fn {bin_low, bin_high} ->
        compute_bin_error(probabilities, labels, mask, bin_low, bin_high, n_total)
      end)
      |> Enum.sum()
    end
  end

  # Compute Maximum Calibration Error
  @spec compute_mce(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), [{float(), float()}]) ::
          float()
  defp compute_mce(probabilities, labels, mask, bins) do
    n_total = Nx.sum(mask) |> Nx.to_number()

    if n_total == 0 do
      0.0
    else
      bins
      |> Enum.map(fn {bin_low, bin_high} ->
        {_weighted_error, abs_error} =
          compute_bin_error_components(probabilities, labels, mask, bin_low, bin_high, n_total)

        abs_error
      end)
      |> Enum.max()
    end
  end

  # Compute calibration error for a single bin
  @spec compute_bin_error(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), float(), float(), float()) ::
          float()
  defp compute_bin_error(probabilities, labels, mask, bin_low, bin_high, n_total) do
    {weighted_error, _abs_error} =
      compute_bin_error_components(probabilities, labels, mask, bin_low, bin_high, n_total)

    weighted_error
  end

  # Compute both weighted and absolute error components for a bin
  @spec compute_bin_error_components(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          float(),
          float(),
          float()
        ) :: {float(), float()}
  defp compute_bin_error_components(probabilities, labels, mask, bin_low, bin_high, n_total) do
    # Get samples in bin
    in_bin = in_bin_mask(probabilities, mask, bin_low, bin_high)

    n_bin = Nx.sum(in_bin) |> Nx.to_number()

    if n_bin > 0 do
      # Average confidence in bin
      bin_probs = Nx.select(in_bin, probabilities, 0.0)
      avg_confidence = (Nx.sum(bin_probs) |> Nx.to_number()) / n_bin

      # Accuracy in bin
      bin_labels = Nx.select(in_bin, labels, 0)
      accuracy = (Nx.sum(bin_labels) |> Nx.to_number()) / n_bin

      # Calibration error
      abs_error = abs(accuracy - avg_confidence)
      weighted_error = n_bin / n_total * abs_error

      {weighted_error, abs_error}
    else
      {0.0, 0.0}
    end
  end

  # Generate plain language interpretation
  @spec generate_interpretation(float(), float(), float(), boolean(), float()) :: String.t()
  defp generate_interpretation(ece_a, ece_b, disparity, passes, threshold) do
    ece_a_pct = Float.round(ece_a * 100, 2)
    ece_b_pct = Float.round(ece_b * 100, 2)
    disparity_pct = Float.round(disparity * 100, 2)

    base_msg = """
    Calibration Analysis:
    - Group A ECE: #{ece_a_pct}% (lower is better)
    - Group B ECE: #{ece_b_pct}% (lower is better)
    - Disparity: #{disparity_pct} percentage points

    """

    pass_msg =
      if passes do
        "✓ Calibration disparity is within acceptable threshold (#{Float.round(threshold * 100, 1)}%). " <>
          "The model shows similar calibration quality across both groups."
      else
        "✗ Calibration disparity exceeds acceptable threshold (#{Float.round(threshold * 100, 1)}%). " <>
          "The model shows different calibration quality across groups, meaning predicted probabilities " <>
          "may be more reliable for one group than the other."
      end

    interpretation =
      if ece_a < 0.05 and ece_b < 0.05 do
        " Both groups show excellent calibration (ECE < 5%)."
      else
        if ece_a < 0.1 and ece_b < 0.1 do
          " Both groups show good calibration (ECE < 10%)."
        else
          " Consider calibration techniques (e.g., Platt scaling, isotonic regression) to improve reliability."
        end
      end

    String.trim(base_msg <> pass_msg <> interpretation)
  end

  @spec bin_group_stats(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), float(), float()) :: map()
  defp bin_group_stats(probabilities, labels, mask, bin_low, bin_high) do
    in_bin = in_bin_mask(probabilities, mask, bin_low, bin_high)
    count = Nx.sum(in_bin) |> Nx.to_number()

    if count > 0 do
      bin_probs = Nx.select(in_bin, probabilities, 0.0)
      bin_labels = Nx.select(in_bin, labels, 0)

      %{
        count: count,
        confidence: (Nx.sum(bin_probs) |> Nx.to_number()) / count,
        accuracy: (Nx.sum(bin_labels) |> Nx.to_number()) / count
      }
    else
      %{count: 0, confidence: 0.0, accuracy: 0.0}
    end
  end

  @spec in_bin_mask(Nx.Tensor.t(), Nx.Tensor.t(), float(), float()) :: Nx.Tensor.t()
  defp in_bin_mask(probabilities, mask, bin_low, bin_high) do
    Nx.logical_and(
      Nx.logical_and(
        Nx.greater_equal(probabilities, bin_low),
        Nx.less(probabilities, bin_high)
      ),
      mask
    )
  end
end
