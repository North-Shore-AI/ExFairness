defmodule ExFairness.Mitigation.Reweighting do
  @moduledoc """
  Sample reweighting for fairness-aware machine learning.

  Reweighting is a pre-processing technique that assigns different weights to
  training samples to achieve fairness. Samples from underrepresented groups
  or combinations receive higher weights.

  ## How It Works

  For demographic parity, the weight for sample (a, y) is:

      w(a, y) = P(Y = y) / P(A = a, Y = y)

  This ensures that all group-label combinations have equal expected weight,
  which helps achieve demographic parity after reweighting.

  For equalized odds, weights are computed to balance both positive and
  negative outcomes across groups.

  ## Usage

  Compute weights during data preparation, then pass them to your training
  algorithm's `sample_weight` parameter.

  ## References

  - Kamiran, F., & Calders, T. (2012). "Data preprocessing techniques for
    classification without discrimination." KAIS.

  ## Examples

      iex> labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> weights = ExFairness.Mitigation.Reweighting.compute_weights(labels, sensitive)
      iex> Nx.size(weights)
      20

  """

  import Nx.Defn

  alias ExFairness.Validation

  @default_min_per_group 10

  @doc """
  Computes sample weights for fairness-aware training.

  ## Parameters

    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options:
      * `:target` - Target fairness metric (`:demographic_parity` or `:equalized_odds`, default: `:demographic_parity`)
      * `:min_per_group` - Minimum samples per group for validation (default: 10)

  ## Returns

  A tensor of sample weights (same shape as labels). Weights are normalized to
  have mean 1.0.

  ## Examples

      iex> labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> weights = ExFairness.Mitigation.Reweighting.compute_weights(labels, sensitive)
      iex> mean = Nx.mean(weights) |> Nx.to_number()
      iex> Float.round(mean, 2)
      1.0

  """
  @spec compute_weights(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def compute_weights(labels, sensitive_attr, opts \\ []) do
    target = Keyword.get(opts, :target, :demographic_parity)
    min_per_group = Keyword.get(opts, :min_per_group, @default_min_per_group)

    # Validate inputs
    Validation.validate_labels!(labels)
    Validation.validate_matching_shapes!([labels, sensitive_attr], ["labels", "sensitive_attr"])

    Validation.validate_sensitive_attr!(sensitive_attr, min_per_group: min_per_group)

    # Compute weights based on target
    case target do
      :demographic_parity -> compute_demographic_parity_weights(labels, sensitive_attr)
      :equalized_odds -> compute_equalized_odds_weights(labels, sensitive_attr)
      _ -> raise ExFairness.Error, "Unknown target: #{inspect(target)}"
    end
  end

  # Compute weights for demographic parity
  # w(a, y) = P(Y = y) / P(A = a, Y = y)
  @spec compute_demographic_parity_weights(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defnp compute_demographic_parity_weights(labels, sensitive_attr) do
    n = Nx.axis_size(labels, 0)

    # Compute probabilities for each combination
    # P(A=0, Y=0)
    p_a0_y0 = count_combination(sensitive_attr, labels, 0, 0) / n
    # P(A=0, Y=1)
    p_a0_y1 = count_combination(sensitive_attr, labels, 0, 1) / n
    # P(A=1, Y=0)
    p_a1_y0 = count_combination(sensitive_attr, labels, 1, 0) / n
    # P(A=1, Y=1)
    p_a1_y1 = count_combination(sensitive_attr, labels, 1, 1) / n

    # P(Y=0) and P(Y=1)
    p_y0 = p_a0_y0 + p_a1_y0
    p_y1 = p_a0_y1 + p_a1_y1

    # Compute weight for each sample: w = P(Y=y) / P(A=a, Y=y)
    # Avoid division by zero with a small epsilon
    epsilon = 1.0e-6

    weights =
      Nx.select(
        Nx.logical_and(Nx.equal(sensitive_attr, 0), Nx.equal(labels, 0)),
        p_y0 / (p_a0_y0 + epsilon),
        Nx.select(
          Nx.logical_and(Nx.equal(sensitive_attr, 0), Nx.equal(labels, 1)),
          p_y1 / (p_a0_y1 + epsilon),
          Nx.select(
            Nx.logical_and(Nx.equal(sensitive_attr, 1), Nx.equal(labels, 0)),
            p_y0 / (p_a1_y0 + epsilon),
            p_y1 / (p_a1_y1 + epsilon)
          )
        )
      )

    # Normalize to mean 1.0
    normalize_weights(weights)
  end

  # Compute weights for equalized odds (same as demographic parity for this implementation)
  @spec compute_equalized_odds_weights(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defnp compute_equalized_odds_weights(labels, sensitive_attr) do
    # For equalized odds, we use the same reweighting scheme
    # This ensures balanced representation across all (A, Y) combinations
    compute_demographic_parity_weights(labels, sensitive_attr)
  end

  # Count samples matching specific combination
  @spec count_combination(Nx.Tensor.t(), Nx.Tensor.t(), number(), number()) :: Nx.Tensor.t()
  defnp count_combination(sensitive_attr, labels, a_val, y_val) do
    mask =
      Nx.logical_and(
        Nx.equal(sensitive_attr, a_val),
        Nx.equal(labels, y_val)
      )

    Nx.sum(mask)
  end

  # Normalize weights to have mean 1.0
  @spec normalize_weights(Nx.Tensor.t()) :: Nx.Tensor.t()
  defnp normalize_weights(weights) do
    mean_weight = Nx.mean(weights)
    weights / mean_weight
  end
end
