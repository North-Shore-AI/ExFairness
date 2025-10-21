defmodule ExFairness do
  @moduledoc """
  ExFairness - Fairness and bias detection library for Elixir AI/ML systems.

  ExFairness provides comprehensive fairness metrics, bias detection algorithms,
  and mitigation techniques to ensure equitable predictions across different
  demographic groups.

  ## Features

  - **Fairness Metrics**: Demographic parity, equalized odds, equal opportunity, and more
  - **Bias Detection**: Statistical testing, disparate impact analysis, intersectional bias
  - **Mitigation**: Reweighting, resampling, threshold optimization, adversarial debiasing
  - **Reporting**: Comprehensive fairness reports with interpretations

  ## Quick Start

      # Compute demographic parity
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = ExFairness.demographic_parity(predictions, sensitive)
      # => %{disparity: 0.0, passes: true, ...}

  ## Metrics

  - `demographic_parity/3` - Demographic parity (statistical parity)
  - `equalized_odds/4` - Equalized odds (equal TPR and FPR)
  - More metrics coming soon...

  """

  alias ExFairness.Metrics.DemographicParity
  alias ExFairness.Metrics.EqualizedOdds

  @doc """
  Computes demographic parity disparity between groups.

  Demographic parity requires that the probability of a positive prediction
  is equal across groups defined by the sensitive attribute.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options (see `ExFairness.Metrics.DemographicParity.compute/3`)

  ## Returns

  A map containing fairness metrics. See `ExFairness.Metrics.DemographicParity.compute/3`
  for details.

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.demographic_parity(predictions, sensitive)
      iex> result.passes
      true

  """
  @spec demographic_parity(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          DemographicParity.result()
  def demographic_parity(predictions, sensitive_attr, opts \\ []) do
    DemographicParity.compute(predictions, sensitive_attr, opts)
  end

  @doc """
  Computes equalized odds disparity between groups.

  Equalized odds requires that both TPR and FPR are equal across groups.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options (see `ExFairness.Metrics.EqualizedOdds.compute/4`)

  ## Returns

  A map containing fairness metrics. See `ExFairness.Metrics.EqualizedOdds.compute/4`
  for details.

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
      iex> labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.equalized_odds(predictions, labels, sensitive)
      iex> result.passes
      true

  """
  @spec equalized_odds(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          EqualizedOdds.result()
  def equalized_odds(predictions, labels, sensitive_attr, opts \\ []) do
    EqualizedOdds.compute(predictions, labels, sensitive_attr, opts)
  end
end
