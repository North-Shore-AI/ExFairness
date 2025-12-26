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
  - `equal_opportunity/4` - Equal opportunity (equal TPR)
  - `predictive_parity/4` - Predictive parity (equal PPV)
  - More metrics coming soon...

  """

  alias ExFairness.Metrics.Calibration
  alias ExFairness.Metrics.DemographicParity
  alias ExFairness.Metrics.EqualizedOdds
  alias ExFairness.Metrics.EqualOpportunity
  alias ExFairness.Metrics.PredictiveParity
  alias ExFairness.Report

  # Support for CrucibleIR integration (conditionally compiled)
  # The evaluate/5 function is only available when crucible_ir is loaded

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

  @doc """
  Computes equal opportunity disparity between groups.

  Equal opportunity requires that TPR is equal across groups.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options (see `ExFairness.Metrics.EqualOpportunity.compute/4`)

  ## Returns

  A map containing fairness metrics. See `ExFairness.Metrics.EqualOpportunity.compute/4`
  for details.
  """
  @spec equal_opportunity(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          EqualOpportunity.result()
  def equal_opportunity(predictions, labels, sensitive_attr, opts \\ []) do
    EqualOpportunity.compute(predictions, labels, sensitive_attr, opts)
  end

  @doc """
  Computes predictive parity disparity between groups.

  Predictive parity requires that PPV/precision is equal across groups.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options (see `ExFairness.Metrics.PredictiveParity.compute/4`)

  ## Returns

  A map containing fairness metrics. See `ExFairness.Metrics.PredictiveParity.compute/4`
  for details.
  """
  @spec predictive_parity(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          PredictiveParity.result()
  def predictive_parity(predictions, labels, sensitive_attr, opts \\ []) do
    PredictiveParity.compute(predictions, labels, sensitive_attr, opts)
  end

  @doc """
  Computes calibration fairness between groups using predicted probabilities.

  Calibration checks whether predicted probabilities align with actual outcomes
  equally across groups, reporting ECE/MCE per group and the disparity.

  ## Parameters

    * `probabilities` - Predicted probabilities (0.0 to 1.0)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options (see `ExFairness.Metrics.Calibration.compute/4`)

  ## Examples

      iex> probs = Nx.tensor([0.1, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.8, 0.5, 0.3,
      ...>                    0.1, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.8, 0.5, 0.3])
      iex> labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.calibration(probs, labels, sensitive, n_bins: 5)
      iex> result.passes
      true

  """
  @spec calibration(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Calibration.result()
  def calibration(probabilities, labels, sensitive_attr, opts \\ []) do
    Calibration.compute(probabilities, labels, sensitive_attr, opts)
  end

  @doc """
  Generates a comprehensive fairness report across multiple metrics.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options (see `ExFairness.Report.generate/4`). To include calibration, pass `probabilities: probs`.

  ## Returns

  A comprehensive fairness report. See `ExFairness.Report.generate/4` for details.

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
      iex> labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> report = ExFairness.fairness_report(predictions, labels, sensitive)
      iex> report.total_count
      4

  """
  @spec fairness_report(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Report.report()
  def fairness_report(predictions, labels, sensitive_attr, opts \\ []) do
    Report.generate(predictions, labels, sensitive_attr, opts)
  end

  @doc """
  Evaluates fairness using a CrucibleIR.Reliability.Fairness configuration.

  This function provides a bridge between the Crucible framework and ExFairness,
  allowing fairness evaluation to be configured using CrucibleIR's configuration
  structures.

  Note: This function is available when the `crucible_ir` dependency is loaded.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `config` - CrucibleIR.Reliability.Fairness configuration struct
    * `probabilities` - (Optional) Prediction probabilities for calibration metrics

  ## Returns

  A map containing:
    * `:metrics` - Map of metric results for each configured metric
    * `:overall_passes` - Boolean indicating if all metrics pass
    * `:violations` - List of metrics that failed to pass

  ## Examples

      iex> config = %CrucibleIR.Reliability.Fairness{
      ...>   enabled: true,
      ...>   metrics: [:demographic_parity],
      ...>   threshold: 0.1
      ...> }
      iex> predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      iex> labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.evaluate(predictions, labels, sensitive, config)
      iex> result.overall_passes
      true

  """
  @spec evaluate(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), struct(), Nx.Tensor.t() | nil) ::
          %{
            metrics: map(),
            overall_passes: boolean(),
            violations: list(map())
          }
  def evaluate(predictions, labels, sensitive_attr, config, probabilities \\ nil)

  def evaluate(
        predictions,
        labels,
        sensitive_attr,
        %{__struct__: CrucibleIR.Reliability.Fairness} = config,
        probabilities
      ) do
    if config.enabled do
      evaluate_enabled(predictions, labels, sensitive_attr, config, probabilities)
    else
      %{metrics: %{}, overall_passes: true, violations: []}
    end
  end

  defp evaluate_enabled(predictions, labels, sensitive_attr, config, probabilities) do
    opts = build_metric_opts(config)

    metrics_results =
      compute_metrics_results(
        config.metrics,
        predictions,
        labels,
        sensitive_attr,
        probabilities,
        opts
      )

    violations = build_violations(metrics_results)

    %{
      metrics: metrics_results,
      overall_passes: Enum.empty?(violations),
      violations: violations
    }
  end

  defp build_metric_opts(config) do
    extra_opts = if config.options, do: Map.to_list(config.options), else: []
    [threshold: config.threshold] ++ extra_opts
  end

  defp compute_metrics_results(
         metrics,
         predictions,
         labels,
         sensitive_attr,
         probabilities,
         opts
       ) do
    Enum.reduce(metrics, %{}, fn metric, acc ->
      Map.put(
        acc,
        metric,
        compute_metric(metric, predictions, labels, sensitive_attr, probabilities, opts)
      )
    end)
  end

  defp compute_metric(:demographic_parity, predictions, _labels, sensitive_attr, _probs, opts) do
    demographic_parity(predictions, sensitive_attr, opts)
  end

  defp compute_metric(:equalized_odds, predictions, labels, sensitive_attr, _probs, opts) do
    equalized_odds(predictions, labels, sensitive_attr, opts)
  end

  defp compute_metric(:equal_opportunity, predictions, labels, sensitive_attr, _probs, opts) do
    equal_opportunity(predictions, labels, sensitive_attr, opts)
  end

  defp compute_metric(:predictive_parity, predictions, labels, sensitive_attr, _probs, opts) do
    predictive_parity(predictions, labels, sensitive_attr, opts)
  end

  defp compute_metric(:calibration, _preds, _labels, _sensitive_attr, nil, _opts) do
    %{error: "Calibration requires probabilities", passes: false}
  end

  defp compute_metric(:calibration, _preds, labels, sensitive_attr, probabilities, opts) do
    calibration(probabilities, labels, sensitive_attr, opts)
  end

  defp compute_metric(metric, _preds, _labels, _sensitive_attr, _probs, _opts) do
    %{error: "Unknown metric: #{metric}", passes: false}
  end

  defp build_violations(metrics_results) do
    metrics_results
    |> Enum.filter(fn {_metric, result} -> not Map.get(result, :passes, false) end)
    |> Enum.map(fn {metric, result} -> %{metric: metric, details: result} end)
  end
end
