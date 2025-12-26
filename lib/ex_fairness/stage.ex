defmodule ExFairness.Stage do
  @moduledoc """
  Pipeline stage for fairness evaluation in CrucibleIR-based experiments.

  This stage integrates ExFairness metrics into the Crucible framework pipeline,
  allowing fairness evaluation to be seamlessly incorporated into LLM reliability
  experiments and model evaluations.

  ## Configuration

  The stage uses `CrucibleIR.Reliability.Fairness` configuration from the experiment context:

      %CrucibleIR.Reliability.Fairness{
        enabled: true,                    # Enable fairness evaluation
        metrics: [:demographic_parity, :equalized_odds, :equal_opportunity, :predictive_parity],
        group_by: :gender,                # Sensitive attribute field name
        threshold: 0.1,                   # Maximum acceptable disparity
        fail_on_violation: false,         # Whether to fail experiment on fairness violation
        options: %{}                      # Additional metric-specific options
      }

  ## Context Requirements

  The stage expects the context to contain:

    * `experiment.reliability.fairness` - Fairness configuration (CrucibleIR.Reliability.Fairness struct)
    * `outputs` - List of model outputs, where each output is a map containing:
      * `:prediction` - Binary prediction (0 or 1)
      * `:label` - Ground truth label (0 or 1)
      * `:probabilities` - (Optional) Prediction probabilities for calibration
      * Sensitive attribute field (e.g., `:gender`, `:race`) matching `group_by`

  ## Returns

  The stage returns `{:ok, updated_context}` with fairness results added to the context:

      context.fairness = %{
        metrics: %{
          demographic_parity: %{disparity: 0.05, passes: true, ...},
          equalized_odds: %{tpr_disparity: 0.03, fpr_disparity: 0.04, passes: true, ...},
          ...
        },
        overall_passes: true,
        violations: []
      }

  If `fail_on_violation` is true and fairness violations are detected, returns `{:error, reason}`.

  ## Example Usage

      # In a Crucible experiment configuration
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity, :equalized_odds],
        group_by: :gender,
        threshold: 0.1,
        fail_on_violation: false
      }

      # In pipeline
      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: [
          %{prediction: 1, label: 1, gender: 0},
          %{prediction: 0, label: 0, gender: 0},
          %{prediction: 1, label: 1, gender: 1},
          %{prediction: 0, label: 0, gender: 1}
        ]
      }

      {:ok, result_context} = ExFairness.Stage.run(context)
      # result_context.fairness contains fairness evaluation results

  ## Integration with Crucible Framework

  This stage is designed to work with the Crucible framework's experiment orchestration.
  It can be added to any pipeline that processes model outputs and requires fairness
  evaluation.

  See the Crucible documentation for more details on pipeline stages and experiment
  configuration.
  """

  @type context :: map()
  @type result :: {:ok, context()} | {:error, String.t()}

  @doc """
  Runs fairness evaluation on model outputs in the context.

  ## Parameters

    * `context` - Experiment context containing fairness config and model outputs
    * `opts` - Additional options (currently unused, reserved for future extensions)

  ## Returns

    * `{:ok, updated_context}` - Context with fairness results added
    * `{:error, reason}` - If configuration is invalid or fairness violations detected
      (when `fail_on_violation` is true)

  ## Examples

      iex> config = %CrucibleIR.Reliability.Fairness{
      ...>   enabled: true,
      ...>   metrics: [:demographic_parity],
      ...>   group_by: :gender,
      ...>   threshold: 0.1
      ...> }
      iex> context = %{
      ...>   experiment: %{reliability: %{fairness: config}},
      ...>   outputs: [
      ...>     %{prediction: 1, label: 1, gender: 0},
      ...>     %{prediction: 1, label: 1, gender: 0},
      ...>     %{prediction: 0, label: 0, gender: 0},
      ...>     %{prediction: 0, label: 0, gender: 0},
      ...>     %{prediction: 1, label: 1, gender: 1},
      ...>     %{prediction: 1, label: 1, gender: 1},
      ...>     %{prediction: 0, label: 0, gender: 1},
      ...>     %{prediction: 0, label: 0, gender: 1}
      ...>   ]
      ...> }
      iex> {:ok, result} = ExFairness.Stage.run(context)
      iex> is_map(result.fairness)
      true

  """
  @spec run(context(), keyword()) :: result()
  def run(context, opts \\ [])

  def run(%{experiment: %{reliability: %{fairness: %{enabled: false}}}} = context, _opts) do
    # Fairness evaluation disabled, pass through
    {:ok, context}
  end

  def run(%{experiment: %{reliability: %{fairness: config}}, outputs: outputs} = context, _opts)
      when is_list(outputs) and is_map(config) do
    with {:ok, tensors} <- extract_tensors(outputs, config.group_by),
         {:ok, metrics_results} <- compute_metrics(tensors, config),
         {:ok, fairness_results} <- build_results(metrics_results, config) do
      updated_context = Map.put(context, :fairness, fairness_results)

      if config.fail_on_violation and not fairness_results.overall_passes do
        {:error, "Fairness violations detected: #{format_violations(fairness_results.violations)}"}
      else
        {:ok, updated_context}
      end
    end
  end

  def run(_context, _opts) do
    {:error, "Invalid context: missing experiment.reliability.fairness config or outputs"}
  end

  @doc """
  Returns a description of the stage for pipeline documentation.

  ## Parameters

    * `opts` - Options (currently unused)

  ## Returns

  A string describing the stage's purpose and behavior.

  ## Examples

      iex> ExFairness.Stage.describe()
      "Fairness evaluation stage: Computes fairness metrics (demographic parity, equalized odds, etc.) on model outputs"

  """
  @spec describe(keyword()) :: String.t()
  def describe(_opts \\ []) do
    "Fairness evaluation stage: Computes fairness metrics (demographic parity, equalized odds, etc.) on model outputs"
  end

  # Private helper functions

  @spec extract_tensors(list(map()), atom()) ::
          {:ok,
           %{
             predictions: Nx.Tensor.t(),
             labels: Nx.Tensor.t(),
             sensitive: Nx.Tensor.t(),
             probabilities: Nx.Tensor.t() | nil
           }}
          | {:error, String.t()}
  defp extract_tensors(outputs, group_by_field) do
    if Enum.empty?(outputs) do
      {:error, "No outputs provided for fairness evaluation"}
    else
      try do
        predictions = outputs |> Enum.map(& &1.prediction) |> Nx.tensor()
        labels = outputs |> Enum.map(& &1.label) |> Nx.tensor()
        sensitive = outputs |> Enum.map(&Map.get(&1, group_by_field)) |> Nx.tensor()

        # Probabilities are optional (only needed for calibration)
        probabilities =
          if Enum.all?(outputs, &Map.has_key?(&1, :probabilities)) do
            outputs |> Enum.map(& &1.probabilities) |> Nx.tensor()
          else
            nil
          end

        {:ok,
         %{
           predictions: predictions,
           labels: labels,
           sensitive: sensitive,
           probabilities: probabilities
         }}
      rescue
        e ->
          {:error, "Failed to extract tensors from outputs: #{Exception.message(e)}"}
      end
    end
  end

  @spec compute_metrics(map(), map()) :: {:ok, map()} | {:error, String.t()}
  defp compute_metrics(tensors, config) do
    results =
      Enum.reduce(config.metrics, %{}, fn metric, acc ->
        result = compute_single_metric(metric, tensors, config)
        Map.put(acc, metric, result)
      end)

    {:ok, results}
  rescue
    e ->
      {:error, "Failed to compute metrics: #{Exception.message(e)}"}
  end

  @spec compute_single_metric(atom(), map(), map()) :: map()
  defp compute_single_metric(:demographic_parity, tensors, config) do
    opts = build_metric_opts(config)
    ExFairness.demographic_parity(tensors.predictions, tensors.sensitive, opts)
  end

  defp compute_single_metric(:equalized_odds, tensors, config) do
    opts = build_metric_opts(config)
    ExFairness.equalized_odds(tensors.predictions, tensors.labels, tensors.sensitive, opts)
  end

  defp compute_single_metric(:equal_opportunity, tensors, config) do
    opts = build_metric_opts(config)
    ExFairness.equal_opportunity(tensors.predictions, tensors.labels, tensors.sensitive, opts)
  end

  defp compute_single_metric(:predictive_parity, tensors, config) do
    opts = build_metric_opts(config)
    ExFairness.predictive_parity(tensors.predictions, tensors.labels, tensors.sensitive, opts)
  end

  defp compute_single_metric(:calibration, tensors, config) do
    if tensors.probabilities do
      opts = build_metric_opts(config)
      ExFairness.calibration(tensors.probabilities, tensors.labels, tensors.sensitive, opts)
    else
      %{
        error: "Calibration requires probabilities in outputs",
        passes: false
      }
    end
  end

  defp compute_single_metric(metric, _tensors, _config) do
    %{
      error: "Unknown metric: #{metric}",
      passes: false
    }
  end

  @spec build_metric_opts(map()) :: keyword()
  defp build_metric_opts(config) do
    extra_opts = if config.options, do: Map.to_list(config.options), else: []
    [threshold: config.threshold] ++ extra_opts
  end

  @spec build_results(map(), map()) :: {:ok, map()}
  defp build_results(metrics_results, _config) do
    violations =
      metrics_results
      |> Enum.filter(fn {_metric, result} -> not Map.get(result, :passes, false) end)
      |> Enum.map(fn {metric, result} ->
        %{
          metric: metric,
          details: result
        }
      end)

    overall_passes = Enum.empty?(violations)

    {:ok,
     %{
       metrics: metrics_results,
       overall_passes: overall_passes,
       violations: violations
     }}
  end

  @spec format_violations(list(map())) :: String.t()
  defp format_violations(violations) do
    violations
    |> Enum.map(fn %{metric: metric, details: details} ->
      disparity = Map.get(details, :disparity, Map.get(details, :tpr_disparity, "N/A"))
      "#{metric} (disparity: #{disparity})"
    end)
    |> Enum.join(", ")
  end
end
