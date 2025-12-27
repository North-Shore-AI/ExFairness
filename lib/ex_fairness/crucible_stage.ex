defmodule ExFairness.CrucibleStage do
  @moduledoc """
  Crucible.Stage implementation for fairness evaluation.

  This stage integrates ExFairness into crucible_framework pipelines,
  providing fairness metric evaluation on model outputs.

  ## Configuration

  The stage uses fairness configuration from the experiment context:

      %CrucibleIR.Reliability.Fairness{
        enabled: true,                    # Enable fairness evaluation
        metrics: [:demographic_parity, :equalized_odds, :equal_opportunity, :predictive_parity],
        group_by: :gender,                # Sensitive attribute field name
        threshold: 0.1,                   # Maximum acceptable disparity
        fail_on_violation: false,         # Whether to fail experiment on fairness violation
        options: %{}                      # Additional metric-specific options
      }

  ## Data Sources

  The stage extracts data from two possible sources (in order of preference):

  1. **From assigns** (preferred when pre-computed tensors available):
     - `context.assigns.fairness_predictions` - Binary predictions tensor
     - `context.assigns.fairness_labels` - Ground truth labels tensor
     - `context.assigns.fairness_sensitive` - Sensitive attribute tensor
     - `context.assigns.fairness_probabilities` - (Optional) Probabilities for calibration

  2. **From outputs** (fallback):
     - `context.outputs` - List of maps with :prediction, :label, sensitive attribute

  ## Results

  Results are stored in `context.metrics.fairness`:

      %{
        metrics: %{
          demographic_parity: %{disparity: 0.05, passes: true, ...},
          equalized_odds: %{tpr_disparity: 0.03, fpr_disparity: 0.04, passes: true, ...},
          ...
        },
        overall_passes: true,
        violations: []
      }

  ## Example Usage

      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity, :equalized_odds],
        group_by: :gender,
        threshold: 0.1
      }

      context = %Crucible.Context{
        experiment: %{reliability: %{fairness: config}},
        outputs: [
          %{prediction: 1, label: 1, gender: 0},
          %{prediction: 0, label: 0, gender: 1}
        ]
      }

      {:ok, result} = ExFairness.CrucibleStage.run(context, %{})
      result.metrics.fairness
      # => %{metrics: %{...}, overall_passes: true, violations: []}
  """

  @behaviour Crucible.Stage

  @version "0.5.0"

  @supported_metrics [
    :demographic_parity,
    :equalized_odds,
    :equal_opportunity,
    :predictive_parity,
    :calibration
  ]

  @type fairness_result :: %{
          metrics: map(),
          overall_passes: boolean(),
          violations: list(map())
        }

  @type context :: map()
  @type opts :: map()

  # ============================================================================
  # Crucible.Stage Callbacks
  # ============================================================================

  @doc """
  Returns metadata about the fairness evaluation stage in canonical schema format.

  ## Parameters

    * `opts` - Options (currently unused)

  ## Returns

  A map containing stage metadata in canonical schema format with:
  - `__schema_version__` - Schema version marker
  - `name` - Stage identifier (atom)
  - `description` - Human-readable description
  - `required` - Required option keys
  - `optional` - Optional option keys
  - `types` - Type specifications for options
  - `defaults` - Default values for optional options
  - `version` - Package version
  - `__extensions__` - Fairness-specific metadata

  ## Examples

      iex> schema = ExFairness.CrucibleStage.describe(%{})
      iex> schema.name
      :fairness
      iex> schema.__schema_version__
      "1.0.0"
      iex> is_list(schema.required)
      true
  """
  @impl Crucible.Stage
  @spec describe(opts()) :: map()
  def describe(_opts \\ %{}) do
    %{
      __schema_version__: "1.0.0",
      name: :fairness,
      description: "Evaluates fairness metrics on model predictions",
      required: [],
      optional: [:metrics, :group_by, :threshold, :fail_on_violation, :options],
      types: %{
        metrics:
          {:list,
           {:enum,
            [
              :demographic_parity,
              :equalized_odds,
              :equal_opportunity,
              :predictive_parity,
              :calibration
            ]}},
        group_by: :atom,
        threshold: :float,
        fail_on_violation: :boolean,
        options: :map
      },
      defaults: %{
        metrics: [:demographic_parity, :equalized_odds],
        group_by: :gender,
        threshold: 0.1,
        fail_on_violation: false,
        options: %{}
      },
      version: @version,
      __extensions__: %{
        fairness: %{
          supported_metrics: @supported_metrics,
          data_sources: [
            %{
              name: :assigns,
              fields: [
                :fairness_predictions,
                :fairness_labels,
                :fairness_sensitive,
                :fairness_probabilities
              ],
              preferred: true
            },
            %{
              name: :outputs,
              fields: [:prediction, :label, :group_by_field],
              fallback: true
            }
          ],
          output_location: [:metrics, :fairness]
        }
      }
    }
  end

  @doc """
  Runs fairness evaluation on model outputs in the context.

  ## Parameters

    * `context` - Experiment context containing fairness config and model outputs
    * `opts` - Additional options (currently unused, reserved for future extensions)

  ## Returns

    * `{:ok, updated_context}` - Context with fairness results added to `context.metrics.fairness`
    * `{:error, reason}` - If configuration is invalid, data missing, or fairness violations detected
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
      ...>     %{prediction: 0, label: 0, gender: 1}
      ...>   ],
      ...>   metrics: %{}
      ...> }
      iex> {:ok, result} = ExFairness.CrucibleStage.run(context, %{})
      iex> is_map(result.metrics.fairness)
      true
  """
  @impl Crucible.Stage
  @spec run(context(), opts()) :: {:ok, context()} | {:error, term()}
  def run(context, opts \\ %{})

  # Handle disabled fairness - pass through unchanged
  def run(%{experiment: %{reliability: %{fairness: %{enabled: false}}}} = context, _opts) do
    {:ok, context}
  end

  # Handle enabled fairness with proper config
  def run(%{experiment: %{reliability: %{fairness: config}}} = context, _opts)
      when is_map(config) do
    with {:ok, tensors} <- extract_tensors(context, config),
         {:ok, metrics_results} <- compute_metrics(tensors, config),
         {:ok, fairness_results} <- build_results(metrics_results, config) do
      updated_context = put_in_metrics(context, :fairness, fairness_results)

      if Map.get(config, :fail_on_violation, false) and not fairness_results.overall_passes do
        {:error, "Fairness violations detected: #{format_violations(fairness_results.violations)}"}
      else
        {:ok, updated_context}
      end
    end
  end

  # Handle missing config
  def run(_context, _opts) do
    {:error, "Invalid context: missing experiment.reliability.fairness configuration"}
  end

  # ============================================================================
  # Data Extraction
  # ============================================================================

  @spec extract_tensors(context(), map()) ::
          {:ok,
           %{
             predictions: Nx.Tensor.t(),
             labels: Nx.Tensor.t(),
             sensitive: Nx.Tensor.t(),
             probabilities: Nx.Tensor.t() | nil
           }}
          | {:error, String.t()}
  defp extract_tensors(context, config) do
    assigns = Map.get(context, :assigns, %{})
    outputs = Map.get(context, :outputs, [])
    group_by = Map.get(config, :group_by, :gender)

    cond do
      # Prefer assigns when fairness tensors are available
      has_fairness_assigns?(assigns) ->
        extract_from_assigns(assigns)

      # Fall back to outputs
      is_list(outputs) and not Enum.empty?(outputs) ->
        extract_from_outputs(outputs, group_by)

      # No data available
      true ->
        {:error, "No data available for fairness evaluation (empty outputs and no assigns)"}
    end
  end

  @spec has_fairness_assigns?(map()) :: boolean()
  defp has_fairness_assigns?(assigns) do
    Map.has_key?(assigns, :fairness_predictions) and
      Map.has_key?(assigns, :fairness_labels) and
      Map.has_key?(assigns, :fairness_sensitive)
  end

  @spec extract_from_assigns(map()) ::
          {:ok,
           %{
             predictions: Nx.Tensor.t(),
             labels: Nx.Tensor.t(),
             sensitive: Nx.Tensor.t(),
             probabilities: Nx.Tensor.t() | nil
           }}
  defp extract_from_assigns(assigns) do
    {:ok,
     %{
       predictions: assigns.fairness_predictions,
       labels: assigns.fairness_labels,
       sensitive: assigns.fairness_sensitive,
       probabilities: Map.get(assigns, :fairness_probabilities)
     }}
  end

  @spec extract_from_outputs(list(map()), atom()) ::
          {:ok,
           %{
             predictions: Nx.Tensor.t(),
             labels: Nx.Tensor.t(),
             sensitive: Nx.Tensor.t(),
             probabilities: Nx.Tensor.t() | nil
           }}
          | {:error, String.t()}
  defp extract_from_outputs(outputs, group_by_field) do
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

  # ============================================================================
  # Metric Computation
  # ============================================================================

  @spec compute_metrics(map(), map()) :: {:ok, map()} | {:error, String.t()}
  defp compute_metrics(tensors, config) do
    metrics = Map.get(config, :metrics, [])

    try do
      results =
        Enum.reduce(metrics, %{}, fn metric, acc ->
          result = compute_single_metric(metric, tensors, config)
          Map.put(acc, metric, result)
        end)

      {:ok, results}
    rescue
      e ->
        {:error, "Failed to compute metrics: #{Exception.message(e)}"}
    end
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
        error: "Calibration requires probabilities in outputs or assigns",
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
    threshold = Map.get(config, :threshold, 0.1)
    options = Map.get(config, :options, %{})
    extra_opts = if is_map(options), do: Map.to_list(options), else: []
    [threshold: threshold] ++ extra_opts
  end

  # ============================================================================
  # Result Building
  # ============================================================================

  @spec build_results(map(), map()) :: {:ok, fairness_result()}
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

  # ============================================================================
  # Utilities
  # ============================================================================

  @spec put_in_metrics(context(), atom(), term()) :: context()
  defp put_in_metrics(context, key, value) do
    metrics = Map.get(context, :metrics, %{})
    updated_metrics = Map.put(metrics, key, value)
    Map.put(context, :metrics, updated_metrics)
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
