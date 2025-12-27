defmodule ExFairness.CrucibleStageTest do
  @moduledoc """
  Tests for ExFairness.CrucibleStage - Crucible.Stage behaviour implementation.
  """
  use ExUnit.Case, async: true

  alias ExFairness.CrucibleStage

  # Mock Crucible.Context struct for testing without crucible_framework dependency
  defmodule MockContext do
    @moduledoc false
    defstruct [
      :experiment_id,
      :run_id,
      :experiment,
      outputs: [],
      metrics: %{},
      artifacts: %{},
      trace: nil,
      telemetry_context: %{},
      assigns: %{}
    ]

    @type t :: %__MODULE__{}

    def put_metric(%__MODULE__{} = ctx, key, value) when is_atom(key) do
      %__MODULE__{ctx | metrics: Map.put(ctx.metrics, key, value)}
    end
  end

  # Mock CrucibleIR.Reliability.Fairness struct for testing
  defmodule MockFairnessConfig do
    @moduledoc false
    defstruct enabled: true,
              metrics: [],
              group_by: :gender,
              threshold: 0.1,
              fail_on_violation: false,
              options: %{}
  end

  # Helper to build a test context
  defp build_context(opts) do
    config = Keyword.get(opts, :config, %MockFairnessConfig{enabled: true})
    outputs = Keyword.get(opts, :outputs, [])
    assigns = Keyword.get(opts, :assigns, %{})
    metrics = Keyword.get(opts, :metrics, %{})

    %MockContext{
      experiment_id: "test-exp-1",
      run_id: "test-run-1",
      experiment: %{reliability: %{fairness: config}},
      outputs: outputs,
      metrics: metrics,
      assigns: assigns
    }
  end

  # Sample balanced outputs (50/50 for each group)
  defp balanced_outputs do
    [
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 0, label: 0, gender: 0},
      %{prediction: 0, label: 0, gender: 0},
      %{prediction: 0, label: 0, gender: 0},
      %{prediction: 0, label: 0, gender: 0},
      %{prediction: 0, label: 0, gender: 0},
      %{prediction: 1, label: 1, gender: 1},
      %{prediction: 1, label: 1, gender: 1},
      %{prediction: 1, label: 1, gender: 1},
      %{prediction: 1, label: 1, gender: 1},
      %{prediction: 1, label: 1, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1}
    ]
  end

  # Sample biased outputs (80% positive for group 0, 20% for group 1)
  defp biased_outputs do
    [
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 1, label: 1, gender: 0},
      %{prediction: 0, label: 0, gender: 0},
      %{prediction: 0, label: 0, gender: 0},
      %{prediction: 1, label: 1, gender: 1},
      %{prediction: 1, label: 1, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1},
      %{prediction: 0, label: 0, gender: 1}
    ]
  end

  describe "behaviour" do
    test "implements Crucible.Stage behaviour callbacks" do
      assert function_exported?(CrucibleStage, :run, 2)
      assert function_exported?(CrucibleStage, :describe, 1)
    end

    test "run/2 has correct arity" do
      # Verify run/2 exists with context and opts
      assert {:run, 2} in CrucibleStage.__info__(:functions)
    end

    test "describe/1 has correct arity" do
      # Verify describe/1 exists with opts
      assert {:describe, 1} in CrucibleStage.__info__(:functions)
    end
  end

  describe "describe/1" do
    test "returns canonical schema format" do
      schema = CrucibleStage.describe(%{})

      # Must use :name key, not :stage
      assert Map.has_key?(schema, :name)
      refute Map.has_key?(schema, :stage)

      # Name must be atom
      assert schema.name == :fairness
      assert is_atom(schema.name)

      # Core fields
      assert is_binary(schema.description)
      assert is_list(schema.required)
      assert is_list(schema.optional)
      assert is_map(schema.types)
    end

    test "returns map with stage metadata" do
      result = CrucibleStage.describe(%{})
      assert is_map(result)
    end

    test "contains :name key (canonical format)" do
      result = CrucibleStage.describe(%{})
      assert Map.has_key?(result, :name)
      assert result.name == :fairness
      assert is_atom(result.name)
    end

    test "contains :description key" do
      result = CrucibleStage.describe(%{})
      assert Map.has_key?(result, :description)
      assert is_binary(result.description)
      assert result.description =~ "fairness"
    end

    test "contains :version key" do
      result = CrucibleStage.describe(%{})
      assert Map.has_key?(result, :version)
    end

    test "contains __schema_version__ key" do
      result = CrucibleStage.describe(%{})
      assert Map.has_key?(result, :__schema_version__)
      assert result.__schema_version__ == "1.0.0"
    end

    test "all optional fields have types" do
      schema = CrucibleStage.describe(%{})

      for key <- schema.optional do
        assert Map.has_key?(schema.types, key),
               "Optional field #{key} missing from types"
      end
    end

    test "contains defaults for optional fields" do
      schema = CrucibleStage.describe(%{})
      assert Map.has_key?(schema, :defaults)
      assert is_map(schema.defaults)

      # All defaults keys must be in optional
      for key <- Map.keys(schema.defaults) do
        assert key in schema.optional,
               "Default key #{key} not in optional list"
      end
    end

    test "extensions contain supported metrics" do
      schema = CrucibleStage.describe(%{})

      assert Map.has_key?(schema, :__extensions__)
      assert Map.has_key?(schema.__extensions__, :fairness)

      assert schema.__extensions__.fairness.supported_metrics == [
               :demographic_parity,
               :equalized_odds,
               :equal_opportunity,
               :predictive_parity,
               :calibration
             ]
    end

    test "extensions contain data sources" do
      schema = CrucibleStage.describe(%{})

      assert Map.has_key?(schema.__extensions__.fairness, :data_sources)
      data_sources = schema.__extensions__.fairness.data_sources
      assert is_list(data_sources)
      assert length(data_sources) == 2

      # Check assigns source
      assigns_source = Enum.find(data_sources, &(&1.name == :assigns))
      assert assigns_source != nil
      assert assigns_source.preferred == true

      # Check outputs source
      outputs_source = Enum.find(data_sources, &(&1.name == :outputs))
      assert outputs_source != nil
      assert outputs_source.fallback == true
    end
  end

  describe "run/2 with disabled fairness" do
    test "passes through context when disabled" do
      config = %MockFairnessConfig{enabled: false}
      context = build_context(config: config)

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert result == context
    end

    test "does not add fairness metrics when disabled" do
      config = %MockFairnessConfig{enabled: false}
      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      refute Map.has_key?(result.metrics, :fairness)
    end
  end

  describe "run/2 data extraction from outputs" do
    test "extracts predictions, labels, and sensitive_attr from outputs" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics, :fairness)
    end

    test "uses custom group_by field" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :race,
        threshold: 0.1
      }

      outputs =
        Enum.map(balanced_outputs(), fn output ->
          output
          |> Map.delete(:gender)
          |> Map.put(:race, Map.get(output, :gender))
        end)

      context = build_context(config: config, outputs: outputs)

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert result.metrics.fairness.overall_passes == true
    end
  end

  describe "run/2 data extraction from assigns" do
    test "extracts from assigns when fairness_predictions provided" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      # Balanced predictions
      predictions = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
      labels = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assigns = %{
        fairness_predictions: predictions,
        fairness_labels: labels,
        fairness_sensitive: sensitive
      }

      context = build_context(config: config, assigns: assigns)

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics, :fairness)
      assert result.metrics.fairness.overall_passes == true
    end

    test "prefers assigns over outputs when both present" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      # Balanced predictions in assigns
      predictions = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
      labels = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assigns = %{
        fairness_predictions: predictions,
        fairness_labels: labels,
        fairness_sensitive: sensitive
      }

      # Biased outputs (should be ignored)
      context = build_context(config: config, outputs: biased_outputs(), assigns: assigns)

      assert {:ok, result} = CrucibleStage.run(context, %{})
      # Should pass because assigns are balanced (not biased outputs)
      assert result.metrics.fairness.overall_passes == true
    end
  end

  describe "run/2 error handling for missing data" do
    test "returns error when no data available (empty outputs and no assigns)" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender
      }

      context = build_context(config: config, outputs: [])

      assert {:error, reason} = CrucibleStage.run(context, %{})
      assert is_binary(reason) or is_atom(reason)
    end

    test "returns error when config is missing" do
      context = %MockContext{
        experiment_id: "test",
        run_id: "test",
        experiment: %{},
        outputs: balanced_outputs()
      }

      assert {:error, _reason} = CrucibleStage.run(context, %{})
    end
  end

  describe "run/2 metric computation" do
    test "computes demographic parity" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness.metrics, :demographic_parity)
      dp = result.metrics.fairness.metrics.demographic_parity
      assert Map.has_key?(dp, :disparity)
      assert Map.has_key?(dp, :passes)
    end

    test "computes equalized_odds" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:equalized_odds],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness.metrics, :equalized_odds)
      eo = result.metrics.fairness.metrics.equalized_odds
      assert Map.has_key?(eo, :tpr_disparity)
      assert Map.has_key?(eo, :fpr_disparity)
      assert Map.has_key?(eo, :passes)
    end

    test "computes equal_opportunity" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:equal_opportunity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness.metrics, :equal_opportunity)
    end

    test "computes predictive_parity" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:predictive_parity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness.metrics, :predictive_parity)
    end

    test "computes multiple metrics" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity, :equalized_odds, :equal_opportunity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness.metrics, :demographic_parity)
      assert Map.has_key?(result.metrics.fairness.metrics, :equalized_odds)
      assert Map.has_key?(result.metrics.fairness.metrics, :equal_opportunity)
    end

    test "computes calibration with probabilities" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:calibration],
        group_by: :gender,
        threshold: 0.1
      }

      outputs = [
        %{prediction: 0, label: 0, gender: 0, probabilities: 0.1},
        %{prediction: 0, label: 0, gender: 0, probabilities: 0.2},
        %{prediction: 0, label: 1, gender: 0, probabilities: 0.3},
        %{prediction: 1, label: 1, gender: 0, probabilities: 0.6},
        %{prediction: 1, label: 1, gender: 0, probabilities: 0.7},
        %{prediction: 1, label: 1, gender: 0, probabilities: 0.8},
        %{prediction: 1, label: 1, gender: 0, probabilities: 0.9},
        %{prediction: 1, label: 0, gender: 0, probabilities: 0.8},
        %{prediction: 1, label: 1, gender: 0, probabilities: 0.7},
        %{prediction: 0, label: 0, gender: 0, probabilities: 0.4},
        %{prediction: 0, label: 0, gender: 1, probabilities: 0.1},
        %{prediction: 0, label: 0, gender: 1, probabilities: 0.2},
        %{prediction: 0, label: 1, gender: 1, probabilities: 0.3},
        %{prediction: 1, label: 1, gender: 1, probabilities: 0.6},
        %{prediction: 1, label: 1, gender: 1, probabilities: 0.7},
        %{prediction: 1, label: 1, gender: 1, probabilities: 0.8},
        %{prediction: 1, label: 1, gender: 1, probabilities: 0.9},
        %{prediction: 1, label: 0, gender: 1, probabilities: 0.8},
        %{prediction: 1, label: 1, gender: 1, probabilities: 0.7},
        %{prediction: 0, label: 0, gender: 1, probabilities: 0.4}
      ]

      context = build_context(config: config, outputs: outputs)

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness.metrics, :calibration)
    end

    test "reports calibration error when probabilities missing" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:calibration],
        group_by: :gender,
        threshold: 0.1
      }

      # Outputs without probabilities
      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness.metrics, :calibration)
      assert result.metrics.fairness.metrics.calibration.passes == false
      assert Map.has_key?(result.metrics.fairness.metrics.calibration, :error)
    end
  end

  describe "run/2 result merging" do
    test "stores results in context.metrics.fairness" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics, :fairness)
      assert is_map(result.metrics.fairness)
    end

    test "includes overall_passes in results" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness, :overall_passes)
      assert is_boolean(result.metrics.fairness.overall_passes)
    end

    test "includes violations list in results" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness, :violations)
      assert is_list(result.metrics.fairness.violations)
    end

    test "preserves existing metrics" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      existing_metrics = %{accuracy: 0.95, loss: 0.05}

      context =
        build_context(config: config, outputs: balanced_outputs(), metrics: existing_metrics)

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert result.metrics.accuracy == 0.95
      assert result.metrics.loss == 0.05
      assert Map.has_key?(result.metrics, :fairness)
    end
  end

  describe "run/2 violation detection" do
    test "detects fairness violations" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: biased_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert result.metrics.fairness.overall_passes == false
      refute Enum.empty?(result.metrics.fairness.violations)
    end

    test "fails when fail_on_violation is true and violations detected" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1,
        fail_on_violation: true
      }

      context = build_context(config: config, outputs: biased_outputs())

      assert {:error, message} = CrucibleStage.run(context, %{})
      assert is_binary(message)
      assert message =~ "violation" or message =~ "Fairness"
    end

    test "passes with no violations when data is fair" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1,
        fail_on_violation: true
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert result.metrics.fairness.overall_passes == true
      assert result.metrics.fairness.violations == []
    end
  end

  describe "run/2 threshold handling" do
    test "uses custom threshold from config" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.05
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert result.metrics.fairness.metrics.demographic_parity.threshold == 0.05
    end

    test "threshold affects pass/fail determination" do
      # Relaxed threshold (0.2) allows disparity up to 0.2
      relaxed_config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.2
      }

      # Create outputs with 10% disparity: group 0 has 60% positive, group 1 has 50%
      outputs_with_disparity = [
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 0, label: 0, gender: 0},
        %{prediction: 0, label: 0, gender: 0},
        %{prediction: 0, label: 0, gender: 0},
        %{prediction: 0, label: 0, gender: 0},
        %{prediction: 1, label: 1, gender: 1},
        %{prediction: 1, label: 1, gender: 1},
        %{prediction: 1, label: 1, gender: 1},
        %{prediction: 1, label: 1, gender: 1},
        %{prediction: 1, label: 1, gender: 1},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 0, label: 0, gender: 1}
      ]

      context = build_context(config: relaxed_config, outputs: outputs_with_disparity)

      # With 0.2 threshold, 10% disparity should pass
      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert result.metrics.fairness.metrics.demographic_parity.passes == true

      # Now test with strict threshold (0.05)
      strict_config = %MockFairnessConfig{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.05
      }

      strict_context = build_context(config: strict_config, outputs: outputs_with_disparity)

      # With 0.05 threshold, 10% disparity should fail
      assert {:ok, strict_result} = CrucibleStage.run(strict_context, %{})
      assert strict_result.metrics.fairness.metrics.demographic_parity.passes == false
    end
  end

  describe "run/2 unknown metric handling" do
    test "handles unknown metric gracefully" do
      config = %MockFairnessConfig{
        enabled: true,
        metrics: [:unknown_metric],
        group_by: :gender,
        threshold: 0.1
      }

      context = build_context(config: config, outputs: balanced_outputs())

      assert {:ok, result} = CrucibleStage.run(context, %{})
      assert Map.has_key?(result.metrics.fairness.metrics, :unknown_metric)
      assert result.metrics.fairness.metrics.unknown_metric.passes == false
      assert Map.has_key?(result.metrics.fairness.metrics.unknown_metric, :error)
    end
  end
end
