defmodule ExFairness.StageTest do
  use ExUnit.Case, async: true

  alias ExFairness.Stage

  # Mock CrucibleIR.Reliability.Fairness struct for testing
  # This allows tests to run even if crucible_ir is not available
  defmodule CrucibleIR.Reliability.Fairness do
    @moduledoc false
    defstruct enabled: true,
              metrics: [],
              group_by: :gender,
              threshold: 0.1,
              fail_on_violation: false,
              options: %{}
  end

  describe "describe/1" do
    test "returns stage description" do
      description = Stage.describe()
      assert is_binary(description)
      assert description =~ "Fairness evaluation"
      assert description =~ "demographic parity"
    end
  end

  describe "run/2 with disabled fairness" do
    test "passes through context when fairness is disabled" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: false
      }

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: []
      }

      assert {:ok, result} = Stage.run(context)
      assert result == context
      refute Map.has_key?(result, :fairness)
    end
  end

  describe "run/2 with valid context" do
    test "evaluates demographic parity successfully" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      outputs = [
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

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      assert {:ok, result} = Stage.run(context)
      assert Map.has_key?(result, :fairness)
      assert Map.has_key?(result.fairness, :metrics)
      assert Map.has_key?(result.fairness.metrics, :demographic_parity)
      assert result.fairness.metrics.demographic_parity.passes == true
      assert result.fairness.overall_passes == true
      assert result.fairness.violations == []
    end

    test "evaluates multiple metrics successfully" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity, :equalized_odds, :equal_opportunity],
        group_by: :gender,
        threshold: 0.1
      }

      outputs = [
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

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      assert {:ok, result} = Stage.run(context)
      assert Map.has_key?(result.fairness.metrics, :demographic_parity)
      assert Map.has_key?(result.fairness.metrics, :equalized_odds)
      assert Map.has_key?(result.fairness.metrics, :equal_opportunity)
      assert result.fairness.overall_passes == true
    end

    test "evaluates calibration with probabilities" do
      config = %CrucibleIR.Reliability.Fairness{
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

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      assert {:ok, result} = Stage.run(context)
      assert Map.has_key?(result.fairness.metrics, :calibration)
    end

    test "reports calibration error when probabilities missing" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:calibration],
        group_by: :gender,
        threshold: 0.1
      }

      outputs = [
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 0, label: 0, gender: 1}
      ]

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      assert {:ok, result} = Stage.run(context)
      assert Map.has_key?(result.fairness.metrics, :calibration)
      assert result.fairness.metrics.calibration.passes == false
      assert result.fairness.metrics.calibration.error =~ "probabilities"
    end

    test "uses custom threshold from config" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.05
      }

      outputs = [
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

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      assert {:ok, result} = Stage.run(context)
      assert result.fairness.metrics.demographic_parity.threshold == 0.05
    end

    test "detects fairness violations" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }

      # Biased outputs - group 0 gets 80% positive, group 1 gets 20%
      outputs = [
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

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      assert {:ok, result} = Stage.run(context)
      assert result.fairness.overall_passes == false
      refute Enum.empty?(result.fairness.violations)
      assert Enum.any?(result.fairness.violations, fn v -> v.metric == :demographic_parity end)
    end

    test "fails when fail_on_violation is true and violations detected" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1,
        fail_on_violation: true
      }

      # Biased outputs
      outputs = [
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

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      assert {:error, message} = Stage.run(context)
      assert message =~ "Fairness violations detected"
      assert message =~ "demographic_parity"
    end
  end

  describe "run/2 with invalid context" do
    test "returns error when config missing" do
      context = %{
        outputs: []
      }

      assert {:error, message} = Stage.run(context)
      assert message =~ "Invalid context"
    end

    test "returns error when outputs missing" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender
      }

      context = %{
        experiment: %{reliability: %{fairness: config}}
      }

      assert {:error, message} = Stage.run(context)
      assert message =~ "Invalid context"
    end

    test "returns error when outputs empty" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender
      }

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: []
      }

      assert {:error, message} = Stage.run(context)
      assert message =~ "No outputs provided"
    end

    test "handles unknown metric gracefully" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:unknown_metric],
        group_by: :gender
      }

      outputs = [
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 0, label: 0, gender: 1}
      ]

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      assert {:ok, result} = Stage.run(context)
      assert Map.has_key?(result.fairness.metrics, :unknown_metric)
      assert result.fairness.metrics.unknown_metric.passes == false
      assert result.fairness.metrics.unknown_metric.error =~ "Unknown metric"
    end
  end

  describe "run/2 with custom options" do
    test "passes through custom options to metrics" do
      config = %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1,
        options: %{min_per_group: 5}
      }

      outputs = [
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 0, label: 0, gender: 1}
      ]

      context = %{
        experiment: %{reliability: %{fairness: config}},
        outputs: outputs
      }

      # Should succeed with min_per_group: 5 (5 samples per group)
      assert {:ok, _result} = Stage.run(context)
    end
  end
end
