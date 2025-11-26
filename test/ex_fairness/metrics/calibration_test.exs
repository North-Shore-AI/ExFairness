defmodule ExFairness.Metrics.CalibrationTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Metrics.Calibration

  alias ExFairness.Metrics.Calibration

  describe "compute/4" do
    test "returns valid result structure" do
      probs =
        Nx.tensor([
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3,
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive)

      assert is_map(result)
      assert Map.has_key?(result, :group_a_ece)
      assert Map.has_key?(result, :group_b_ece)
      assert Map.has_key?(result, :disparity)
      assert Map.has_key?(result, :passes)
      assert Map.has_key?(result, :threshold)
      assert Map.has_key?(result, :group_a_mce)
      assert Map.has_key?(result, :group_b_mce)
      assert Map.has_key?(result, :n_bins)
      assert Map.has_key?(result, :strategy)
      assert Map.has_key?(result, :interpretation)
    end

    test "ECE values are non-negative" do
      probs =
        Nx.tensor([
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3,
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive)

      assert result.group_a_ece >= 0.0
      assert result.group_b_ece >= 0.0
      assert result.disparity >= 0.0
    end

    test "MCE values are non-negative" do
      probs =
        Nx.tensor([
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3,
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive)

      assert result.group_a_mce >= 0.0
      assert result.group_b_mce >= 0.0
    end

    test "perfectly calibrated model has low ECE" do
      # Create perfectly calibrated data
      # For 0.1 probability: 10% positive, 0.5: 50%, 0.9: 90%
      probs =
        Nx.tensor([
          0.1,
          0.1,
          0.1,
          0.1,
          0.1,
          0.1,
          0.1,
          0.1,
          0.1,
          0.1,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5
        ])

      # Matches probabilities: 1 positive out of 10 at 0.1, 5 out of 10 at 0.5
      labels = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive, n_bins: 5)

      # ECE should be low for well-calibrated data
      assert result.group_a_ece < 0.3
      assert result.group_b_ece < 0.3
    end

    test "completely miscalibrated model has high ECE" do
      # High confidence but all wrong predictions
      probs =
        Nx.tensor([
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9,
          0.9
        ])

      # All actually negative
      labels = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive, n_bins: 5)

      # ECE should be high for miscalibrated data
      assert result.group_a_ece > 0.5
      assert result.group_b_ece > 0.5
    end

    test "respects threshold parameter" do
      probs =
        Nx.tensor([
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3,
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result_strict = Calibration.compute(probs, labels, sensitive, threshold: 0.05)
      result_lenient = Calibration.compute(probs, labels, sensitive, threshold: 0.5)

      assert result_strict.threshold == 0.05
      assert result_lenient.threshold == 0.5

      # Same disparity but different pass/fail
      assert result_strict.disparity == result_lenient.disparity
    end

    test "respects n_bins parameter" do
      probs =
        Nx.tensor([
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3,
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result_5 = Calibration.compute(probs, labels, sensitive, n_bins: 5)
      result_10 = Calibration.compute(probs, labels, sensitive, n_bins: 10)

      assert result_5.n_bins == 5
      assert result_10.n_bins == 10
    end

    test "supports uniform binning strategy" do
      probs =
        Nx.tensor([
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3,
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive, strategy: :uniform)

      assert result.strategy == :uniform
      assert is_float(result.group_a_ece)
      assert is_float(result.group_b_ece)
    end

    test "supports quantile binning strategy" do
      probs =
        Nx.tensor([
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3,
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive, strategy: :quantile)

      assert result.strategy == :quantile
      assert is_float(result.group_a_ece)
      assert is_float(result.group_b_ece)
    end

    test "includes interpretation" do
      probs =
        Nx.tensor([
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3,
          0.1,
          0.3,
          0.6,
          0.9,
          0.2,
          0.4,
          0.7,
          0.8,
          0.5,
          0.3
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive)

      assert is_binary(result.interpretation)
      assert String.contains?(result.interpretation, "Calibration")
      assert String.contains?(result.interpretation, "ECE")
      assert String.contains?(result.interpretation, "Group A")
      assert String.contains?(result.interpretation, "Group B")
    end

    test "raises error for probabilities out of range" do
      # Probabilities > 1.0
      probs =
        Nx.tensor([
          1.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, fn ->
        Calibration.compute(probs, labels, sensitive)
      end
    end

    test "raises error for negative probabilities" do
      # Probabilities < 0.0
      probs =
        Nx.tensor([
          -0.1,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, fn ->
        Calibration.compute(probs, labels, sensitive)
      end
    end

    test "handles edge case with all same probabilities" do
      # All probabilities are 0.5
      probs =
        Nx.tensor([
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5,
          0.5
        ])

      labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive, n_bins: 5)

      # Should not crash
      assert is_float(result.group_a_ece)
      assert is_float(result.group_b_ece)
    end

    test "handles different group sizes" do
      # 15 samples in group A, 5 in group B
      probs =
        Nx.tensor([
          0.1,
          0.2,
          0.3,
          0.4,
          0.5,
          0.6,
          0.7,
          0.8,
          0.9,
          0.3,
          0.4,
          0.5,
          0.6,
          0.7,
          0.8,
          0.2,
          0.4,
          0.6,
          0.8,
          0.9
        ])

      labels = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1])

      sensitive =
        Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

      result = Calibration.compute(probs, labels, sensitive, n_bins: 5)

      assert is_float(result.group_a_ece)
      assert is_float(result.group_b_ece)
      assert is_float(result.disparity)
    end
  end
end
