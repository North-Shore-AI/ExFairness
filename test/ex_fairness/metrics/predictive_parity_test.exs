defmodule ExFairness.Metrics.PredictiveParityTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Metrics.PredictiveParity

  alias ExFairness.Metrics.PredictiveParity

  describe "compute/4" do
    test "detects perfect predictive parity" do
      # Both groups have same PPV (precision)
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = PredictiveParity.compute(predictions, labels, sensitive)

      # Group A: TP=2, FP=1 -> PPV=2/3 ≈ 0.667
      # Group B: TP=2, FP=1 -> PPV=2/3 ≈ 0.667
      assert_in_delta(result.disparity, 0.0, 0.01)
      assert result.passes == true
    end

    test "detects PPV disparity" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = PredictiveParity.compute(predictions, labels, sensitive)

      # Group A: TP=3, FP=1 -> PPV=0.75
      # Group B: TP=1, FP=2 -> PPV=0.33
      assert_in_delta(result.disparity, 0.42, 0.05)
      assert result.passes == false
    end

    test "accepts custom threshold" do
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = PredictiveParity.compute(predictions, labels, sensitive, threshold: 0.5)

      assert result.passes == true
      assert result.threshold == 0.5
    end

    test "validates inputs" do
      predictions =
        Nx.tensor([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      labels = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        PredictiveParity.compute(predictions, labels, sensitive)
      end
    end

    test "returns interpretation" do
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = PredictiveParity.compute(predictions, labels, sensitive)

      assert Map.has_key?(result, :interpretation)
      assert is_binary(result.interpretation)
    end

    test "handles edge case: no positive predictions in group" do
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = PredictiveParity.compute(predictions, labels, sensitive)

      # Group B has no positive predictions, so PPV = 0
      assert result.group_b_ppv == 0.0
    end

    test "handles edge case: all predictions correct" do
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = PredictiveParity.compute(predictions, labels, sensitive)

      # Perfect precision for both
      assert result.group_a_ppv == 1.0
      assert result.group_b_ppv == 1.0
      assert result.disparity == 0.0
    end
  end
end
