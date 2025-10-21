defmodule ExFairness.Metrics.EqualOpportunityTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Metrics.EqualOpportunity

  alias ExFairness.Metrics.EqualOpportunity

  describe "compute/4" do
    test "detects perfect equal opportunity" do
      # Both groups have same TPR
      predictions =
        Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualOpportunity.compute(predictions, labels, sensitive)

      # Group A: TP=2, FN=2 -> TPR=0.5
      # Group B: TP=2, FN=2 -> TPR=0.5
      assert_in_delta(result.disparity, 0.0, 0.01)
      assert result.passes == true
    end

    test "detects TPR disparity" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualOpportunity.compute(predictions, labels, sensitive)

      # Group A: TP=4, FN=0 -> TPR=1.0
      # Group B: TP=1, FN=3 -> TPR=0.25
      assert_in_delta(result.disparity, 0.75, 0.01)
      assert result.passes == false
    end

    test "accepts custom threshold" do
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualOpportunity.compute(predictions, labels, sensitive, threshold: 0.5)

      assert result.passes == true
      assert result.threshold == 0.5
    end

    test "validates inputs" do
      predictions =
        Nx.tensor([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      labels = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        EqualOpportunity.compute(predictions, labels, sensitive)
      end
    end

    test "returns interpretation" do
      predictions =
        Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualOpportunity.compute(predictions, labels, sensitive)

      assert Map.has_key?(result, :interpretation)
      assert is_binary(result.interpretation)
    end

    test "handles edge case: all positive labels" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualOpportunity.compute(predictions, labels, sensitive)

      # Both groups have TPR computable
      assert Map.has_key?(result, :group_a_tpr)
      assert Map.has_key?(result, :group_b_tpr)
    end

    test "handles edge case: no positive labels" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

      labels = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualOpportunity.compute(predictions, labels, sensitive)

      # TPR should be 0 for both groups (no positives)
      assert result.group_a_tpr == 0.0
      assert result.group_b_tpr == 0.0
      assert result.disparity == 0.0
    end
  end
end
