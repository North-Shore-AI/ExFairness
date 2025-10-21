defmodule ExFairness.Metrics.EqualizedOddsTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Metrics.EqualizedOdds

  alias ExFairness.Metrics.EqualizedOdds

  describe "compute/4" do
    test "detects perfect equalized odds" do
      # Both groups have same TPR and FPR
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualizedOdds.compute(predictions, labels, sensitive)

      # Group A: TP=2, FN=2, FP=2, TN=4 -> TPR=0.5, FPR=0.33
      # Group B: TP=2, FN=2, FP=2, TN=4 -> TPR=0.5, FPR=0.33
      assert_in_delta(result.tpr_disparity, 0.0, 0.01)
      assert_in_delta(result.fpr_disparity, 0.0, 0.01)
      assert result.passes == true
    end

    test "detects TPR disparity" do
      # Group A has higher TPR than Group B
      predictions =
        Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualizedOdds.compute(predictions, labels, sensitive)

      # Group A: TP=4, FN=0 -> TPR=1.0
      # Group B: TP=1, FN=3 -> TPR=0.25
      # TPR disparity = 0.75
      assert_in_delta(result.tpr_disparity, 0.75, 0.01)
      assert result.passes == false
    end

    test "detects FPR disparity" do
      # Group A has higher FPR than Group B
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualizedOdds.compute(predictions, labels, sensitive)

      # Group A: FP=4, TN=4 -> FPR=0.5
      # Group B: FP=0, TN=8 -> FPR=0.0
      # FPR disparity = 0.5
      assert_in_delta(result.fpr_disparity, 0.5, 0.01)
      assert result.passes == false
    end

    test "detects both TPR and FPR disparities" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualizedOdds.compute(predictions, labels, sensitive)

      # Group A: TP=5, FN=0, FP=5, TN=0 -> TPR=1.0, FPR=1.0
      # Group B: TP=0, FN=5, FP=0, TN=5 -> TPR=0.0, FPR=0.0
      assert_in_delta(result.tpr_disparity, 1.0, 0.01)
      assert_in_delta(result.fpr_disparity, 1.0, 0.01)
      assert result.passes == false
    end

    test "accepts custom threshold" do
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualizedOdds.compute(predictions, labels, sensitive, threshold: 0.5)

      # Small disparities should pass with high threshold
      assert result.passes == true
      assert result.threshold == 0.5
    end

    test "validates predictions are binary" do
      predictions =
        Nx.tensor([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      labels = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        EqualizedOdds.compute(predictions, labels, sensitive)
      end
    end

    test "validates labels are binary" do
      predictions =
        Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      labels = Nx.tensor([1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        EqualizedOdds.compute(predictions, labels, sensitive)
      end
    end

    test "validates shapes match" do
      predictions =
        Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      labels = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, ~r/shape mismatch/, fn ->
        EqualizedOdds.compute(predictions, labels, sensitive)
      end
    end

    test "returns interpretation in result" do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualizedOdds.compute(predictions, labels, sensitive)

      assert Map.has_key?(result, :interpretation)
      assert is_binary(result.interpretation)
    end

    test "handles edge case: all positive labels" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

      labels = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualizedOdds.compute(predictions, labels, sensitive)

      # TPR can be computed, FPR should be 0 (no negatives)
      assert Map.has_key?(result, :group_a_tpr)
      assert Map.has_key?(result, :group_a_fpr)
      assert result.group_a_fpr == 0.0
      assert result.group_b_fpr == 0.0
    end

    test "handles edge case: all negative labels" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

      labels = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = EqualizedOdds.compute(predictions, labels, sensitive)

      # FPR can be computed, TPR should be 0 (no positives)
      assert Map.has_key?(result, :group_a_tpr)
      assert Map.has_key?(result, :group_a_fpr)
      assert result.group_a_tpr == 0.0
      assert result.group_b_tpr == 0.0
    end
  end
end
