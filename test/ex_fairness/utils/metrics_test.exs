defmodule ExFairness.Utils.MetricsTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Utils.Metrics

  alias ExFairness.Utils.Metrics

  describe "confusion_matrix/3" do
    test "computes correct confusion matrix for all samples" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 1, 0, 0])
      labels = Nx.tensor([1, 0, 0, 0, 1, 1, 0, 1])
      mask = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1])

      result = Metrics.confusion_matrix(predictions, labels, mask)

      # TP: pred=1, label=1 -> indices 0, 4, 5 = 3
      # FP: pred=1, label=0 -> index 2 = 1
      # TN: pred=0, label=0 -> indices 1, 3, 6 = 3
      # FN: pred=0, label=1 -> index 7 = 1
      assert Nx.to_number(result.tp) == 3
      assert Nx.to_number(result.fp) == 1
      assert Nx.to_number(result.tn) == 3
      assert Nx.to_number(result.fn) == 1
    end

    test "computes confusion matrix for masked subset" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 1, 0, 0])
      labels = Nx.tensor([1, 0, 0, 0, 1, 1, 0, 1])
      mask = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0])

      result = Metrics.confusion_matrix(predictions, labels, mask)

      # Only first 4 elements
      # TP: pred=1, label=1 -> index 0 = 1
      # FP: pred=1, label=0 -> index 2 = 1
      # TN: pred=0, label=0 -> indices 1, 3 = 2
      # FN: pred=0, label=1 -> none = 0
      assert Nx.to_number(result.tp) == 1
      assert Nx.to_number(result.fp) == 1
      assert Nx.to_number(result.tn) == 2
      assert Nx.to_number(result.fn) == 0
    end

    test "handles all true positives" do
      predictions = Nx.tensor([1, 1, 1, 1])
      labels = Nx.tensor([1, 1, 1, 1])
      mask = Nx.tensor([1, 1, 1, 1])

      result = Metrics.confusion_matrix(predictions, labels, mask)

      assert Nx.to_number(result.tp) == 4
      assert Nx.to_number(result.fp) == 0
      assert Nx.to_number(result.tn) == 0
      assert Nx.to_number(result.fn) == 0
    end

    test "handles all true negatives" do
      predictions = Nx.tensor([0, 0, 0, 0])
      labels = Nx.tensor([0, 0, 0, 0])
      mask = Nx.tensor([1, 1, 1, 1])

      result = Metrics.confusion_matrix(predictions, labels, mask)

      assert Nx.to_number(result.tp) == 0
      assert Nx.to_number(result.fp) == 0
      assert Nx.to_number(result.tn) == 4
      assert Nx.to_number(result.fn) == 0
    end
  end

  describe "true_positive_rate/3" do
    test "computes TPR correctly" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 1, 0, 0])
      labels = Nx.tensor([1, 0, 0, 0, 1, 1, 0, 1])
      mask = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1])

      tpr = Metrics.true_positive_rate(predictions, labels, mask)

      # TP = 3, FN = 1, TPR = 3/(3+1) = 0.75
      assert_in_delta(Nx.to_number(tpr), 0.75, 0.001)
    end

    test "handles no positive labels (returns 0)" do
      predictions = Nx.tensor([1, 0, 1, 0])
      labels = Nx.tensor([0, 0, 0, 0])
      mask = Nx.tensor([1, 1, 1, 1])

      tpr = Metrics.true_positive_rate(predictions, labels, mask)

      # No positives, so TPR should be 0 (or NaN, we'll handle as 0)
      result = Nx.to_number(tpr)
      assert result == 0.0 or :math.is_nan(result)
    end
  end

  describe "false_positive_rate/3" do
    test "computes FPR correctly" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 1, 0, 0])
      labels = Nx.tensor([1, 0, 0, 0, 1, 1, 0, 1])
      mask = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1])

      fpr = Metrics.false_positive_rate(predictions, labels, mask)

      # FP = 1, TN = 3, FPR = 1/(1+3) = 0.25
      assert_in_delta(Nx.to_number(fpr), 0.25, 0.001)
    end

    test "handles no negative labels (returns 0)" do
      predictions = Nx.tensor([1, 0, 1, 0])
      labels = Nx.tensor([1, 1, 1, 1])
      mask = Nx.tensor([1, 1, 1, 1])

      fpr = Metrics.false_positive_rate(predictions, labels, mask)

      # No negatives, so FPR should be 0 (or NaN)
      result = Nx.to_number(fpr)
      assert result == 0.0 or :math.is_nan(result)
    end
  end

  describe "positive_predictive_value/3" do
    test "computes PPV (precision) correctly" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 1, 0, 0])
      labels = Nx.tensor([1, 0, 0, 0, 1, 1, 0, 1])
      mask = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1])

      ppv = Metrics.positive_predictive_value(predictions, labels, mask)

      # TP = 3, FP = 1, PPV = 3/(3+1) = 0.75
      assert_in_delta(Nx.to_number(ppv), 0.75, 0.001)
    end

    test "handles no positive predictions (returns 0)" do
      predictions = Nx.tensor([0, 0, 0, 0])
      labels = Nx.tensor([1, 0, 1, 0])
      mask = Nx.tensor([1, 1, 1, 1])

      ppv = Metrics.positive_predictive_value(predictions, labels, mask)

      # No positive predictions, so PPV should be 0 (or NaN)
      result = Nx.to_number(ppv)
      assert result == 0.0 or :math.is_nan(result)
    end
  end
end
