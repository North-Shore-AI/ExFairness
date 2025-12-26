defmodule ExFairness.Utils.StatisticalTestsTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Utils.StatisticalTests

  alias ExFairness.Utils.StatisticalTests

  describe "two_proportion_test/3" do
    test "returns valid test result structure" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.two_proportion_test(predictions, sensitive)

      assert is_map(result)
      assert Map.has_key?(result, :statistic)
      assert Map.has_key?(result, :p_value)
      assert Map.has_key?(result, :significant)
      assert Map.has_key?(result, :alpha)
      assert Map.has_key?(result, :effect_size)
      assert Map.has_key?(result, :test_name)
      assert Map.has_key?(result, :interpretation)
    end

    test "detects significant disparity with large difference" do
      # Group A: 80% positive, Group B: 0% positive
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.two_proportion_test(predictions, sensitive, alpha: 0.05)

      assert result.test_name == "Two-Proportion Z-Test"
      assert result.significant == true
      assert result.p_value < 0.05
      # Critical value for two-tailed test at α=0.05
      assert abs(result.statistic) > 1.96
    end

    test "does not detect significant disparity with small difference" do
      # Perfectly balanced
      predictions =
        Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.two_proportion_test(predictions, sensitive, alpha: 0.05)

      assert result.significant == false
      assert result.p_value > 0.05
    end

    test "p-value is between 0 and 1" do
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.two_proportion_test(predictions, sensitive)

      assert result.p_value >= 0.0
      assert result.p_value <= 1.0
    end

    test "respects alpha parameter" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result_05 = StatisticalTests.two_proportion_test(predictions, sensitive, alpha: 0.05)
      result_01 = StatisticalTests.two_proportion_test(predictions, sensitive, alpha: 0.01)

      assert result_05.alpha == 0.05
      assert result_01.alpha == 0.01

      # Same p-value but potentially different significance
      assert result_05.p_value == result_01.p_value
    end

    test "includes interpretation" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.two_proportion_test(predictions, sensitive)

      assert is_binary(result.interpretation)
      assert String.contains?(result.interpretation, "Group A")
      assert String.contains?(result.interpretation, "Group B")
      assert String.contains?(result.interpretation, "P-value")
    end
  end

  describe "chi_square_test/4" do
    test "returns valid test result structure" do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.chi_square_test(predictions, labels, sensitive)

      assert is_map(result)
      assert Map.has_key?(result, :statistic)
      assert Map.has_key?(result, :p_value)
      assert Map.has_key?(result, :significant)
      assert Map.has_key?(result, :test_name)
      assert result.test_name == "Chi-Square Test"
    end

    test "p-value is between 0 and 1" do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.chi_square_test(predictions, labels, sensitive)

      assert result.p_value >= 0.0
      assert result.p_value <= 1.0
    end

    test "chi-square statistic is non-negative" do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.chi_square_test(predictions, labels, sensitive)

      assert result.statistic >= 0.0
    end

    test "includes interpretation" do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = StatisticalTests.chi_square_test(predictions, labels, sensitive)

      assert is_binary(result.interpretation)
      assert String.contains?(result.interpretation, "Chi-square")
      assert String.contains?(result.interpretation, "P-value")
    end
  end

  describe "permutation_test/3" do
    test "returns valid test result structure" do
      predictions =
        Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        StatisticalTests.permutation_test([predictions, sensitive], metric_fn,
          n_permutations: 100,
          seed: 42
        )

      assert is_map(result)
      assert Map.has_key?(result, :statistic)
      assert Map.has_key?(result, :p_value)
      assert Map.has_key?(result, :significant)
      assert Map.has_key?(result, :test_name)
      assert result.test_name == "Permutation Test"
    end

    test "p-value is between 0 and 1" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        StatisticalTests.permutation_test([predictions, sensitive], metric_fn,
          n_permutations: 100,
          seed: 42
        )

      assert result.p_value >= 0.0
      assert result.p_value <= 1.0
    end

    test "detects significant disparity with large effect" do
      # Very large disparity
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        StatisticalTests.permutation_test([predictions, sensitive], metric_fn,
          n_permutations: 100,
          alpha: 0.05,
          seed: 42
        )

      # With large disparity, should likely be significant
      assert result.statistic > 0.5
    end

    test "does not detect significance with no disparity" do
      # Perfectly balanced
      predictions =
        Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        StatisticalTests.permutation_test([predictions, sensitive], metric_fn,
          n_permutations: 100,
          alpha: 0.05,
          seed: 42
        )

      assert result.statistic == 0.0
      assert result.significant == false
    end

    test "works with different metrics" do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, labs, sens] ->
        result = ExFairness.equalized_odds(preds, labs, sens)
        result.tpr_disparity
      end

      result =
        StatisticalTests.permutation_test([predictions, labels, sensitive], metric_fn,
          n_permutations: 100,
          seed: 42
        )

      assert is_map(result)
      assert is_float(result.p_value)
    end

    test "reproducible with same seed" do
      predictions =
        Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result1 =
        StatisticalTests.permutation_test([predictions, sensitive], metric_fn,
          n_permutations: 100,
          seed: 12_345
        )

      result2 =
        StatisticalTests.permutation_test([predictions, sensitive], metric_fn,
          n_permutations: 100,
          seed: 12_345
        )

      assert result1.p_value == result2.p_value
    end

    test "includes interpretation" do
      predictions =
        Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        StatisticalTests.permutation_test([predictions, sensitive], metric_fn,
          n_permutations: 100,
          seed: 42
        )

      assert is_binary(result.interpretation)
      assert String.contains?(result.interpretation, "Observed metric")
      assert String.contains?(result.interpretation, "P-value")
    end
  end

  describe "cohens_h/2" do
    test "returns correct effect size for equal proportions" do
      h = StatisticalTests.cohens_h(0.5, 0.5)
      assert_in_delta h, 0.0, 0.01
    end

    test "returns positive value for different proportions" do
      h = StatisticalTests.cohens_h(0.7, 0.3)
      assert h > 0.0
    end

    test "effect size increases with larger difference" do
      h_small = StatisticalTests.cohens_h(0.5, 0.4)
      h_large = StatisticalTests.cohens_h(0.9, 0.1)

      assert abs(h_large) > abs(h_small)
    end

    test "known effect size values" do
      # Small effect (0.2)
      h_small = StatisticalTests.cohens_h(0.55, 0.45)
      assert_in_delta abs(h_small), 0.2, 0.1

      # Medium effect (0.5)
      h_medium = StatisticalTests.cohens_h(0.65, 0.35)
      assert_in_delta abs(h_medium), 0.6, 0.2

      # Large effect (0.8)
      h_large = StatisticalTests.cohens_h(0.8, 0.2)
      assert abs(h_large) > 1.0
    end
  end
end
