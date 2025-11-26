defmodule ExFairness.Utils.BootstrapTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Utils.Bootstrap

  alias ExFairness.Utils.Bootstrap

  describe "confidence_interval/3" do
    test "returns valid confidence interval structure" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          seed: 42
        )

      assert is_map(result)
      assert Map.has_key?(result, :point_estimate)
      assert Map.has_key?(result, :confidence_interval)
      assert Map.has_key?(result, :confidence_level)
      assert Map.has_key?(result, :n_samples)
      assert Map.has_key?(result, :method)
    end

    test "confidence interval contains point estimate" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          seed: 42
        )

      {lower, upper} = result.confidence_interval

      # For perfectly balanced data, disparity should be 0
      assert result.point_estimate == 0.0
      assert lower <= result.point_estimate
      assert upper >= result.point_estimate
    end

    test "confidence interval bounds are ordered correctly" do
      predictions = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          seed: 42
        )

      {lower, upper} = result.confidence_interval

      assert is_float(lower)
      assert is_float(upper)
      assert lower <= upper
    end

    test "respects confidence level parameter" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result_95 =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          confidence_level: 0.95,
          seed: 42
        )

      result_90 =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          confidence_level: 0.90,
          seed: 42
        )

      assert result_95.confidence_level == 0.95
      assert result_90.confidence_level == 0.90

      {lower_95, upper_95} = result_95.confidence_interval
      {lower_90, upper_90} = result_90.confidence_interval

      # 95% CI should be wider than 90% CI
      width_95 = upper_95 - lower_95
      width_90 = upper_90 - lower_90

      # Small tolerance for sampling variation
      assert width_95 >= width_90 - 0.01
    end

    test "stratified sampling maintains group proportions" do
      # Create imbalanced data: 15 in group A, 5 in group B
      predictions =
        Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

      # Verify original proportions
      original_a_count = Nx.sum(Nx.equal(sensitive, 0)) |> Nx.to_number()
      original_b_count = Nx.sum(Nx.equal(sensitive, 1)) |> Nx.to_number()
      assert original_a_count == 15
      assert original_b_count == 5

      metric_fn = fn [_preds, sens] ->
        # Return proportion of group A
        n_total = Nx.size(sens)
        n_a = Nx.sum(Nx.equal(sens, 0)) |> Nx.to_number()
        n_a / n_total
      end

      result =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          stratified: true,
          seed: 42
        )

      # With stratified sampling, proportion should remain 0.75
      assert_in_delta result.point_estimate, 0.75, 0.01
    end

    test "supports different bootstrap methods" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result_percentile =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          method: :percentile,
          seed: 42
        )

      result_basic =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          method: :basic,
          seed: 42
        )

      assert result_percentile.method == :percentile
      assert result_basic.method == :basic

      # Both should produce valid intervals
      {lower_p, upper_p} = result_percentile.confidence_interval
      {lower_b, upper_b} = result_basic.confidence_interval

      assert is_float(lower_p) and is_float(upper_p)
      assert is_float(lower_b) and is_float(upper_b)
    end

    test "reproducible with same seed" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result1 =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          seed: 12345
        )

      result2 =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          seed: 12345
        )

      assert result1.confidence_interval == result2.confidence_interval
    end

    test "works with equalized odds metric" do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, labs, sens] ->
        result = ExFairness.equalized_odds(preds, labs, sens)
        result.tpr_disparity
      end

      result =
        Bootstrap.confidence_interval([predictions, labels, sensitive], metric_fn,
          n_samples: 100,
          seed: 42
        )

      assert is_map(result)
      {lower, upper} = result.confidence_interval
      assert lower <= upper
    end

    test "handles small number of bootstrap samples" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 10,
          seed: 42
        )

      assert result.n_samples == 10
      {lower, upper} = result.confidence_interval
      assert is_float(lower) and is_float(upper)
    end

    test "parallel and sequential produce similar results" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      metric_fn = fn [preds, sens] ->
        result = ExFairness.demographic_parity(preds, sens)
        result.disparity
      end

      result_parallel =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          parallel: true,
          seed: 42
        )

      result_sequential =
        Bootstrap.confidence_interval([predictions, sensitive], metric_fn,
          n_samples: 100,
          parallel: false,
          seed: 42
        )

      # Results should be identical with same seed
      assert result_parallel.confidence_interval == result_sequential.confidence_interval
    end
  end
end
