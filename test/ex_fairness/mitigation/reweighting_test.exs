defmodule ExFairness.Mitigation.ReweightingTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Mitigation.Reweighting

  alias ExFairness.Mitigation.Reweighting

  describe "compute_weights/3" do
    test "computes demographic parity weights correctly" do
      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      weights = Reweighting.compute_weights(labels, sensitive, target: :demographic_parity)

      # Weights should be normalized
      assert Nx.size(weights) == 20
      # Mean should be approximately 1.0
      mean_weight = Nx.mean(weights) |> Nx.to_number()
      assert_in_delta(mean_weight, 1.0, 0.01)
    end

    test "computes equalized odds weights correctly" do
      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      weights = Reweighting.compute_weights(labels, sensitive, target: :equalized_odds)

      assert Nx.size(weights) == 20
      mean_weight = Nx.mean(weights) |> Nx.to_number()
      assert_in_delta(mean_weight, 1.0, 0.01)
    end

    test "weights are all positive" do
      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      weights = Reweighting.compute_weights(labels, sensitive)

      min_weight = Nx.reduce_min(weights) |> Nx.to_number()
      assert min_weight > 0.0
    end

    test "returns tensor of correct shape" do
      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      weights = Reweighting.compute_weights(labels, sensitive)

      assert Nx.shape(weights) == Nx.shape(labels)
    end

    test "validates inputs" do
      labels = Nx.tensor([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        Reweighting.compute_weights(labels, sensitive)
      end
    end

    test "handles balanced data gracefully" do
      # Perfectly balanced data - all weights should be close to 1.0
      labels = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      weights = Reweighting.compute_weights(labels, sensitive)

      # All weights should be close to 1.0 for balanced data
      mean_weight = Nx.mean(weights) |> Nx.to_number()
      std_weight = Nx.standard_deviation(weights) |> Nx.to_number()

      assert_in_delta(mean_weight, 1.0, 0.01)
      # Low variance
      assert std_weight < 0.5
    end

    test "default target is demographic parity" do
      labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      weights1 = Reweighting.compute_weights(labels, sensitive)
      weights2 = Reweighting.compute_weights(labels, sensitive, target: :demographic_parity)

      # Should be identical
      assert Nx.all(Nx.equal(weights1, weights2)) |> Nx.to_number() == 1
    end
  end
end
