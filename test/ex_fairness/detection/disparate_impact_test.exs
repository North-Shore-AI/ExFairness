defmodule ExFairness.Detection.DisparateImpactTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Detection.DisparateImpact

  alias ExFairness.Detection.DisparateImpact

  describe "detect/3" do
    test "passes the 80% rule with equal rates" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = DisparateImpact.detect(predictions, sensitive)

      # Both groups: 5/10 = 0.5, ratio = 1.0
      assert result.ratio == 1.0
      assert result.passes_80_percent_rule == true
    end

    test "fails the 80% rule with significant disparity" do
      # Group A: 8/10 = 0.8, Group B: 2/10 = 0.2
      # Ratio = 0.2/0.8 = 0.25 < 0.8
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = DisparateImpact.detect(predictions, sensitive)

      assert_in_delta(result.ratio, 0.25, 0.01)
      assert result.passes_80_percent_rule == false
    end

    test "passes at exactly 80%" do
      # Group A: 10/10 = 1.0, Group B: 8/10 = 0.8
      # Ratio = 0.8/1.0 = 0.8 (exactly at threshold)
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = DisparateImpact.detect(predictions, sensitive)

      assert_in_delta(result.ratio, 0.8, 0.01)
      assert result.passes_80_percent_rule == true
    end

    test "handles reverse disparity (minority favored)" do
      # Group A: 2/10 = 0.2, Group B: 8/10 = 0.8
      # Ratio = 0.2/0.8 = 0.25 (computed as min/max, so same as above)
      predictions =
        Nx.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = DisparateImpact.detect(predictions, sensitive)

      # Should compute ratio as min/max to detect disparity in either direction
      assert result.ratio < 0.8
      assert result.passes_80_percent_rule == false
    end

    test "includes interpretation" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = DisparateImpact.detect(predictions, sensitive)

      assert Map.has_key?(result, :interpretation)
      assert is_binary(result.interpretation)
    end

    test "includes selection rates" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = DisparateImpact.detect(predictions, sensitive)

      assert Map.has_key?(result, :group_a_rate)
      assert Map.has_key?(result, :group_b_rate)
      assert result.group_a_rate == 0.5
      assert result.group_b_rate == 0.5
    end

    test "validates inputs" do
      predictions =
        Nx.tensor([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        DisparateImpact.detect(predictions, sensitive)
      end
    end

    test "handles edge case: all zeros" do
      predictions =
        Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = DisparateImpact.detect(predictions, sensitive)

      # Both rates are 0, ratio is 1.0 (no disparity when both are 0)
      assert result.ratio == 1.0
      assert result.passes_80_percent_rule == true
    end

    test "handles edge case: all ones" do
      predictions =
        Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = DisparateImpact.detect(predictions, sensitive)

      # Both rates are 1.0, ratio is 1.0
      assert result.ratio == 1.0
      assert result.passes_80_percent_rule == true
    end
  end
end
