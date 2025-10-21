defmodule ExFairness.UtilsTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Utils

  alias ExFairness.Utils

  describe "positive_rate/2" do
    test "computes positive rate for all samples when mask is all 1s" do
      predictions = Nx.tensor([1, 0, 1, 1, 0, 1, 0, 1])
      mask = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1])

      rate = Utils.positive_rate(predictions, mask)
      assert_in_delta(Nx.to_number(rate), 0.625, 0.001)
    end

    test "computes positive rate for masked subset" do
      predictions = Nx.tensor([1, 0, 1, 1, 0, 1, 0, 1])
      mask = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0])

      # First 4 elements: [1, 0, 1, 1] -> 3 out of 4 = 0.75
      rate = Utils.positive_rate(predictions, mask)
      assert_in_delta(Nx.to_number(rate), 0.75, 0.001)
    end

    test "handles all zeros predictions" do
      predictions = Nx.tensor([0, 0, 0, 0])
      mask = Nx.tensor([1, 1, 1, 1])

      rate = Utils.positive_rate(predictions, mask)
      assert_in_delta(Nx.to_number(rate), 0.0, 0.001)
    end

    test "handles all ones predictions" do
      predictions = Nx.tensor([1, 1, 1, 1])
      mask = Nx.tensor([1, 1, 1, 1])

      rate = Utils.positive_rate(predictions, mask)
      assert_in_delta(Nx.to_number(rate), 1.0, 0.001)
    end

    test "handles single element" do
      predictions = Nx.tensor([1])
      mask = Nx.tensor([1])

      rate = Utils.positive_rate(predictions, mask)
      assert_in_delta(Nx.to_number(rate), 1.0, 0.001)
    end
  end

  describe "create_group_mask/2" do
    test "creates mask for group 0" do
      sensitive_attr = Nx.tensor([0, 0, 1, 1, 0, 1])
      mask = Utils.create_group_mask(sensitive_attr, 0)

      expected = Nx.tensor([1, 1, 0, 0, 1, 0])
      assert Nx.equal(mask, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "creates mask for group 1" do
      sensitive_attr = Nx.tensor([0, 0, 1, 1, 0, 1])
      mask = Utils.create_group_mask(sensitive_attr, 1)

      expected = Nx.tensor([0, 0, 1, 1, 0, 1])
      assert Nx.equal(mask, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles all same group" do
      sensitive_attr = Nx.tensor([0, 0, 0, 0])
      mask = Utils.create_group_mask(sensitive_attr, 0)

      expected = Nx.tensor([1, 1, 1, 1])
      assert Nx.equal(mask, expected) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "group_count/2" do
    test "counts samples in group 0" do
      sensitive_attr = Nx.tensor([0, 0, 1, 1, 0, 1])
      count = Utils.group_count(sensitive_attr, 0)

      assert Nx.to_number(count) == 3
    end

    test "counts samples in group 1" do
      sensitive_attr = Nx.tensor([0, 0, 1, 1, 0, 1])
      count = Utils.group_count(sensitive_attr, 1)

      assert Nx.to_number(count) == 3
    end

    test "counts zero when group not present" do
      sensitive_attr = Nx.tensor([0, 0, 0, 0])
      count = Utils.group_count(sensitive_attr, 1)

      assert Nx.to_number(count) == 0
    end
  end
end
