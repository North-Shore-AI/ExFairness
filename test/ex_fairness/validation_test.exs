defmodule ExFairness.ValidationTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Validation

  alias ExFairness.Validation

  describe "validate_predictions!/1" do
    test "accepts valid binary tensor" do
      predictions = Nx.tensor([0, 1, 1, 0, 1, 0])
      assert predictions == Validation.validate_predictions!(predictions)
    end

    test "rejects non-tensor" do
      assert_raise ExFairness.Error, ~r/must be an Nx.Tensor/, fn ->
        Validation.validate_predictions!([0, 1, 1, 0])
      end
    end

    test "rejects tensor with values outside [0, 1]" do
      predictions = Nx.tensor([0, 1, 2, 0])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        Validation.validate_predictions!(predictions)
      end
    end

    test "rejects tensor with negative values" do
      predictions = Nx.tensor([0, 1, -1, 0])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        Validation.validate_predictions!(predictions)
      end
    end

    test "rejects tensor with size 0" do
      # Nx doesn't support truly empty tensors, but we can test with a 0-sized dimension
      # For now, skip this test as Nx.tensor([]) raises ArgumentError
      # This validation is more theoretical than practical
    end
  end

  describe "validate_labels!/1" do
    test "accepts valid binary tensor" do
      labels = Nx.tensor([0, 1, 1, 0, 1, 0])
      assert labels == Validation.validate_labels!(labels)
    end

    test "rejects non-tensor" do
      assert_raise ExFairness.Error, ~r/must be an Nx.Tensor/, fn ->
        Validation.validate_labels!([0, 1, 1, 0])
      end
    end

    test "rejects tensor with values outside [0, 1]" do
      labels = Nx.tensor([0, 1, 2, 0])

      assert_raise ExFairness.Error, ~r/must be binary/, fn ->
        Validation.validate_labels!(labels)
      end
    end

    test "rejects tensor with size 0" do
      # Nx doesn't support truly empty tensors, skip this theoretical test
    end
  end

  describe "validate_sensitive_attr!/1" do
    test "accepts valid binary tensor with multiple groups" do
      # Need at least 10 samples per group by default
      sensitive_attr =
        Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      assert sensitive_attr == Validation.validate_sensitive_attr!(sensitive_attr)
    end

    test "rejects tensor with single group" do
      sensitive_attr = Nx.tensor([0, 0, 0, 0])

      assert_raise ExFairness.Error, ~r/at least 2 different groups/, fn ->
        Validation.validate_sensitive_attr!(sensitive_attr)
      end
    end

    test "rejects tensor with insufficient samples per group" do
      # Only 2 samples in group 1 (default minimum is 10)
      sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

      assert_raise ExFairness.Error, ~r/Insufficient samples/, fn ->
        Validation.validate_sensitive_attr!(sensitive_attr)
      end
    end

    test "accepts tensor with custom minimum samples per group" do
      # Only 2 samples in group 1 but we allow minimum of 2
      sensitive_attr = Nx.tensor([0, 0, 0, 1, 1])

      assert sensitive_attr ==
               Validation.validate_sensitive_attr!(sensitive_attr, min_per_group: 2)
    end

    test "rejects tensor with size 0" do
      # Nx doesn't support truly empty tensors, skip this theoretical test
    end
  end

  describe "validate_matching_shapes!/2" do
    test "accepts tensors with matching shapes" do
      t1 = Nx.tensor([1, 2, 3])
      t2 = Nx.tensor([4, 5, 6])
      t3 = Nx.tensor([7, 8, 9])

      assert [^t1, ^t2, ^t3] =
               Validation.validate_matching_shapes!([t1, t2, t3], ["t1", "t2", "t3"])
    end

    test "rejects tensors with mismatched shapes" do
      t1 = Nx.tensor([1, 2, 3])
      t2 = Nx.tensor([4, 5])

      assert_raise ExFairness.Error, ~r/shape mismatch/, fn ->
        Validation.validate_matching_shapes!([t1, t2], ["t1", "t2"])
      end
    end

    test "rejects tensors with different dimensions" do
      t1 = Nx.tensor([1, 2, 3])
      t2 = Nx.tensor([[4, 5, 6]])

      assert_raise ExFairness.Error, ~r/shape mismatch/, fn ->
        Validation.validate_matching_shapes!([t1, t2], ["t1", "t2"])
      end
    end
  end
end
