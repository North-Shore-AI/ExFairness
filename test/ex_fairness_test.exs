defmodule ExFairnessTest do
  use ExUnit.Case, async: true
  doctest ExFairness

  describe "demographic_parity/3" do
    test "delegates to DemographicParity module" do
      predictions =
        Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = ExFairness.demographic_parity(predictions, sensitive)

      assert result.disparity == 0.0
      assert result.passes == true
    end
  end
end
