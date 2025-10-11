defmodule ExFairnessTest do
  use ExUnit.Case
  doctest ExFairness

  test "greets the world" do
    assert ExFairness.hello() == :world
  end
end
