defmodule ExFairness.ConformanceTest do
  @moduledoc """
  Conformance tests for Crucible.Stage behaviour implementation.

  These tests validate that ExFairness.CrucibleStage adheres to the
  canonical schema format defined in the Stage describe/1 contract.
  """
  use ExUnit.Case, async: true

  alias ExFairness.CrucibleStage

  describe "stage conformance" do
    test "implements Crucible.Stage behaviour" do
      assert function_exported?(CrucibleStage, :run, 2)
      assert function_exported?(CrucibleStage, :describe, 1)
    end

    test "describe/1 returns valid canonical schema" do
      schema = CrucibleStage.describe(%{})

      # Must use :name, not :stage
      assert Map.has_key?(schema, :name)
      refute Map.has_key?(schema, :stage)

      # Name must be atom
      assert is_atom(schema.name)
      assert schema.name == :fairness

      # Required core fields exist
      assert Map.has_key?(schema, :description)
      assert Map.has_key?(schema, :required)
      assert Map.has_key?(schema, :optional)
      assert Map.has_key?(schema, :types)

      # No overlap between required and optional
      overlap =
        MapSet.intersection(
          MapSet.new(schema.required),
          MapSet.new(schema.optional)
        )

      assert MapSet.size(overlap) == 0
    end

    test "schema has valid __schema_version__" do
      schema = CrucibleStage.describe(%{})

      assert Map.has_key?(schema, :__schema_version__)
      assert is_binary(schema.__schema_version__)
      # Should be semver format
      assert schema.__schema_version__ =~ ~r/^\d+\.\d+\.\d+$/
    end

    test "all required fields have types defined" do
      schema = CrucibleStage.describe(%{})

      for key <- schema.required do
        assert Map.has_key?(schema.types, key),
               "Required field #{key} missing from types"
      end
    end

    test "all optional fields have types defined" do
      schema = CrucibleStage.describe(%{})

      for key <- schema.optional do
        assert Map.has_key?(schema.types, key),
               "Optional field #{key} missing from types"
      end
    end

    test "defaults only contain optional keys" do
      schema = CrucibleStage.describe(%{})

      if Map.has_key?(schema, :defaults) do
        for key <- Map.keys(schema.defaults) do
          assert key in schema.optional,
                 "Default key #{key} is not in optional list"
        end
      end
    end

    test "extensions are namespaced under fairness" do
      schema = CrucibleStage.describe(%{})

      if Map.has_key?(schema, :__extensions__) do
        assert is_map(schema.__extensions__)
        # Fairness-specific metadata should be under :fairness namespace
        assert Map.has_key?(schema.__extensions__, :fairness)
      end
    end

    test "description is a non-empty string" do
      schema = CrucibleStage.describe(%{})

      assert is_binary(schema.description)
      assert String.length(schema.description) > 0
    end

    test "required is a list of atoms" do
      schema = CrucibleStage.describe(%{})

      assert is_list(schema.required)

      for key <- schema.required do
        assert is_atom(key), "Required key #{inspect(key)} is not an atom"
      end
    end

    test "optional is a list of atoms" do
      schema = CrucibleStage.describe(%{})

      assert is_list(schema.optional)

      for key <- schema.optional do
        assert is_atom(key), "Optional key #{inspect(key)} is not an atom"
      end
    end

    test "types is a map with atom keys" do
      schema = CrucibleStage.describe(%{})

      assert is_map(schema.types)

      for key <- Map.keys(schema.types) do
        assert is_atom(key), "Types key #{inspect(key)} is not an atom"
      end
    end

    test "version is present and is a string" do
      schema = CrucibleStage.describe(%{})

      assert Map.has_key?(schema, :version)
      assert is_binary(schema.version)
    end
  end
end
